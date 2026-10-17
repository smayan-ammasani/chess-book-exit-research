#!/usr/bin/env python3
"""
Script 03: Stockfish Value-of-Computation Covariate
==============================================================================
PURPOSE
    For each unique board position appearing as a last-book or first-OOB
    position in boundary_pairs.csv, run Stockfish at depth 1 and depth 15
    and compute the "value of computation" (VOC):

        VOC = |eval(depth_15) - eval(depth_1)|    (in centipawns)

    A large VOC means additional thinking substantially changes the evaluation
    — the position objectively rewards deeper calculation. A small VOC means
    the position is clear even at shallow analysis.

    This covariate is related to, but not the same as, the measure in
    Russek et al. (2025, Cognitive Science), who compared the deep-search
    value of the best move with the deep-search value of the move a depth-1
    search would choose.
    By including VOC as a control variable in the mixed-effects model
    (Script 05), this paper isolates the book-exit memory cost from the
    generic position difficulty cost — the key methodological contribution
    over prior work. If the book-exit time spike survives after controlling
    for VOC, it reflects a memory-specific cost rather than just a complexity
    spike.

INPUT
    data/processed/boundary_pairs.csv        — output of Script 02

OUTPUT
    data/processed/stockfish_covariate.csv
    Columns:
        fen_normalized   Normalized FEN (4 fields: position, turn, castling, en passant)
        voc              |eval_depth15 - eval_depth1| in centipawns
        eval_depth1      Stockfish centipawn eval at depth 1 (White perspective)
        eval_depth15     Stockfish centipawn eval at depth 15 (White perspective)

KEY EFFICIENCY
    22.8M boundary pairs produce only 114,083 UNIQUE positions (opening
    positions repeat heavily across games). This script analyzes each unique
    position ONCE and caches the result, cutting Stockfish calls from ~45M
    to ~114K — a 400x reduction. Script 04 joins the results back to the
    full boundary_pairs.csv using the normalized FEN as a key.

ESTIMATED RUNTIME
    3–8 hours depending on CPU speed. Safe to interrupt with Ctrl+C —
    checkpoints save every 500 positions (~every few minutes).
==============================================================================
"""

import csv
import logging
import pickle
import sys
from pathlib import Path

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

# ── Project root on path so we can import config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    STOCKFISH_PATH,
    STOCKFISH_DEPTH_SHALLOW,
    STOCKFISH_DEPTH_DEEP,
    STOCKFISH_EVAL_CAP_CP,
    STOCKFISH_THREADS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH   = PROCESSED_DIR / 'boundary_pairs.csv'
OUTPUT_PATH  = PROCESSED_DIR / 'stockfish_covariate.csv'
CHECKPOINT_A = PROCESSED_DIR / 'script03_checkpoint_a.pkl'
CHECKPOINT_B = PROCESSED_DIR / 'script03_checkpoint_b.pkl'

CHECKPOINT_INTERVAL = 500   # Save every 500 positions (~every few minutes)
_checkpoint_slot    = 0


# ==============================================================================
# FEN UTILITIES
# ==============================================================================

def normalize_fen(fen: str) -> str:
    """
    Normalize a FEN string to its first 4 space-separated fields:
        1. Piece placement
        2. Active color (w/b)
        3. Castling availability
        4. En passant target square

    Fields 5 (halfmove clock) and 6 (fullmove number) are dropped.
    The same board position reached via different move sequences has
    different move counters but identical first-4-field FENs — normalizing
    ensures all paths to the same position map to the same cache key.

    This is the same normalization used during deduplication in the
    diagnostic that found 114,083 unique positions.
    """
    return ' '.join(fen.split()[:4])


def make_board(normalized_fen: str) -> chess.Board:
    """
    Build a chess.Board from a 4-field normalized FEN by appending
    placeholder move counters '0 1'. These counters don't affect
    Stockfish's evaluation of the position.
    """
    return chess.Board(normalized_fen + ' 0 1')


# ==============================================================================
# PHASE 1: COLLECT UNIQUE POSITIONS
# ==============================================================================

def collect_unique_fens() -> list:
    """
    Stream boundary_pairs.csv in chunks and collect every unique normalized
    FEN from both the last_book_fen and first_oob_fen columns.

    Returns a sorted list for deterministic processing order.
    Processes the full 4.9 GB file without loading it into memory at once.
    """
    log.info('Streaming boundary_pairs.csv to collect unique positions...')
    log.info('(This takes a few minutes for a 4.9 GB file.)')

    unique_fens = set()
    chunk_size  = 100_000

    for i, chunk in enumerate(pd.read_csv(
        INPUT_PATH,
        chunksize=chunk_size,
        usecols=['last_book_fen', 'first_oob_fen'],
    )):
        for col in ['last_book_fen', 'first_oob_fen']:
            unique_fens.update(chunk[col].dropna().apply(normalize_fen))

        if (i + 1) % 50 == 0:
            log.info(
                f'  {(i+1)*chunk_size:,} rows processed — '
                f'{len(unique_fens):,} unique positions so far...'
            )

    result = sorted(unique_fens)
    log.info(f'Total unique positions to analyze: {len(result):,}')
    return result


# ==============================================================================
# PHASE 2: CHECKPOINT SYSTEM
# ==============================================================================

def load_checkpoint() -> dict:
    """
    Load previously computed VOC results from the most recent valid checkpoint.

    Returns a dict mapping normalized_fen → result_dict, where result_dict
    contains 'voc', 'eval_depth1', and 'eval_depth15'.

    Tries both rotating checkpoint files and picks whichever has more entries.
    """
    best = None

    for path in [CHECKPOINT_A, CHECKPOINT_B]:
        if not path.exists():
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            log.info(f'  Found {path.name}: {len(data):,} positions computed')
            if best is None or len(data) > len(best):
                best = data
        except Exception as e:
            log.warning(f'  {path.name} is corrupt: {e}')

    if best is None:
        log.info('No checkpoint found — starting from scratch.')
        return {}

    log.info(f'Resuming with {len(best):,} positions already done.')
    return best


def save_checkpoint(results: dict) -> None:
    """
    Save current results to the next rotating checkpoint file.
    Uses Path.replace() which overwrites atomically on Windows.
    """
    global _checkpoint_slot
    target = CHECKPOINT_A if _checkpoint_slot % 2 == 0 else CHECKPOINT_B
    tmp    = target.with_suffix('.pkl.tmp')

    with open(tmp, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)

    _checkpoint_slot += 1
    log.info(f'  [Checkpoint → {target.name}] {len(results):,} positions saved')


# ==============================================================================
# PHASE 3: STOCKFISH ANALYSIS
# ==============================================================================

def analyze_position(board: chess.Board, engine: chess.engine.SimpleEngine) -> dict:
    """
    Compute the value-of-computation covariate for one board position.

    Runs Stockfish twice:
        1. At STOCKFISH_DEPTH_SHALLOW (depth 1) — simulates no thinking
        2. At STOCKFISH_DEPTH_DEEP (depth 15) — simulates deep thinking

    VOC = |eval_deep - eval_shallow| in centipawns.

    WHY THIS CAPTURES POSITION COMPLEXITY:
        In a "forcing" position where only one move is good, even shallow
        analysis finds it — VOC is near zero. In a "quiet" position with many
        plausible continuations, shallow and deep evals often disagree — VOC
        is large, meaning extra thinking genuinely changes the picture.

    Evaluations are from White's perspective and capped at STOCKFISH_EVAL_CAP_CP
    to prevent mate-distance scores (which can be ±100,000 in python-chess
    units) from distorting the distribution.

    Returns dict with 'voc', 'eval_depth1', 'eval_depth15'.
    Returns zeros for terminal positions or on Stockfish errors.
    """
    if board.is_game_over():
        # Checkmate, stalemate, etc. — no computation is valuable here
        return {'voc': 0.0, 'eval_depth1': 0, 'eval_depth15': 0}

    try:
        # Shallow analysis
        info_s  = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH_SHALLOW), game=board.fen())
        score_s = info_s['score'].white().score(mate_score=STOCKFISH_EVAL_CAP_CP)
        score_s = max(-STOCKFISH_EVAL_CAP_CP, min(STOCKFISH_EVAL_CAP_CP, int(score_s)))

        # Deep analysis
        info_d  = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH_DEEP), game=board.fen())
        score_d = info_d['score'].white().score(mate_score=STOCKFISH_EVAL_CAP_CP)
        score_d = max(-STOCKFISH_EVAL_CAP_CP, min(STOCKFISH_EVAL_CAP_CP, int(score_d)))

        return {
            'voc':          float(abs(score_d - score_s)),
            'eval_depth1':  score_s,
            'eval_depth15': score_d,
        }

    except Exception as e:
        log.warning(f'Stockfish error: {e}')
        return {'voc': 0.0, 'eval_depth1': 0, 'eval_depth15': 0}


def run_analysis(all_fens: list, results: dict) -> dict:
    """
    Run Stockfish on every position not yet in the results cache.
    Saves checkpoints every CHECKPOINT_INTERVAL positions.
    """
    # Locate Stockfish binary — try the configured path, then the .exe variant
    stockfish_path = Path(STOCKFISH_PATH)
    if not stockfish_path.exists():
        exe_path = stockfish_path.with_suffix('.exe')
        if exe_path.exists():
            stockfish_path = exe_path
        else:
            log.error(f'Stockfish binary not found.')
            log.error(f'Tried: {STOCKFISH_PATH}')
            log.error(f'Tried: {exe_path}')
            log.error('Download from stockfishchess.org and update STOCKFISH_PATH in config.py')
            sys.exit(1)

    log.info(f'Stockfish binary: {stockfish_path}')
    log.info(f'Threads: {STOCKFISH_THREADS}')
    log.info(f'Depths: {STOCKFISH_DEPTH_SHALLOW} (shallow) and {STOCKFISH_DEPTH_DEEP} (deep)')

    todo = [fen for fen in all_fens if fen not in results]
    log.info(f'Positions remaining: {len(todo):,}  (already done: {len(all_fens)-len(todo):,})')

    if not todo:
        log.info('All positions already analyzed. Nothing to do.')
        return results

    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
    engine.configure({'Threads': STOCKFISH_THREADS})
    log.info('Stockfish engine started.')

    try:
        with tqdm(
            total=len(todo),
            desc='Stockfish',
            unit=' pos',
            dynamic_ncols=True,
        ) as pbar:
            for i, fen in enumerate(todo):
                try:
                    board      = make_board(fen)
                    results[fen] = analyze_position(board, engine)
                except Exception as e:
                    log.warning(f'Failed on FEN ({fen}): {e}')
                    results[fen] = {'voc': 0.0, 'eval_depth1': 0, 'eval_depth15': 0}

                pbar.update(1)

                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(results)

    finally:
        # Always close the engine cleanly, even on Ctrl+C
        engine.quit()
        log.info('Stockfish engine closed.')

    save_checkpoint(results)
    return results


# ==============================================================================
# PHASE 4: SAVE OUTPUT
# ==============================================================================

def save_output(results: dict) -> None:
    """
    Write the final covariate lookup table to CSV.
    One row per unique normalized FEN position.

    Script 04 will join this table to boundary_pairs.csv on the
    fen_normalized column to add last_book_voc and first_oob_voc
    to every boundary pair record.
    """
    log.info(f'Writing {len(results):,} rows to {OUTPUT_PATH}...')

    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['fen_normalized', 'voc', 'eval_depth1', 'eval_depth15'],
        )
        writer.writeheader()
        for fen, vals in results.items():
            writer.writerow({
                'fen_normalized': fen,
                'voc':            round(vals['voc'], 2),
                'eval_depth1':    vals['eval_depth1'],
                'eval_depth15':   vals['eval_depth15'],
            })

    log.info(f'Saved: {OUTPUT_PATH}')

    # Quick sanity check — print distribution of VOC values
    vocs = [v['voc'] for v in results.values()]
    import statistics
    log.info('')
    log.info('VOC distribution sanity check:')
    log.info(f'  min:    {min(vocs):.1f}')
    log.info(f'  median: {statistics.median(vocs):.1f}')
    log.info(f'  mean:   {statistics.mean(vocs):.1f}')
    log.info(f'  max:    {max(vocs):.1f}')
    log.info(f'  pct with VOC=0: {100*sum(1 for v in vocs if v==0)/len(vocs):.1f}%')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    log.info('Script 03: Stockfish Value-of-Computation Covariate')
    log.info(f'Shallow depth:      {STOCKFISH_DEPTH_SHALLOW}')
    log.info(f'Deep depth:         {STOCKFISH_DEPTH_DEEP}')
    log.info(f'Eval cap:           {STOCKFISH_EVAL_CAP_CP} centipawns')
    log.info(f'Checkpoint every:   {CHECKPOINT_INTERVAL} positions')
    log.info('')

    if not INPUT_PATH.exists():
        log.error(f'Input not found: {INPUT_PATH}')
        log.error('Run Script 02 first.')
        sys.exit(1)

    # ── Phase 1: Collect unique FENs ──────────────────────────────────────────
    all_fens = collect_unique_fens()
    log.info('')

    # ── Check if already complete ─────────────────────────────────────────────
    if OUTPUT_PATH.exists():
        try:
            existing_count = sum(1 for _ in open(OUTPUT_PATH, encoding='utf-8')) - 1
            if existing_count >= len(all_fens):
                log.info(f'Output already complete ({existing_count:,} rows). Nothing to do.')
                log.info(f'Output: {OUTPUT_PATH}')
                log.info('Next step: python scripts/04_assemble_dataset.py')
                return
            else:
                log.info(f'Output exists but has only {existing_count:,} rows — continuing.')
        except Exception:
            pass

    # ── Phase 2: Load checkpoint ──────────────────────────────────────────────
    log.info('Checking for checkpoint...')
    results = load_checkpoint()
    log.info('')

    # ── Phase 3: Stockfish analysis ───────────────────────────────────────────
    log.info('Running Stockfish analysis...')
    log.info('Safe to interrupt with Ctrl+C — progress saves every 500 positions.')
    log.info('')
    results = run_analysis(all_fens, results)

    # ── Phase 4: Save output ──────────────────────────────────────────────────
    log.info('')
    log.info('Saving output...')
    save_output(results)

    # Clean up checkpoint files now that we have the final output
    for ck in [CHECKPOINT_A, CHECKPOINT_B]:
        if ck.exists():
            ck.unlink()
            log.info(f'Checkpoint deleted: {ck.name}')

    log.info('')
    log.info('Script 03 complete.')
    log.info(f'Output: {OUTPUT_PATH}')
    log.info('Next step: python scripts/04_assemble_dataset.py')


if __name__ == '__main__':
    main()
