#!/usr/bin/env python3
"""
Script 02: Parse and Tag Games
==============================================================================
PURPOSE
    Download the December 2024 Lichess Rapid dataset (ALL rating bands),
    compute thinking time per move from embedded clock annotations, classify
    each move as "book" or "out of book" using the reference corpus built in
    Script 01, identify each player's book-exit boundary (last book move →
    first out-of-book move), and write one record per boundary pair to a CSV.

    This is the main analysis dataset. Unlike Script 01 (which filtered for
    2200+ only to build a reference corpus), this script keeps all rating
    bands so the book-exit time cost can be compared across skill levels.

INPUT
    data/reference_corpus/book_sequences.csv    output of Script 01
    Lichess monthly PGN dump for ANALYSIS_MONTH (downloaded here)

OUTPUT
    data/processed/boundary_pairs.csv
    One row per player per game where a valid book-exit transition exists.
    Two rows per game when both White and Black have valid transitions.

THINKING TIME FORMULA
    Lichess %clk annotations show remaining time AFTER the move and AFTER
    the increment has been added. Therefore:

        curr_clock = prev_clock - thinking_time + increment
        → thinking_time = prev_clock - curr_clock + increment

    Where prev_clock is base_time for the first move of each color,
    or the clock annotation from that player's previous move otherwise.

SAFE TO INTERRUPT
    Checkpoints save every 25,000 games. Restart resumes from the last
    checkpoint with no lost work beyond ~60 seconds of processing.

ESTIMATED RUNTIME
    Download: 1–4 hours (~29 GB).
    Processing: 2–4 days (streaming about 97 million games).
==============================================================================
"""

import csv
import io
import logging
import pickle
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import pandas as pd
import requests
import zstandard as zstd
from tqdm import tqdm

# ── Project root on path so config can be imported ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ANALYSIS_MONTH,
    ALLOWED_TIME_CONTROLS,
    MIN_GAMES_PLAYED_PROXY,
    MIN_PLIES_FOR_INCLUSION,
    PREMOVE_THRESHOLD_SECONDS,
    PRIMARY_BOOK_THRESHOLD,
    RATING_BANDS,
    MAX_GAMES_TO_PARSE,
)

# ── Transition-window extraction ─────────────────────────────────────────────
# Window is expressed in the player's OWN moves, not absolute plies.
# 0 = first out-of-book move; negative = still in book.
WINDOW_BEFORE = -5
WINDOW_AFTER  = 10

# Writing the full window for all 22.8M boundary pairs would produce roughly
# 365M rows (30-40 GB). Sample instead: strides below target about 200,000
# player-games per band, derived from the band counts measured on the previous
# run (band_1 4,597,238 / band_2 8,479,639 / band_3 8,010,394 /
# band_4 1,641,301 / band_5 140,423). Band 5 is taken in full.
WINDOW_STRIDES = {
    'band_1': 22,
    'band_2': 42,
    'band_3': 40,
    'band_4': 8,
    'band_5': 1,
}

WINDOW_CSV_COLUMNS = [
    'game_id', 'player_color', 'player_rating', 'rating_band',
    'player_username', 'ply_at_exit', 'relative_move', 'absolute_ply',
    'thinking_time', 'is_premove', 'in_book', 'move_uci', 'time_control',
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / 'data' / 'raw'
CORPUS_DIR    = PROJECT_ROOT / 'data' / 'reference_corpus'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_PATH = RAW_DIR       / f'lichess_db_standard_rated_{ANALYSIS_MONTH}.pgn.zst'
BOOK_CSV_PATH = CORPUS_DIR    / 'book_sequences.csv'
OUTPUT_PATH   = PROCESSED_DIR / 'boundary_pairs.csv'
WINDOW_OUTPUT_PATH = PROCESSED_DIR / 'transition_window.csv'
CHECKPOINT_A  = PROCESSED_DIR / 'script02_checkpoint_a.pkl'
CHECKPOINT_B  = PROCESSED_DIR / 'script02_checkpoint_b.pkl'

CHECKPOINT_INTERVAL = 25_000   # Save every 25K games (~60 sec at ~400 games/sec)
_checkpoint_slot    = 0        # Alternates between 0 (→ A) and 1 (→ B)

# Fixed column order for the output CSV
CSV_COLUMNS = [
    'game_id', 'player_color', 'player_rating', 'rating_band',
    'player_username', 'ply_at_exit',
    'last_book_move_uci', 'first_oob_move_uci',
    'last_book_thinking_time', 'first_oob_thinking_time',
    'last_book_is_premove', 'first_oob_is_premove',
    'premove_suspect',
    'time_control', 'game_result',
    'last_book_fen', 'first_oob_fen',
]


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# Matches Lichess clock annotations: { [%clk H:MM:SS] }
_CLK_RE = re.compile(r'\[%clk (\d+):(\d+):(\d+)\]')


def parse_clock(comment: str):
    """
    Extract remaining clock time in seconds from a Lichess move comment.

    Lichess annotates each move: { [%clk 0:09:55] } where the time shown
    is the clock AFTER the move and AFTER the increment has been added.

    Returns float seconds, or None if no annotation is present.
    """
    if not comment:
        return None
    m = _CLK_RE.search(comment)
    if not m:
        return None
    h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return float(h * 3600 + mn * 60 + s)


def parse_time_control(tc_string: str):
    """
    Parse a Lichess time control string into (base_seconds, increment_seconds).

    "600+5"  → (600, 5)
    "900+0"  → (900, 0)
    Returns (0, 0) on failure.
    """
    if not tc_string or '+' not in tc_string:
        return 0, 0
    try:
        base, inc = tc_string.split('+', 1)
        return int(base), int(inc)
    except (ValueError, IndexError):
        return 0, 0


def get_rating_band(rating: int):
    """
    Assign a rating to one of the five pre-defined bands from config.py.
    Returns the band name string, or None if it fits no band.
    """
    for band_name, (low, high) in RATING_BANDS.items():
        if low <= rating <= high:
            return band_name
    return None


# ==============================================================================
# BOOK SET LOADING
# ==============================================================================

def load_book_set(csv_path: Path, threshold: int) -> set:
    """
    Load book_sequences.csv and build a Python set of in-book sequences.

    A sequence is "in book" if it appeared at least `threshold` times in
    the June 2024 reference corpus of 2200+ Rapid games.

    Each sequence is stored as an immutable tuple of UCI move strings,
    e.g. ('e2e4', 'e7e5', 'g1f3'), allowing O(1) lookup during game
    processing. With 14,533 sequences at threshold 10, this set is small
    (a few MB) and fast.

    WHY NOT ECO CODES: see the Script 01 docstring. In short, ECO codes
    stop classifying lines after 10-20 moves regardless of how deeply
    players prepare, while a frequency-based book follows the lines that
    strong players actually play.
    """
    log.info(f'Loading book sequences from {csv_path}...')
    df = pd.read_csv(csv_path)
    filtered = df[df['count'] >= threshold]
    book_set  = {tuple(seq.split()) for seq in filtered['move_sequence']}
    log.info(f'Loaded {len(book_set):,} book sequences (count >= {threshold})')
    return book_set


# ==============================================================================
# DOWNLOAD (identical logic to Script 01)
# ==============================================================================

def download_with_resume(month: str, local_path: Path) -> None:
    """
    Download a Lichess monthly database file with resume and auto-retry.

    If the connection drops mid-download, the script waits 30 seconds and
    resumes from the last successfully written byte using an HTTP Range
    header. Up to 20 retries before giving up.
    """
    url = (
        f'https://database.lichess.org/standard/'
        f'lichess_db_standard_rated_{month}.pgn.zst'
    )

    MAX_RETRIES = 20
    RETRY_DELAY = 30

    log.info(f'URL: {url}')
    log.info('The download resumes automatically if the connection drops.')

    attempt = 0
    while attempt < MAX_RETRIES:
        existing_bytes = local_path.stat().st_size if local_path.exists() else 0

        if existing_bytes > 0 and attempt == 0:
            log.info(f'Resuming from {existing_bytes / 1e9:.2f} GB...')
        elif attempt > 0:
            log.info(f'Reconnecting from {existing_bytes / 1e9:.2f} GB (retry {attempt})...')

        headers = {}
        if existing_bytes > 0:
            headers['Range'] = f'bytes={existing_bytes}-'

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=60)
        except requests.exceptions.ConnectionError:
            attempt += 1
            log.warning(f'Cannot connect. Retry {attempt}/{MAX_RETRIES} in {RETRY_DELAY}s...')
            time.sleep(RETRY_DELAY)
            continue

        if response.status_code == 416:
            log.info('File already fully downloaded.')
            return

        if response.status_code not in (200, 206):
            log.error(f'HTTP {response.status_code}. Cannot download.')
            sys.exit(1)

        content_length = int(response.headers.get('content-length', 0))
        total_bytes    = existing_bytes + content_length

        if attempt == 0:
            log.info(f'Saving to: {local_path}')
            log.info(f'Total size: ~{total_bytes / 1e9:.1f} GB')

        write_mode = 'ab' if existing_bytes > 0 else 'wb'

        try:
            with open(local_path, write_mode) as f:
                with tqdm(
                    total=total_bytes, initial=existing_bytes,
                    unit='B', unit_scale=True, unit_divisor=1024,
                    desc='Downloading', dynamic_ncols=True,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=65_536):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            log.info('Download complete.')
            return

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            ConnectionResetError,
        ):
            attempt += 1
            new_size = local_path.stat().st_size if local_path.exists() else 0
            log.warning(
                f'Connection dropped at {new_size / 1e9:.2f} GB. '
                f'Retry {attempt}/{MAX_RETRIES} in {RETRY_DELAY}s...'
            )
            time.sleep(RETRY_DELAY)

    log.error(f'Download failed after {MAX_RETRIES} retries.')
    sys.exit(1)


# ==============================================================================
# GAME FILTER
# ==============================================================================

def game_passes_filter(headers: dict):
    """
    Return (True, '') if this game should be included in the analysis sample.
    Return (False, reason_string) if it should be excluded.

    KEY DIFFERENCE FROM SCRIPT 01: no rating minimum. All five rating bands
    are kept so the book-exit time cost can be compared across skill levels.

    Inclusion criteria:
    1. Rated Rapid game (not Blitz, Bullet, Classical, or Casual)
    2. Time control is one of the allowed Rapid controls (from config.py)
    3. Both players have parseable integer ratings
    4. RatingDiff headers present (proxy for rating reliability; see config.py)
    5. Game ended normally or by time forfeit; not abandoned or rules infraction
    """
    if headers.get('Event', '') != 'Rated Rapid game':
        return False, 'not_rapid'

    if headers.get('TimeControl', '') not in ALLOWED_TIME_CONTROLS:
        return False, 'time_control_excluded'

    try:
        int(headers.get('WhiteElo', ''))
        int(headers.get('BlackElo', ''))
    except (ValueError, TypeError):
        return False, 'missing_or_unparseable_rating'

    if MIN_GAMES_PLAYED_PROXY:
        wrd = headers.get('WhiteRatingDiff', '')
        brd = headers.get('BlackRatingDiff', '')
        if not wrd or wrd == '?' or not brd or brd == '?':
            return False, 'no_rating_diff'

    termination = headers.get('Termination', '').lower()
    if 'abandoned' in termination or 'rules infraction' in termination:
        return False, 'abnormal_termination'

    return True, ''


import zlib


def _in_window_sample(game_id: str, color_name: str, band: str) -> bool:
    """
    Deterministic sampling gate.

    Uses a CRC32 of the game/colour key rather than a running counter, so the
    same player-games are selected regardless of how many times the run is
    interrupted and resumed from a checkpoint. Python's built-in hash() is
    randomised per process for strings and would NOT be reproducible here.
    """
    stride = WINDOW_STRIDES.get(band, 1)
    if stride <= 1:
        return True
    return zlib.crc32(f'{game_id}:{color_name}'.encode()) % stride == 0


def extract_window_rows(game_id, color_name, rating, band, username,
                        player_plies, first_oob_ply, thinking_times,
                        book_at_ply, moves_uci, tc_str) -> list:
    """
    Build long-format rows for moves WINDOW_BEFORE..WINDOW_AFTER around the
    book exit, in this player's own move sequence.

    Positions that fall outside the game (before move 1, or after the game
    ended) are simply absent. The window is ragged by design, and the
    analysis must handle unbalanced panels rather than assume 16 rows per
    player-game.
    """
    if not _in_window_sample(game_id, color_name, band):
        return []

    try:
        exit_idx = player_plies.index(first_oob_ply)
    except ValueError:
        return []

    rows = []
    for rel in range(WINDOW_BEFORE, WINDOW_AFTER + 1):
        idx = exit_idx + rel
        if idx < 0 or idx >= len(player_plies):
            continue

        p = player_plies[idx]
        t = thinking_times[p]
        if t is None or t < 0:
            continue

        rows.append({
            'game_id':         game_id,
            'player_color':    color_name,
            'player_rating':   rating,
            'rating_band':     band,
            'player_username': username,
            'ply_at_exit':     first_oob_ply + 1,
            'relative_move':   rel,
            'absolute_ply':    p + 1,
            'thinking_time':   round(t, 3),
            'is_premove':      t <= PREMOVE_THRESHOLD_SECONDS,
            'in_book':         book_at_ply[p],
            'move_uci':        moves_uci[p],
            'time_control':    tc_str,
        })

    return rows

# ==============================================================================
# GAME PROCESSING
# ==============================================================================

def process_game(game: chess.pgn.Game, headers: dict, book_set: set) -> list:
    """
    Walk through one game move by move and return 0, 1, or 2 boundary pair
    records, one per player where a valid book-exit transition was found.

    BOOK STATUS:
        A move at ply P is "in book" if the full game sequence from the
        start up to and including that move appears in book_set as a tuple
        of UCI strings. This is a single O(1) set lookup.

        Once a player's first out-of-book move is found, the search stops
        for that player: all later moves are necessarily also OOB (the game
        sequence can never re-enter the reference corpus).

    THINKING TIME:
        Lichess %clk annotations show time AFTER the move and AFTER
        increment. The formula therefore is:

            thinking_time = prev_clock - curr_clock + increment

        For each color's first move, prev_clock = base_time (from the
        time control header). For all later moves, prev_clock = that
        player's previous %clk annotation.

        Derivation:
            curr_clock = prev_clock − thinking_time + increment
            → thinking_time = prev_clock − curr_clock + increment

    BOUNDARY PAIR:
        For each player, the walk locates:
        - last_book_ply : their last ply where the game was still in book
        - first_oob_ply : their immediately following move (first OOB)

        These form the paired comparison at the heart of H1 and H2.
        The FEN positions at both plies are recorded for Stockfish (Script 03).
    """
    game_id  = headers.get('Site', '').rsplit('/', 1)[-1]
    tc_str   = headers.get('TimeControl', '')
    result   = headers.get('Result', '*')

    try:
        white_rating = int(headers.get('WhiteElo', 0))
        black_rating = int(headers.get('BlackElo', 0))
    except (ValueError, TypeError):
        return [], []

    white_username = headers.get('White', '')
    black_username = headers.get('Black', '')

    base_time, increment = parse_time_control(tc_str)
    if base_time == 0:
        return [], []

    # ── Walk through all moves ────────────────────────────────────────────────
    board            = game.board()
    node             = game
    moves_uci        = []
    thinking_times   = []   # float seconds or None, one per ply
    fens_before      = []   # board FEN before the move at this ply
    book_at_ply      = []   # True if the sequence through this ply is in book
    moves_so_far     = []   # accumulating UCI prefix for book lookup

    white_prev_clock = float(base_time)
    black_prev_clock = float(base_time)

    while node.variations:
        next_node = node.variations[0]
        move      = next_node.move
        if move is None:
            break

        is_white  = (board.turn == chess.WHITE)   # check BEFORE push
        fen_pre   = board.fen()
        curr_clk  = parse_clock(next_node.comment)

        # Thinking time = prev_clock − curr_clock + increment
        if curr_clk is not None:
            if is_white:
                t                = white_prev_clock - curr_clk + increment
                white_prev_clock = curr_clk
            else:
                t                = black_prev_clock - curr_clk + increment
                black_prev_clock = curr_clk
        else:
            t = None

        # Book status: does the full sequence including this move appear in the corpus?
        moves_so_far.append(move.uci())
        in_book = tuple(moves_so_far) in book_set

        fens_before.append(fen_pre)
        moves_uci.append(move.uci())
        thinking_times.append(t)
        book_at_ply.append(in_book)

        try:
            board.push(move)
        except Exception:
            break  # malformed PGN: keep the moves read so far

        node = next_node

    total_plies = len(moves_uci)
    if total_plies < MIN_PLIES_FOR_INCLUSION:
        return [], []

    # ── Find boundary pair for each player ───────────────────────────────────
    records = []
    window_records = []

    for is_white_player, color_name, rating, username in [
        (True,  'white', white_rating, white_username),
        (False, 'black', black_rating, black_username),
    ]:
        band = get_rating_band(rating)
        if band is None:
            continue

        # Plies where this player moved (White = even plies, Black = odd plies,
        # both 0-indexed from game start)
        player_plies = [p for p in range(total_plies)
                        if (p % 2 == 0) == is_white_player]

        last_book_ply = None
        first_oob_ply = None

        for p in player_plies:
            if book_at_ply[p]:
                last_book_ply = p
            elif first_oob_ply is None:
                first_oob_ply = p
                break   # first OOB found; all subsequent are also OOB

        # Need both endpoints to form a valid boundary pair
        if last_book_ply is None or first_oob_ply is None:
            continue

        lb_time  = thinking_times[last_book_ply]
        oob_time = thinking_times[first_oob_ply]

        # Exclude pairs with missing or negative thinking times.
        # Negative times arise from clock parsing edge cases (flag/timeout)
        # and have no meaningful cognitive interpretation.
        if lb_time is None or oob_time is None:
            continue
        if lb_time < 0 or oob_time < 0:
            continue

        lb_is_premove  = lb_time  <= PREMOVE_THRESHOLD_SECONDS
        oob_is_premove = oob_time <= PREMOVE_THRESHOLD_SECONDS

        # premove_suspect flags pairs where the OOB move itself is likely
        # a premove; included for transparency, filtered in Script 04.
        premove_suspect = oob_is_premove

        records.append({
            'game_id':                  game_id,
            'player_color':             color_name,
            'player_rating':            rating,
            'rating_band':              band,
            'player_username':          username,
            'ply_at_exit':              first_oob_ply + 1,   # 1-indexed
            'last_book_move_uci':       moves_uci[last_book_ply],
            'first_oob_move_uci':       moves_uci[first_oob_ply],
            'last_book_thinking_time':  round(lb_time,  3),
            'first_oob_thinking_time':  round(oob_time, 3),
            'last_book_is_premove':     lb_is_premove,
            'first_oob_is_premove':     oob_is_premove,
            'premove_suspect':          premove_suspect,
            'time_control':             tc_str,
            'game_result':              result,
            'last_book_fen':            fens_before[last_book_ply],
            'first_oob_fen':            fens_before[first_oob_ply],
        })

        window_records.extend(extract_window_rows(
            game_id, color_name, rating, band, username,
            player_plies, first_oob_ply, thinking_times,
            book_at_ply, moves_uci, tc_str))

    return records, window_records


# ==============================================================================
# CHECKPOINT SYSTEM
# ==============================================================================

def save_checkpoint(games_read: int, records_written: int,
                    games_kept: int, exclusion_counts: dict) -> None:
    """
    Save progress to the next rotating checkpoint slot (A or B).

    Stores the game count and records-written count only, not the records
    themselves, which are written directly to the CSV as they are produced.
    On restart, truncate_csv_to() removes any rows written after the
    last checkpoint.

    Uses replace() rather than rename(): replace() overwrites atomically on
    Windows, while rename() raises FileExistsError when the destination exists.
    """
    global _checkpoint_slot
    target = CHECKPOINT_A if _checkpoint_slot % 2 == 0 else CHECKPOINT_B
    tmp    = target.with_suffix('.pkl.tmp')

    with open(tmp, 'wb') as f:
        pickle.dump({
            'games_read':       games_read,
            'records_written':  records_written,
            'games_kept':       games_kept,
            'exclusion_counts': exclusion_counts,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp.replace(target)
    _checkpoint_slot += 1

    log.info(
        f'  [Checkpoint → {target.name}] '
        f'{games_read:,} read | {games_kept:,} kept | '
        f'{records_written:,} pairs written'
    )


def load_checkpoint():
    """
    Load the most recent valid checkpoint from either rotating slot.
    Returns (games_read, records_written, games_kept, exclusion_counts).
    Falls back to zeros if no checkpoint exists.
    """
    best = None

    for path in [CHECKPOINT_A, CHECKPOINT_B]:
        if not path.exists():
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            log.info(f'  Found {path.name}: {data["games_read"]:,} games read')
            if best is None or data['games_read'] > best['games_read']:
                best = data
        except Exception as e:
            log.warning(f'  {path.name} is corrupt: {e}')

    if best is None:
        log.info('No checkpoint found; starting from scratch.')
        return 0, 0, 0, {}

    log.info(
        f'Resuming from game {best["games_read"]:,} '
        f'({best["records_written"]:,} boundary pairs already written).'
    )
    return (
        best['games_read'],
        best['records_written'],
        best['games_kept'],
        best.get('exclusion_counts', {}),
    )


def truncate_csv_to(path: Path, n_data_rows: int) -> None:
    """
    Keep only the header row plus the first n_data_rows data rows.

    Called on restart to discard any rows written after the last checkpoint.
    Without this, restarted runs would append duplicate records to the CSV.
    """
    if not path.exists() or n_data_rows == 0:
        return

    lines_to_keep = n_data_rows + 1   # +1 for the header
    kept = []

    with open(path, 'r', encoding='utf-8', newline='') as f:
        for i, line in enumerate(f):
            if i >= lines_to_keep:
                break
            kept.append(line)

    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.writelines(kept)

    log.info(f'Truncated output CSV to {n_data_rows:,} data rows (removed post-crash writes).')


# ==============================================================================
# FAST SKIP (identical approach to Script 01)
# ==============================================================================

def fast_skip_games(text_stream, n: int, pbar) -> None:
    """
    Skip n games by scanning for game-result tokens at the line level,
    without parsing moves or doing any board operations.

    5-10x faster than calling chess.pgn.read_game() n times, because
    it avoids the expensive legal-move generation needed for SAN parsing.

    In Lichess PGN format, every game ends with a result token (1-0,
    0-1, 1/2-1/2, or *) as the last word on the final movetext line.
    Scanning for these is sufficient to count game boundaries.
    """
    RESULT_TOKENS = {'1-0', '0-1', '1/2-1/2', '*'}
    skipped = 0
    for line in text_stream:
        stripped = line.strip()
        if stripped:
            last_word = stripped.split()[-1]
            if last_word in RESULT_TOKENS:
                skipped += 1
                if skipped % 500_000 == 0:
                    pbar.update(500_000)
                    log.info(f'  Fast-skipped {skipped:,} / {n:,}...')
                if skipped >= n:
                    break
    pbar.update(skipped % 500_000)
    log.info(f'Skip complete. Resuming processing from game {n:,}.')


# ==============================================================================
# MAIN PROCESSING LOOP
# ==============================================================================

def run_processing(book_set: set, start_games_read: int,
                   start_records_written: int, start_games_kept: int,
                   start_exclusions: dict) -> None:
    """
    Stream through the December 2024 analysis file, process each qualifying
    Rapid game, and write boundary pairs to the output CSV.

    CSV is written in append mode ('a') when resuming from a checkpoint
    so that previously written rows are preserved. A fresh run uses 'w'
    and writes the header row first.

    The CSV is flushed to disk before every checkpoint, so the file on
    disk is always consistent with what the checkpoint reports.
    """
    games_read       = start_games_read
    records_written  = start_records_written
    games_kept       = start_games_kept
    exclusion_counts = start_exclusions.copy()

    csv_mode = 'a' if (OUTPUT_PATH.exists() and start_records_written > 0) else 'w'

    dctx = zstd.ZstdDecompressor()

    with open(DOWNLOAD_PATH, 'rb') as compressed_file, \
         open(OUTPUT_PATH, csv_mode, newline='', encoding='utf-8') as csv_file, \
         open(WINDOW_OUTPUT_PATH, csv_mode, newline='', encoding='utf-8') as win_file:

        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if csv_mode == 'w':
            writer.writeheader()

        window_writer = csv.DictWriter(win_file, fieldnames=WINDOW_CSV_COLUMNS)
        if csv_mode == 'w':
            window_writer.writeheader()
        with dctx.stream_reader(compressed_file) as zst_reader:
            text_stream = io.TextIOWrapper(zst_reader, encoding='utf-8', errors='replace')

            with tqdm(desc='Processing games', unit=' games', dynamic_ncols=True) as pbar:

                # ── Fast-skip already-processed games ─────────────────────────
                if start_games_read > 0:
                    log.info(f'Fast-skipping {start_games_read:,} already-processed games...')
                    fast_skip_games(text_stream, start_games_read, pbar)

                # ── Main loop ─────────────────────────────────────────────────
                while True:
                    if MAX_GAMES_TO_PARSE is not None and games_read >= MAX_GAMES_TO_PARSE:
                        log.info(f'Reached MAX_GAMES_TO_PARSE = {MAX_GAMES_TO_PARSE:,}. Stopping.')
                        break

                    try:
                        game = chess.pgn.read_game(text_stream)
                    except Exception as exc:
                        log.debug(f'Skipping malformed game: {exc}')
                        continue

                    if game is None:
                        break   # end of file

                    games_read += 1
                    pbar.update(1)

                    # ── Filter ────────────────────────────────────────────────
                    headers = dict(game.headers)
                    passed, reason = game_passes_filter(headers)

                    if not passed:
                        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
                        continue

                    # ── Process ───────────────────────────────────────────────
                    try:
                        pairs, window_rows = process_game(game, headers, book_set)
                    except Exception as exc:
                        log.debug(f'Error processing {headers.get("Site", "?")}: {exc}')
                        exclusion_counts['processing_error'] = \
                            exclusion_counts.get('processing_error', 0) + 1
                        continue

                    if not pairs:
                        exclusion_counts['no_valid_boundary'] = \
                            exclusion_counts.get('no_valid_boundary', 0) + 1
                        continue

                    games_kept += 1
                    for pair in pairs:
                        writer.writerow(pair)
                        records_written += 1
                    for wrow in window_rows:
                        window_writer.writerow(wrow)

                    pbar.set_postfix(
                        {'kept': f'{games_kept:,}', 'pairs': f'{records_written:,}'},
                        refresh=False,
                    )

                    # ── Checkpoint ────────────────────────────────────────────
                    if games_read % CHECKPOINT_INTERVAL == 0:
                        csv_file.flush()   # ensure rows are on disk before checkpoint
                        win_file.flush()   # ensure rows are on disk before checkpoint
                        save_checkpoint(games_read, records_written,
                                        games_kept, exclusion_counts)

        # Final flush and checkpoint after the file ends
        csv_file.flush()
        win_file.flush() 
        save_checkpoint(games_read, records_written, games_kept, exclusion_counts)

    # ── Filtration log ────────────────────────────────────────────────────────
    log.info('')
    log.info('=' * 65)
    log.info('FILTRATION LOG')
    log.info(f'  Total games read:                       {games_read:>10,}')
    for reason, n in sorted(exclusion_counts.items(), key=lambda x: -x[1]):
        label = f'  Excluded ({reason}):'
        log.info(f'{label:<45} {n:>10,}')
    log.info(f'  Games producing valid boundary pairs:   {games_kept:>10,}')
    log.info(f'  Total boundary pair records written:    {records_written:>10,}')
    log.info('=' * 65)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    log.info('Script 02: Parse and Tag Games')
    log.info(f'Analysis month:      {ANALYSIS_MONTH}')
    log.info(f'Book threshold:      count >= {PRIMARY_BOOK_THRESHOLD}')
    log.info(f'Rating bands:        {list(RATING_BANDS.keys())}')
    log.info(f'Checkpoint every:    {CHECKPOINT_INTERVAL:,} games')
    if MAX_GAMES_TO_PARSE:
        log.info(f'*** TEST MODE: stopping after {MAX_GAMES_TO_PARSE:,} games ***')
    log.info('')

    # ── Verify Script 01 output exists ───────────────────────────────────────
    if not BOOK_CSV_PATH.exists():
        log.error(f'Book sequences not found: {BOOK_CSV_PATH}')
        log.error('Run Script 01 first.')
        sys.exit(1)

    # ── Phase 1: Download ─────────────────────────────────────────────────────
    log.info('Phase 1: Checking download status...')
    if DOWNLOAD_PATH.exists():
        size_gb = DOWNLOAD_PATH.stat().st_size / 1e9
        log.info(f'  Found {size_gb:.1f} GB on disk. Checking if complete...')
    download_with_resume(ANALYSIS_MONTH, DOWNLOAD_PATH)
    log.info('')

    # ── Phase 2: Load book set ────────────────────────────────────────────────
    log.info('Phase 2: Loading book sequences into memory...')
    book_set = load_book_set(BOOK_CSV_PATH, PRIMARY_BOOK_THRESHOLD)
    log.info('')

    # ── Phase 3: Resume from checkpoint ──────────────────────────────────────
    log.info('Phase 3: Checking for checkpoint...')
    games_read, records_written, games_kept, exclusions = load_checkpoint()

    if records_written > 0 and OUTPUT_PATH.exists():
        truncate_csv_to(OUTPUT_PATH, records_written)
    log.info('')

    # ── Phase 4–6: Stream, filter, process ───────────────────────────────────
    log.info('Phase 4–6: Streaming through games...')
    log.info('Safe to interrupt; progress is saved every 25,000 games.')
    log.info('')

    run_processing(book_set, games_read, records_written, games_kept, exclusions)

    log.info('')
    log.info('Script 02 complete.')
    log.info(f'Output: {OUTPUT_PATH}')
    log.info('Next step: python scripts/03_stockfish_covariate.py')


if __name__ == '__main__':
    main()
