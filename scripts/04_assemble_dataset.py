#!/usr/bin/env python3
"""
Script 04: Assemble Final Analysis Dataset
==============================================================================
PURPOSE
    Merge the Stockfish value-of-computation covariate (Script 03) into the
    boundary pair dataset (Script 02), apply pre-registered exclusion filters,
    add log-transformed thinking times, and produce three output datasets:

    1. filtered_pairs.csv    all clean boundary pairs (one row per pair)
    2. analysis_sample.csv   stratified systematic sample of up to 1M pairs
                             (200,000 per band, or all of a smaller band)
                             for the mixed-effects models
    3. player_aggregated.csv one row per player with mean statistics, for
                             the player-level Kruskal-Wallis test

INPUT
    data/processed/boundary_pairs.csv        from Script 02
    data/processed/stockfish_covariate.csv   from Script 03

OUTPUT
    data/processed/filtered_pairs.csv
    data/processed/analysis_sample.csv
    data/processed/player_aggregated.csv

FILTERS APPLIED
    1. Exclude pairs where EITHER the last-book move OR the first-OOB move
       is a premove (thinking time <= PREMOVE_THRESHOLD_SECONDS). The filter
       must be symmetric: the outcome is (OOB time - book time), so excluding
       fast moves on one side only inflates the difference by construction.
    2. Exclude pairs where either VOC is missing (Stockfish failure).
    3. Exclude players with fewer than MIN_TRANSITIONS_PER_PLAYER usable
       pairs. Applied during sampling, so it reaches analysis_sample.csv
       and not only player_aggregated.csv.

    Corrected analysis pre-registered at https://osf.io/n6tkx/ (2026-08-04),
    before these filters were re-run. Prior results are withdrawn; see
    docs/findings/2026-08-03-pipeline-defects.md.

LOG TRANSFORMATION
    Move times are right-skewed, violating the normality assumption of
    linear regression, so both thinking times are log-transformed. No
    offset is added: filter 1 floors every retained time above the premove
    threshold, so log(0) cannot occur. A previous version added 0.001,
    which turned any retained near-zero time into an extreme negative
    outlier (log(0.001) = -6.9 against roughly 1.4 for a typical move).

SAMPLING
    The mixed-effects model in Script 05 cannot run on 20M+ observations.
    analysis_sample.csv holds a stratified sample balanced across rating
    bands so band 5 is adequately represented. Selection within each band
    is systematic (evenly spaced across the band's full sequence) rather
    than sequential, because boundary_pairs.csv inherits the chronological
    order of the source dump and taking the first N rows would restrict
    the sample to the earliest days of the month.

ESTIMATED RUNTIME
    20-40 minutes (streaming 4.9 GB in chunks + aggregation).
==============================================================================
"""

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MIN_TRANSITIONS_PER_PLAYER,
    RATING_BANDS,
    PREMOVE_THRESHOLD_SECONDS,
    RANDOM_SEED,
)

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

PAIRS_PATH      = PROCESSED_DIR / 'boundary_pairs.csv'
VOC_PATH        = PROCESSED_DIR / 'stockfish_covariate.csv'
FILTERED_PATH   = PROCESSED_DIR / 'filtered_pairs.csv'
SAMPLE_PATH     = PROCESSED_DIR / 'analysis_sample.csv'
PLAYER_AGG_PATH = PROCESSED_DIR / 'player_aggregated.csv'

# Target size of analysis_sample.csv, split evenly across the five bands
SAMPLE_SIZE = 1_000_000

# Chunk size for streaming the 4.9 GB boundary_pairs.csv
CHUNK_SIZE = 100_000

# Offset added before the log transform. Zero: the premove filter keeps every
# retained time above the threshold, so log(0) cannot occur (see docstring).
LOG_OFFSET = 0.0


# ==============================================================================
# UTILITY
# ==============================================================================

def normalize_fen(fen: str) -> str:
    """
    Normalize a FEN string to its first 4 fields (piece placement, active
    color, castling, en passant), dropping move counters.

    Matches the normalization used in Scripts 02 and 03 so the join works.
    """
    return ' '.join(fen.split()[:4])


# ==============================================================================
# PHASE 1: LOAD VOC LOOKUP
# ==============================================================================

def load_voc_lookup(voc_path: Path) -> dict:
    """
    Load stockfish_covariate.csv into a dict mapping normalized_fen → voc.

    114,083 rows loads fully into memory in under a second (~10 MB).
    Dict lookup during chunk processing is O(1) per row.
    """
    log.info(f'Loading VOC lookup from {voc_path}...')
    df  = pd.read_csv(voc_path)
    lkp = dict(zip(df['fen_normalized'], df['voc']))
    log.info(f'Loaded {len(lkp):,} positions into VOC lookup.')
    return lkp


# ==============================================================================
# PHASE 2: FILTER AND JOIN
# ==============================================================================

def make_output_columns() -> list:
    """
    Define the complete set of output columns for filtered_pairs.csv.
    """
    return [
        # Original pair metadata
        'game_id', 'player_color', 'player_rating', 'rating_band',
        'player_username', 'ply_at_exit',
        'last_book_move_uci', 'first_oob_move_uci',
        # Raw thinking times (seconds)
        'last_book_thinking_time', 'first_oob_thinking_time',
        # Log-transformed thinking times
        'log_last_book_time', 'log_first_oob_time',
        # Time delta (primary dependent variable)
        'time_delta', 'log_time_delta',
        # Premove flags
        'last_book_is_premove', 'first_oob_is_premove',
        # Stockfish VOC covariate
        'last_book_voc', 'first_oob_voc',
        # Metadata
        'time_control', 'game_result',
    ]


def process_chunk(chunk: pd.DataFrame, voc_lookup: dict,
                  exclusion_counts: dict) -> pd.DataFrame:
    """
    Apply all pre-registered filters and transformations to one chunk.

    Returns the filtered, enriched rows ready for writing.

    FILTERS:
    1. Exclude pairs where either move is a premove (last_book_is_premove or
       first_oob_is_premove). A premove is queued during the opponent's turn,
       so its recorded time is a UI artifact (near zero however long the
       player considered the position), not a cognitive signal. Both sides
       are filtered because the outcome is a difference: removing fast moves
       on one side only would inflate it.

    2. Exclude pairs where either VOC value is missing from the Stockfish
       lookup. This happens only if normalize_fen produced a string not in
       the covariate table (very rare edge case from malformed PGN FENs).

    TRANSFORMATIONS:
    - log_last_book_time  = log(last_book_thinking_time + LOG_OFFSET)
    - log_first_oob_time  = log(first_oob_thinking_time + LOG_OFFSET)
    - time_delta          = first_oob_thinking_time - last_book_thinking_time
    - log_time_delta      = log_first_oob_time - log_last_book_time

    The log delta measures the multiplicative change in thinking time at the
    book-exit boundary: positive values mean the OOB move took longer.
    """
    n_in = len(chunk)

    # ── Filter 1: premove on either side of the boundary ──────────────────────
    premove_mask = (chunk['first_oob_is_premove'].astype(bool)
                    | chunk['last_book_is_premove'].astype(bool))
    n_premove    = premove_mask.sum()
    chunk        = chunk[~premove_mask].copy()
    exclusion_counts['either_move_premove'] += n_premove

    if chunk.empty:
        return chunk

    # ── Join VOC covariate ────────────────────────────────────────────────────
    chunk['_lb_fen_norm']  = chunk['last_book_fen'].apply(normalize_fen)
    chunk['_oob_fen_norm'] = chunk['first_oob_fen'].apply(normalize_fen)

    chunk['last_book_voc']  = chunk['_lb_fen_norm'].map(voc_lookup)
    chunk['first_oob_voc']  = chunk['_oob_fen_norm'].map(voc_lookup)

    # ── Filter 2: missing VOC ─────────────────────────────────────────────────
    voc_missing_mask = chunk['last_book_voc'].isna() | chunk['first_oob_voc'].isna()
    n_voc_missing    = voc_missing_mask.sum()
    chunk            = chunk[~voc_missing_mask].copy()
    exclusion_counts['voc_missing'] += n_voc_missing

    if chunk.empty:
        return chunk

    # No offset: filter 1 floors every retained time above the premove
    # threshold, so log(0) cannot occur.
    chunk['log_last_book_time'] = np.log(chunk['last_book_thinking_time'] + LOG_OFFSET)
    chunk['log_first_oob_time'] = np.log(chunk['first_oob_thinking_time'] + LOG_OFFSET)

    # ── Time delta (the key outcome variable) ─────────────────────────────────
    chunk['time_delta']     = (chunk['first_oob_thinking_time']
                               - chunk['last_book_thinking_time'])
    chunk['log_time_delta'] = (chunk['log_first_oob_time']
                               - chunk['log_last_book_time'])

    # ── Select output columns ─────────────────────────────────────────────────
    out_cols = make_output_columns()
    # Only keep columns that exist (avoid KeyError if input is missing columns)
    out_cols = [c for c in out_cols if c in chunk.columns]
    chunk    = chunk[out_cols]

    n_out  = len(chunk)
    n_filt = n_in - n_out
    exclusion_counts['_total_in']  += n_in
    exclusion_counts['_total_out'] += n_out

    return chunk


def filter_and_join(voc_lookup: dict) -> None:
    """
    Stream boundary_pairs.csv in chunks, apply filters, write filtered rows.

    Prints a filtration log at the end.
    """
    log.info('Phase 2: Filtering boundary pairs and joining VOC...')
    log.info(f'Input:  {PAIRS_PATH}')
    log.info(f'Output: {FILTERED_PATH}')

    exclusion_counts = defaultdict(int)
    first_chunk      = True

    for i, chunk in enumerate(pd.read_csv(
        PAIRS_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )):
        filtered = process_chunk(chunk, voc_lookup, exclusion_counts)

        if not filtered.empty:
            filtered.to_csv(
                FILTERED_PATH,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False,
            )
            first_chunk = False

        if (i + 1) % 50 == 0:
            log.info(
                f'  {(i+1)*CHUNK_SIZE:>12,} rows read | '
                f'{exclusion_counts["_total_out"]:>10,} kept so far'
            )

    # ── Filtration log ────────────────────────────────────────────────────────
    n_in  = exclusion_counts.pop('_total_in',  0)
    n_out = exclusion_counts.pop('_total_out', 0)

    log.info('')
    log.info('=' * 65)
    log.info('FILTRATION LOG (Script 04)')
    log.info(f'  Input boundary pairs:              {n_in:>12,}')
    for reason, n in sorted(exclusion_counts.items(), key=lambda x: -x[1]):
        log.info(f'  Excluded ({reason}):  {n:>12,}')
    log.info(f'  Final filtered pairs:              {n_out:>12,}')
    log.info('=' * 65)
    log.info('')


# ==============================================================================
# PHASE 3: STRATIFIED SAMPLE
# ==============================================================================



def create_stratified_sample() -> None:
    """
    Build a stratified sample of SAMPLE_SIZE pairs from filtered_pairs.csv.

    Players contributing fewer than MIN_TRANSITIONS_PER_PLAYER pairs are
    excluded here, so the exclusion applies to the dataset used by the
    mixed-effects model and not only to the player-level aggregate.

    Selection within each band is systematic: every k-th qualifying row,
    where k = band_count // target, starting from a seeded random offset.
    Because filtered_pairs.csv inherits the chronological order of the
    source PGN dump, taking the first N rows of each band would restrict
    the sample to the earliest days of the month. Systematic selection
    spans the full corpus.
    """
    log.info(f'Phase 3: Creating stratified sample ({SAMPLE_SIZE:,} pairs)...')

    # ── Pass 1: pairs per player ─────────────────────────────────────────────
    log.info('  Pass 1/3: counting pairs per player...')
    player_counts = defaultdict(int)
    for chunk in pd.read_csv(FILTERED_PATH, chunksize=CHUNK_SIZE,
                             usecols=['player_username']):
        for user, count in chunk['player_username'].value_counts().items():
            player_counts[user] += count

    qualified = {u for u, n in player_counts.items()
                 if n >= MIN_TRANSITIONS_PER_PLAYER}
    log.info(f'    Players total:     {len(player_counts):,}')
    log.info(f'    Players qualified: {len(qualified):,} '
             f'(>= {MIN_TRANSITIONS_PER_PLAYER} pairs)')
    log.info(f'    Players excluded:  {len(player_counts) - len(qualified):,}')

    # ── Pass 2: qualifying rows per band ─────────────────────────────────────
    log.info('  Pass 2/3: counting qualifying rows per band...')
    band_counts = defaultdict(int)
    for chunk in pd.read_csv(FILTERED_PATH, chunksize=CHUNK_SIZE,
                             usecols=['player_username', 'rating_band']):
        keep = chunk['player_username'].isin(qualified)
        for band, count in chunk.loc[keep, 'rating_band'].value_counts().items():
            band_counts[band] += count

    if not band_counts:
        log.warning('No qualifying rows found; sample not created.')
        return

    total = sum(band_counts.values())
    log.info('    Band distribution (qualifying rows only):')
    for band in sorted(band_counts):
        log.info(f'      {band}: {band_counts[band]:>10,}  '
                 f'({100 * band_counts[band] / total:.1f}%)')

    per_band = SAMPLE_SIZE // len(band_counts)
    targets = {b: min(per_band, n) for b, n in band_counts.items()}

    log.info(f'    Target per band: {per_band:,} (capped at available)')
    for band in sorted(band_counts):
        log.info(f'      {band}: target={targets[band]:>8,}  '
                 f'of {band_counts[band]:>10,} available')

    # ── Pass 3: systematic selection ─────────────────────────────────────────
    # A row at position p within its band's sequence of N rows is selected iff
    #   (p * T) // N  !=  ((p + 1) * T) // N
    # This yields exactly T rows spaced evenly across the whole sequence,
    # independent of whether N divides evenly by T.
    log.info('  Pass 3/3: selecting rows...')
    seen = {b: 0 for b in band_counts}
    parts = {b: [] for b in band_counts}
    taken = {b: 0 for b in band_counts}

    for chunk in pd.read_csv(FILTERED_PATH, chunksize=CHUNK_SIZE,
                             low_memory=False):
        chunk = chunk[chunk['player_username'].isin(qualified)]
        if chunk.empty:
            continue

        for band in band_counts:
            if taken[band] >= targets[band]:
                continue

            band_chunk = chunk[chunk['rating_band'] == band]
            if band_chunk.empty:
                continue

            n_total = band_counts[band]
            t_target = targets[band]

            pos = np.arange(seen[band], seen[band] + len(band_chunk))
            seen[band] += len(band_chunk)

            hit = band_chunk[((pos * t_target) // n_total)
                             != (((pos + 1) * t_target) // n_total)]
            if hit.empty:
                continue

            room = t_target - taken[band]
            if len(hit) > room:
                hit = hit.iloc[:room]

            parts[band].append(hit)
            taken[band] += len(hit)

    frames = [pd.concat(p, ignore_index=True) for p in parts.values() if p]
    if not frames:
        log.warning('No rows selected; sample not created.')
        return

    sample = pd.concat(frames, ignore_index=True)
    sample = sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    sample.to_csv(SAMPLE_PATH, index=False)

    log.info(f'  Sample saved: {len(sample):,} rows -> {SAMPLE_PATH}')
    log.info('  Band distribution in sample:')
    for band in sorted(sample['rating_band'].unique()):
        log.info(f'    {band}: {(sample["rating_band"] == band).sum():>8,}')
    log.info('')


# ==============================================================================
# PHASE 4: PLAYER-LEVEL AGGREGATION
# ==============================================================================

def aggregate_to_player_level() -> None:
    """
    Aggregate filtered_pairs.csv to one row per player with mean statistics.

    This player-level dataset supports the simpler ANOVA / Kruskal-Wallis
    approach in Script 05, where the unit of analysis is the player rather
    than the individual boundary pair. Using player-level means avoids the
    non-independence problem without requiring a mixed-effects model.

    EXCLUSION: Players with fewer than MIN_TRANSITIONS_PER_PLAYER usable
    pairs are excluded. With fewer than 3 pairs, a player's mean is
    unreliable and a random intercept for that player cannot be estimated
    stably.

    Statistics computed per player:
        n_pairs              Number of usable boundary pairs
        mean_time_delta      Mean(first_oob_time - last_book_time) in seconds
        mean_log_time_delta  Mean(log_first_oob_time - log_last_book_time)
        mean_last_book_time  Mean thinking time on last book moves
        mean_first_oob_time  Mean thinking time on first OOB moves
        mean_last_book_voc   Mean VOC at last book positions
        mean_first_oob_voc   Mean VOC at first OOB positions
        sd_time_delta        Std dev of time deltas (within-player variability)
        rating_band          Most common rating band for this player
    """
    log.info('Phase 4: Aggregating to player level...')

    # Accumulators per player
    # Stored as: {username: [n, sum_delta, sum_log_delta, sum_lb, sum_oob,
    #                        sum_lb_voc, sum_oob_voc, sum_sq_delta,
    #                        band_list]}
    player_data = {}

    for i, chunk in enumerate(pd.read_csv(
        FILTERED_PATH,
        chunksize=CHUNK_SIZE,
        usecols=[
            'player_username', 'rating_band',
            'time_delta', 'log_time_delta',
            'last_book_thinking_time', 'first_oob_thinking_time',
            'last_book_voc', 'first_oob_voc',
        ],
        low_memory=False,
    )):
        for username, grp in chunk.groupby('player_username'):
            if username not in player_data:
                player_data[username] = [
                    0,    # n
                    0.0,  # sum_delta
                    0.0,  # sum_log_delta
                    0.0,  # sum_lb_time
                    0.0,  # sum_oob_time
                    0.0,  # sum_lb_voc
                    0.0,  # sum_oob_voc
                    0.0,  # sum_sq_delta (for std dev)
                    [],   # band_list
                ]
            d = player_data[username]
            d[0] += len(grp)
            d[1] += grp['time_delta'].sum()
            d[2] += grp['log_time_delta'].sum()
            d[3] += grp['last_book_thinking_time'].sum()
            d[4] += grp['first_oob_thinking_time'].sum()
            d[5] += grp['last_book_voc'].sum()
            d[6] += grp['first_oob_voc'].sum()
            d[7] += (grp['time_delta'] ** 2).sum()
            d[8].extend(grp['rating_band'].tolist())

        if (i + 1) % 50 == 0:
            log.info(f'  {(i+1)*CHUNK_SIZE:>12,} rows processed | '
                     f'{len(player_data):>8,} unique players seen')

    log.info(f'Total unique players before filtering: {len(player_data):,}')

    # Build output rows, applying MIN_TRANSITIONS_PER_PLAYER filter
    rows          = []
    n_excluded    = 0

    for username, d in player_data.items():
        n = d[0]
        if n < MIN_TRANSITIONS_PER_PLAYER:
            n_excluded += 1
            continue

        mean_delta    = d[1] / n
        mean_log_d    = d[2] / n
        mean_lb       = d[3] / n
        mean_oob      = d[4] / n
        mean_lb_voc   = d[5] / n
        mean_oob_voc  = d[6] / n

        # Population std dev of time_delta
        variance_delta = (d[7] / n) - (mean_delta ** 2)
        sd_delta       = variance_delta ** 0.5 if variance_delta > 0 else 0.0

        # Rating band: most common across all pairs for this player
        band_list     = d[8]
        rating_band   = max(set(band_list), key=band_list.count)

        rows.append({
            'player_username':     username,
            'rating_band':         rating_band,
            'n_pairs':             n,
            'mean_time_delta':     round(mean_delta,   3),
            'mean_log_time_delta': round(mean_log_d,   4),
            'mean_last_book_time': round(mean_lb,      3),
            'mean_first_oob_time': round(mean_oob,     3),
            'mean_last_book_voc':  round(mean_lb_voc,  2),
            'mean_first_oob_voc':  round(mean_oob_voc, 2),
            'sd_time_delta':       round(sd_delta,     3),
        })

    log.info(f'Players excluded (< {MIN_TRANSITIONS_PER_PLAYER} pairs): {n_excluded:,}')
    log.info(f'Players in final aggregated dataset: {len(rows):,}')

    # Write player-level CSV
    player_df = pd.DataFrame(rows)
    player_df.sort_values('rating_band', inplace=True)
    player_df.to_csv(PLAYER_AGG_PATH, index=False)
    log.info(f'Saved: {PLAYER_AGG_PATH}')

    # Summary per band
    log.info('\nPlayer-level summary by rating band:')
    log.info(f'{"Band":<10} {"N players":>10} {"Mean Δt":>10} {"Median Δt":>10}')
    for band in sorted(player_df['rating_band'].unique()):
        sub = player_df[player_df['rating_band'] == band]
        log.info(
            f'{band:<10} {len(sub):>10,} '
            f'{sub["mean_time_delta"].mean():>10.2f} '
            f'{sub["mean_time_delta"].median():>10.2f}'
        )

    log.info('')


# ==============================================================================
# PHASE 5: SUMMARY STATISTICS
# ==============================================================================

def print_summary_statistics() -> None:
    """
    Print key statistics from the filtered pairs, as a description of the
    data and a sanity check.
    """
    log.info('Phase 5: Summary statistics (sample of first 500K rows)...')

    # Use a sample rather than loading the full file
    chunks = []
    for chunk in pd.read_csv(FILTERED_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        chunks.append(chunk)
        if sum(len(c) for c in chunks) >= 500_000:
            break

    df = pd.concat(chunks, ignore_index=True).head(500_000)

    log.info('')
    log.info('=== SUMMARY STATISTICS (first 500K pairs) ===')
    log.info('')
    log.info('Thinking time, last book move (seconds):')
    desc = df['last_book_thinking_time'].describe()
    for stat, val in desc.items():
        log.info(f'  {stat:8s}: {val:.3f}')

    log.info('')
    log.info('Thinking time, first OOB move (seconds):')
    desc = df['first_oob_thinking_time'].describe()
    for stat, val in desc.items():
        log.info(f'  {stat:8s}: {val:.3f}')

    log.info('')
    log.info('Time delta (first_oob - last_book) (seconds):')
    desc = df['time_delta'].describe()
    for stat, val in desc.items():
        log.info(f'  {stat:8s}: {val:.3f}')

    log.info('')
    log.info('VOC, last book positions:')
    log.info(f'  mean: {df["last_book_voc"].mean():.1f}  '
             f'median: {df["last_book_voc"].median():.1f}  '
             f'max: {df["last_book_voc"].max():.1f}')

    log.info('')
    log.info('VOC, first OOB positions:')
    log.info(f'  mean: {df["first_oob_voc"].mean():.1f}  '
             f'median: {df["first_oob_voc"].median():.1f}  '
             f'max: {df["first_oob_voc"].max():.1f}')

    log.info('')
    log.info('Rating band distribution:')
    counts = df['rating_band'].value_counts().sort_index()
    for band, count in counts.items():
        pct = 100 * count / len(df)
        log.info(f'  {band}: {count:>8,}  ({pct:.1f}%)')

    log.info('')
    log.info('Preliminary H1 signal (before controlling for VOC):')
    mean_lb  = df['last_book_thinking_time'].mean()
    mean_oob = df['first_oob_thinking_time'].mean()
    log.info(f'  Mean last_book_time:  {mean_lb:.3f} sec')
    log.info(f'  Mean first_oob_time:  {mean_oob:.3f} sec')
    log.info(f'  Raw difference:       {mean_oob - mean_lb:+.3f} sec')
    log.info(f'  Raw ratio:            {mean_oob/mean_lb:.3f}x longer')
    log.info('  (Raw effect, before VOC adjustment; the models adjust for VOC)')
    log.info('')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    log.info('Script 04: Assemble Final Analysis Dataset')
    log.info(f'Min transitions per player: {MIN_TRANSITIONS_PER_PLAYER}')
    log.info(f'Sample size:                {SAMPLE_SIZE:,}')
    log.info(f'Random seed:                {RANDOM_SEED}')
    log.info(f'Log offset:                 {LOG_OFFSET}')
    log.info('')

    # Verify inputs
    for path in [PAIRS_PATH, VOC_PATH]:
        if not path.exists():
            log.error(f'Input not found: {path}')
            log.error('Run Scripts 02 and 03 first.')
            sys.exit(1)

    # ── Phase 1: VOC lookup ───────────────────────────────────────────────────
    voc_lookup = load_voc_lookup(VOC_PATH)
    log.info('')

    # ── Phase 2: Filter and join ──────────────────────────────────────────────
    filter_and_join(voc_lookup)

    # ── Phase 3: Stratified sample ────────────────────────────────────────────
    create_stratified_sample()

    # ── Phase 4: Player aggregation ───────────────────────────────────────────
    aggregate_to_player_level()

    # ── Phase 5: Summary statistics ───────────────────────────────────────────
    print_summary_statistics()

    log.info('Script 04 complete.')
    log.info(f'Outputs:')
    log.info(f'  {FILTERED_PATH}')
    log.info(f'  {SAMPLE_PATH}')
    log.info(f'  {PLAYER_AGG_PATH}')
    log.info('Next step: python scripts/05_statistical_analysis.py')


if __name__ == '__main__':
    main()
