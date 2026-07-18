#!/usr/bin/env python3
"""
Script 01: Build Reference Corpus
==============================================================================
PURPOSE
    Build an empirical "opening book" frequency table from a large sample of
    high-rated Lichess Rapid games. This table defines what counts as
    "documented opening theory" for the main analysis in Script 02.

    A position is classified as "in book" if the exact sequence of moves
    leading to it was played at least N times by strong players in this
    independent reference dataset (where N is PRIMARY_BOOK_THRESHOLD in
    config.py, default = 10).

WHY NOT ECO CODES?
    The Encyclopaedia of Chess Openings (ECO) assigns labels like "C54" to
    opening positions, but stops labeling lines after roughly 10-20 moves.
    Modern preparation at the 1800+ online level routinely extends 25-35+
    moves deep in sharp theoretical lines that no ECO label covers.

    If we defined "in book" using ECO codes, we would mislabel a strong
    player's genuine, deeply memorized preparation as "out of book" the moment
    it ran past the ECO cutoff — manufacturing a false effect that has nothing
    to do with cognition, purely because of our measurement tool's limitations.

    The frequency-based approach scales naturally: a position is "book" if
    many real players of comparable or higher strength actually played there,
    regardless of whether a textbook assigned it a code.

WHY A SEPARATE REFERENCE MONTH?
    We use games from June 2024 to build the frequency table, and games from
    December 2024 as the main analysis sample (both set in config.py).

    This separation is critical for avoiding circularity: if we defined "book"
    using the same games we measure time costs in, a sequence might look
    "book" partly because it appears in our own analysis data — a self-fulfilling
    artifact. Using an independent time period eliminates this confound.

INPUT
    None — this script downloads its own data from Lichess.

OUTPUT
    data/reference_corpus/book_sequences.csv
    Columns:
        move_sequence   Space-separated UCI moves from game start (e.g. "e2e4 e7e5 g1f3")
        count           How many reference games followed this exact prefix
        last_move_uci   The final move of this sequence (for readability)
        depth           Number of half-moves in this sequence

USAGE
    python scripts/01_build_reference_corpus.py

    Safe to interrupt with Ctrl+C. Progress is checkpointed every 100,000
    games — rerunning the script will resume from the last checkpoint.

ESTIMATED RUNTIME
    Download: 1-4 hours depending on internet speed (file is 15-30 GB).
    Processing: 2-6 hours depending on CPU speed.
    Both phases run unattended. Leave the script running in the background.
==============================================================================
"""

import os
import sys
import io
import csv
import pickle
import logging
import time
from collections import defaultdict
from pathlib import Path

import requests
import zstandard as zstd
import chess
import chess.pgn
from tqdm import tqdm

# ── Import project configuration ───────────────────────────────────────────────
# sys.path.insert allows Python to find config.py in the project root,
# regardless of which directory you run this script from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    REFERENCE_MONTH,
    ALLOWED_TIME_CONTROLS,
    MIN_PLIES_FOR_INCLUSION,
    MAX_PLIES_FOR_BOOK,
    PRIMARY_BOOK_THRESHOLD,
    MIN_COUNT_TO_SAVE,
    MAX_GAMES_TO_PARSE,
)

# ── Logging ────────────────────────────────────────────────────────────────────
# Using Python's logging module (not print) gives timestamps on every line,
# which helps you track how long each phase is taking.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / 'data' / 'raw'
CORPUS_DIR    = PROJECT_ROOT / 'data' / 'reference_corpus'

# Create directories if they don't exist yet
RAW_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_PATH   = RAW_DIR    / f'lichess_db_standard_rated_{REFERENCE_MONTH}.pgn.zst'
OUTPUT_PATH     = CORPUS_DIR / 'book_sequences.csv'

# Two rotating checkpoint files. Each save alternates between A and B so that
# a crash or OS filesystem rollback during a write can never destroy BOTH copies.
# On restart, load_checkpoint() picks whichever file has the higher game count.
CHECKPOINT_A    = CORPUS_DIR / 'checkpoint_a.pkl'
CHECKPOINT_B    = CORPUS_DIR / 'checkpoint_b.pkl'
# Legacy path from earlier runs — loaded as fallback if neither A nor B exists.
CHECKPOINT_LEGACY = CORPUS_DIR / 'checkpoint.pkl'

# Save every 25,000 games (~60 seconds at 400 games/sec).
# Worst-case lost work on crash: ~1 minute of processing.
CHECKPOINT_INTERVAL = 25_000


# ==============================================================================
# PHASE 1 — DOWNLOAD
# ==============================================================================

def download_with_resume(month: str, local_path: Path) -> None:
    """
    Download a Lichess monthly database file to disk, with resume support
    and automatic retry on connection drops.

    Lichess provides complete monthly PGN archives at a public URL. The files
    are Zstandard-compressed (.pgn.zst). We keep them compressed on disk and
    stream-decompress during processing — this avoids writing a 100-200 GB
    uncompressed file to disk.

    A 29 GB download over a home internet connection will very likely experience
    at least one connection reset mid-transfer. The retry loop handles this
    automatically: when the connection drops, the script waits 30 seconds,
    checks how many bytes were successfully written, and resumes from exactly
    that byte using an HTTP Range header. No progress is lost.

    Args:
        month:      "YYYY-MM" string, e.g. "2024-06"
        local_path: where to save the compressed file on disk
    """
    url = (
        f'https://database.lichess.org/standard/'
        f'lichess_db_standard_rated_{month}.pgn.zst'
    )

    MAX_RETRIES  = 20    # Maximum reconnection attempts before giving up
    RETRY_DELAY  = 30    # Seconds to wait between retries

    log.info(f'URL: {url}')
    log.info('The script will auto-retry if the connection drops. Leave it running.')

    attempt = 0

    while attempt < MAX_RETRIES:

        # Check how many bytes are already on disk (0 if fresh start)
        existing_bytes = local_path.stat().st_size if local_path.exists() else 0

        if existing_bytes > 0 and attempt == 0:
            log.info(f'Partial download found ({existing_bytes / 1e9:.2f} GB). Resuming...')
        elif attempt > 0:
            log.info(f'Reconnecting... resuming from {existing_bytes / 1e9:.2f} GB.')

        # Send Range header to resume from the last written byte
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

        # HTTP 416 = server says we already have all the bytes — file is complete
        if response.status_code == 416:
            log.info('File already fully downloaded.')
            return

        if response.status_code not in (200, 206):
            log.error(f'HTTP {response.status_code}. Cannot download file.')
            log.error('Check https://database.lichess.org/ to verify the file exists.')
            sys.exit(1)

        content_length = int(response.headers.get('content-length', 0))
        total_bytes    = existing_bytes + content_length

        if attempt == 0:
            log.info(f'Saving to: {local_path}')
            log.info(f'Total size: ~{total_bytes / 1e9:.1f} GB')

        # 'ab' = append-binary (continues partial); 'wb' = write-binary (fresh)
        write_mode = 'ab' if existing_bytes > 0 else 'wb'

        try:
            with open(local_path, write_mode) as f:
                with tqdm(
                    total=total_bytes,
                    initial=existing_bytes,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc='Downloading',
                    dynamic_ncols=True,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=65_536):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Reaching here means the download loop finished without error
            log.info('Download complete.')
            return

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            ConnectionResetError,
        ):
            # Connection dropped mid-transfer. The bytes written so far are safe
            # on disk (opened with 'ab', flushed on each chunk). Loop back to
            # the top to reconnect and resume from the new file size.
            attempt += 1
            new_size = local_path.stat().st_size if local_path.exists() else 0
            log.warning(
                f'Connection dropped at {new_size / 1e9:.2f} GB. '
                f'Retry {attempt}/{MAX_RETRIES} in {RETRY_DELAY}s...'
            )
            time.sleep(RETRY_DELAY)
            # Loop back — next iteration resumes from new_size bytes

    log.error(f'Download failed after {MAX_RETRIES} retries.')
    log.error('Check your internet connection and re-run the script to try again.')
    sys.exit(1)


# ==============================================================================
# PHASE 2 — FILTERING
# ==============================================================================

def game_passes_filter(headers: dict) -> tuple[bool, str]:
    """
    Decide whether a game should contribute to the reference corpus.

    Returns (True, '') if the game passes all filters.
    Returns (False, reason) if it should be excluded.

    Inclusion criteria — ALL must be met:
    1. Rated Rapid game (not Blitz, Bullet, Classical, or Casual)
    2. Both players rated 2200 or above on Lichess Rapid
    3. Time control matches one of our allowed Rapid controls (from config.py)
    4. Game ended normally or by time forfeit (not abandoned or rules infraction)
       Note: time forfeit games are INCLUDED — the moves played are legitimate
       chess, the game just ended when a player's clock hit zero. Abandoned
       games are excluded because the move record may be incomplete.

    Why 2200+?
        We want the reference corpus to capture what genuinely-prepared players
        actually know in theory. Players rated 2200+ online routinely prepare
        20-30+ move theoretical lines. Using their games to define "book" means
        our threshold reflects real preparation depth — not just the first few
        moves any player might know from habit.
    """
    # ── Check 1: Event type ──────────────────────────────────────────────────
    # Lichess encodes the event type exactly as "Rated Rapid game" in the PGN
    # header for rated rapid games. Other values include "Rated Blitz game",
    # "Rated Bullet game", "Casual Rapid game", etc.
    event = headers.get('Event', '')
    if event != 'Rated Rapid game':
        return False, 'not_rapid'

    # ── Check 2: Ratings ─────────────────────────────────────────────────────
    try:
        white_elo = int(headers.get('WhiteElo', '0'))
        black_elo = int(headers.get('BlackElo', '0'))
    except ValueError:
        # Non-numeric Elo field means the rating is unknown/provisional
        return False, 'bad_rating_format'

    if white_elo < 2200 or black_elo < 2200:
        return False, 'rating_below_2200'

    # ── Check 3: Time control ────────────────────────────────────────────────
    time_control = headers.get('TimeControl', '')
    if time_control not in ALLOWED_TIME_CONTROLS:
        return False, 'time_control_excluded'

    # ── Check 4: Termination ─────────────────────────────────────────────────
    termination = headers.get('Termination', '').lower()
    if 'abandoned' in termination or 'rules infraction' in termination:
        return False, 'abnormal_termination'

    return True, ''


# ==============================================================================
# PHASE 3 — SEQUENCE EXTRACTION
# ==============================================================================

def extract_prefixes(game: chess.pgn.Game):
    """
    Walk through a game's main line and yield every move-sequence prefix
    up to MAX_PLIES_FOR_BOOK half-moves.

    Each prefix is a tuple of UCI move strings representing the complete
    sequence of moves from the start of the game to that point.

    Example: after White plays e4 and Black plays e5:
        yields ("e2e4",)           — after White's first move
        yields ("e2e4", "e7e5")    — after Black's first move

    Why tuples?
        Tuples are hashable and can be used as dictionary keys. This allows
        O(1) counting via a defaultdict. Lists cannot be dictionary keys.

    Why ALL prefixes (not just the full game sequence)?
        In Script 02, we need to check any mid-game position: "is THIS
        specific sequence still in book?" — not just the final position.
        A game that goes 25 moves deep in theory contributes 25 separate
        prefix entries, each counted independently.

    This function is a generator (uses 'yield') to be memory-efficient:
    it produces one prefix at a time rather than building a list of all
    prefixes in memory before returning.
    """
    moves_so_far = []
    node = game

    for ply in range(MAX_PLIES_FOR_BOOK):
        # Stop if there are no more moves in the main line
        if not node.variations:
            break

        next_node = node.variations[0]  # always follow the main line only

        if next_node.move is None:
            break

        moves_so_far.append(next_node.move.uci())

        # tuple() creates an immutable snapshot — necessary because
        # moves_so_far is a mutable list that keeps growing
        yield tuple(moves_so_far)

        node = next_node


# ==============================================================================
# PHASE 4 — CHECKPOINTING
# ==============================================================================

# Tracks which checkpoint slot to write to next (0 = A, 1 = B, alternating)
_checkpoint_slot = 0

def save_checkpoint(
    counts: defaultdict,
    games_read: int,
    games_kept: int,
    exclusion_counts: dict,
) -> None:
    """
    Write current progress to disk using two rotating checkpoint files.

    Alternates writes between checkpoint_a.pkl and checkpoint_b.pkl.
    If one file is corrupt or partially written (e.g. OS killed mid-write),
    the other still holds the previous good checkpoint.

    Uses .replace() instead of .rename() — replace() overwrites atomically
    on both Windows and Unix; rename() raises FileExistsError on Windows.
    """
    global _checkpoint_slot

    target = CHECKPOINT_A if _checkpoint_slot % 2 == 0 else CHECKPOINT_B
    tmp    = target.with_suffix('.pkl.tmp')

    data = {
        'counts':           dict(counts),
        'games_read':       games_read,
        'games_kept':       games_kept,
        'exclusion_counts': exclusion_counts,
    }

    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)

    _checkpoint_slot += 1

    log.info(
        f'  [Checkpoint → {target.name}] {games_read:,} read | '
        f'{games_kept:,} kept | {len(counts):,} sequences'
    )


def load_checkpoint() -> tuple:
    """
    Load the most recent valid checkpoint from either rotating slot,
    with fallback to the legacy checkpoint.pkl from earlier runs.

    Returns:
        (counts, games_read, games_kept, exclusion_counts)

    Tries checkpoint_a.pkl, checkpoint_b.pkl, and checkpoint.pkl (legacy).
    Picks whichever has the highest games_read value and is not corrupt.
    """
    best      = None
    best_path = None

    candidates = [CHECKPOINT_A, CHECKPOINT_B, CHECKPOINT_LEGACY]

    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if best is None or data['games_read'] > best['games_read']:
                best      = data
                best_path = path
            log.info(f'  Found {path.name}: {data["games_read"]:,} games read')
        except Exception as e:
            log.warning(f'  {path.name} is corrupt and will be ignored: {e}')

    if best is None:
        log.info('No valid checkpoint found — starting from the beginning.')
        return defaultdict(int), 0, 0, {}

    log.info(f'Loading from {best_path.name} ({best["games_read"]:,} games read, '
             f'{best["games_kept"]:,} kept, {len(best["counts"]):,} sequences).')

    counts           = defaultdict(int, best['counts'])
    games_read       = best['games_read']
    games_kept       = best['games_kept']
    exclusion_counts = best.get('exclusion_counts', {})

    return counts, games_read, games_kept, exclusion_counts


# ==============================================================================
# PHASE 5 — MAIN PROCESSING LOOP
# ==============================================================================

def build_sequence_counts() -> defaultdict:
    """
    Stream through the compressed PGN file and count how often each
    move-sequence prefix appears in games that pass our filters.

    This is the core computational step of Script 01. It:
        1. Opens the .pgn.zst file and decompresses it on-the-fly
           (never writes the full uncompressed file to disk — saves 100+ GB)
        2. Reads games one by one using python-chess's PGN parser
        3. Applies the inclusion filter (Rapid, 2200+, normal time control)
        4. For games that pass, counts every opening prefix up to 40 plies
        5. Saves a checkpoint every CHECKPOINT_INTERVAL games in case of interruption

    Memory note:
        The sequence_counts dictionary maps tuples of UCI strings to integers.
        With 100,000+ qualifying games contributing up to 40 prefixes each,
        this dictionary may grow to 500 MB – 2 GB of RAM. This is expected
        and normal for a research pipeline of this scale.

    Returns:
        defaultdict mapping sequence tuples → occurrence count
    """
    counts, start_at, games_kept, exclusion_counts = load_checkpoint()

    games_read = start_at

    dctx = zstd.ZstdDecompressor()

    with open(DOWNLOAD_PATH, 'rb') as compressed_file:
        with dctx.stream_reader(compressed_file) as zst_reader:

            # Wrap the binary Zstandard stream in a text decoder.
            # 'errors=replace' handles the rare malformed UTF-8 byte in PGN comments.
            text_stream = io.TextIOWrapper(zst_reader, encoding='utf-8', errors='replace')

            with tqdm(
                desc='Reading games',
                unit=' games',
                dynamic_ncols=True,
            ) as pbar:

                # ── Resume: fast-skip games already processed ───────────
                if start_at > 0:
                    log.info(f'Fast-skipping {start_at:,} already-processed games...')
                    log.info('(Scanning for game boundaries without parsing moves — '
                             'much faster than before.)')

                    # Scan line-by-line and count game-result tokens.
                    # In Lichess PGN format, every game ends with a result token
                    # (1-0, 0-1, 1/2-1/2, or *) as the last word on a line.
                    # This avoids expensive board operations (legal move generation,
                    # piece tracking) used by chess.pgn.read_game(), making the
                    # skip 5-10x faster than the previous implementation.
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
                                    log.info(f'  Fast-skipped {skipped:,} / {start_at:,}...')
                                if skipped >= start_at:
                                    break
                    pbar.update(skipped % 500_000)  # update remainder
                    log.info(f'Skip complete. Resuming processing from game {start_at:,}.')

                # ── Main loop ─────────────────────────────────────────────────
                while True:

                    # Optional cap for testing (set MAX_GAMES_TO_PARSE in config.py)
                    if MAX_GAMES_TO_PARSE is not None and games_read >= MAX_GAMES_TO_PARSE:
                        log.info(f'Reached MAX_GAMES_TO_PARSE = {MAX_GAMES_TO_PARSE:,}. Stopping.')
                        break

                    # Read the next game from the PGN stream
                    try:
                        game = chess.pgn.read_game(text_stream)
                    except Exception as exc:
                        # Malformed PGN record — log at debug level and skip.
                        # This happens occasionally in large public datasets.
                        log.debug(f'Skipping malformed game record: {exc}')
                        continue

                    if game is None:
                        # None signals end of file — we're done
                        break

                    games_read += 1
                    pbar.update(1)

                    # ── Apply inclusion filter ────────────────────────────────
                    headers = dict(game.headers)
                    passed, reason = game_passes_filter(headers)

                    if not passed:
                        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
                        continue

                    # ── Extract opening prefixes and count them ───────────────
                    # We materialize the generator into a list FIRST so we can
                    # check the game's length before deciding whether to count it.
                    prefixes = list(extract_prefixes(game))

                    if len(prefixes) < MIN_PLIES_FOR_INCLUSION:
                        # Game too short to be useful (checkmate in <10 moves, etc.)
                        exclusion_counts['too_short'] = exclusion_counts.get('too_short', 0) + 1
                        continue

                    # Game is valid — count all its opening prefixes
                    for prefix_tuple in prefixes:
                        counts[prefix_tuple] += 1

                    games_kept += 1
                    pbar.set_postfix({'kept': f'{games_kept:,}'}, refresh=False)

                    # ── Checkpoint ────────────────────────────────────────────
                    if games_read % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(counts, games_read, games_kept, exclusion_counts)

    # Final checkpoint after processing is complete
    save_checkpoint(counts, games_read, games_kept, exclusion_counts)

    # ── Print filtration log ───────────────────────────────────────────────────
    # This output should be copied verbatim into your paper's Methods section
    # under "Reference corpus construction" or similar.
    log.info('')
    log.info('=' * 65)
    log.info('FILTRATION LOG — copy this into your Methods section:')
    log.info(f'  Total games read from raw file:         {games_read:>10,}')
    for reason, n in sorted(exclusion_counts.items(), key=lambda x: -x[1]):
        log.info(f'  Excluded ({reason}):  {n:>10,}')
    log.info(f'  Games used for reference corpus:        {games_kept:>10,}')
    log.info('=' * 65)

    return counts


# ==============================================================================
# PHASE 6 — SAVE OUTPUT
# ==============================================================================

def save_output(counts: defaultdict) -> None:
    """
    Filter the sequence counts and write the final CSV output.

    We save all sequences with count >= MIN_COUNT_TO_SAVE (default = 3).
    The actual analysis threshold (PRIMARY_BOOK_THRESHOLD = 10, or 5 or 20
    for robustness checks) is applied at load time in Script 02.

    This way we only need to run Script 01 once. Changing the threshold
    for a robustness check just requires reloading the CSV with a different
    filter, not rerunning the entire multi-hour processing job.

    Output columns:
        move_sequence   Space-separated UCI moves (e.g., "e2e4 e7e5 g1f3")
        count           How many reference games followed this exact sequence
        last_move_uci   The final move in the sequence (convenience field)
        depth           Number of half-moves in the sequence
    """
    log.info('Filtering and saving...')

    # Count how many sequences survive each threshold (for your Methods section)
    log.info('')
    log.info('Sequence counts by threshold:')
    log.info(f'  Total unique sequences (all counts):   {len(counts):>10,}')
    for t in [MIN_COUNT_TO_SAVE, 5, 10, 20, 50]:
        n = sum(1 for c in counts.values() if c >= t)
        marker = '  ← primary threshold' if t == PRIMARY_BOOK_THRESHOLD else ''
        log.info(f'  count >= {t:2d}:                            {n:>10,}{marker}')

    # Filter to MIN_COUNT_TO_SAVE before writing to keep the CSV manageable
    to_save = {seq: cnt for seq, cnt in counts.items() if cnt >= MIN_COUNT_TO_SAVE}

    log.info(f'\nWriting {len(to_save):,} sequences to {OUTPUT_PATH}...')

    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['move_sequence', 'count', 'last_move_uci', 'depth'])

        # Sort by descending count so the most common sequences come first
        for seq_tuple, count in sorted(to_save.items(), key=lambda x: -x[1]):
            writer.writerow([
                ' '.join(seq_tuple),  # space-separated UCI string
                count,
                seq_tuple[-1],        # last move (convenience)
                len(seq_tuple),       # depth in half-moves
            ])

    log.info(f'Saved: {OUTPUT_PATH}')

    # Print depth distribution — useful for verifying the corpus looks reasonable
    from collections import Counter
    depth_dist = Counter(len(seq) for seq in to_save)
    log.info('\nDepth distribution of saved sequences:')
    log.info('  (each unit of depth = 1 half-move = 0.5 full moves)')
    max_bar = max(depth_dist.values()) if depth_dist else 1
    for depth in sorted(depth_dist):
        bar_len = int(40 * depth_dist[depth] / max_bar)
        bar = '█' * bar_len
        log.info(f'  Depth {depth:2d}: {depth_dist[depth]:>8,}  {bar}')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    log.info('Script 01: Build Reference Corpus')
    log.info(f'Reference month:    {REFERENCE_MONTH}')
    log.info(f'Rating minimum:     2200 (both players)')
    log.info(f'Max book depth:     {MAX_PLIES_FOR_BOOK} half-moves ({MAX_PLIES_FOR_BOOK // 2} full moves)')
    log.info(f'Primary threshold:  {PRIMARY_BOOK_THRESHOLD} occurrences (applied in Script 02)')
    log.info(f'Checkpoint every:   {CHECKPOINT_INTERVAL:,} games')
    if MAX_GAMES_TO_PARSE:
        log.info(f'*** TEST MODE: stopping after {MAX_GAMES_TO_PARSE:,} games ***')
    log.info('')

    # ── Phase 1: Download ──────────────────────────────────────────────────────
    # Always call download_with_resume — it handles three cases automatically:
    #   1. File does not exist yet: starts a fresh download
    #   2. File exists but is incomplete (connection dropped): resumes from
    #      the last byte written, no progress lost
    #   3. File is already complete: server returns HTTP 416 and the function
    #      returns immediately with "File already fully downloaded."
    #
    # Never skip this call just because the file exists on disk. A partial
    # file looks identical to a complete one — only the server can tell the
    # difference via the Range header response.
    log.info('Phase 1: Checking download status...')
    if DOWNLOAD_PATH.exists():
        size_gb = DOWNLOAD_PATH.stat().st_size / 1e9
        log.info(f'  Found {size_gb:.1f} GB on disk. Checking if complete...')
    download_with_resume(REFERENCE_MONTH, DOWNLOAD_PATH)

    log.info('')

    # ── Phase 2–5: Process ────────────────────────────────────────────────────
    if OUTPUT_PATH.exists() and not CHECKPOINT_PATH.exists():
        log.info(f'Output already exists and no checkpoint found: {OUTPUT_PATH}')
        log.info('Script 01 appears already complete. Delete the output to rerun.')
        log.info('Next step: python scripts/02_parse_and_tag_games.py')
        return

    log.info('Phase 2-5: Streaming through games and counting opening sequences...')
    log.info('Safe to interrupt with Ctrl+C — progress is saved every 100,000 games.')
    log.info('')

    counts = build_sequence_counts()

    log.info('')

    # ── Phase 6: Save ─────────────────────────────────────────────────────────
    log.info('Phase 6: Saving output...')
    save_output(counts)

    # Clean up checkpoint files now that we have the final output
    for ck in [CHECKPOINT_A, CHECKPOINT_B, CHECKPOINT_LEGACY]:
        if ck.exists():
            ck.unlink()
            log.info(f'Checkpoint file deleted: {ck.name}')

    log.info('')
    log.info('Script 01 complete.')
    log.info(f'Output: {OUTPUT_PATH}')
    log.info('Next step: python scripts/02_parse_and_tag_games.py')


if __name__ == '__main__':
    main()
