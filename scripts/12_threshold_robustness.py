#!/usr/bin/env python3
"""
Script 12: Robustness check 1, book thresholds 5 and 20
==============================================================================
The registration (https://osf.io/n6tkx/) lists as robustness check 1:
"Book frequency threshold at 5 and 20, alongside the primary value of 10."

Changing the threshold changes where each player leaves the book, so the
boundary pairs must be rebuilt from the games, new positions need Stockfish,
and H1 must be refitted. This script does that in three stages:

    python scripts/12_threshold_robustness.py parse     (about 1-3 hours)
    python scripts/12_threshold_robustness.py voc       (about 1-3 hours)
    python scripts/12_threshold_robustness.py analyze   (about 10-20 minutes)

Each stage can be stopped with Ctrl+C and restarted with the same command;
it picks up where it left off.

DESIGN
    Players, not games, are sampled: every game of a fixed 10% of players
    (chosen by a hash of the username, so the choice never changes between
    runs). Each sampled player keeps all of their games, so the 3-pair
    minimum works the same way it did in the main analysis. All three
    thresholds are computed on the same games in one pass, so the threshold
    10 result on this subsample is the anchor the other two are compared
    with. Then each threshold goes through the same steps as the main
    pipeline: symmetric premove filter, registered 40-ply exit limit,
    3-pair minimum, the 200,000-per-band systematic sample, and the
    registered H1 model.

WHY YOU CAN TRUST THE RE-PARSE
    The script imports your own scripts/02_parse_and_tag_games.py and uses
    its game filter, clock parser, time-control parser and rating bands.
    Only the boundary search is rebuilt, so that all three thresholds come
    out of one pass. To prove the rebuilt search matches yours, it runs your
    process_game() on every 50th parsed game and stops with an error if the
    threshold-10 pairs differ in any field. The analyze stage then checks
    every threshold-10 pair against boundary_pairs.csv.

    The voc stage imports analyze_position() from your
    scripts/03_stockfish_covariate.py and first re-evaluates 20 positions
    that are already in stockfish_covariate.csv. If any result differs, it
    stops before doing any new work.

SPEED
    Script 02 fully parsed all 96.6 million games. This script reads the
    headers of every game as plain text and fully parses only Rapid games
    that pass your filter and involve a sampled player.

OUTPUTS
    data/processed/threshold_pairs.csv               (parse)
    data/processed/stockfish_covariate_thresholds.csv (voc)
    outputs/tables/threshold_robustness_parse.txt    (parse log)
    outputs/tables/threshold_robustness.txt          (results)
==============================================================================
"""

import csv
import importlib.util
import io
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = (10, 5, 20)          # 10 first: it is the anchor
PLAYER_SAMPLE_ONE_IN = 10         # 10% of players
SELF_CHECK_ONE_IN = 50            # compare with your process_game() this often
CHECKPOINT_EVERY = 200_000        # games
SAMPLE_SIZE = 1_000_000           # same as Script 04 (200,000 per band)
CHUNK = 500_000
D_THRESHOLD = 0.20
N_VOC_CHECK = 20
FIRST_MOVE_MAX_EXIT = 4    # exit ply 3 or 4: the book move was the untimed first move

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / 'scripts'
PROC = ROOT / 'data' / 'processed'
TABLES = ROOT / 'outputs' / 'tables'
PAIRS_OUT = PROC / 'threshold_pairs.csv'
PAIRS_PART = PROC / 'threshold_pairs.partial.csv'
PROGRESS = PROC / 'threshold_parse_progress.json'
VOC_MAIN = PROC / 'stockfish_covariate.csv'
VOC_NEW = PROC / 'stockfish_covariate_thresholds.csv'

OUT_COLS = ['threshold', 'game_id', 'player_color', 'player_rating',
            'rating_band', 'player_username', 'ply_at_exit',
            'last_book_thinking_time', 'first_oob_thinking_time',
            'last_book_is_premove', 'first_oob_is_premove',
            'last_book_fen', 'first_oob_fen']

EVENT_RE = re.compile(r'^\[Event "(.*)"\]\s*$')
TAG_RE = re.compile(r'^\[([A-Za-z0-9][A-Za-z0-9_+#=:-]*)\s+"(.*)"\]\s*$')


class Tee:
    def __init__(self, path, mode='w'):
        self.f = open(path, mode, encoding='utf-8')

    def write(self, t):
        sys.__stdout__.write(t)
        self.f.write(t)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def load_module(filename, name):
    path = SCRIPTS / filename
    if not path.exists():
        sys.exit(f'ERROR: {path} not found. This script uses your own copy.')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crc(s: str) -> int:
    return zlib.crc32(s.encode('utf-8'))


def is_sampled_player(username: str) -> bool:
    return crc(username) % PLAYER_SAMPLE_ONE_IN == 0


def normalize_fen(fen) -> str:
    return ' '.join(str(fen).split()[:4])


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])


# ==============================================================================
# STAGE 1: PARSE
# ==============================================================================

def load_counts(book_csv, min_threshold):
    df = pd.read_csv(book_csv)
    lowest = int(df['count'].min())
    print(f'book_sequences.csv: {len(df):,} sequences, lowest stored count {lowest}')
    if lowest > min_threshold:
        sys.exit(f'ERROR: the book file only keeps counts >= {lowest}, so a '
                 f'threshold of {min_threshold} cannot be applied. Script 01 '
                 'would need to be rerun with a lower cutoff.')
    df = df[df['count'] >= min_threshold]
    counts = {}
    for seq, c in zip(df['move_sequence'], df['count']):
        key = tuple(str(seq).split())
        counts[key] = max(counts.get(key, 0), int(c))
    maxlen = max(len(k) for k in counts)
    for t in sorted(set(THRESHOLDS)):
        print(f'  sequences with count >= {t:>2}: '
              f'{sum(1 for c in counts.values() if c >= t):,}')
    print(f'  longest sequence: {maxlen} plies')
    return counts, maxlen


def walk_game(game, headers, counts, maxlen, colors, s02):
    """
    Walk one game exactly as Script 02's process_game does, but compute the
    boundary at every threshold, for the sampled colours only.

    Returns ({threshold: {colour: record}}, {threshold: {colour: outcome}}).
    """
    import chess
    game_id = headers.get('Site', '').rsplit('/', 1)[-1]
    tc_str = headers.get('TimeControl', '')
    result = headers.get('Result', '*')
    try:
        white_rating = int(headers.get('WhiteElo', 0))
        black_rating = int(headers.get('BlackElo', 0))
    except (ValueError, TypeError):
        return {}, {t: {c: 'bad_rating' for c in colors} for t in THRESHOLDS}
    base_time, increment = s02.parse_time_control(tc_str)
    if base_time == 0:
        return {}, {t: {c: 'bad_time_control' for c in colors} for t in THRESHOLDS}

    board = game.board()
    node = game
    moves_uci, times, cnts, fens = [], [], [], []
    w_prev = b_prev = float(base_time)
    seq = []
    # A boundary can only fall at ply index <= maxlen + 1, because no longer
    # sequence is in the book. Later moves are only counted (for the 20-ply
    # minimum). python-chess has already checked every mainline move, so the
    # push that Script 02 guards with try/except cannot fail on them.
    limit = maxlen + 2
    extra = 0
    while node.variations:
        nxt = node.variations[0]
        move = nxt.move
        if move is None:
            break
        ply = len(moves_uci)
        if ply >= limit:
            while node.variations and node.variations[0].move is not None:
                node = node.variations[0]
                extra += 1
            break
        is_white = board.turn == chess.WHITE
        fen_pre = board.fen()
        clk = s02.parse_clock(nxt.comment)
        if clk is not None:
            if is_white:
                t = w_prev - clk + increment
                w_prev = clk
            else:
                t = b_prev - clk + increment
                b_prev = clk
        else:
            t = None
        u = move.uci()
        if ply < maxlen:
            seq.append(u)
            c = counts.get(tuple(seq), 0)
        else:
            c = 0
        moves_uci.append(u)
        times.append(t)
        cnts.append(c)
        fens.append(fen_pre)
        try:
            board.push(move)
        except Exception:
            break
        node = nxt

    total = len(moves_uci) + extra
    if total < s02.MIN_PLIES_FOR_INCLUSION:
        return {}, {t: {c: 'game_too_short' for c in colors} for t in THRESHOLDS}

    premove = s02.PREMOVE_THRESHOLD_SECONDS
    records = {t: {} for t in THRESHOLDS}
    outcomes = {t: {} for t in THRESHOLDS}
    for is_w, color, rating, username in (
            (True, 'white', white_rating, headers.get('White', '')),
            (False, 'black', black_rating, headers.get('Black', ''))):
        if color not in colors:
            continue
        band = s02.get_rating_band(rating)
        if band is None:
            for t in THRESHOLDS:
                outcomes[t][color] = 'no_band'
            continue
        plies = range(0 if is_w else 1, total, 2)
        for T in THRESHOLDS:
            last_b = first_o = None
            for p in plies:
                if (cnts[p] if p < len(cnts) else 0) >= T:
                    last_b = p
                elif first_o is None:
                    first_o = p
                    break
            if last_b is None or first_o is None:
                outcomes[T][color] = ('left_book_on_first_move' if last_b is None
                                      else 'never_left_book')
                continue
            lt, ot = times[last_b], times[first_o]
            if lt is None or ot is None:
                outcomes[T][color] = 'missing_clock'
                continue
            if lt < 0 or ot < 0:
                outcomes[T][color] = 'negative_time'
                continue
            outcomes[T][color] = 'pair'
            records[T][color] = {
                'game_id': game_id,
                'player_color': color,
                'player_rating': rating,
                'rating_band': band,
                'player_username': username,
                'ply_at_exit': first_o + 1,
                'last_book_move_uci': moves_uci[last_b],
                'first_oob_move_uci': moves_uci[first_o],
                'last_book_thinking_time': round(lt, 3),
                'first_oob_thinking_time': round(ot, 3),
                'last_book_is_premove': lt <= premove,
                'first_oob_is_premove': ot <= premove,
                'premove_suspect': ot <= premove,
                'time_control': tc_str,
                'game_result': result,
                'last_book_fen': fens[last_b],
                'first_oob_fen': fens[first_o],
            }
    return records, outcomes


def self_check(game, headers, book_set_primary, mine, colors, s02, game_id):
    out = s02.process_game(game, headers, book_set_primary)
    theirs = out[0] if isinstance(out, tuple) else out
    theirs = {r['player_color']: r for r in theirs}
    for color in colors:
        a = mine.get(color)
        b = theirs.get(color)
        if a is None and b is None:
            continue
        if a is None or b is None:
            return f'{game_id} {color}: one side has no pair ({a is None}, {b is None})'
        for k in b:
            if k in a and a[k] != b[k]:
                return f'{game_id} {color}: field {k}: this script {a[k]!r}, Script 02 {b[k]!r}'
    return None


def iter_games(text_stream, skip):
    """
    Yield one item per game: ('other', event) for games that cannot pass the
    Rapid filter, ('rapid', lines) for candidates. The first `skip` games are
    passed over without being yielded (used when resuming).
    """
    n = 0
    buf = None
    for line in text_stream:
        if line.startswith('[Event '):
            if buf is not None:
                yield 'rapid', buf
                buf = None
            n += 1
            if n <= skip:
                continue
            m = EVENT_RE.match(line)
            if m and m.group(1) == 'Rated Rapid game':
                buf = [line]
            else:
                yield 'other', (m.group(1) if m else '')
        elif buf is not None:
            buf.append(line)
    if buf is not None:
        yield 'rapid', buf


def header_dict(lines):
    h = {}
    for line in lines:
        if not line.strip():
            break
        m = TAG_RE.match(line)
        if m:
            h[m.group(1)] = m.group(2).replace('\\"', '"').replace('\\\\', '\\')
    return h


def truncate_rows(path, n_rows):
    """Keep the header and the first n_rows data rows; cut the rest."""
    with open(path, 'rb+') as f:
        pos = 0
        for i, line in enumerate(f):
            if i > n_rows:
                break
            pos += len(line)
        f.seek(pos)
        f.truncate()


def stage_parse():
    import chess.pgn
    import zstandard as zstd
    from tqdm import tqdm

    TABLES.mkdir(parents=True, exist_ok=True)
    s02 = load_module('02_parse_and_tag_games.py', 's02')
    if PAIRS_OUT.exists():
        print(f'{PAIRS_OUT.name} already exists. Delete it to parse again.')
        return

    resume = PROGRESS.exists() and PAIRS_PART.exists()
    tee = Tee(TABLES / 'threshold_robustness_parse.txt', 'a' if resume else 'w')
    sys.stdout = tee
    try:
        print('Script 12, stage 1: parse')
        print(f'Thresholds {THRESHOLDS}; players sampled 1 in '
              f'{PLAYER_SAMPLE_ONE_IN} by CRC32 of username')
        print(f'PGN source: {s02.DOWNLOAD_PATH}')
        if not Path(s02.DOWNLOAD_PATH).exists():
            sys.exit('ERROR: the PGN file is not there. Script 02 downloads it to '
                     'that path.')
        counts, maxlen = load_counts(s02.BOOK_CSV_PATH, min(THRESHOLDS))
        primary = int(s02.PRIMARY_BOOK_THRESHOLD)
        if primary not in THRESHOLDS:
            sys.exit(f'ERROR: primary threshold {primary} is not in {THRESHOLDS}.')
        book_set_primary = s02.load_book_set(s02.BOOK_CSV_PATH, primary)

        if resume:
            st = json.loads(PROGRESS.read_text())
            truncate_rows(PAIRS_PART, st['rows_written'])
            print(f"Resuming after game {st['games_read']:,} "
                  f"({st['rows_written']:,} rows kept)")
        else:
            st = {'games_read': 0, 'rows_written': 0, 'parsed': 0,
                  'self_checks': 0, 'game_excl': {}, 'outcomes': {}}

        def bump(d, *keys):
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = d.get(keys[-1], 0) + 1

        mode = 'a' if resume else 'w'
        failed = None
        with open(s02.DOWNLOAD_PATH, 'rb') as fz, \
                open(PAIRS_PART, mode, newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=OUT_COLS, extrasaction='ignore')
            if not resume:
                writer.writeheader()
            reader = zstd.ZstdDecompressor().stream_reader(fz)
            text = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            bar = tqdm(desc='games read', unit=' games', initial=st['games_read'],
                       dynamic_ncols=True, mininterval=2.0)
            for kind, lines in iter_games(text, st['games_read']):
                st['games_read'] += 1
                if st['games_read'] % 10_000 == 0:
                    bar.update(10_000)
                if kind == 'other':
                    bump(st['game_excl'], 'not_rapid')
                    if 'rapid' in lines.lower():
                        # Rapid games whose Event header is not exactly
                        # "Rated Rapid game" (for example tournament games).
                        bump(st.setdefault('rapid_other', {}),
                             lines.split(' http')[0])
                else:
                    headers = header_dict(lines)
                    ok, reason = s02.game_passes_filter(headers)
                    if not ok:
                        bump(st['game_excl'], reason)
                    else:
                        colors = [c for c, k in (('white', 'White'), ('black', 'Black'))
                                  if is_sampled_player(headers.get(k, ''))]
                        if not colors:
                            bump(st['game_excl'], 'passed_filter_not_sampled')
                        else:
                            bump(st['game_excl'], 'passed_filter_sampled')
                            game = chess.pgn.read_game(io.StringIO(''.join(lines)))
                            if game is None:
                                bump(st['game_excl'], 'unparseable')
                            else:
                                hdr = dict(game.headers)
                                if not s02.game_passes_filter(hdr)[0]:
                                    bump(st['game_excl'], 'header_parse_mismatch')
                                    continue
                                st['parsed'] += 1
                                recs, outs = walk_game(game, hdr, counts, maxlen,
                                                       colors, s02)
                                for T, d in outs.items():
                                    for color, outcome in d.items():
                                        rating = hdr.get('WhiteElo' if color == 'white'
                                                         else 'BlackElo', '0')
                                        try:
                                            band = s02.get_rating_band(int(rating))
                                        except ValueError:
                                            band = None
                                        bump(st['outcomes'], str(T), str(band), outcome)
                                gid = hdr.get('Site', '').rsplit('/', 1)[-1]
                                if crc(gid) % SELF_CHECK_ONE_IN == 0:
                                    err = self_check(game, hdr, book_set_primary,
                                                     recs.get(primary, {}), colors,
                                                     s02, gid)
                                    st['self_checks'] += 1
                                    if err:
                                        failed = err
                                        break
                                for T in THRESHOLDS:
                                    for rec in recs.get(T, {}).values():
                                        writer.writerow({'threshold': T, **rec})
                                        st['rows_written'] += 1
                if st['games_read'] % CHECKPOINT_EVERY == 0:
                    fout.flush()
                    tmp = PROGRESS.with_suffix('.tmp')
                    tmp.write_text(json.dumps(st))
                    tmp.replace(PROGRESS)
            bar.close()

        if failed:
            PAIRS_PART.unlink(missing_ok=True)
            PROGRESS.unlink(missing_ok=True)
            print('\nSELF-CHECK FAILED: this script and your Script 02 disagree on '
                  'a pair.')
            print(f'  {failed}')
            print('The partial output was deleted. Send this message and your '
                  'current Script 02.')
            sys.exit(1)

        print('\nPARSE LOG')
        print(f'games read: {st["games_read"]:,}   (Script 02 read 96,587,411)')
        print('game-level filter results (compare the exclusion counts with the '
              'Script 02 log):')
        for k, v in sorted(st['game_excl'].items(), key=lambda kv: -kv[1]):
            print(f'  {k:<32}{v:>14,}')
        ro = st.get('rapid_other', {})
        if ro:
            print('\nGames counted as not_rapid although their Event header '
                  'mentions Rapid\n(Script 02 keeps only the exact header '
                  '"Rated Rapid game"):')
            for k, v in sorted(ro.items(), key=lambda kv: -kv[1]):
                print(f'  {k[:60]:<62}{v:>12,}')
        print(f'games fully parsed: {st["parsed"]:,}')
        print(f'self-checks against your process_game(): {st["self_checks"]:,}, '
              f'all matched')
        print(f'rows written: {st["rows_written"]:,}')
        print('\nPer-band outcome for each sampled player-game, by threshold.')
        print('This is also the per-band missingness the registration asks for '
              '(threshold 10).')
        for T in THRESHOLDS:
            o = st['outcomes'].get(str(T), {})
            kinds = sorted({k for d in o.values() for k in d})
            print(f'\nthreshold {T}')
            print(f'{"band":<8}' + ''.join(f'{k[:22]:>24}' for k in kinds))
            for band in sorted(o):
                tot = sum(o[band].values())
                print(f'{band:<8}' + ''.join(
                    f'{o[band].get(k, 0):>14,} ({100 * o[band].get(k, 0) / tot:4.1f}%)'
                    for k in kinds))
        # Finish the log first, then publish the file, so an interruption at
        # any point leaves a state the next run can resume from.
        PAIRS_PART.replace(PAIRS_OUT)
        PROGRESS.unlink(missing_ok=True)
        print(f'\nSaved: {PAIRS_OUT}\nNext: python scripts/12_threshold_robustness.py voc')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


# ==============================================================================
# STAGE 2: VOC
# ==============================================================================

def trim_partial_line(path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, 'rb+') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read()
        if tail.endswith(b'\n'):
            return
        cut = tail.rfind(b'\n')
        f.truncate(size - len(tail) + cut + 1 if cut >= 0 else 0)


def stage_voc():
    import chess.engine
    from tqdm import tqdm

    if not PAIRS_OUT.exists():
        sys.exit('ERROR: run the parse stage first.')
    s03 = load_module('03_stockfish_covariate.py', 's03')
    make_board = getattr(s03, 'make_board', None)
    if make_board is None:
        import chess
        make_board = lambda f: chess.Board(f + ' 0 1')  # noqa: E731

    print('Script 12, stage 2: Stockfish for new positions')
    need = set()
    for ch in pd.read_csv(PAIRS_OUT, usecols=['last_book_fen', 'first_oob_fen'],
                          chunksize=CHUNK):
        need.update(ch['last_book_fen'].map(normalize_fen))
        need.update(ch['first_oob_fen'].map(normalize_fen))
    cached = pd.read_csv(VOC_MAIN)
    have = set(cached['fen_normalized'])
    trim_partial_line(VOC_NEW)
    done = set()
    if VOC_NEW.exists() and VOC_NEW.stat().st_size > 0:
        done = set(pd.read_csv(VOC_NEW, on_bad_lines='skip')['fen_normalized'])
    todo = sorted(need - have - done)
    print(f'positions used at any threshold: {len(need):,}')
    print(f'already in stockfish_covariate.csv: {len(need & have):,}')
    print(f'already done by an earlier run of this stage: {len(done):,}')
    print(f'to evaluate now: {len(todo):,}')

    path = Path(s03.STOCKFISH_PATH)
    if not path.exists() and path.with_suffix('.exe').exists():
        path = path.with_suffix('.exe')
    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    engine.configure({'Threads': s03.STOCKFISH_THREADS})
    try:
        # Self-check: reproduce positions computed by Script 03.
        ref = cached[~((cached['eval_depth1'] == 0) & (cached['eval_depth15'] == 0))]
        ref = ref.sort_values('fen_normalized').reset_index(drop=True)
        pick = ref.iloc[np.linspace(0, len(ref) - 1, N_VOC_CHECK).astype(int)]
        bad = 0
        for _, r in pick.iterrows():
            res = s03.analyze_position(make_board(r['fen_normalized']), engine)
            same = (round(res['voc'], 2) == round(float(r['voc']), 2)
                    and int(res['eval_depth1']) == int(r['eval_depth1'])
                    and int(res['eval_depth15']) == int(r['eval_depth15']))
            if not same:
                bad += 1
                print(f'  MISMATCH {r["fen_normalized"]}: now {res}, before '
                      f'voc {r["voc"]} d1 {r["eval_depth1"]} d15 {r["eval_depth15"]}')
        print(f'self-check: {N_VOC_CHECK - bad} of {N_VOC_CHECK} positions reproduce '
              f'stockfish_covariate.csv exactly')
        if bad:
            sys.exit('STOPPED: Stockfish results differ from Script 03. Check that '
                     'config.py, the Stockfish binary and Script 03 are the ones '
                     'used for stockfish_covariate.csv.')

        new_file = not VOC_NEW.exists() or VOC_NEW.stat().st_size == 0
        with open(VOC_NEW, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['fen_normalized', 'voc', 'eval_depth1', 'eval_depth15'])
            for i, fen in enumerate(tqdm(todo, desc='Stockfish', unit=' pos')):
                try:
                    res = s03.analyze_position(make_board(fen), engine)
                except Exception:
                    res = {'voc': 0.0, 'eval_depth1': 0, 'eval_depth15': 0}
                w.writerow([fen, round(res['voc'], 2), res['eval_depth1'],
                            res['eval_depth15']])
                if (i + 1) % 50 == 0:
                    f.flush()
    finally:
        engine.quit()
    print(f'Saved: {VOC_NEW}\nNext: python scripts/12_threshold_robustness.py analyze')


# ==============================================================================
# STAGE 3: ANALYZE
# ==============================================================================

def fit_ri_reml(y, X, codes):
    """Random-intercept REML; identical to the function in Script 10."""
    from scipy import optimize
    N, p = X.shape
    n_g = np.bincount(codes).astype(float)
    gy = np.bincount(codes, weights=y) / n_g
    gX = np.column_stack([np.bincount(codes, weights=X[:, j]) / n_g for j in range(p)])
    gyr, gXr = gy[codes], gX[codes]

    def pieces(lam):
        th = (1.0 - 1.0 / np.sqrt(1.0 + n_g * lam))[codes]
        ys, Xs = y - th * gyr, X - th[:, None] * gXr
        XtX = Xs.T @ Xs
        b = np.linalg.solve(XtX, Xs.T @ ys)
        r = ys - Xs @ b
        return b, float(r @ r), XtX

    def crit(t):
        lam = np.exp(t)
        b, rss, XtX = pieces(lam)
        return ((N - p) * np.log(rss / (N - p)) + np.sum(np.log1p(n_g * lam))
                + np.linalg.slogdet(XtX)[1])

    res = optimize.minimize_scalar(crit, bounds=(np.log(1e-8), np.log(1e3)),
                                   method='bounded', options={'xatol': 1e-10})
    lam = float(np.exp(res.x))
    b, rss, XtX = pieces(lam)
    s2e = rss / (N - p)
    se = np.sqrt(np.diag(s2e * np.linalg.inv(XtX)))
    return b, se, s2e, lam * s2e


def systematic_sample(D):
    """Script 04's selection: per band, evenly spaced rows in file order."""
    bands = sorted(D['rating_band'].unique())
    per_band = SAMPLE_SIZE // len(bands)
    parts = []
    for band in bands:
        sub = D[D['rating_band'] == band]
        N = len(sub)
        T = min(per_band, N)
        pos = np.arange(N)
        parts.append(sub[(pos * T) // N != ((pos + 1) * T) // N])
    return pd.concat(parts)


def h1_fit(S, with_voc=True):
    n = len(S)
    y = np.concatenate([np.log(S['last_book_thinking_time'].to_numpy(float)),
                        np.log(S['first_oob_thinking_time'].to_numpy(float))])
    is_oob = np.r_[np.zeros(n), np.ones(n)]
    cols = [np.ones(2 * n), is_oob]
    if with_voc:
        cols.append(np.concatenate([S['lb_voc'].to_numpy(float),
                                    S['oob_voc'].to_numpy(float)]))
    codes = pd.factorize(np.tile(S['player_username'].to_numpy(), 2))[0]
    b, se, s2e, s2u = fit_ri_reml(y, np.column_stack(cols), codes)
    return b[1], se[1], b[1] / np.sqrt(s2e), s2e, s2u


def stage_analyze():
    sys.path.insert(0, str(ROOT))
    try:
        from config import MIN_TRANSITIONS_PER_PLAYER as MIN_PAIRS
    except Exception:
        MIN_PAIRS = 3
    try:
        from config import MAX_PLIES_FOR_BOOK as MAX_EXIT_PLY
    except Exception:
        MAX_EXIT_PLY = 40
    if not PAIRS_OUT.exists():
        sys.exit('ERROR: run the parse stage first.')
    TABLES.mkdir(parents=True, exist_ok=True)
    tee = Tee(TABLES / 'threshold_robustness.txt')
    sys.stdout = tee
    try:
        print('Script 12, stage 3: robustness check 1 (book thresholds)')
        print('Registered at https://osf.io/n6tkx/ : "Book frequency threshold at '
              '5 and 20,\nalongside the primary value of 10."')
        print(f'Players sampled: 1 in {PLAYER_SAMPLE_ONE_IN} (all of their games). '
              f'Minimum pairs {MIN_PAIRS}. Exit limit ply {MAX_EXIT_PLY}.')

        V = pd.read_csv(VOC_MAIN)
        if VOC_NEW.exists():
            V = pd.concat([V, pd.read_csv(VOC_NEW, on_bad_lines='skip')])
        V = V.dropna(subset=['voc']).drop_duplicates('fen_normalized')
        voc = dict(zip(V['fen_normalized'], V['voc']))
        zero = set(V.loc[(V['eval_depth1'] == 0) & (V['eval_depth15'] == 0),
                         'fen_normalized'])
        print(f'VOC lookup: {len(voc):,} positions ({len(zero):,} carry the '
              f'zero failure code)')

        results, results_c = [], []
        for T in THRESHOLDS:
            parts = []
            for ch in pd.read_csv(PAIRS_OUT, chunksize=CHUNK, low_memory=False,
                                  usecols=[c for c in OUT_COLS if c != 'player_rating']):
                ch = ch[ch['threshold'] == T]
                if ch.empty:
                    continue
                lb = ch['last_book_fen'].map(normalize_fen)
                ob = ch['first_oob_fen'].map(normalize_fen)
                ch = ch.drop(columns=['last_book_fen', 'first_oob_fen']).assign(
                    lb_voc=lb.map(voc).to_numpy(), oob_voc=ob.map(voc).to_numpy(),
                    zero_coded=(lb.isin(zero) | ob.isin(zero)).to_numpy())
                parts.append(ch)
            D = pd.concat(parts, ignore_index=True)
            D['player_username'] = D['player_username'].astype(str)
            n0 = len(D)
            prem = (to_bool(D['last_book_is_premove'])
                    | to_bool(D['first_oob_is_premove']))
            D = D[~prem]
            n_prem = n0 - len(D)
            miss = D['lb_voc'].isna() | D['oob_voc'].isna()
            D = D[~miss]
            cap = D['ply_at_exit'] > MAX_EXIT_PLY
            D = D[~cap]
            if T == THRESHOLDS[0]:
                anchor = D[['game_id', 'player_color', 'ply_at_exit',
                            'last_book_thinking_time', 'first_oob_thinking_time']].copy()
            D_pre = D
            npp = D.groupby('player_username')['player_username'].transform('size')
            n_before_min = len(D)
            D = D[npp >= MIN_PAIRS]
            S = systematic_sample(D)
            b, se, d, s2e, s2u = h1_fit(S)
            b0, _, d0, _, _ = h1_fit(S, with_voc=False)
            Sz = S[~S['zero_coded']]
            bz, _, dz, _, _ = h1_fit(Sz) if len(Sz) < len(S) else (b, se, d, 0, 0)
            met = b > 0 and d >= D_THRESHOLD
            print(f'\n--- threshold {T} ---')
            print(f'pairs {n0:,}; premove pairs removed {n_prem:,}; missing VOC '
                  f'{int(miss.sum()):,}; exits after ply {MAX_EXIT_PLY} '
                  f'{int(cap.sum()):,}; pairs of players under {MIN_PAIRS} pairs '
                  f'{n_before_min - len(D):,}')
            print(f'filtered pairs {len(D):,}; sample {len(S):,} '
                  f'({", ".join(f"{k} {v:,}" for k, v in S["rating_band"].value_counts().sort_index().items())})')
            print(f'mean exit ply {D["ply_at_exit"].mean():.2f}; mean VOC book '
                  f'{D["lb_voc"].mean():.1f}, out of book {D["oob_voc"].mean():.1f}')
            print(f'H1 (registered model): beta {b:+.4f} (95% CI {b - 1.96 * se:.4f} '
                  f'to {b + 1.96 * se:.4f}), d {d:.4f} -> rule '
                  f'{"MET" if met else "NOT MET"}')
            print(f'  without VOC: beta {b0:+.4f}, d {d0:.4f}')
            if len(Sz) < len(S):
                print(f'  without {len(S) - len(Sz):,} zero-coded pairs: beta {bz:+.4f}, '
                      f'd {dz:.4f}')
            results.append((T, len(D), len(S), S['player_username'].nunique(), b, se,
                            d, met))

            # Corrected for untimed first moves (see Script 13): Lichess does not
            # time a player's first move, so drop pairs whose book move was the
            # first move and repeat the same steps.
            keep_c = D_pre['ply_at_exit'] > FIRST_MOVE_MAX_EXIT
            Dc = D_pre[keep_c]
            nppc = Dc.groupby('player_username')['player_username'].transform('size')
            Dc = Dc[nppc >= MIN_PAIRS]
            Sc = systematic_sample(Dc)
            bc, sec, dc, _, _ = h1_fit(Sc)
            metc = bc > 0 and dc >= D_THRESHOLD
            print(f'  corrected for untimed first moves ({int((~keep_c).sum()):,} pairs '
                  f'with exit ply <= {FIRST_MOVE_MAX_EXIT} removed): beta {bc:+.4f} '
                  f'(95% CI {bc - 1.96 * sec:.4f} to {bc + 1.96 * sec:.4f}), d {dc:.4f} '
                  f'-> rule {"MET" if metc else "NOT MET"}')
            results_c.append((T, len(Dc), len(Sc), Sc['player_username'].nunique(), bc,
                              sec, dc, metc))

        # ── Agreement with the main pipeline at the primary threshold ──────
        print('\n' + '=' * 78)
        print(f'CHECK: threshold {THRESHOLDS[0]} pairs here versus boundary_pairs.csv')
        print('=' * 78)
        bp = PROC / 'boundary_pairs.csv'
        if bp.exists():
            keep = []
            for ch in pd.read_csv(bp, chunksize=CHUNK, low_memory=False,
                                  usecols=['game_id', 'player_color', 'player_username',
                                           'ply_at_exit', 'last_book_thinking_time',
                                           'first_oob_thinking_time',
                                           'last_book_is_premove',
                                           'first_oob_is_premove']):
                ch = ch[ch['player_username'].astype(str).map(is_sampled_player)]
                prem = (to_bool(ch['last_book_is_premove'])
                        | to_bool(ch['first_oob_is_premove']))
                keep.append(ch[~prem & (ch['ply_at_exit'] <= MAX_EXIT_PLY)])
            B = pd.concat(keep, ignore_index=True)
            m = B.merge(anchor, on=['game_id', 'player_color'], how='outer',
                        suffixes=('_main', '_here'), indicator=True)
            both = m[m['_merge'] == 'both']
            same = ((both['ply_at_exit_main'] == both['ply_at_exit_here'])
                    & (both['last_book_thinking_time_main']
                       == both['last_book_thinking_time_here'])
                    & (both['first_oob_thinking_time_main']
                       == both['first_oob_thinking_time_here']))
            print('Both sides: sampled players, premove pairs removed, exits after '
                  f'ply {MAX_EXIT_PLY} removed,\nbefore the 3-pair rule.')
            print(f'pairs in boundary_pairs.csv: {len(B):,};  pairs here: {len(anchor):,}')
            print(f'in both files: {len(both):,}; identical exit ply and times: '
                  f'{int(same.sum()):,}')
            n_left = int((m["_merge"] == "left_only").sum())
            n_right = int((m["_merge"] == "right_only").sum())
            print(f'only in boundary_pairs.csv: {n_left:,};  only here: {n_right:,}')
            if n_left == 0 and n_right == 0 and int(same.sum()) == len(both):
                print('PASS: identical pairs. The re-parse reproduces the main pipeline.')
            else:
                print('MISMATCH: the re-parse does not reproduce boundary_pairs.csv. '
                      'Do not use\nthe threshold results until this is explained.')
        else:
            print('boundary_pairs.csv not found; skipped.')

        print('\n' + '=' * 78)
        print('SUMMARY: robustness check 1')
        print('=' * 78)
        print(f'{"threshold":>10}{"filtered":>12}{"sample":>10}{"players":>10}'
              f'{"beta":>9}{"95% CI":>20}{"d":>8}  rule')
        for T, nd, ns, npl, b, se, d, met in results:
            print(f'{T:>10}{nd:>12,}{ns:>10,}{npl:>10,}{b:>9.4f}'
                  f'{f"[{b - 1.96 * se:.4f}, {b + 1.96 * se:.4f}]":>20}{d:>8.4f}  '
                  f'{"MET" if met else "NOT MET"}')
        print(f'\nThreshold {THRESHOLDS[0]} on this 1-in-{PLAYER_SAMPLE_ONE_IN} '
              'player sample should land close to the main result from Script 10.')
        print('\nCorrected for untimed first moves (pairs with exit ply <= '
              f'{FIRST_MOVE_MAX_EXIT} removed; compare with Script 13):')
        print(f'{"threshold":>10}{"filtered":>12}{"sample":>10}{"players":>10}'
              f'{"beta":>9}{"95% CI":>20}{"d":>8}  rule')
        for T, nd, ns, npl, b, se, d, met in results_c:
            print(f'{T:>10}{nd:>12,}{ns:>10,}{npl:>10,}{b:>9.4f}'
                  f'{f"[{b - 1.96 * se:.4f}, {b + 1.96 * se:.4f}]":>20}{d:>8.4f}  '
                  f'{"MET" if met else "NOT MET"}')
        print('Saved: outputs/tables/threshold_robustness.txt')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == '__main__':
    stages = {'parse': stage_parse, 'voc': stage_voc, 'analyze': stage_analyze}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        print(__doc__)
        print('Usage: python scripts/12_threshold_robustness.py parse|voc|analyze')
        sys.exit(1)
    stages[sys.argv[1]]()
