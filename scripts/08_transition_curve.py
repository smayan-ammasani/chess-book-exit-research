#!/usr/bin/env python3
"""
Script 08: Transition curve analysis (EXPLORATORY), version 3
==============================================================================
STATUS: EXPLORATORY. Proposed after the confirmatory results were observed.
Not a registered test, and it does not substitute for H1 under any outcome.
See https://osf.io/n6tkx/

WHAT CHANGED IN VERSION 3
    First moves are removed. Lichess does not time either player's first
    move (the clock starts after Black's first move, and neither first move
    earns an increment), so the recorded "thinking time" of a first move is
    just the increment: 0 s in games without one, 5 to 20 s in games with
    one. Rows at absolute ply 1 or 2 are dropped, and the script first shows
    how often their recorded time equals the increment. Version 2 kept them,
    which raised the in-book cell means at the earliest positions and pulled
    the within-game in-book slopes down.

WHAT CHANGED FROM VERSION 1 (in version 2)
    1. Duplicate rows are removed. When Script 02 resumed after the crash it
       truncated boundary_pairs.csv but not transition_window.csv, so about
       1,164 player-games were written twice. Rows are now deduplicated on
       (game_id, player_color, relative_move), and the count is printed.

    2. Confidence intervals are fixed. Version 1 bootstrapped a random
       50,000-row subsample for any cell larger than that. Those intervals
       were centered on the subsample, not on the reported mean, and were
       as wide as a 50,000-row cell. Version 2 uses every row, with standard
       errors clustered by player, because one player can appear in many
       games.

    3. Within-game estimates are added. Coverage is 17% of player-games at
       move -5 and 100% at -1, so the cell means at different positions come
       from different sets of games, and a trend across cell means can come
       from that mix rather than from anything happening inside a game.
       Version 2 measures the boundary step, the move-to-move changes around
       it, and the in-book trend inside each player-game.

    4. The recovery section is removed. It extrapolated the in-book trend
       past the boundary, which is not valid.

    5. Complete-window sensitivity: the curve is recomputed on player-games
       that have every position from -5 to +10.

PREMOVES
    Cell means drop individual premoves (recorded time <= 0.5 s), as in
    version 1. Every within-game comparison requires that every move it uses
    is not a premove, the same symmetric rule as H1.

OUTPUT
    outputs/tables/transition_curve.txt   the report
    outputs/tables/transition_curve.csv   the curve, read by Script 09

USAGE
    python scripts/08_transition_curve.py
==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREMOVE_THRESHOLD = 0.5
Z95 = 1.959964
POSITIONS = list(range(-5, 11))
IN_BOOK = [-5, -4, -3, -2, -1]
COLS = ['game_id', 'player_color', 'rating_band', 'player_username',
        'relative_move', 'thinking_time', 'is_premove']
FIRST_MOVE_COLS = ['absolute_ply', 'time_control']   # needed to find first moves


def increment_of(tc) -> float:
    """Increment in seconds from a TimeControl string such as '600+5'."""
    try:
        return float(str(tc).split('+', 1)[1])
    except (IndexError, ValueError):
        return np.nan


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def write(self, t):
        sys.__stdout__.write(t)
        self.f.write(t)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def resolve():
    here = Path(__file__).resolve()
    for base in (here.parent.parent, Path.cwd()):
        p = base / 'data' / 'processed' / 'transition_window.csv'
        if p.exists():
            out = base / 'outputs' / 'tables'
            out.mkdir(parents=True, exist_ok=True)
            return p, out / 'transition_curve.txt', out / 'transition_curve.csv'
    root = here.parent.parent
    return (root / 'data' / 'processed' / 'transition_window.csv',
            root / 'outputs' / 'tables' / 'transition_curve.txt',
            root / 'outputs' / 'tables' / 'transition_curve.csv')


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])


def header(title):
    print('\n' + '=' * 84)
    print(title)
    print('=' * 84)


def clustered_mean(values, clusters):
    """Mean with a standard error clustered on `clusters`."""
    v = np.asarray(values, float)
    n = len(v)
    if n < 2:
        return np.nan, np.nan, n, 0
    m = v.mean()
    codes, uniq = pd.factorize(np.asarray(clusters))
    G = len(uniq)
    s = np.bincount(codes, weights=v - m)
    se = np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n if G > 1 else np.nan
    return m, se, n, G


def fe_slope(pg, x, y, clusters):
    """
    Within-group (fixed-effects) slope of y on x, groups = player-games,
    standard error clustered on player.
    """
    d = pd.DataFrame({'pg': pg, 'x': x, 'y': y, 'c': clusters})
    d['xt'] = d['x'] - d.groupby('pg')['x'].transform('mean')
    d['yt'] = d['y'] - d.groupby('pg')['y'].transform('mean')
    sxx = float((d['xt'] ** 2).sum())
    if sxx == 0:
        return np.nan, np.nan, 0
    b = float((d['xt'] * d['yt']).sum()) / sxx
    d['score'] = d['xt'] * (d['yt'] - b * d['xt'])
    sc = d.groupby('c')['score'].sum().to_numpy()
    G = len(sc)
    se = np.sqrt(G / (G - 1) * np.sum(sc ** 2)) / sxx if G > 1 else np.nan
    return b, se, d['pg'].nunique()


def fmt_ci(m, se):
    return f'[{m - Z95 * se:.4f}, {m + Z95 * se:.4f}]'


def main():
    path, txt_path, csv_path = resolve()
    if not path.exists():
        print(f'ERROR: {path} not found.')
        sys.exit(1)

    tee = Tee(txt_path)
    sys.stdout = tee
    try:
        print('Script 08: Transition curve analysis, version 3')
        print('STATUS: EXPLORATORY. Proposed after the confirmatory results were')
        print('observed. Not a registered test. See https://osf.io/n6tkx/')

        # ── Load and clean ────────────────────────────────────────────────
        header('DATA')
        print(f'Loading {path.name} ({path.stat().st_size / 1e9:.2f} GB) ...')
        available = set(pd.read_csv(path, nrows=0).columns)
        can_drop_first = all(c in available for c in FIRST_MOVE_COLS)
        cols = COLS + (FIRST_MOVE_COLS if can_drop_first else [])
        # Text columns are read as categories: each distinct string is stored
        # once, which keeps a 13-million-row file within a laptop's memory.
        df = pd.read_csv(path, usecols=cols,
                         dtype={'game_id': 'category', 'player_color': 'category',
                                'rating_band': 'category',
                                'player_username': 'category',
                                'time_control': 'category'})
        n_raw = len(df)
        print(f'rows read: {n_raw:,}')
        df['relative_move'] = pd.to_numeric(df['relative_move'], errors='coerce')
        df['thinking_time'] = pd.to_numeric(df['thinking_time'], errors='coerce')
        df = df.dropna(subset=['game_id', 'player_color', 'rating_band',
                               'player_username', 'relative_move', 'thinking_time'])
        df['relative_move'] = df['relative_move'].astype(np.int16)
        df = df[df['relative_move'].isin(POSITIONS)]
        print(f'rows with every needed field: {len(df):,} '
              f'(dropped {n_raw - len(df):,})')

        # Integer key per player-game (game x colour) and per player.
        k = len(df['player_color'].cat.categories)
        pg_raw = (df['game_id'].cat.codes.to_numpy(np.int64) * k
                  + df['player_color'].cat.codes.to_numpy(np.int64))
        dup = pd.DataFrame({'pg': pg_raw, 'r': df['relative_move'].to_numpy()}
                           ).duplicated(keep='first').to_numpy()
        n_dup_pg = len(np.unique(pg_raw[dup]))
        print(f'duplicate rows removed: {int(dup.sum()):,} '
              f'(from {n_dup_pg:,} player-games written twice)')
        keep_cols = ['rating_band', 'relative_move', 'thinking_time', 'is_premove']
        if can_drop_first:
            keep_cols += FIRST_MOVE_COLS
        df = df.loc[~dup, keep_cols].assign(
            pg=pd.factorize(pg_raw[~dup])[0],
            pl=df['player_username'].cat.codes.to_numpy()[~dup])
        n_pg = int(df['pg'].max()) + 1
        print(f'player-games: {n_pg:,}   players: {df["pl"].nunique():,}')

        # First moves. Lichess does not time either player's first move, so
        # the recorded time is the increment, not a thinking time.
        if can_drop_first:
            ply = pd.to_numeric(df['absolute_ply'], errors='coerce').to_numpy()
            first = ply <= 2
            cats = df['time_control'].cat.categories
            lookup = np.array([increment_of(c) for c in cats] + [np.nan])
            inc = lookup[df['time_control'].cat.codes.to_numpy()]
            eq = np.abs(df['thinking_time'].to_numpy(float) - inc) < 1e-6
            print(f"\nfirst moves of the game (absolute ply 1 or 2): {int(first.sum()):,} "
                  f'rows')
            if first.any():
                print(f'  recorded time equals the increment: {100 * eq[first].mean():.2f}% '
                      f'(all other rows: {100 * eq[~first].mean():.2f}%)')
                by_pos = df.loc[first, 'relative_move'].value_counts().sort_index()
                print('  by position: ' + ', '.join(f'{int(r)}: {int(n):,}'
                                                    for r, n in by_pos.items()))
            print('  Lichess does not time first moves, so these rows are removed.')
            df = df.loc[~first].drop(columns=FIRST_MOVE_COLS)
        else:
            print('\nWARNING: absolute_ply or time_control is missing from the window '
                  'file, so\nfirst moves (which Lichess does not time) could not be '
                  'removed.')

        df['is_premove'] = (to_bool(df['is_premove'])
                            | (df['thinking_time'] <= PREMOVE_THRESHOLD))
        df['log_time'] = np.where(df['is_premove'], np.nan,
                                  np.log(df['thinking_time'].clip(lower=1e-9)))
        print(f'premove rows (kept for coverage counts, excluded from means): '
              f'{int(df["is_premove"].sum()):,} '
              f'({100 * df["is_premove"].mean():.1f}%)')
        bands = sorted(df['rating_band'].dropna().unique().tolist())

        present = df.groupby('relative_move')['pg'].nunique()
        print('\nCoverage (player-games with each position present):')
        for r in POSITIONS:
            print(f'  {r:>3}: {present.get(r, 0):>9,}  '
                  f'({100 * present.get(r, 0) / n_pg:5.1f}%)')

        # ── The curve ─────────────────────────────────────────────────────
        header('THE CURVE: mean log thinking time by position (cell means)')
        print('Position 0 = first out-of-book move; negative = still in book.')
        print('95% CIs use standard errors clustered by player.')
        ok = df.dropna(subset=['log_time'])
        present_bp = df.groupby(['rating_band', 'relative_move'],
                                observed=True)['pg'].nunique()
        rows = []
        for band in bands:
            print(f'\n--- {band} ---')
            print(f'{"rel":>5}{"n":>10}{"players":>10}{"mean log":>10}{"95% CI":>22}'
                  f'{"seconds":>9}{"vs -1":>9}')
            sub = ok[ok['rating_band'] == band]
            ref = sub.loc[sub['relative_move'] == -1, 'log_time'].mean()
            for r in POSITIONS:
                c = sub[sub['relative_move'] == r]
                if len(c) < 30:
                    continue
                m, se, n, G = clustered_mean(c['log_time'], c['pl'])
                n_present = int(present_bp.get((band, r), 0))
                rows.append(dict(rating_band=band, relative_move=r, n=n,
                                 n_players=G, n_present=n_present, mean_log=m,
                                 se=se, ci_lo=m - Z95 * se, ci_hi=m + Z95 * se,
                                 mean_sec=float(np.exp(m)), delta_vs_ref=m - ref))
                mark = '  <-- boundary' if r == 0 else ''
                print(f'{r:>5}{n:>10,}{G:>10,}{m:>10.4f}{fmt_ci(m, se):>22}'
                      f'{np.exp(m):>9.2f}{m - ref:>+9.4f}{mark}')
        curve = pd.DataFrame(rows)
        curve.to_csv(csv_path, index=False)

        # ── Wide table for within-game work ───────────────────────────────
        # One row per player-game, one column per position (NaN = absent or,
        # for W, a premove). Built with numpy to keep memory low.
        col = df['relative_move'].to_numpy() - POSITIONS[0]
        pgv = df['pg'].to_numpy()
        W = np.full((n_pg, len(POSITIONS)), np.nan)
        W[pgv, col] = df['log_time'].to_numpy()
        P = np.full((n_pg, len(POSITIONS)), np.nan)
        P[pgv, col] = df['thinking_time'].to_numpy()
        W = pd.DataFrame(W, columns=POSITIONS)
        P = pd.DataFrame(P, columns=POSITIONS)
        meta = df.drop_duplicates('pg').set_index('pg')[['rating_band', 'pl']]
        meta = meta.reindex(W.index)

        # ── Within-game step ──────────────────────────────────────────────
        header('WITHIN-GAME STEP: last in-book move (-1) to first out-of-book (0)')
        print('Same player, same game, both moves not premoves. This is the '
              'boundary pair\nfrom H1, without the VOC adjustment.')
        print(f'{"band":<9}{"player-games":>14}{"mean step":>11}{"95% CI":>22}'
              f'{"ratio":>8}')
        step_by_band = {}
        for band in bands + ['all']:
            idx = meta.index if band == 'all' else meta.index[meta['rating_band'] == band]
            d = (W.loc[idx, 0] - W.loc[idx, -1]).dropna()
            m, se, n, _ = clustered_mean(d, meta.loc[d.index, 'pl'])
            step_by_band[band] = m
            print(f'{band:<9}{n:>14,}{m:>11.4f}{fmt_ci(m, se):>22}{np.exp(m):>8.3f}')

        # ── Increments around the boundary, same games ────────────────────
        header('STEP OR RAMP? Move-to-move changes in the same player-games')
        print('Player-games with moves -2, -1, 0 and +1 all present and not '
              'premoves.\nA steady ramp would make all three changes about equal. '
              'A step at the\nboundary makes the middle one larger than both '
              'neighbours.')
        print(f'\n{"band":<8}{"n":>9}{"-2 to -1":>11}{"-1 to 0":>11}{"0 to +1":>11}'
              f'{"step minus before":>27}{"step minus after":>27}')
        for band in bands + ['all']:
            idx = meta.index if band == 'all' else meta.index[meta['rating_band'] == band]
            w = W.loc[idx, [-2, -1, 0, 1]].dropna()
            if len(w) < 30:
                continue
            cl = meta.loc[w.index, 'pl']
            pre = w[-1] - w[-2]
            mid = w[0] - w[-1]
            post = w[1] - w[0]
            mp, _, n, _ = clustered_mean(pre, cl)
            mm, _, _, _ = clustered_mean(mid, cl)
            mq, _, _, _ = clustered_mean(post, cl)
            a, sa, _, _ = clustered_mean(mid - pre, cl)
            b, sb, _, _ = clustered_mean(mid - post, cl)
            print(f'{band:<8}{n:>9,}{mp:>11.4f}{mm:>11.4f}{mq:>11.4f}'
                  f'{f"{a:.4f} {fmt_ci(a, sa)}":>27}{f"{b:.4f} {fmt_ci(b, sb)}":>27}')

        # ── In-book trend inside games ────────────────────────────────────
        header('IN-BOOK TREND: version 1 method versus within-game method')
        print('Version 1 fitted a line through the cell means at -5..-1, weighted '
              'by cell size.\nThose cells hold different games. The within-game '
              'slope compares positions\ninside the same player-game only '
              '(player-games with 2+ usable in-book moves),\nso the changing mix '
              'of games cannot create a trend.')
        print(f'\n{"band":<9}{"v1 (cell means)":>17}{"within-game":>13}{"95% CI":>22}'
              f'{"player-games":>14}{"step / slope":>14}')
        L = df[df['relative_move'].isin(IN_BOOK)].dropna(subset=['log_time'])
        L = L[L.groupby('pg')['pg'].transform('size') >= 2]
        for band in bands + ['all']:
            if band == 'all':
                c = curve
                lb = L
            else:
                c = curve[curve['rating_band'] == band]
                lb = L[L['rating_band'] == band]
            pre = c[c['relative_move'].isin(IN_BOOK)]
            if band != 'all' and len(pre) >= 3:
                x = pre['relative_move'].to_numpy(float)
                y = pre['mean_log'].to_numpy()
                w = pre['n'].to_numpy(float)
                mx, my = np.average(x, weights=w), np.average(y, weights=w)
                v1 = np.sum(w * (x - mx) * (y - my)) / np.sum(w * (x - mx) ** 2)
                v1s = f'{v1:.4f}'
            else:
                v1s = '--'
            b, se, npg = fe_slope(lb['pg'].to_numpy(), lb['relative_move'].to_numpy(float),
                                  lb['log_time'].to_numpy(), lb['pl'].to_numpy())
            st = step_by_band.get(band, np.nan)
            ratio = f'{st / b:.1f}' if b > 0 else 'slope <= 0'
            print(f'{band:<9}{v1s:>17}{b:>13.4f}{fmt_ci(b, se):>22}{npg:>14,}'
                  f'{ratio:>14}')
        print('\nslope = change in log thinking time per own move while still in '
              'book.\nstep / slope compares the boundary step with one more move of '
              'that trend.')

        # ── Post-boundary trend inside games ──────────────────────────────
        header('OUT-OF-BOOK TREND inside games (moves 0 to +5)')
        print('For reference: how fast thinking time keeps rising after the '
              'exit.')
        Lp = df[df['relative_move'].between(0, 5)].dropna(subset=['log_time'])
        Lp = Lp[Lp.groupby('pg')['pg'].transform('size') >= 2]
        print(f'{"band":<9}{"slope":>10}{"95% CI":>22}{"player-games":>14}')
        for band in bands + ['all']:
            lb = Lp if band == 'all' else Lp[Lp['rating_band'] == band]
            b, se, npg = fe_slope(lb['pg'].to_numpy(), lb['relative_move'].to_numpy(float),
                                  lb['log_time'].to_numpy(), lb['pl'].to_numpy())
            print(f'{band:<9}{b:>10.4f}{fmt_ci(b, se):>22}{npg:>14,}')

        # ── Complete-window sensitivity ───────────────────────────────────
        header('SENSITIVITY: player-games with every position from -5 to +10')
        full = P.notna().all(axis=1)
        print(f'complete windows: {int(full.sum()):,} of {n_pg:,} player-games '
              f'({100 * full.mean():.1f}%).')
        print('These games stayed in book for at least five of the player\'s '
              'moves, so they are\na selected group. Means drop premoves, as '
              'above.')
        Wf = W[full]
        mf = meta[full]
        print(f'\n{"band":<8}' + ''.join(f'{r:>8}' for r in POSITIONS))
        for band in bands:
            wb = Wf[mf['rating_band'] == band]
            if len(wb) < 30:
                continue
            print(f'{band:<8}' + ''.join(f'{wb[r].mean():>8.3f}' for r in POSITIONS))
        print(f'\n{"band":<9}{"games":>9}{"step":>9}{"95% CI":>22}{"in-book slope":>15}')
        for band in bands + ['all']:
            ib = mf.index if band == 'all' else mf.index[mf['rating_band'] == band]
            d = (Wf.loc[ib, 0] - Wf.loc[ib, -1]).dropna()
            if len(d) < 30:
                continue
            m, se, n, _ = clustered_mean(d, mf.loc[d.index, 'pl'])
            lb = L[L['pg'].isin(ib)]
            b, _, _ = fe_slope(lb['pg'].to_numpy(), lb['relative_move'].to_numpy(float),
                               lb['log_time'].to_numpy(), lb['pl'].to_numpy())
            print(f'{band:<9}{n:>9,}{m:>9.4f}{fmt_ci(m, se):>22}{b:>15.4f}')

        header('NOTES')
        print('1. This analysis is exploratory: it was formulated after the '
              'confirmatory results.')
        print('2. Statements about trends rest on the within-game estimates. The '
              'cell means\n   describe the curve, and their trend mixes different games.')
        print('3. Duplicate rows were removed (count above).')
        print('4. Coverage falls quickly on the in-book side (table above).')
        print("5. Each player's first move was removed because Lichess does not "
              'time it\n   (count above).')
        print(f'\nSaved: {txt_path}\nSaved: {csv_path}')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == '__main__':
    main()
