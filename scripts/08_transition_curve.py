#!/usr/bin/env python3
"""
Script 08: Transition curve analysis (EXPLORATORY)
==============================================================================
STATUS: EXPLORATORY. Not pre-registered as a confirmatory test.

This analysis was proposed after H1-H3 results were observed. It is reported
as an exploratory extension and does not substitute for H1 under any outcome.
See https://osf.io/n6tkx/ Section 7.

WHAT IT ANSWERS

    The two-point design (last book move vs first out-of-book move) cannot
    distinguish two very different things:

        STEP                          RAMP
        ....___                       ....--''
        ....   |                      ..--'
        ____---'                  __--'
        -2 -1  0 +1 +2               -2 -1  0 +1 +2

    Both produce the same difference between positions -1 and 0. Only a
    window shows which one is happening.

    It also addresses the strongest objection to H1: the two compared moves
    are consecutive, and thinking time generally rises through a game, so
    part of the jump may just be "the next move is later". The ply-trend
    probe estimated the jump exceeds that trend by 3.6x-19x, but did so
    cross-sectionally. This measures the trend WITHIN games.

WHAT IT COMPUTES

    1. Mean log thinking time at each relative position, per rating band,
       with bootstrap confidence intervals.
    2. A pre-boundary slope (positions -5..-1) as a within-game baseline,
       and the boundary jump (-1 -> 0) measured against it.
    3. A post-boundary decay: how many moves until thinking time returns
       toward its pre-boundary trend.
    4. A step-vs-ramp discrimination statistic.

IMPORTANT: UNBALANCED PANEL

    Only ~16.6% of player-games have data at position -5, versus 100% at
    -1 and 0, because most players leave book within five of their own
    moves. Requiring a complete window would retain only the players who
    stayed in book longest -- a systematically more-prepared group, and
    exactly the confound this analysis exists to address.

    Every statistic here is therefore computed per-position on whatever
    observations exist at that position. Cell counts are reported alongside
    every estimate so the reader can see where the data thins out.

USAGE
    python scripts/08_transition_curve.py
==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREMOVE_THRESHOLD = 0.5
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
PRE_BOUNDARY = [-5, -4, -3, -2, -1]

COLS = ['game_id', 'player_color', 'rating_band', 'player_username',
        'relative_move', 'thinking_time', 'is_premove']


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


def boot_ci(values, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Percentile bootstrap CI for a mean. Returns (lo, hi)."""
    if len(values) < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    # For very large cells, bootstrap a capped subsample -- the CI is already
    # tiny at this n and full resampling would dominate runtime.
    if n > 50_000:
        values = rng.choice(values, 50_000, replace=False)
        n = 50_000
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def wls_slope(x, y, w):
    W = np.sum(w)
    mx = np.sum(w * x) / W
    my = np.sum(w * y) / W
    sxx = np.sum(w * (x - mx) ** 2)
    if sxx == 0:
        return np.nan
    return np.sum(w * (x - mx) * (y - my)) / sxx


def main():
    path, txt_path, csv_path = resolve()
    if not path.exists():
        print(f'ERROR: {path} not found.')
        sys.exit(1)

    tee = Tee(txt_path)
    sys.stdout = tee
    try:
        print('Script 08: Transition curve analysis')
        print('STATUS: EXPLORATORY -- proposed after confirmatory results were')
        print('observed. Not a pre-registered test. See https://osf.io/n6tkx/\n')

        print(f'Loading {path.name} ({path.stat().st_size / 1e9:.2f} GB) ...')
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
        print(f'Loaded {len(df):,} rows.')

        df['is_premove'] = to_bool(df['is_premove'])
        n_before = len(df)
        df = df[~df['is_premove']]
        df = df[df['thinking_time'] > PREMOVE_THRESHOLD]
        df = df[np.isfinite(df['thinking_time'])]
        print(f'After premove exclusion: {len(df):,} '
              f'({100 * len(df) / n_before:.1f}% retained)')

        df['log_time'] = np.log(df['thinking_time'])
        bands = sorted(df['rating_band'].dropna().unique())
        positions = sorted(df['relative_move'].unique())
        print(f'Bands: {len(bands)}   positions: {min(positions)} to {max(positions)}\n')

        # ── The curve ────────────────────────────────────────────────────────
        print('=' * 96)
        print('THE CURVE: mean log(thinking time) by relative move')
        print('=' * 96)
        print('Position 0 = first out-of-book move. Negative = still in book.')
        print('CI is a percentile bootstrap on the mean.\n')

        rows = []
        for band in bands:
            sub = df[df['rating_band'] == band]
            print(f'--- {band} ---')
            print(f'{"rel":>5}{"n":>11}{"mean log":>11}{"95% CI":>22}'
                  f'{"mean sec":>11}{"vs -1":>9}')
            print('-' * 70)
            ref = sub.loc[sub.relative_move == -1, 'log_time'].mean()
            for rel in positions:
                v = sub.loc[sub.relative_move == rel, 'log_time'].to_numpy()
                if len(v) < 30:
                    continue
                m = float(v.mean())
                lo, hi = boot_ci(v)
                rows.append(dict(rating_band=band, relative_move=rel, n=len(v),
                                 mean_log=m, ci_lo=lo, ci_hi=hi,
                                 mean_sec=float(np.exp(m)), delta_vs_ref=m - ref))
                mark = '  <-- boundary' if rel == 0 else ''
                print(f'{rel:>5}{len(v):>11,}{m:>11.4f}'
                      f'{f"[{lo:.4f}, {hi:.4f}]":>22}'
                      f'{np.exp(m):>11.2f}{m - ref:>+9.4f}{mark}')
            print()

        curve = pd.DataFrame(rows)
        curve.to_csv(csv_path, index=False)

        # ── Step vs ramp ─────────────────────────────────────────────────────
        print('=' * 96)
        print('STEP OR RAMP? Boundary jump against the within-game pre-boundary trend')
        print('=' * 96)
        print('The pre-boundary slope is fitted on positions -5..-1, weighted by')
        print('cell size. It is a WITHIN-GAME baseline for how much thinking time')
        print('rises per move with no boundary crossed. The ply-trend probe')
        print('estimated this cross-sectionally; this is the direct measurement.\n')

        print(f'{"band":<9}{"pre-slope":>12}{"predicted":>12}{"observed":>12}'
              f'{"excess":>10}{"ratio":>9}')
        print('-' * 64)
        summary = []
        for band in bands:
            c = curve[curve.rating_band == band]
            pre = c[c.relative_move.isin(PRE_BOUNDARY)]
            if len(pre) < 3:
                continue
            slope = wls_slope(pre.relative_move.to_numpy(float),
                              pre.mean_log.to_numpy(), pre.n.to_numpy(float))
            at_m1 = c.loc[c.relative_move == -1, 'mean_log']
            at_0 = c.loc[c.relative_move == 0, 'mean_log']
            if at_m1.empty or at_0.empty:
                continue
            observed = float(at_0.iloc[0] - at_m1.iloc[0])
            ratio = observed / slope if slope > 0 else np.inf
            summary.append((band, slope, observed, observed - slope, ratio))
            print(f'{band:<9}{slope:>12.5f}{slope:>12.5f}{observed:>12.5f}'
                  f'{observed - slope:>10.5f}{ratio:>9.1f}')

        print()
        print('pre-slope  = log-seconds gained per move while still in book')
        print('predicted  = what the -1 -> 0 step would be if the trend simply continued')
        print('observed   = the actual -1 -> 0 step')
        print('excess     = observed minus predicted, i.e. the boundary-specific part')
        print('ratio      = observed / predicted')
        print()
        print('CAVEAT ON THE PRE-SLOPE. It is fitted across positions -5..-1, but')
        print('those cells contain different players: only ~17% of player-games')
        print('reach -5, and those are the ones who stayed in book longest.')
        print('Validation on synthetic data with a known trend showed this biases')
        print('the fitted slope LOW (recovering 0.010-0.013 where the true value')
        print('was 0.020), which biases the ratio HIGH. On synthetic data built as')
        print('a pure ramp with no boundary effect at all, the ratio still read')
        print('1.6-1.7x rather than 1.0. Treat ratios below roughly 2x as')
        print('consistent with no boundary-specific effect.\n')

        if summary:
            ratios = [s[4] for s in summary if np.isfinite(s[4])]
            worst = min(ratios) if ratios else np.nan
            print(f'Smallest ratio across bands: {worst:.1f}x')
            if worst >= 3:
                print('The jump is several times the within-game trend in every band.')
                print('This is a STEP, not a continuation of a ramp.')
            elif worst >= 1.5:
                print('The jump exceeds the trend but not dramatically in every band.')
                print('Report the excess and the ratio; avoid calling it a clean step.')
            else:
                print('The jump is close to what the pre-boundary trend predicts.')
                print('The two-point H1 result may substantially reflect a general')
                print('ply trend rather than a boundary-specific cost. This would be')
                print('a material qualification and must be reported prominently.')
        print()

        # ── Recovery ─────────────────────────────────────────────────────────
        print('=' * 96)
        print('RECOVERY: how long does the elevation persist?')
        print('=' * 96)
        print('Each post-boundary position is compared against the pre-boundary')
        print('trend extrapolated to that position. "Recovered" means the residual')
        print('elevation has fallen below 25% of the boundary excess.\n')

        for band in bands:
            c = curve[curve.rating_band == band]
            pre = c[c.relative_move.isin(PRE_BOUNDARY)]
            if len(pre) < 3:
                continue
            slope = wls_slope(pre.relative_move.to_numpy(float),
                              pre.mean_log.to_numpy(), pre.n.to_numpy(float))
            at_m1 = c.loc[c.relative_move == -1, 'mean_log']
            if at_m1.empty:
                continue
            base = float(at_m1.iloc[0])

            print(f'--- {band} ---')
            print(f'{"rel":>5}{"observed":>11}{"expected":>11}{"excess":>10}{"% of peak":>11}')
            print('-' * 48)
            peak = None
            recovered_at = None
            for rel in [r for r in positions if r >= 0]:
                row = c[c.relative_move == rel]
                if row.empty:
                    continue
                obs = float(row.mean_log.iloc[0])
                exp = base + slope * (rel - (-1))
                exc = obs - exp
                if peak is None:
                    peak = exc
                pct = 100 * exc / peak if peak else np.nan
                if recovered_at is None and rel > 0 and peak and exc < 0.25 * peak:
                    recovered_at = rel
                print(f'{rel:>5}{obs:>11.4f}{exp:>11.4f}{exc:>+10.4f}{pct:>10.1f}%')
            if recovered_at is not None:
                print(f'  -> falls below 25% of peak excess by move +{recovered_at}')
            else:
                print('  -> still above 25% of peak excess at the end of the window')
            print()

        print('=' * 96)
        print('HOW TO REPORT THIS')
        print('=' * 96)
        print('1. This is exploratory. Label it as such in its own section,')
        print('   separate from the H1-H3 results, and state that it was')
        print('   formulated after the confirmatory results were observed.')
        print('2. The pre-boundary slope here is a within-game measurement and')
        print('   supersedes the cross-sectional ply-trend probe. Report both,')
        print('   and say which is which.')
        print('3. Report cell counts with every estimate. The panel is')
        print('   unbalanced by design: ~16.6% coverage at -5 against 100% at')
        print('   -1 and 0. Do not present the curve as if all positions were')
        print('   equally well estimated.')
        print('4. If the ratio is small, that is a real qualification to H1 and')
        print('   belongs in the abstract, not only in a limitations paragraph.')

        print(f'\nSaved: {txt_path}')
        print(f'Saved: {csv_path}')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == '__main__':
    main()
