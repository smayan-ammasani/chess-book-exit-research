#!/usr/bin/env python3
"""
Ply-trend baseline probe
==============================================================================
THE OBJECTION THIS ADDRESSES

    H1 compares the last in-book move to the first out-of-book move. These are
    the same player's consecutive moves, two plies apart. Thinking time
    generally rises as a game progresses. So some unknown share of the observed
    1.328x is simply "the next move is usually slower than this one," and the
    two-point design has no baseline for what a normal move-to-move increase
    looks like.

WHAT THIS DOES

    boundary_pairs.csv records ply_at_exit for every pair. Players exit the
    book at different depths, so regressing thinking time on exit ply gives a
    cross-sectional estimate of how thinking time scales with ply. Scaled to
    one player move (two plies), that is an approximate baseline for the
    expected increase with no boundary effect at all.

    Compare that baseline against the observed book -> OOB jump.

WHAT THIS IS NOT

    Cross-sectional, not within-game. Games that leave book at ply 30 may
    differ systematically from games that leave at ply 10 -- deeper
    preparation, different openings, possibly stronger players. The slope
    conflates "later plies are slower" with "games that stay in book longer
    are different games."

    It is a first-order check on whether the objection is serious, not a
    substitute for extracting the actual transition curve. If the boundary
    jump is many times the baseline, the objection is likely survivable. If
    they are comparable, the curve extraction is mandatory before H1 can be
    interpreted at all.

USAGE
    python scripts/ply_trend_probe.py
==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_SIZE = 500_000
PREMOVE_THRESHOLD = 0.5

COLS = ['rating_band', 'ply_at_exit',
        'last_book_thinking_time', 'first_oob_thinking_time',
        'last_book_is_premove', 'first_oob_is_premove']

# Exit plies with too few observations give unstable cell means
MIN_N_PER_PLY = 500
PLY_RANGE = (6, 40)


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
        p = base / 'data' / 'processed' / 'boundary_pairs.csv'
        if p.exists():
            out = base / 'outputs' / 'diagnostics'
            out.mkdir(parents=True, exist_ok=True)
            return p, out / 'ply_trend_probe.txt'
    root = here.parent.parent
    return (root / 'data' / 'processed' / 'boundary_pairs.csv',
            root / 'outputs' / 'diagnostics' / 'ply_trend_probe.txt')


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])


def accumulate(path):
    """
    Stream once. For each (band, ply_at_exit) cell accumulate sums of
    log(book time) and log(OOB time) so cell means can be formed exactly.
    Applies the same symmetric premove exclusion as the main analysis.
    """
    acc = {}   # (band, ply) -> [n, sum_log_book, sum_log_oob]
    n_rows = n_kept = 0

    print(f'Streaming {path.name} ({path.stat().st_size / 1e9:.2f} GB) ...')
    for i, chunk in enumerate(pd.read_csv(path, usecols=COLS,
                                          chunksize=CHUNK_SIZE, low_memory=False)):
        n_rows += len(chunk)

        keep = ~(to_bool(chunk['last_book_is_premove'])
                 | to_bool(chunk['first_oob_is_premove']))
        c = chunk[keep]
        c = c[(c['last_book_thinking_time'] > PREMOVE_THRESHOLD)
              & (c['first_oob_thinking_time'] > PREMOVE_THRESHOLD)]
        c = c[(c['ply_at_exit'] >= PLY_RANGE[0]) & (c['ply_at_exit'] <= PLY_RANGE[1])]
        if c.empty:
            continue
        n_kept += len(c)

        lb = np.log(c['last_book_thinking_time'].to_numpy(dtype=float))
        lo = np.log(c['first_oob_thinking_time'].to_numpy(dtype=float))

        for band, ply, a, b in zip(c['rating_band'], c['ply_at_exit'], lb, lo):
            k = (band, int(ply))
            if k not in acc:
                acc[k] = [0, 0.0, 0.0]
            acc[k][0] += 1
            acc[k][1] += a
            acc[k][2] += b

        if (i + 1) % 10 == 0:
            print(f'  {n_rows:,} rows read, {n_kept:,} kept...')

    print(f'Total rows: {n_rows:,}   kept after premove exclusion: {n_kept:,}\n')
    return acc


def wls_slope(x, y, w):
    """Weighted least squares slope and intercept."""
    W = np.sum(w)
    mx = np.sum(w * x) / W
    my = np.sum(w * y) / W
    sxx = np.sum(w * (x - mx) ** 2)
    sxy = np.sum(w * (x - mx) * (y - my))
    slope = sxy / sxx
    return slope, my - slope * mx


def main():
    path, out_path = resolve()
    if not path.exists():
        print(f'ERROR: {path} not found.')
        sys.exit(1)

    tee = Tee(out_path)
    sys.stdout = tee
    try:
        print('Ply-trend baseline probe')
        print('Approximate check on whether the book-exit jump exceeds the')
        print('normal move-to-move increase in thinking time.\n')

        acc = accumulate(path)
        if not acc:
            print('No usable rows.')
            return

        bands = sorted({b for b, _ in acc})

        print('=' * 88)
        print('PER-BAND: cross-sectional ply trend vs observed boundary jump')
        print('=' * 88)
        print('All quantities in log(seconds).\n')
        print(f'{"band":<9}{"cells":>7}{"n":>12}{"slope/ply":>12}'
              f'{"baseline/move":>15}{"observed jump":>15}{"ratio":>9}')
        print('-' * 88)

        summary = []
        for band in bands:
            plies, ns, mbook, moob = [], [], [], []
            for (b, ply), (n, sb, so) in sorted(acc.items()):
                if b != band or n < MIN_N_PER_PLY:
                    continue
                plies.append(ply)
                ns.append(n)
                mbook.append(sb / n)
                moob.append(so / n)
            if len(plies) < 5:
                print(f'{band:<9}{len(plies):>7}   too few stable cells')
                continue

            plies = np.array(plies, float)
            ns = np.array(ns, float)
            mbook = np.array(mbook)
            moob = np.array(moob)

            # Trend estimated from BOOK moves only: these are the moves that
            # have not yet crossed the boundary, so their trend with ply is
            # the closest available proxy for a no-boundary baseline.
            slope, _ = wls_slope(plies, mbook, ns)
            baseline_per_move = slope * 2.0     # one player move = two plies

            n_tot = ns.sum()
            observed = float(np.sum(ns * (moob - mbook)) / n_tot)
            ratio = observed / baseline_per_move if baseline_per_move != 0 else np.inf

            print(f'{band:<9}{len(plies):>7}{int(n_tot):>12,}{slope:>12.5f}'
                  f'{baseline_per_move:>15.5f}{observed:>15.5f}'
                  f'{ratio:>9.1f}' if np.isfinite(ratio)
                  else f'{band:<9} ratio undefined')
            summary.append((band, slope, baseline_per_move, observed, ratio, n_tot))

        print()
        print('=' * 88)
        print('INTERPRETATION')
        print('=' * 88)
        if summary:
            ratios = [s[4] for s in summary if np.isfinite(s[4])]
            worst = min(ratios) if ratios else float('nan')
            print(f'Smallest jump-to-baseline ratio across bands: {worst:.1f}x\n')
            if worst >= 5:
                print('The boundary jump is many times larger than the ply trend')
                print('would predict. The "it is just the next move" objection is')
                print('unlikely to account for the effect. Extracting the full')
                print('transition curve remains worthwhile for characterising the')
                print('SHAPE of the transition, but is not required to defend H1.')
            elif worst >= 2:
                print('The boundary jump exceeds the ply trend, but not by a wide')
                print('margin in every band. Report this probe as a limitation and')
                print('treat the full curve extraction as a priority.')
            else:
                print('The boundary jump is comparable to what the ply trend alone')
                print('would predict. H1 cannot be cleanly separated from a general')
                print('ply effect with the current two-point design. Extract the')
                print('full transition curve before interpreting H1 further, and')
                print('state this limitation prominently.')
        print()
        print('CAVEAT: this trend is cross-sectional. Games leaving book at ply 30')
        print('may differ systematically from games leaving at ply 10, so the slope')
        print('conflates "later plies are slower" with "longer-book games differ".')
        print('It bounds the concern; it does not resolve it. Only a within-game')
        print('window (moves -5 to +10) measures the actual trend.')

        print(f'\nSaved to: {out_path}')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == '__main__':
    main()
