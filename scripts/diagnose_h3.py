#!/usr/bin/env python3
"""
H3 Diagnostic
==============================================================================
PURPOSE
    Decide whether the ex-Gaussian is a usable model for these thinking-time
    distributions, and quantify the truncation asymmetry between the book and
    out-of-book conditions.

    Run this BEFORE trusting any output from Script 06.

WHY
    The ex-Gaussian distribution has a structural skewness ceiling of 2.0:

        Var  = sigma^2 + tau^2
        Skew = 2*tau^3 / (sigma^2 + tau^2)^1.5   -> approaches 2 as tau/sigma grows

    No ex-Gaussian, for any parameter values, can have skewness >= 2. If the
    sample skewness exceeds 2, method-of-moments has no valid solution: tau
    comes out larger than the total SD, (v - tau^2) goes negative, and both
    sigma and mu clamp to their floors. That is what produced the
    mu=0.05 / sigma=0.10 output in every band.

USAGE
    From the project root:
        python scripts/diagnose_h3.py

    Or with an explicit path:
        python scripts/diagnose_h3.py path/to/analysis_sample.csv
==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Must match the constants in Script 06
MIN_TIME_FOR_FIT = 0.5
MAX_TIME_FOR_FIT = 120.0

# The ex-Gaussian cannot represent skewness at or above this value
EXGAUSS_SKEW_CEILING = 2.0

CONDITIONS = [
    ('last_book_thinking_time', 'book'),
    ('first_oob_thinking_time', 'oob'),
]


def resolve_input_path() -> Path:
    """Locate analysis_sample.csv from an argument, or by the usual layout."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / 'data' / 'processed' / 'analysis_sample.csv',
        Path.cwd() / 'data' / 'processed' / 'analysis_sample.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def verdict(skew: float) -> str:
    """Translate a sample skewness into a fitting verdict."""
    if not np.isfinite(skew):
        return 'UNDEFINED'
    if skew >= EXGAUSS_SKEW_CEILING:
        return 'INFEASIBLE (MOM will clamp)'
    if skew >= 1.85:
        return 'marginal'
    return 'feasible'


def main() -> None:
    path = resolve_input_path()

    if not path.exists():
        print(f'ERROR: could not find {path}')
        print('Pass the path explicitly:  python scripts/diagnose_h3.py <path-to-csv>')
        sys.exit(1)

    print(f'Loading {path} ...')
    df = pd.read_csv(
        path,
        usecols=['rating_band'] + [c for c, _ in CONDITIONS],
        low_memory=False,
    )
    print(f'Loaded {len(df):,} pairs.\n')

    bands = sorted(df['rating_band'].dropna().unique())

    # =========================================================================
    # SECTION 1 - Distribution shape inside the Script 06 fitting window
    # =========================================================================
    print('=' * 100)
    print('SECTION 1 - DISTRIBUTION SHAPE INSIDE THE FITTING WINDOW '
          f'[{MIN_TIME_FOR_FIT}, {MAX_TIME_FOR_FIT}] SECONDS')
    print('=' * 100)
    print(f'{"band":<9} {"cond":<5} {"n_kept":>9} {"mean":>8} {"sd":>8} '
          f'{"SKEW":>8}   {"ex-Gaussian verdict"}')
    print('-' * 100)

    infeasible_count = 0
    total_fits = 0

    for band in bands:
        sub = df[df['rating_band'] == band]
        for col, name in CONDITIONS:
            x = sub[col].dropna().values
            w = x[(x >= MIN_TIME_FOR_FIT) & (x <= MAX_TIME_FOR_FIT)]
            if len(w) < 50:
                print(f'{band:<9} {name:<5} {len(w):>9,}   (too few observations)')
                continue
            sk = float(stats.skew(w))
            v = verdict(sk)
            total_fits += 1
            if sk >= EXGAUSS_SKEW_CEILING:
                infeasible_count += 1
            print(f'{band:<9} {name:<5} {len(w):>9,} {w.mean():>8.2f} {w.std():>8.2f} '
                  f'{sk:>8.2f}   {v}')
        print()

    # =========================================================================
    # SECTION 2 - Truncation asymmetry (the 0.5s floor)
    # =========================================================================
    print('=' * 100)
    print(f'SECTION 2 - TRUNCATION ASYMMETRY: HOW MUCH DOES THE '
          f'{MIN_TIME_FOR_FIT}s FLOOR REMOVE FROM EACH CONDITION?')
    print('=' * 100)
    print(f'{"band":<9} {"book <0.5s":>12} {"oob <0.5s":>12} {"ratio":>10}   '
          f'{"book >120s":>12} {"oob >120s":>12}')
    print('-' * 100)

    for band in bands:
        sub = df[df['rating_band'] == band]
        vals = {}
        for col, name in CONDITIONS:
            x = sub[col].dropna().values
            if len(x) == 0:
                vals[name] = (np.nan, np.nan)
                continue
            vals[name] = (
                100.0 * float(np.mean(x < MIN_TIME_FOR_FIT)),
                100.0 * float(np.mean(x > MAX_TIME_FOR_FIT)),
            )
        b_lo, b_hi = vals['book']
        o_lo, o_hi = vals['oob']
        ratio = (b_lo / o_lo) if (o_lo and np.isfinite(o_lo) and o_lo > 0) else np.inf
        ratio_str = f'{ratio:>9.1f}x' if np.isfinite(ratio) else '      inf'
        print(f'{band:<9} {b_lo:>11.2f}% {o_lo:>11.2f}% {ratio_str}   '
              f'{b_hi:>11.2f}% {o_hi:>11.2f}%')

    print()
    print('Interpretation: a large book/oob ratio in the "<0.5s" columns means the')
    print('fitting filter removes far more book moves than out-of-book moves, so any')
    print('book-vs-oob comparison inside the window is between unequally truncated')
    print('samples. Note also whether the book percentage falls as rating rises --')
    print('that trend is itself a directly observable discrete-switch signal.')
    print()

    # =========================================================================
    # SECTION 3 - Does a tighter upper window bring skewness under the ceiling?
    # =========================================================================
    print('=' * 100)
    print('SECTION 3 - WINDOW SENSITIVITY: SAMPLE SKEWNESS AT DIFFERENT UPPER CUTOFFS')
    print('=' * 100)

    cutoffs = [15.0, 30.0, 60.0, 120.0]
    header = f'{"band":<9} {"cond":<5}' + ''.join(f'{f"<={c:g}s":>12}' for c in cutoffs)
    print(header)
    print('-' * 100)

    for band in bands:
        sub = df[df['rating_band'] == band]
        for col, name in CONDITIONS:
            x = sub[col].dropna().values
            cells = ''
            for c in cutoffs:
                w = x[(x >= MIN_TIME_FOR_FIT) & (x <= c)]
                if len(w) < 50:
                    cells += f'{"--":>12}'
                else:
                    cells += f'{float(stats.skew(w)):>12.2f}'
            print(f'{band:<9} {name:<5}{cells}')
        print()

    print('A cutoff whose skewness sits comfortably below 2.0 would make the')
    print('ex-Gaussian fittable -- but changing the window to obtain a fittable')
    print('result is a researcher degree of freedom. If the window is narrowed,')
    print('the choice and its rationale must be stated explicitly in Methods,')
    print('and results at the original window reported alongside.')
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('=' * 100)
    print('SUMMARY')
    print('=' * 100)
    print(f'Fits attempted:            {total_fits}')
    print(f'Above skewness ceiling:    {infeasible_count} / {total_fits}')
    print()
    if total_fits == 0:
        print('No fits attempted -- check the input file and column names.')
    elif infeasible_count == 0:
        print('All conditions sit below the ex-Gaussian skewness ceiling.')
        print('-> The ex-Gaussian is a defensible model here. Refit Script 06 by')
        print('   maximum likelihood rather than method-of-moments, and re-check')
        print('   that mu and sigma no longer land on their floors.')
    elif infeasible_count == total_fits:
        print('EVERY condition exceeds the ex-Gaussian skewness ceiling.')
        print('-> No parameterisation of the ex-Gaussian can describe these data,')
        print('   and no fitting method recovers this. Method-of-moments clamps;')
        print('   maximum likelihood degenerates toward a shifted exponential.')
        print('   H3 needs a different operationalisation -- the truncation')
        print('   asymmetry in Section 2 is the strongest candidate already in hand.')
    else:
        print('MIXED: some conditions are fittable and others are not.')
        print('-> Do not fit only the feasible subset and report those; that')
        print('   selects on the outcome. Either adopt one approach that applies')
        print('   to every band, or report the infeasibility itself as the result.')
    print()


if __name__ == '__main__':
    main()
