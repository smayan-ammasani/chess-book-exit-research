#!/usr/bin/env python3
"""
VOC robustness check
==============================================================================
Compares the value-of-computation covariate before and after the Stockfish
transposition-table fix, across all analysed positions.

Pre-registered as robustness check 2 at https://osf.io/n6tkx/

Before the fix, ucinewgame was emitted only once per run, so Stockfish's
transposition table was never cleared between positions. Positions were
processed in sorted FEN order, which places similar positions adjacently, so
each depth-1 search ran against a hash populated by the depth-15 search of a
closely related position. The depth-1 evaluation is supposed to represent
minimal deliberation; a warm hash lets it return a deep-search result.

This script quantifies the resulting measurement error on the actual dataset.
A 180-position simulation estimated r = 0.878 between contaminated and clean
VOC. This measures the real value.

USAGE
    python scripts/voc_robustness_check.py
==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD_NAME = 'stockfish_covariate_PRE_FIX.csv'
NEW_NAME = 'stockfish_covariate.csv'

# Candidate names for the position key and the VOC column, in priority order
KEY_CANDIDATES = ['fen', 'fen_normalized', 'normalized_fen', 'position_fen']
VOC_CANDIDATES = ['voc', 'voc_cp', 'value_of_computation']


def resolve_dir() -> Path:
    here = Path(__file__).resolve()
    for c in [here.parent.parent / 'data' / 'processed',
              Path.cwd() / 'data' / 'processed']:
        if (c / NEW_NAME).exists():
            return c
    return here.parent.parent / 'data' / 'processed'


def pick(cols, candidates, label):
    for c in candidates:
        if c in cols:
            return c
    print(f'ERROR: could not find a {label} column.')
    print(f'  looked for: {candidates}')
    print(f'  available:  {list(cols)}')
    sys.exit(1)


def main() -> None:
    d = resolve_dir()
    old_path, new_path = d / OLD_NAME, d / NEW_NAME

    for p in (old_path, new_path):
        if not p.exists():
            print(f'ERROR: missing {p}')
            sys.exit(1)

    print(f'Loading {old_path.name} and {new_path.name} ...')
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)

    key = pick(new.columns, KEY_CANDIDATES, 'position key')
    voc = pick(new.columns, VOC_CANDIDATES, 'VOC')
    print(f'  position key column: {key}')
    print(f'  VOC column:          {voc}')
    print(f'  rows: old={len(old):,}  new={len(new):,}\n')

    m = old[[key, voc]].merge(new[[key, voc]], on=key,
                              suffixes=('_old', '_new'), how='inner')
    a = m[f'{voc}_old'].astype(float).values
    b = m[f'{voc}_new'].astype(float).values
    n = len(m)
    print(f'Positions matched on {key}: {n:,}')
    if n == 0:
        print('No overlap. Check that both files describe the same positions.')
        sys.exit(1)
    print()

    diff = b - a
    changed = int(np.sum(a != b))
    r = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    sd_new = float(np.std(b))

    print('=' * 74)
    print('MEASUREMENT ERROR IN VOC FROM TRANSPOSITION-TABLE CONTAMINATION')
    print('=' * 74)
    print(f'{"":34s}{"contaminated":>18}{"corrected":>18}')
    print('-' * 74)
    print(f'{"mean VOC (cp)":34s}{np.mean(a):>18.2f}{np.mean(b):>18.2f}')
    print(f'{"median VOC (cp)":34s}{np.median(a):>18.2f}{np.median(b):>18.2f}')
    print(f'{"SD (cp)":34s}{np.std(a):>18.2f}{np.std(b):>18.2f}')
    print(f'{"max (cp)":34s}{np.max(a):>18.2f}{np.max(b):>18.2f}')
    print(f'{"positions with VOC = 0":34s}'
          f'{100*np.mean(a==0):>17.2f}%{100*np.mean(b==0):>17.2f}%')
    print()
    print(f'positions whose VOC changed:      {changed:,} / {n:,} '
          f'({100*changed/n:.1f}%)')
    print(f'correlation (contaminated, clean): r = {r:.4f}')
    print(f'implied reliability r^2:          {r**2:.4f}  '
          f'-> {100*(1-r**2):.1f}% of control power lost')
    print(f'RMS difference:                   {rmse:.2f} cp')
    print(f'SD of corrected VOC:              {sd_new:.2f} cp')
    print(f'noise-to-signal (RMSE / SD):      {rmse/sd_new:.3f}')
    print()
    print(f'mean shift (corrected - contaminated): {np.mean(diff):+.2f} cp')
    print(f'  under-measured before fix: {int(np.sum(diff > 0)):,}')
    print(f'  over-measured before fix:  {int(np.sum(diff < 0)):,}')
    print(f'  unchanged:                 {int(np.sum(diff == 0)):,}')
    print()

    print('Distribution of absolute change:')
    for q in [50, 75, 90, 95, 99]:
        print(f'  p{q:<3d} {np.percentile(np.abs(diff), q):>8.1f} cp')
    print(f'  max  {np.max(np.abs(diff)):>8.1f} cp')
    print()

    print('=' * 74)
    if abs(np.mean(diff)) < 0.1 * rmse:
        print('Pattern: near-zero mean shift with substantial RMS difference.')
        print('The contamination is measurement NOISE rather than systematic')
        print('bias. Noise in a covariate attenuates its coefficient toward')
        print('zero, so complexity was under-controlled and residual complexity')
        print('variance would be absorbed by whichever predictor correlates')
        print('with it, here is_oob.')
    else:
        print('Pattern: the mean shift is large relative to the RMS difference,')
        print('indicating systematic bias rather than noise alone. Both the')
        print('shift and the spread matter.')
    print('=' * 74)


if __name__ == '__main__':
    main()
