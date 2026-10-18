#!/usr/bin/env python3
"""
Validate transition_window.csv
==============================================================================
Validates the transition-window extraction in Script 02 on a small test run
(MAX_GAMES_TO_PARSE = 500_000), before the full 2-3 day run.

Four checks:
  1. ALIGNMENT   - is the window correctly positioned against the boundary?
  2. REPRODUCTION- do positions -1 and 0 reproduce the existing H1 estimate?
  3. SAMPLING    - did the band strides land near their targets?
  4. PANEL SHAPE - how ragged is the window, and where?

Check 1 is critical: if it fails, the window is misaligned and nothing
downstream is meaningful.

USAGE
    python scripts/validate_window.py
==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREMOVE_THRESHOLD = 0.5
EXPECTED_H1_LOG_DELTA = 0.2837   # from the 05b refit, for comparison


def resolve():
    here = Path(__file__).resolve()
    for base in (here.parent.parent, Path.cwd()):
        p = base / 'data' / 'processed' / 'transition_window.csv'
        if p.exists():
            return p
    return here.parent.parent / 'data' / 'processed' / 'transition_window.csv'


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])


def main():
    path = resolve()
    if not path.exists():
        print(f'ERROR: {path} not found.')
        print('The window file is written by Script 02.')
        sys.exit(1)

    size_gb = path.stat().st_size / 1e9
    print(f'Loading {path.name} ({size_gb:.2f} GB) ...')
    if size_gb > 3:
        print('Large file; loading only the columns needed.')
        df = pd.read_csv(path, usecols=['game_id', 'player_color', 'rating_band',
                                        'relative_move', 'thinking_time',
                                        'in_book', 'ply_at_exit'],
                         low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    df['in_book'] = to_bool(df['in_book'])
    print(f'Loaded {len(df):,} rows.\n')

    failures = []

    # ── CHECK 1: alignment ───────────────────────────────────────────────────
    print('=' * 74)
    print('CHECK 1 - ALIGNMENT (critical)')
    print('=' * 74)
    print('Book membership is defined on move-sequence prefixes, so it is')
    print('monotone within a game: once out of book, always out. Therefore')
    print('every relative_move < 0 must be in book, and every relative_move')
    print('>= 0 must be out of book.\n')

    bad_neg = df[(df.relative_move < 0) & (~df.in_book)]
    bad_pos = df[(df.relative_move >= 0) & (df.in_book)]
    total_bad = len(bad_neg) + len(bad_pos)

    print(f'  negative positions marked out-of-book: {len(bad_neg):,}')
    print(f'  non-negative positions marked in-book: {len(bad_pos):,}')
    print(f'  TOTAL MISALIGNED: {total_bad:,}   (must be 0)')

    if total_bad == 0:
        print('\n  PASS - window is correctly aligned to the boundary.')
    else:
        failures.append('alignment')
        print('\n  FAIL - the window is misaligned.')
        print('  Most likely cause: exit_idx points at the wrong element of')
        print('  player_plies, or relative_move 0 was defined as the last book')
        print('  move rather than the first out-of-book move.')
        print('\n  Sample of offending rows:')
        print(pd.concat([bad_neg.head(3), bad_pos.head(3)]).to_string())
    print()

    # ── CHECK 2: reproduces the existing H1 estimate ─────────────────────────
    print('=' * 74)
    print('CHECK 2 - REPRODUCTION OF THE EXISTING BOUNDARY PAIR')
    print('=' * 74)
    print('Positions -1 and 0 are the same observations as boundary_pairs.csv,')
    print('so their mean log difference should land near the H1 estimate.\n')

    w = df[df.relative_move.isin([-1, 0])]
    piv = w.pivot_table(index=['game_id', 'player_color'],
                        columns='relative_move', values='thinking_time')
    if -1 in piv.columns and 0 in piv.columns:
        piv = piv[(piv[-1] > PREMOVE_THRESHOLD) & (piv[0] > PREMOVE_THRESHOLD)]
        delta = np.log(piv[0]) - np.log(piv[-1])
        print(f'  complete (-1, 0) pairs after premove exclusion: {len(piv):,}')
        print(f'  mean log delta: {delta.mean():.4f}')
        print(f'  H1 estimate from 05b refit: {EXPECTED_H1_LOG_DELTA:.4f}')
        diff = abs(delta.mean() - EXPECTED_H1_LOG_DELTA)
        print(f'  difference: {diff:.4f}')
        if diff < 0.08:
            print('\n  PASS - consistent with the existing estimate.')
        else:
            failures.append('reproduction')
            print('\n  WARN - larger gap than expected. Some difference is')
            print('  normal (this is a sampled subset, no VOC control, no')
            print('  random intercept), but a large gap suggests the window is')
            print('  not capturing the same observations.')
    else:
        failures.append('reproduction')
        print('  FAIL - positions -1 and/or 0 missing entirely.')
    print()

    # ── CHECK 3: sampling ────────────────────────────────────────────────────
    print('=' * 74)
    print('CHECK 3 - SAMPLING RATES BY BAND')
    print('=' * 74)
    print('Target was ~200,000 player-games per band, band_5 taken in full.')
    print('On a reduced test run these will be proportionally smaller.\n')

    pg = df.groupby('rating_band').apply(
        lambda g: g.groupby(['game_id', 'player_color']).ngroups)
    print(f'{"band":<10}{"player-games":>16}')
    print('-' * 26)
    for band in sorted(pg.index):
        print(f'{band:<10}{pg[band]:>16,}')
    print(f'\n  total player-games: {pg.sum():,}')
    print(f'  total rows:         {len(df):,}')
    print(f'  mean rows per player-game: {len(df) / pg.sum():.2f}  (max possible 16)')
    print()

    # ── CHECK 4: panel shape ─────────────────────────────────────────────────
    print('=' * 74)
    print('CHECK 4 - PANEL SHAPE')
    print('=' * 74)
    print('The window is ragged by design: negative positions are missing when')
    print('a player left book early, positive positions when the game ended.')
    print('Positions -1 and 0 should have the highest counts.\n')

    counts = df.groupby('relative_move').size()
    peak = counts.max()
    print(f'{"rel":>6}{"n":>12}{"% of peak":>12}')
    print('-' * 30)
    for rel in sorted(counts.index):
        bar = '#' * int(40 * counts[rel] / peak)
        print(f'{rel:>6}{counts[rel]:>12,}{100 * counts[rel] / peak:>11.1f}%  {bar}')

    top = counts.idxmax()
    if top in (-1, 0):
        print(f'\n  PASS - peak at relative_move = {top}, as expected.')
    else:
        failures.append('panel shape')
        print(f'\n  WARN - peak at relative_move = {top}, expected -1 or 0.')
    print()
    print('  Incomplete windows are kept. Requiring a full 16-position window')
    print('  would select for long games that stayed in book a while, which is')
    print('  exactly the confound the curve is meant to address, so the')
    print('  analysis must handle unbalanced panels.')
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print('=' * 74)
    print('SUMMARY')
    print('=' * 74)
    if not failures:
        print('  All checks passed. The window extraction is ready for the full')
        print('  run (MAX_GAMES_TO_PARSE = None).')
    elif 'alignment' in failures:
        print('  ALIGNMENT FAILED. extract_window_rows must be fixed first;')
        print('  every downstream number depends on it.')
    else:
        print(f'  Checks needing attention: {", ".join(failures)}')
        print('  Alignment passed, so the window is positioned correctly.')
        print('  See the warnings above.')


if __name__ == '__main__':
    main()
