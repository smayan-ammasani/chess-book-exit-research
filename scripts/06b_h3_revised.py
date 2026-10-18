#!/usr/bin/env python3
"""
Script 06b: H3-revised, immediate execution as a discrete-transition signal
==============================================================================
Pre-registered at https://osf.io/n6tkx/, Section 5.3.

The original H3 (ex-Gaussian tau-dominance) is withdrawn: sample skewness in
every band-by-condition cell (4.35-6.95) exceeds the ex-Gaussian's structural
skewness ceiling of 2.0. See docs/findings/2026-08-03-pipeline-defects.md.

REPLACEMENT PREDICTION
    The proportion of moves executed at or below the premove threshold is
    higher for last-in-book moves than for first-out-of-book moves, in every
    rating band. A retrieved move can be emitted without deliberation; a move
    requiring search cannot.

WHY THIS RUNS ON boundary_pairs.csv, NOT analysis_sample.csv
    The corrected Script 04 now excludes any pair where EITHER move is a
    premove (that is Defect 2's fix). analysis_sample.csv therefore has an
    out-of-book premove rate of essentially zero by construction, and the
    premove rate is exactly what this test measures. This script reads the
    unfiltered boundary_pairs.csv, which has never had any premove exclusion
    applied, using the last_book_is_premove / first_oob_is_premove flags
    Script 02 already computed for every pair.

REGISTERED TEST
    Two-proportion z-test per band (book rate vs OOB rate), one-tailed,
    Holm-corrected across the five bands.

REGISTERED CRITERION
    Supported only if the book proportion exceeds the OOB proportion in ALL
    FIVE bands, AND the ratio is >= 2.0 in every band.

SUPPLEMENTARY TEST (not in the pre-registration, added here, before any
result exists, for a documented reason)
    Book-premove and OOB-premove status are measured on the same pair, so
    they are paired binary outcomes, not independent samples. McNemar's test
    is the statistically appropriate test for paired binary data and is
    reported alongside the registered z-test rather than in place of it.

USAGE
    python scripts/06b_h3_revised.py
==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

CHUNK_SIZE   = 500_000
RATIO_TARGET = 2.0

COLS = ['rating_band', 'last_book_is_premove', 'first_oob_is_premove']


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


def resolve_paths():
    here = Path(__file__).resolve()
    for base in (here.parent.parent, Path.cwd()):
        p = base / 'data' / 'processed' / 'boundary_pairs.csv'
        if p.exists():
            out = base / 'outputs' / 'tables'
            out.mkdir(parents=True, exist_ok=True)
            return p, out / 'h3_revised_results.txt'
    root = here.parent.parent
    return (root / 'data' / 'processed' / 'boundary_pairs.csv',
            root / 'outputs' / 'tables' / 'h3_revised_results.txt')


def to_bool(series):
    """
    Handle True/False, 'True'/'False' and 1/0, as the CSV round trip may give.
    Returns (bool_series, n_missing). Missing values map to False, so they are
    counted here and reported rather than silently absorbed into the not-a-premove
    group, which would understate the rate.
    """
    n_missing = int(series.isna().sum())
    if series.dtype == bool:
        return series, n_missing
    out = series.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    return out, n_missing


def stream_counts(path):
    """One pass over boundary_pairs.csv. Returns per-band 2x2 counts."""
    counts = {}  # band -> [n_both, n_book_only, n_oob_only, n_neither]
    n_rows = 0
    n_missing_book = 0
    n_missing_oob = 0
    n_missing_band = 0

    print(f'Streaming {path.name} ({path.stat().st_size / 1e9:.2f} GB) ...')
    for i, chunk in enumerate(pd.read_csv(path, usecols=COLS,
                                          chunksize=CHUNK_SIZE, low_memory=False)):
        book, mb = to_bool(chunk['last_book_is_premove'])
        oob, mo = to_bool(chunk['first_oob_is_premove'])
        n_missing_book += mb
        n_missing_oob += mo
        n_missing_band += int(chunk['rating_band'].isna().sum())
        n_rows += len(chunk)

        for band, b, o in zip(chunk['rating_band'], book, oob):
            if pd.isna(band):
                continue
            if band not in counts:
                counts[band] = [0, 0, 0, 0]  # both, book_only, oob_only, neither
            if b and o:
                counts[band][0] += 1
            elif b and not o:
                counts[band][1] += 1
            elif o and not b:
                counts[band][2] += 1
            else:
                counts[band][3] += 1

        if (i + 1) % 10 == 0:
            print(f'  {n_rows:,} rows processed...')

    print(f'Total rows: {n_rows:,}')
    counted = sum(sum(v) for v in counts.values())
    print(f'Rows counted: {counted:,}')

    if n_missing_book or n_missing_oob or n_missing_band:
        print('\nMISSING VALUES DETECTED:')
        print(f'  last_book_is_premove missing:  {n_missing_book:,}')
        print(f'  first_oob_is_premove missing:  {n_missing_oob:,}')
        print(f'  rating_band missing:           {n_missing_band:,}')
        print('  Missing premove flags are treated as NOT a premove, which')
        print('  understates that side\'s rate. Rows with a missing rating_band')
        print('  are excluded entirely. Counts that are large relative to the')
        print('  totals below would point to a problem in Script 02.')
    else:
        print('No missing values in any of the three columns.')
    print()
    return counts


def main():
    path, out_path = resolve_paths()
    if not path.exists():
        print(f'ERROR: {path} not found.')
        sys.exit(1)

    tee = Tee(out_path)
    sys.stdout = tee
    try:
        print('Script 06b: H3-revised, immediate execution as a')
        print('discrete-transition signal')
        print('Pre-registration: https://osf.io/n6tkx/  (Section 5.3)\n')

        counts = stream_counts(path)
        bands = sorted(counts)

        # ── Per-band rates and registered z-test ────────────────────────────
        rows = []
        for band in bands:
            both, book_only, oob_only, neither = counts[band]
            n = both + book_only + oob_only + neither
            n_book = both + book_only
            n_oob  = both + oob_only

            book_rate = n_book / n
            oob_rate  = n_oob / n
            ratio = book_rate / oob_rate if oob_rate > 0 else float('inf')

            z, p = proportions_ztest(count=[n_book, n_oob], nobs=[n, n],
                                     alternative='larger')

            # McNemar's test, supplementary: uses only the discordant pairs
            table = [[both, book_only], [oob_only, neither]]
            mc = mcnemar(table, exact=False, correction=True)

            rows.append(dict(band=band, n=n, n_book=n_book, n_oob=n_oob,
                             book_rate=book_rate, oob_rate=oob_rate,
                             ratio=ratio, z=z, p=p,
                             both=both, book_only=book_only, oob_only=oob_only,
                             mcnemar_stat=mc.statistic, mcnemar_p=mc.pvalue))

        # Holm correction across the five registered z-test p-values
        pvals = [r['p'] for r in rows]
        reject, p_holm, _, _ = multipletests(pvals, method='holm', alpha=0.05)
        for r, ph, rej in zip(rows, p_holm, reject):
            r['p_holm'] = ph
            r['reject_holm'] = rej

        # ── Report ────────────────────────────────────────────────────────
        print('=' * 100)
        print('PREMOVE RATES BY BAND (unfiltered boundary_pairs.csv)')
        print('=' * 100)
        print(f'{"band":<9}{"n":>12}{"book rate":>12}{"oob rate":>12}'
              f'{"ratio":>10}{"z":>9}{"p (raw)":>12}{"p (Holm)":>12}')
        print('-' * 100)
        for r in rows:
            ratio_str = f'{r["ratio"]:.2f}' if np.isfinite(r['ratio']) else 'inf'
            print(f'{r["band"]:<9}{r["n"]:>12,}{100*r["book_rate"]:>11.2f}%'
                  f'{100*r["oob_rate"]:>11.2f}%{ratio_str:>10}'
                  f'{r["z"]:>9.2f}{r["p"]:>12.2e}{r["p_holm"]:>12.2e}')
        print()

        print('=' * 100)
        print('SUPPLEMENTARY: McNemar\'s test (paired binary data)')
        print('=' * 100)
        print('Not in the pre-registration. Book/OOB premove status is measured on')
        print('the same pair, so this is the statistically matched test for the')
        print('registered z-test\'s independent-samples assumption. Added before')
        print('any result was observed.\n')
        print(f'{"band":<9}{"both":>10}{"book only":>12}{"oob only":>12}'
              f'{"statistic":>12}{"p-value":>12}')
        print('-' * 100)
        for r in rows:
            print(f'{r["band"]:<9}{r["both"]:>10,}{r["book_only"]:>12,}'
                  f'{r["oob_only"]:>12,}{r["mcnemar_stat"]:>12.2f}'
                  f'{r["mcnemar_p"]:>12.2e}')
        print()
        print('book_only > oob_only in every band would mean book-side premoves')
        print('outnumber OOB-side premoves even among pairs discordant on the')
        print('other dimension: the same direction as the registered test,')
        print('via a different, paired-appropriate mechanism.')
        print()

        # ── Registered decision rule ─────────────────────────────────────────
        print('=' * 100)
        print('PRE-REGISTERED DECISION RULE FOR H3-REVISED')
        print('=' * 100)
        print('Criterion: book proportion > OOB proportion in ALL five bands,')
        print(f'AND ratio >= {RATIO_TARGET} in every band.\n')

        direction_ok = all(r['book_rate'] > r['oob_rate'] for r in rows)
        ratio_ok = all(np.isfinite(r['ratio']) and r['ratio'] >= RATIO_TARGET
                       for r in rows)

        for r in rows:
            d_mark = 'OK' if r['book_rate'] > r['oob_rate'] else 'FAIL'
            ratio_str = f'{r["ratio"]:.2f}' if np.isfinite(r['ratio']) else 'inf'
            r_mark = ('OK' if np.isfinite(r['ratio']) and r['ratio'] >= RATIO_TARGET
                     else 'FAIL')
            print(f'  {r["band"]}: book {100*r["book_rate"]:.2f}% vs '
                  f'oob {100*r["oob_rate"]:.2f}%  [{d_mark}]   '
                  f'ratio {ratio_str}  [{r_mark}]')

        print()
        if direction_ok and ratio_ok:
            print('H3-REVISED SUPPORTED.')
        elif direction_ok and not ratio_ok:
            failing = [r['band'] for r in rows
                      if not (np.isfinite(r['ratio']) and r['ratio'] >= RATIO_TARGET)]
            print('H3-REVISED NOT SUPPORTED.')
            print(f'Direction is correct in every band, but the ratio falls below '
                 f'{RATIO_TARGET} in: {", ".join(failing)}.')
            print('The direction is consistent, but the registered criterion is')
            print('not met; the registered threshold applies.')
        else:
            failing = [r['band'] for r in rows if r['book_rate'] <= r['oob_rate']]
            print('H3-REVISED NOT SUPPORTED.')
            print(f'Direction fails in: {", ".join(failing)}.')

        n_holm_reject = sum(r['reject_holm'] for r in rows)
        print(f'\nHolm-corrected significance: {n_holm_reject} / {len(rows)} bands '
             f'reject at alpha = 0.05.')
        print('Note: with n in the hundreds of thousands to millions per band, the')
        print('z-test will reach significance for effects too small to matter. The')
        print('ratio criterion above, not the p-value, is what decides this')
        print('hypothesis, consistent with H1 and H2.')

        print(f'\nSaved to: {out_path}')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == '__main__':
    main()
