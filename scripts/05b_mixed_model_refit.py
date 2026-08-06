#!/usr/bin/env python3
"""
Script 05b: Mixed-effects model refit (H1, VOC-controlled)
==============================================================================
The Script 05 mixed model was degenerate: random effects covariance singular,
Hessian not positive definite, Group Var = 0.0000, intercept SE = 2.45e6.

Cause: the subsample drew 25,000 pairs from 20,288 players, i.e. 1.23 pairs
per player. A random intercept needs repeated observations per player to be
identified. With one pair each, the between-player variance collapses to the
boundary of the parameter space and the model reports numbers that look like
estimates but are not.

Fix: sample PLAYERS who have at least MIN_PAIRS_PER_PLAYER pairs, then take
their pairs. Every player in the model contributes several observations.

Decision rule is the one registered at https://osf.io/n6tkx/ :
H1 is supported only if the is_oob coefficient is positive AND corresponds to
Cohen's d >= 0.20 in the model that includes VOC. With N in the hundreds of
thousands the p-value is uninformative and is not the criterion.

USAGE
    python scripts/05b_mixed_model_refit.py
==============================================================================
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ── Parameters ───────────────────────────────────────────────────────────────
MIN_PAIRS_PER_PLAYER = 3      # must match MIN_TRANSITIONS_PER_PLAYER
PLAYERS_PER_BAND     = 1500   # capped at available
MAX_PAIRS_PER_PLAYER = 10     # keeps the design balanced and the fit tractable
RANDOM_SEED          = 42
REFERENCE_BAND       = 'band_3'

D_THRESHOLD          = 0.20   # pre-registered

SAMPLE_NAME = 'analysis_sample.csv'
OUT_NAME    = 'mixed_model_refit.txt'

BOOK_TIME_CANDS = ['log_last_book_time']
OOB_TIME_CANDS  = ['log_first_oob_time']
BOOK_VOC_CANDS  = ['last_book_voc']
OOB_VOC_CANDS   = ['first_oob_voc']


class Tee:
    """Write to stdout and to a file at the same time."""

    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')

    def write(self, text):
        sys.__stdout__.write(text)
        self.file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def resolve_paths():
    here = Path(__file__).resolve()
    root = here.parent.parent
    for base in (root, Path.cwd()):
        p = base / 'data' / 'processed' / SAMPLE_NAME
        if p.exists():
            out = base / 'outputs' / 'tables'
            out.mkdir(parents=True, exist_ok=True)
            return p, out / OUT_NAME
    return root / 'data' / 'processed' / SAMPLE_NAME, root / 'outputs' / 'tables' / OUT_NAME


def pick(cols, cands, label):
    for c in cands:
        if c in cols:
            return c
    print(f'ERROR: no {label} column found. Looked for {cands}.')
    print(f'Available: {list(cols)}')
    sys.exit(1)


def build_long_format(df, rng):
    """Select players with enough pairs, then reshape to one row per move."""
    book_t = pick(df.columns, BOOK_TIME_CANDS, 'book log-time')
    oob_t  = pick(df.columns, OOB_TIME_CANDS, 'OOB log-time')
    book_v = pick(df.columns, BOOK_VOC_CANDS, 'book VOC')
    oob_v  = pick(df.columns, OOB_VOC_CANDS, 'OOB VOC')

    counts = df['player_username'].value_counts()
    eligible = counts[counts >= MIN_PAIRS_PER_PLAYER].index
    print(f'Players in sample:                {len(counts):,}')
    print(f'Players with >= {MIN_PAIRS_PER_PLAYER} pairs:          {len(eligible):,}')

    elig = df[df['player_username'].isin(eligible)]

    # One band per player (a player's rating band is fixed in this dataset)
    player_band = elig.groupby('player_username')['rating_band'].first()

    chosen = []
    print('\nPlayers selected per band:')
    for band in sorted(player_band.unique()):
        pool = player_band[player_band == band].index.to_numpy()
        take = min(PLAYERS_PER_BAND, len(pool))
        sel = rng.choice(pool, size=take, replace=False)
        chosen.append(sel)
        print(f'  {band}: {take:>6,} of {len(pool):>8,} eligible')
    chosen = np.concatenate(chosen)

    sub = elig[elig['player_username'].isin(chosen)]

    # Cap pairs per player so a few prolific players cannot dominate.
    # Shuffle first so the retained pairs are a random subset, not the
    # first ones in file order.
    sub = sub.sample(frac=1, random_state=RANDOM_SEED)
    sub = sub.groupby('player_username', sort=False).head(MAX_PAIRS_PER_PLAYER)

    pp = sub['player_username'].value_counts()
    print(f'\nPairs selected:                   {len(sub):,}')
    print(f'Players in model:                 {sub["player_username"].nunique():,}')
    print(f'Pairs per player: mean {pp.mean():.2f}  min {pp.min()}  max {pp.max()}')
    if pp.mean() < 2.0:
        print('\nWARNING: mean pairs per player below 2. The random intercept')
        print('will likely remain unidentified.')

    book = pd.DataFrame({
        'log_thinking_time': sub[book_t].to_numpy(),
        'voc':               sub[book_v].to_numpy(),
        'is_oob':            0,
        'player_username':   sub['player_username'].to_numpy(),
        'rating_band':       sub['rating_band'].to_numpy(),
    })
    oob = pd.DataFrame({
        'log_thinking_time': sub[oob_t].to_numpy(),
        'voc':               sub[oob_v].to_numpy(),
        'is_oob':            1,
        'player_username':   sub['player_username'].to_numpy(),
        'rating_band':       sub['rating_band'].to_numpy(),
    })
    long = pd.concat([book, oob], ignore_index=True)
    long = long.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['log_thinking_time', 'voc'])
    print(f'Long-format observations:         {len(long):,}')
    return long


def fit_mixed(long, formula):
    """Try the optimizer chain. Return (result, method, singular_flag)."""
    for method in ['lbfgs', 'powell', 'nm']:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                res = smf.mixedlm(formula, data=long,
                                  groups=long['player_username']).fit(
                                      method=method, maxiter=500, disp=False)
            except Exception as e:
                print(f'  {method:7s} FAILED: {type(e).__name__}: {str(e)[:70]}')
                continue

            msgs = ' '.join(str(w.message) for w in caught)
            group_var = float(res.cov_re.iloc[0, 0]) if res.cov_re.size else 0.0
            singular = (group_var < 1e-6
                        or 'singular' in msgs.lower()
                        or 'not positive definite' in msgs.lower()
                        or 'boundary of the parameter space' in msgs.lower())

            print(f'  {method:7s} converged={res.converged}  '
                  f'GroupVar={group_var:.6f}  singular={singular}')
            if res.converged and not singular:
                return res, method, False
            last = (res, method, True)
    return last if 'last' in dir() else (None, None, True)


def fit_ols_clustered(long, formula):
    """Pre-registered fallback: OLS with SEs clustered on player."""
    return smf.ols(formula, data=long).fit(
        cov_type='cluster', cov_kwds={'groups': long['player_username']})


def report(res, long, kind):
    print('\n' + '=' * 74)
    print(f'MODEL: {kind}')
    print('=' * 74)

    beta = float(res.params['is_oob'])
    se   = float(res.bse['is_oob'])
    pval = float(res.pvalues['is_oob'])

    if kind.startswith('Mixed'):
        resid_var = float(res.scale)
        group_var = float(res.cov_re.iloc[0, 0]) if res.cov_re.size else 0.0
    else:
        resid_var = float(np.var(res.resid, ddof=len(res.params)))
        group_var = float('nan')

    d_resid = beta / np.sqrt(resid_var)
    d_total = (beta / np.sqrt(resid_var + group_var)
               if np.isfinite(group_var) else float('nan'))

    print(f'\nis_oob coefficient:')
    print(f'  beta                    = {beta:.4f}')
    print(f'  SE                      = {se:.4f}')
    print(f'  p                       = {pval:.3e}')
    print(f'  multiplicative effect   = {np.exp(beta):.4f}x')
    print(f'\nVariance components:')
    print(f'  residual variance       = {resid_var:.4f}  (SD {np.sqrt(resid_var):.4f})')
    if np.isfinite(group_var):
        print(f'  between-player variance = {group_var:.4f}')

    print(f'\nCohen\'s d (registered: beta / residual SD)')
    print(f'  d                       = {d_resid:.4f}')
    if np.isfinite(d_total):
        print(f'  d vs total SD (context) = {d_total:.4f}')

    voc_beta = float(res.params.get('voc', np.nan))
    if np.isfinite(voc_beta):
        print(f'\nVOC covariate:')
        print(f'  beta                    = {voc_beta:.6f}  '
              f'(p = {float(res.pvalues["voc"]):.3e})')
        gap = (long.loc[long.is_oob == 1, 'voc'].mean()
               - long.loc[long.is_oob == 0, 'voc'].mean())
        print(f'  book/OOB VOC gap        = {gap:.2f} cp')
        print(f'  gap explains            = {gap * voc_beta:.5f} log units '
              f'({100 * gap * voc_beta / beta:.2f}% of the is_oob effect)')

    print('\n' + '-' * 74)
    print('PRE-REGISTERED DECISION RULE FOR H1')
    print('-' * 74)
    print(f'  Criterion: is_oob positive AND d >= {D_THRESHOLD}')
    print(f'  Observed:  beta = {beta:+.4f},  d = {d_resid:.4f}')
    if beta > 0 and d_resid >= D_THRESHOLD:
        print(f'\n  H1 SUPPORTED.')
    elif beta > 0:
        print(f'\n  H1 NOT SUPPORTED. Effect is positive but d = {d_resid:.4f}')
        print(f'  falls below the registered threshold of {D_THRESHOLD}.')
        print('  Report as a non-confirmation. Do not substitute a')
        print('  different effect-size metric or threshold after the fact.')
    else:
        print(f'\n  H1 NOT SUPPORTED. Coefficient is not positive.')

    # H2 context only: the registered H2 test is the player-level
    # Kruskal-Wallis in Script 05, which already failed on both criteria.
    inter = [p for p in res.params.index if p.startswith('is_oob:')]
    if inter:
        print('\n' + '-' * 74)
        print('is_oob x rating_band interactions (supplementary; the registered')
        print('H2 test is the player-level Kruskal-Wallis in Script 05)')
        print('-' * 74)
        totals = {REFERENCE_BAND: beta}
        for p in inter:
            band = p.split('T.')[-1].rstrip(']')
            totals[band] = beta + float(res.params[p])
        for band in sorted(totals):
            mark = '  [reference]' if band == REFERENCE_BAND else ''
            print(f'  {band}: total is_oob effect = {totals[band]:+.4f}{mark}')
        ordered = [totals[b] for b in sorted(totals)]
        if all(x > y for x, y in zip(ordered, ordered[1:])):
            print('\n  Monotonically DECREASING across bands (H2 direction).')
        elif all(x < y for x, y in zip(ordered, ordered[1:])):
            print('\n  Monotonically INCREASING across bands (opposite to H2).')
        else:
            print('\n  Not monotonic in either direction.')


def main():
    sample_path, out_path = resolve_paths()
    if not sample_path.exists():
        print(f'ERROR: missing {sample_path}')
        sys.exit(1)

    tee = Tee(out_path)
    sys.stdout = tee
    try:
        rng = np.random.default_rng(RANDOM_SEED)
        print('Script 05b: Mixed-effects model refit (H1, VOC-controlled)')
        print(f'Pre-registration: https://osf.io/n6tkx/')
        print(f'Seed: {RANDOM_SEED}   d threshold: {D_THRESHOLD}\n')

        print(f'Loading {sample_path.name} ...')
        df = pd.read_csv(sample_path, low_memory=False)
        print(f'Loaded {len(df):,} pairs.\n')

        long = build_long_format(df, rng)

        formula = (f"log_thinking_time ~ is_oob + voc "
                   f"+ C(rating_band, Treatment('{REFERENCE_BAND}')) "
                   f"+ is_oob:C(rating_band, Treatment('{REFERENCE_BAND}'))")
        print(f'\nFormula: {formula}')
        print(f'Random effect: (1 | player_username)\n')
        print('Fitting mixed model...')

        res, method, singular = fit_mixed(long, formula)

        if res is not None and not singular:
            report(res, long, f'Mixed-effects (statsmodels MixedLM, {method})')
        else:
            print('\nAll optimizers produced a singular or non-converged fit.')
            print('Falling back to the pre-registered alternative: OLS with')
            print('cluster-robust standard errors clustered on player.\n')
            ols = fit_ols_clustered(long, formula)
            report(ols, long, 'OLS, cluster-robust SEs (player)')
            print('\nNOTE: this substitution is specified in the pre-registration')
            print('and must be stated in the manuscript.')

        print(f'\nSaved to: {out_path}')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == '__main__':
    main()
