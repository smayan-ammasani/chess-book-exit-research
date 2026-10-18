#!/usr/bin/env python3
"""
Script 06: Ex-Gaussian Distribution Fitting (original H3, withdrawn)
==============================================================================
STATUS
    WITHDRAWN. Sample skewness in every band-by-condition cell exceeds the
    ex-Gaussian's structural ceiling of 2.0, so the model cannot describe
    these data (see diagnose_h3.py and
    docs/findings/2026-08-03-pipeline-defects.md). The registered
    replacement, H3-revised, is tested in Script 06b. This script is kept as
    a record.

PURPOSE
    Test the original H3: whether the book-exit time cost appears primarily
    as a change in the slow, deliberate tail of the response-time distribution
    (supporting a discrete cognitive switch) or as a uniform shift of the
    whole distribution (supporting a continuous difficulty gradient).

THE EX-GAUSSIAN DISTRIBUTION
    Reaction time distributions in cognitive psychology are often described by
    the ex-Gaussian: a convolution of a normal distribution and an exponential.
    It has three parameters:

        mu      mean of the Gaussian component (routine, automatic processing)
        sigma   SD of the Gaussian component
        tau     mean of the exponential component (slow, deliberate episodes)

    An increase in tau means the slow tail of the distribution has grown:
    more observations fall in the slow, effortful range. It is often read as
    a sign that a different process (for example, search instead of
    recognition) occurs on some trials, rather than every response slowing
    uniformly.

    An increase in mu with unchanged tau means the whole distribution shifted
    right, which fits a continuous gradient in which the same process takes
    somewhat more time rather than a discrete mode switch.

H3 PREDICTION
    If the book-exit time cost represents a discrete switch from System 1
    (recognition/cached policy) to System 2 (search/online computation):
        → tau_OOB >> tau_book  (large increase in slow tail)
        → mu_OOB ≈ mu_book     (small or no change in Gaussian body)

    If it represents a continuous gradient (just harder positions):
        → tau_OOB ≈ tau_book   (little change in tail)
        → mu_OOB > mu_book     (uniform shift of whole distribution)

    The ratio (tau_OOB - tau_book) / (mu_OOB - mu_book) is the test statistic:
    a ratio >> 1 supports the discrete switch; a ratio ≈ 1 supports continuity.

IMPLEMENTATION
    scipy.stats.exponnorm is the built-in ex-Gaussian implementation.
    Parametrisation: K = tau/sigma, loc ≈ mu, scale = sigma.
    It is fitted separately to book and OOB thinking times for each band.

    Bootstrap 95% CIs are computed for each parameter to quantify uncertainty.

INPUT
    data/processed/analysis_sample.csv

OUTPUT
    outputs/tables/exgaussian_parameters.csv
    outputs/figures/fig5_exgaussian.png / .pdf

ESTIMATED RUNTIME
    10-30 minutes depending on CPU speed (bootstrap is the bottleneck).
==============================================================================
"""

import csv
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RATING_BANDS, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUTS_DIR   = PROJECT_ROOT / 'outputs'
TABLES_DIR    = OUTPUTS_DIR / 'tables'
FIGURES_DIR   = OUTPUTS_DIR / 'figures'
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH  = PROCESSED_DIR / 'analysis_sample.csv'
OUTPUT_CSV   = TABLES_DIR    / 'exgaussian_parameters.csv'
OUTPUT_FIG   = FIGURES_DIR   / 'fig5_exgaussian'

BAND_ORDER   = sorted(RATING_BANDS.keys())
BAND_LABELS  = {
    'band_1': '<1200',
    'band_2': '1200–1599',
    'band_3': '1600–1999',
    'band_4': '2000–2299',
    'band_5': '2300+',
}

# Minimum thinking time for fitting (exclude near-zero premove artifacts)
MIN_TIME_FOR_FIT = 0.5   # seconds

# Number of bootstrap samples for CIs
N_BOOTSTRAP = 500

# Max thinking time to include (prevents extreme outliers from dominating fit)
MAX_TIME_FOR_FIT = 120.0  # seconds

# Pairs per band for fitting (more = slower but more stable)
PAIRS_PER_BAND = 5_000


# ==============================================================================
# EX-GAUSSIAN UTILITIES
# ==============================================================================

def exgauss_params_from_scipy(K: float, loc: float, scale: float) -> dict:
    """
    Convert scipy.stats.exponnorm parametrisation to standard ex-Gaussian.

    scipy uses: K, loc, scale where
        K     = tau / sigma
        loc   ≈ mu   (Gaussian component location)
        scale = sigma

    Returns dict with mu, sigma, tau, and derived statistics.
    """
    sigma = scale
    tau   = K * scale
    mu    = loc

    return {
        'mu':    mu,
        'sigma': sigma,
        'tau':   tau,
        'mean':  mu + tau,          # theoretical mean
        'sd':    np.sqrt(sigma**2 + tau**2),  # theoretical SD
    }


def fit_exgaussian(data: np.ndarray, seed: int = 42) -> dict | None:
    """
    Fit an ex-Gaussian distribution using the method of moments.

    MLE (scipy.stats.exponnorm.fit) is numerically unstable with the
    integer-second clock data produced by Lichess. Method of moments (MOM)
    is the standard alternative for discretized RT data (Ratcliff 1979;
    Van Zandt 2000).

    Jitter of [-0.4, +0.4] seconds is added to convert integer-second
    values to pseudo-continuous data before parameter estimation.
    """
    rng   = np.random.default_rng(seed)
    clean = data[(data >= MIN_TIME_FOR_FIT) & (data <= MAX_TIME_FOR_FIT)]
    clean = clean[np.isfinite(clean)]

    if len(clean) < 50:
        return None

    # Jitter to pseudo-continuous (standard for RT distributions)
    jittered = clean + rng.uniform(-0.4, 0.4, size=len(clean))
    jittered = np.clip(jittered, MIN_TIME_FOR_FIT, MAX_TIME_FOR_FIT)

    m    = float(np.mean(jittered))
    v    = float(np.var(jittered))
    skew = float(stats.skew(jittered))

    tau   = max(0.05, (v**1.5 * max(skew, 0.01) / 2) ** (1/3))
    sigma = max(0.05, float(np.sqrt(max(0.01, v - tau**2))))
    mu    = max(0.05, m - tau)

    return {
        'mu':    round(mu,    3),
        'sigma': round(sigma, 3),
        'tau':   round(tau,   3),
        'mean':  round(mu + tau, 3),
        'sd':    round(float(np.sqrt(sigma**2 + tau**2)), 3),
        'n':     len(clean),
    }


def bootstrap_ci(data: np.ndarray, n_boot: int, seed: int) -> dict:
    """
    Bootstrap 95% CIs for ex-Gaussian mu, sigma, and tau.

    Returns dict with lower/upper bounds for each parameter,
    or NaN bounds if fitting fails consistently.
    """
    rng        = np.random.default_rng(seed)
    boot_mu    = []
    boot_sigma = []
    boot_tau   = []

    clean = data[(data >= MIN_TIME_FOR_FIT) & (data <= MAX_TIME_FOR_FIT)]
    clean = clean[np.isfinite(clean)]

    if len(clean) < 50:
        return {'mu_lo': np.nan, 'mu_hi': np.nan,
                'sigma_lo': np.nan, 'sigma_hi': np.nan,
                'tau_lo': np.nan, 'tau_hi': np.nan}

    seed_counter = 0
    for _ in range(n_boot):
        sample = rng.choice(clean, size=len(clean), replace=True)
        result = fit_exgaussian(sample, seed=seed_counter)
        seed_counter += 1
        if result is not None:
            boot_mu.append(result['mu'])
            boot_sigma.append(result['sigma'])
            boot_tau.append(result['tau'])

    if len(boot_mu) < 10:
        return {'mu_lo': np.nan, 'mu_hi': np.nan,
                'sigma_lo': np.nan, 'sigma_hi': np.nan,
                'tau_lo': np.nan, 'tau_hi': np.nan}

    def ci(vals):
        arr = np.array(vals)
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    mu_lo, mu_hi       = ci(boot_mu)
    sigma_lo, sigma_hi = ci(boot_sigma)
    tau_lo, tau_hi     = ci(boot_tau)

    return {
        'mu_lo': mu_lo, 'mu_hi': mu_hi,
        'sigma_lo': sigma_lo, 'sigma_hi': sigma_hi,
        'tau_lo': tau_lo, 'tau_hi': tau_hi,
    }


# ==============================================================================
# MAIN FITTING LOOP
# ==============================================================================

def run_fitting(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit ex-Gaussian distributions to book and OOB thinking times for each
    rating band. Return a DataFrame with all fitted parameters and CIs.
    """
    rng  = np.random.default_rng(RANDOM_SEED)
    rows = []

    for band in BAND_ORDER:
        band_data = sample_df[sample_df['rating_band'] == band]
        n_avail   = len(band_data)
        n_use     = min(PAIRS_PER_BAND, n_avail)

        if n_use < 100:
            log.warning(f'{band}: too few pairs ({n_avail}), skipping.')
            continue

        idx = rng.choice(n_avail, size=n_use, replace=False)
        sub = band_data.iloc[idx]

        book_times = sub['last_book_thinking_time'].values
        oob_times  = sub['first_oob_thinking_time'].values

        log.info(f'{band}: fitting on {n_use:,} pairs '
                 f'({n_avail:,} available)...')

        for condition, times in [('book', book_times), ('oob', oob_times)]:
            params = fit_exgaussian(times)
            if params is None:
                log.warning(f'  {band} {condition}: fitting failed')
                continue

            log.info(f'  {condition}: mu={params["mu"]:.2f}  '
                     f'sigma={params["sigma"]:.2f}  '
                     f'tau={params["tau"]:.2f}  '
                     f'(n={params["n"]:,})')

            log.info(f'    Computing bootstrap CIs ({N_BOOTSTRAP} samples)...')
            seed = hash((band, condition)) % (2**31)
            cis  = bootstrap_ci(times, N_BOOTSTRAP, seed)

            rows.append({
                'rating_band': band,
                'band_label':  BAND_LABELS[band],
                'condition':   condition,
                'mu':          round(params['mu'],    3),
                'sigma':       round(params['sigma'], 3),
                'tau':         round(params['tau'],   3),
                'mean':        round(params['mean'],  3),
                'sd':          round(params['sd'],    3),
                'n':           params['n'],
                **{k: round(v, 3) if not np.isnan(v) else np.nan
                   for k, v in cis.items()},
            })

    return pd.DataFrame(rows)


# ==============================================================================
# FIGURE 5: EX-GAUSSIAN DECOMPOSITION
# ==============================================================================

def figure5_exgaussian(results: pd.DataFrame) -> None:
    """
    Three-panel figure showing mu, sigma, and tau for book vs OOB conditions
    across rating bands.

    The key panel is tau: if tau_OOB >> tau_book, this supports H3 (discrete
    cognitive switch). If tau is similar but mu differs, it suggests a
    continuous gradient.
    """
    if results.empty or 'condition' not in results.columns:
        log.warning('No fitted parameters to plot; skipping Figure 5.')
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    book_color = '#4878CF'
    oob_color  = '#D65F5F'
    params_to_plot = [
        ('mu',    'μ (Gaussian body, sec)',     axes[0]),
        ('sigma', 'σ (Gaussian spread, sec)',   axes[1]),
        ('tau',   'τ (Exponential tail, sec)',  axes[2]),
    ]

    x = np.arange(len(BAND_ORDER))
    width = 0.35

    for param, ylabel, ax in params_to_plot:
        for i, (condition, color, offset) in enumerate([
            ('book', book_color, -width/2),
            ('oob',  oob_color,  +width/2),
        ]):
            sub    = results[results['condition'] == condition]
            values = []
            lo_err = []
            hi_err = []

            for band in BAND_ORDER:
                row = sub[sub['rating_band'] == band]
                if row.empty:
                    values.append(np.nan)
                    lo_err.append(0)
                    hi_err.append(0)
                else:
                    v = float(row[param].iloc[0])
                    lo_k = f'{param}_lo'
                    hi_k = f'{param}_hi'
                    lo = float(row[lo_k].iloc[0]) if lo_k in row.columns else np.nan
                    hi = float(row[hi_k].iloc[0]) if hi_k in row.columns else np.nan
                    values.append(v)
                    lo_err.append(v - lo if not np.isnan(lo) else 0)
                    hi_err.append(hi - v if not np.isnan(hi) else 0)

            ax.bar(x + offset, values, width, color=color,
                   label='Book' if i == 0 else 'Out-of-book',
                   alpha=0.85, edgecolor='white')
            ax.errorbar(x + offset, values,
                        yerr=[lo_err, hi_err],
                        fmt='none', color='#333333',
                        capsize=3, linewidth=1.2)

        ax.set_xticks(list(x))
        ax.set_xticklabels([BAND_LABELS[b] for b in BAND_ORDER],
                           fontsize=8, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(param.upper(), fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend on last axis
    book_patch = mpatches.Patch(color=book_color, alpha=0.85, label='Last book move')
    oob_patch  = mpatches.Patch(color=oob_color,  alpha=0.85, label='First OOB move')
    axes[2].legend(handles=[book_patch, oob_patch], loc='upper right', fontsize=9)

    fig.suptitle(
        'Figure 5. Ex-Gaussian decomposition of thinking time distributions\n'
        'by condition (book vs. OOB) and rating band\n'
        '(error bars = 95% bootstrap CI)',
        fontsize=11,
    )
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        path = f'{OUTPUT_FIG}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        log.info(f'  Saved: {Path(path).name}')
    plt.close(fig)


# ==============================================================================
# INTERPRET RESULTS
# ==============================================================================

def interpret_results(results: pd.DataFrame) -> None:
    """
    Print an H3 interpretation based on the fitted parameters.

    The key comparison: does tau change more than mu when moving from
    book to OOB? If so, the effect is tail-driven (discrete switch).
    """
    log.info('')
    log.info('=' * 65)
    log.info('H3 INTERPRETATION')
    log.info('=' * 65)
    log.info('')
    log.info(f'{"Band":<15} {"Δmu (OOB-book)":>16} {"Δtau (OOB-book)":>16} '
             f'{"Δtau/Δmu":>10} {"Support for H3"}')
    log.info('-' * 75)

    if results.empty or 'condition' not in results.columns:
        log.warning('No results to interpret.')
        return

    for band in BAND_ORDER:
        book_row = results[(results['rating_band'] == band) &
                           (results['condition'] == 'book')]
        oob_row  = results[(results['rating_band'] == band) &
                           (results['condition'] == 'oob')]

        if book_row.empty or oob_row.empty:
            continue

        mu_book  = float(book_row['mu'].iloc[0])
        mu_oob   = float(oob_row['mu'].iloc[0])
        tau_book = float(book_row['tau'].iloc[0])
        tau_oob  = float(oob_row['tau'].iloc[0])

        delta_mu  = mu_oob  - mu_book
        delta_tau = tau_oob - tau_book

        if delta_mu != 0:
            ratio = delta_tau / delta_mu
        else:
            ratio = float('inf')

        # Ratio > 1 means tau changed more than mu → discrete switch
        # Ratio < 1 means mu changed more than tau → continuous gradient
        support = ('Discrete switch (τ dominates)'
                   if ratio > 1.5
                   else 'Mixed / continuous gradient'
                   if ratio > 0.5
                   else 'Continuous gradient (μ dominates)')

        log.info(f'{band:<15} {delta_mu:>16.3f} {delta_tau:>16.3f} '
                 f'{ratio:>10.2f}  {support}')

    log.info('')
    log.info('Δtau/Δmu > 1: increase in slow-tail component exceeds Gaussian shift')
    log.info('  → the pattern expected under a discrete switch (original H3)')
    log.info('Δtau/Δmu < 1: Gaussian shift exceeds tail change')
    log.info('  → the pattern expected under a continuous gradient')
    log.info('')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    log.info('Script 06: Ex-Gaussian Distribution Fitting (original H3, withdrawn)')
    log.info(f'Bootstrap samples: {N_BOOTSTRAP}')
    log.info(f'Pairs per band:    {PAIRS_PER_BAND:,}')
    log.info(f'Time range:        [{MIN_TIME_FOR_FIT}, {MAX_TIME_FOR_FIT}] seconds')
    log.info('')
    log.info('NOTE: The original H3 was withdrawn because the data exceed the')
    log.info('ex-Gaussian skewness ceiling. H3-revised is tested in Script 06b.')
    log.info('')

    if not SAMPLE_PATH.exists():
        log.error(f'Input not found: {SAMPLE_PATH}')
        log.error('Run Script 04 first.')
        sys.exit(1)

    log.info('Loading analysis sample...')
    sample_df = pd.read_csv(
        SAMPLE_PATH,
        usecols=['rating_band', 'last_book_thinking_time', 'first_oob_thinking_time'],
        low_memory=False,
    )
    log.info(f'Loaded {len(sample_df):,} pairs.')
    log.info('')

    log.info('Fitting ex-Gaussian distributions...')
    results = run_fitting(sample_df)

    log.info('')
    log.info('Saving parameter table...')
    results.to_csv(OUTPUT_CSV, index=False)
    log.info(f'Saved: {OUTPUT_CSV}')

    log.info('')
    log.info('Generating Figure 5...')
    import matplotlib.patches as mpatches
    globals()['mpatches'] = mpatches
    figure5_exgaussian(results)

    interpret_results(results)

    log.info('Script 06 complete.')
    log.info(f'Table: {OUTPUT_CSV}')
    log.info(f'Figure: {OUTPUT_FIG}.png / .pdf')


if __name__ == '__main__':
    import matplotlib.patches as mpatches
    main()
