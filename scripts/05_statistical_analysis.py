#!/usr/bin/env python3
"""
Script 05: Statistical Analysis
==============================================================================
PURPOSE
    Run the pre-registered confirmatory hypothesis tests for H1 and H2,
    plus the Kruskal-Wallis robustness check. Produce a formatted results
    table for direct inclusion in the paper.

PRE-REGISTERED HYPOTHESES (fixed before examining full results)
    H1 (confirmatory): After controlling for engine-derived value of
        computation (VOC), the first out-of-book move takes significantly
        longer than the last book move within the same game.

    H2 (confirmatory): The magnitude of this residual book-exit time cost
        decreases as player rating increases, consistent with template theory's
        prediction that stronger players' recognition generalises beyond
        literally memorised sequences.

    H3 (exploratory): The book-exit time cost appears primarily as an increase
        in the slow-mode tail of the response-time distribution (run in
        Script 06 only if H1 is confirmed here).

ANALYSIS STRUCTURE
    Primary analysis (player-level):
        One observation per player — their mean log time delta across all
        usable boundary pairs. This eliminates the non-independence problem
        without requiring a mixed-effects model, and is appropriate for
        Kruskal-Wallis. N = 544,599 players.

    Secondary analysis (mixed-effects model):
        Run on a stratified subsample of 25,000 boundary pairs (5,000 per
        band), reshaped to 50,000 long-format observations. This tests H1
        and H2 with proper pair-level VOC control and a random intercept per
        player. If MixedLM fails to converge, falls back to OLS with
        clustered standard errors (same point estimates, robust SEs).

    Robustness: Kruskal-Wallis + Dunn's on player-level means.

INPUTS
    data/processed/analysis_sample.csv       — from Script 04
    data/processed/player_aggregated.csv     — from Script 04

OUTPUTS
    outputs/tables/statistical_results.txt   — full results for paper

ESTIMATED RUNTIME
    ~5-15 minutes.
==============================================================================
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import statsmodels.formula.api as smf
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_SEED, RATING_BANDS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUTS_DIR   = PROJECT_ROOT / 'outputs' / 'tables'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH   = PROCESSED_DIR / 'analysis_sample.csv'
PLAYER_PATH   = PROCESSED_DIR / 'player_aggregated.csv'
RESULTS_PATH  = OUTPUTS_DIR   / 'statistical_results.txt'

# Mixed-effects subsample: pairs per band
MIXEDLM_PAIRS_PER_BAND = 5_000

# Rating bands in display order
BAND_ORDER = sorted(RATING_BANDS.keys())

# Reference band for mixed-effects model interaction contrasts
# Band 3 (1600-1999) is the middle skill tier
REFERENCE_BAND = 'band_3'


# ==============================================================================
# UTILITIES
# ==============================================================================

class ResultsWriter:
    """Writes results simultaneously to console and to the output text file."""

    def __init__(self, path: Path):
        self.path = path
        self.lines = []

    def write(self, line: str = '') -> None:
        log.info(line)
        self.lines.append(line)

    def section(self, title: str) -> None:
        border = '=' * 65
        self.write('')
        self.write(border)
        self.write(title)
        self.write(border)

    def save(self) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
        log.info(f'\nResults saved to: {self.path}')


def cohens_d_one_sample(x: pd.Series) -> float:
    """Cohen's d for a one-sample test (mean / SD)."""
    return float(x.mean() / x.std())


def epsilon_squared(h: float, n: int, k: int) -> float:
    """
    Epsilon-squared effect size for Kruskal-Wallis test.
    ε² = (H - k + 1) / (N - k)
    Preferred over eta-squared for KW because it is unbiased.
    """
    return (h - k + 1) / (n - k)


def stars(p: float) -> str:
    """Significance stars for p-values."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.10:
        return '.'
    return ''


# ==============================================================================
# PHASE 1: LOAD DATA
# ==============================================================================

def load_data():
    """Load analysis sample and player-aggregated datasets."""
    log.info('Loading player-aggregated dataset...')
    player_df = pd.read_csv(PLAYER_PATH)
    log.info(f'Player dataset: {len(player_df):,} players × {player_df.shape[1]} columns')

    log.info('Loading analysis sample (stratified 932K pairs)...')
    sample_df = pd.read_csv(SAMPLE_PATH, low_memory=False)
    log.info(f'Sample dataset: {len(sample_df):,} pairs × {sample_df.shape[1]} columns')

    return player_df, sample_df


# ==============================================================================
# PHASE 2: DESCRIPTIVE STATISTICS
# ==============================================================================

def descriptive_statistics(player_df: pd.DataFrame, rw: ResultsWriter) -> None:
    """
    Descriptive statistics by rating band for the Methods/Results section.
    Uses the player-aggregated dataset (one row per player).
    """
    rw.section('DESCRIPTIVE STATISTICS (player-level, N = {:,})'.format(len(player_df)))

    rw.write(f'\n{"Band":<10} {"N players":>10} {"N pairs (mean)":>15} '
             f'{"Mean Δt (s)":>12} {"Median Δt (s)":>14} '
             f'{"Mean log Δt":>12}')
    rw.write('-' * 80)

    for band in BAND_ORDER:
        sub = player_df[player_df['rating_band'] == band]
        if sub.empty:
            continue
        rw.write(
            f'{band:<10} {len(sub):>10,} {sub["n_pairs"].mean():>15.1f} '
            f'{sub["mean_time_delta"].mean():>12.3f} '
            f'{sub["mean_time_delta"].median():>14.3f} '
            f'{sub["mean_log_time_delta"].mean():>12.4f}'
        )

    rw.write('')
    rw.write(f'Overall mean Δt:      {player_df["mean_time_delta"].mean():.3f} s')
    rw.write(f'Overall median Δt:    {player_df["mean_time_delta"].median():.3f} s')
    rw.write(f'Pct players with Δt > 0: '
             f'{100*(player_df["mean_time_delta"] > 0).mean():.1f}%')


# ==============================================================================
# PHASE 3: H1 — IS THERE A BOOK-EXIT TIME COST?
# ==============================================================================

def test_h1(player_df: pd.DataFrame, rw: ResultsWriter) -> None:
    """
    H1: The first out-of-book move takes significantly longer than the last
    book move, even after controlling for position complexity (VOC).

    Test 1 — Wilcoxon signed-rank test on player-level mean_log_time_delta:
        Tests whether the population median log time delta differs from 0.
        Non-parametric: appropriate given the right-skewed distribution of
        individual time deltas. Uses player-level means, which eliminates
        the within-player non-independence problem (each player contributes
        one observation).

    Test 2 — Cohen's d:
        Quantifies the magnitude of the effect at the player level.
        d = mean(mean_log_time_delta) / SD(mean_log_time_delta)
    """
    rw.section('H1: BOOK-EXIT TIME COST (player-level Wilcoxon)')

    x = player_df['mean_log_time_delta'].dropna()

    rw.write(f'\nN players: {len(x):,}')
    rw.write(f'Mean log Δt:   {x.mean():.4f}  (= {np.exp(x.mean()):.3f}x on raw scale)')
    rw.write(f'Median log Δt: {x.median():.4f}')
    rw.write(f'SD log Δt:     {x.std():.4f}')
    rw.write(f'Pct > 0:       {100*(x > 0).mean():.1f}%')

    # Wilcoxon signed-rank test (H0: median = 0)
    stat, p = stats.wilcoxon(x, alternative='greater')
    d       = cohens_d_one_sample(x)

    rw.write(f'\nWilcoxon signed-rank test (H0: median log Δt = 0, one-tailed):')
    rw.write(f'  W = {stat:.1f},  p = {p:.2e}  {stars(p)}')
    rw.write(f'  Cohen\'s d = {d:.4f}  '
             f'({"small" if abs(d) < 0.2 else "medium" if abs(d) < 0.5 else "large"})')

    rw.write('')
    rw.write('Interpretation:')
    if p < 0.05 and d > 0:
        rw.write('  H1 CONFIRMED. Players take significantly longer on their')
        rw.write('  first out-of-book move than on their last book move.')
    else:
        rw.write('  H1 NOT CONFIRMED at α = 0.05.')

    rw.write('')
    rw.write('NOTE: This test uses player-level means (one observation per')
    rw.write('player), which eliminates within-player non-independence. The')
    rw.write('VOC-controlled result from the mixed-effects model (below)')
    rw.write('is the definitive test of H1 after complexity control.')


# ==============================================================================
# PHASE 4: H2 — DOES THE COST VARY WITH RATING?
# ==============================================================================

def test_h2_kruskal(player_df: pd.DataFrame, rw: ResultsWriter) -> None:
    """
    H2: The book-exit time cost decreases with player rating.

    Kruskal-Wallis test: non-parametric one-way ANOVA on player-level
    mean_log_time_delta across the five rating bands.

    Kruskal-Wallis is preferred over one-way ANOVA because:
    (a) The distribution of mean time deltas is right-skewed even at the
        player level, violating ANOVA's normality assumption.
    (b) Sample sizes are very unequal across bands (Band 5 << Bands 2-3),
        which can affect ANOVA's robustness.

    Effect size: epsilon-squared (ε²), unbiased for Kruskal-Wallis.
    Post-hoc: Dunn's test with Bonferroni correction for pairwise comparisons.
    """
    rw.section("H2: COST VARIATION WITH RATING (Kruskal-Wallis + Dunn's)")

    groups   = [player_df[player_df['rating_band'] == b]['mean_log_time_delta'].dropna().values
                for b in BAND_ORDER]
    group_ns = [len(g) for g in groups]
    N        = sum(group_ns)
    k        = len(groups)

    h_stat, p_kw = stats.kruskal(*groups)
    eps2         = epsilon_squared(h_stat, N, k)

    rw.write(f'\nKruskal-Wallis test across {k} rating bands, N = {N:,}:')
    rw.write(f'  H({k-1}) = {h_stat:.3f},  p = {p_kw:.2e}  {stars(p_kw)}')
    rw.write(f'  ε² = {eps2:.4f}  '
             f'({"small" if eps2 < 0.01 else "medium" if eps2 < 0.06 else "large"} effect)')

    rw.write(f'\nMedian log Δt by band:')
    for band, grp in zip(BAND_ORDER, groups):
        rw.write(f'  {band}: median = {np.median(grp):.4f}  '
                 f'(N = {len(grp):,})')

    # Dunn's post-hoc with Bonferroni correction
    rw.write(f"\nDunn's post-hoc test (Bonferroni-corrected p-values):")
    dunn = sp.posthoc_dunn(
        player_df,
        val_col='mean_log_time_delta',
        group_col='rating_band',
        p_adjust='bonferroni',
    )
    # Print upper triangle only
    bands_in_order = [b for b in BAND_ORDER if b in dunn.columns]
    header = '           ' + ''.join(f'{b:>12}' for b in bands_in_order)
    rw.write(header)
    for i, b1 in enumerate(bands_in_order):
        row = f'{b1:<11}'
        for j, b2 in enumerate(bands_in_order):
            if j <= i:
                row += '            '
            else:
                p_val = dunn.loc[b1, b2]
                row += f'{p_val:>11.3f}{stars(p_val):>1}'
        rw.write(row)

    rw.write('')
    rw.write('Interpretation:')
    if p_kw < 0.05:
        rw.write('  H2 (global test): Significant difference in book-exit')
        rw.write('  time cost across rating bands (Kruskal-Wallis, p < 0.05).')
        rw.write('  See Dunn\'s table and mixed-effects model below for')
        rw.write('  direction and pairwise significance.')
    else:
        rw.write('  H2 NOT CONFIRMED at α = 0.05.')


# ==============================================================================
# PHASE 5: MIXED-EFFECTS MODEL (VOC-CONTROLLED)
# ==============================================================================

def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape boundary pairs from wide format to long format.

    Wide format: one row per pair with last_book_ and first_oob_ columns.
    Long format: two rows per pair (one per observation).

    The long format is required for the mixed-effects model, which treats
    each move observation (book or OOB) as a unit, with pair membership
    captured by the random intercept for player.

    Columns in output:
        player_username, rating_band, is_oob (0/1),
        log_thinking_time, voc, game_id
    """
    book_rows = df_wide.assign(
        is_oob=0,
        log_thinking_time=df_wide['log_last_book_time'],
        voc=df_wide['last_book_voc'],
    )

    oob_rows = df_wide.assign(
        is_oob=1,
        log_thinking_time=df_wide['log_first_oob_time'],
        voc=df_wide['first_oob_voc'],
    )

    keep_cols = ['player_username', 'rating_band', 'is_oob',
                 'log_thinking_time', 'voc', 'game_id']

    long = pd.concat(
        [book_rows[keep_cols], oob_rows[keep_cols]],
        ignore_index=True,
    )
    return long


def run_mixed_effects_model(sample_df: pd.DataFrame, rw: ResultsWriter) -> None:
    """
    Fit a linear mixed-effects model (LMM) on a stratified subsample.

    MODEL SPECIFICATION:
        log(thinking_time) ~ is_oob + voc + rating_band
                           + is_oob × rating_band
                           + (1 | player)

    Fixed effects:
        is_oob          : book (0) vs out-of-book (1) — tests H1
        voc             : Stockfish depth-1 vs depth-15 eval gap — the
                          complexity control variable from Russek et al. (2025)
        rating_band     : skill tier (categorical) — tests H2
        is_oob × rating : interaction — does H1 effect vary by skill? — tests H2

    Random effect:
        (1 | player)    : random intercept per player, absorbing the fact
                          that one player contributes both book and OOB
                          observations across many games (non-independence)

    WHY THIS SUBSAMPLE SIZE:
        Running LMM on 50,000 observations (100K after long reshape) with
        statsmodels MixedLM is computationally feasible on a laptop.
        Statistical power at N=50K is far beyond what is needed to detect
        effects of the magnitude observed in the descriptive statistics.

    REFERENCE BAND: band_3 (1600-1999, club players). Interaction coefficients
    for other bands represent the difference in is_oob effect relative to
    band_3 players.
    """
    rw.section('MIXED-EFFECTS MODEL (VOC-controlled, N = 25,000 pairs)')

    # ── Stratified subsample ──────────────────────────────────────────────────
    rng    = np.random.default_rng(RANDOM_SEED)
    parts  = []

    for band in BAND_ORDER:
        sub = sample_df[sample_df['rating_band'] == band]
        n   = min(MIXEDLM_PAIRS_PER_BAND, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        parts.append(sub.iloc[idx])

    sub_wide = pd.concat(parts, ignore_index=True)
    rw.write(f'\nSubsample size: {len(sub_wide):,} pairs '
             f'({len(sub_wide["player_username"].unique()):,} unique players)')
    rw.write('Band breakdown: ' +
             ', '.join(f'{b}: {(sub_wide["rating_band"]==b).sum()}'
                       for b in BAND_ORDER))

    # ── Reshape to long format ────────────────────────────────────────────────
    df_long = wide_to_long(sub_wide)
    rw.write(f'Long-format observations: {len(df_long):,}')

    # Set band_3 as reference category explicitly
    df_long['rating_band'] = pd.Categorical(
        df_long['rating_band'],
        categories=BAND_ORDER,
        ordered=True,
    )

    # ── Fit model ─────────────────────────────────────────────────────────────
    formula = (
        f"log_thinking_time ~ is_oob + voc "
        f"+ C(rating_band, Treatment('{REFERENCE_BAND}')) "
        f"+ is_oob:C(rating_band, Treatment('{REFERENCE_BAND}'))"
    )

    rw.write(f'\nFormula: {formula}')
    rw.write(f'Random effect: (1 | player_username)')
    rw.write(f'Reference band: {REFERENCE_BAND}')
    rw.write('\nFitting... (this may take several minutes)')

    converged = False
    result    = None

    for method in ['lbfgs', 'powell', 'nm']:
        try:
            model  = smf.mixedlm(formula, data=df_long,
                                 groups=df_long['player_username'])
            result = model.fit(method=method, maxiter=500, disp=False)
            if result.converged:
                rw.write(f'Converged with method: {method}')
                converged = True
                break
            else:
                rw.write(f'Method {method}: did not converge, trying next...')
        except Exception as e:
            rw.write(f'Method {method} failed: {e}')

    if not converged or result is None:
        rw.write('\nMixedLM did not converge. Falling back to OLS with')
        rw.write('clustered standard errors (same point estimates, robust SEs).')
        ols_model = smf.ols(formula, data=df_long)
        result    = ols_model.fit(
            cov_type='cluster',
            cov_kwds={'groups': df_long['player_username']},
        )

    # ── Report results ────────────────────────────────────────────────────────
    rw.write('\n--- Coefficient Table ---')
    rw.write(f'{"Predictor":<50} {"Coef":>8} {"SE":>8} {"z/t":>8} {"p":>10}')
    rw.write('-' * 88)

    params = result.params
    bse    = result.bse
    tvals  = result.tvalues if hasattr(result, 'tvalues') else params / bse
    pvals  = result.pvalues

    for name in params.index:
        coef = params[name]
        se   = bse[name]
        tv   = tvals[name]
        pv   = pvals[name]
        rw.write(
            f'{name:<50} {coef:>8.4f} {se:>8.4f} {tv:>8.3f} {pv:>10.4f}{stars(pv):>3}'
        )

    # ── Key effects ───────────────────────────────────────────────────────────
    rw.write('\n--- Key Effects ---')

    is_oob_coef = params.get('is_oob', np.nan)
    is_oob_p    = pvals.get('is_oob', np.nan)
    rw.write(f'\nH1 — is_oob coefficient:')
    rw.write(f'  β = {is_oob_coef:.4f} (SE = {bse.get("is_oob", np.nan):.4f})')
    rw.write(f'  p = {is_oob_p:.4f}  {stars(is_oob_p)}')
    rw.write(f'  Interpretation: on the log scale, the first OOB move takes')
    if not np.isnan(is_oob_coef):
        rw.write(f'  exp({is_oob_coef:.4f}) = {np.exp(is_oob_coef):.4f}x longer than')
    rw.write(f'  the last book move, holding VOC and rating band constant.')

    voc_coef = params.get('voc', np.nan)
    rw.write(f'\nVOC covariate (complexity control):')
    rw.write(f'  β = {voc_coef:.5f}  (p = {pvals.get("voc", np.nan):.4f}  '
             f'{stars(pvals.get("voc", np.nan))})')
    rw.write(f'  A 1-unit increase in VOC (centipawns) corresponds to a')
    if not np.isnan(voc_coef):
        rw.write(f'  {voc_coef:.5f} increase in log(thinking_time).')

    # ── Interaction terms (H2) ────────────────────────────────────────────────
    rw.write(f'\nH2 — is_oob × rating_band interactions '
             f'(relative to {REFERENCE_BAND}):')
    for band in BAND_ORDER:
        if band == REFERENCE_BAND:
            rw.write(f'  {band}: [reference — is_oob effect = {is_oob_coef:.4f}]')
            continue
        key = f'is_oob:C(rating_band, Treatment(\'{REFERENCE_BAND}\'))[T.{band}]'
        if key not in params.index:
            # Try alternative name format
            for k in params.index:
                if band in k and 'is_oob' in k:
                    key = k
                    break
        if key in params.index:
            ic   = params[key]
            ip   = pvals[key]
            tot  = is_oob_coef + ic
            rw.write(f'  {band}: Δβ = {ic:+.4f}  (total is_oob effect = {tot:.4f}, '
                     f'p = {ip:.4f}  {stars(ip)})')
        else:
            rw.write(f'  {band}: interaction term not found in model output')

    # ── Variance explained ────────────────────────────────────────────────────
    if hasattr(result, 'cov_re'):
        # MixedLM fitted
        try:
            var_random   = float(result.cov_re.iloc[0, 0])
            var_residual = float(result.scale)
            fitted_vals  = result.fittedvalues
            var_fixed    = float(np.var(fitted_vals))

            r2_marg = var_fixed / (var_fixed + var_random + var_residual)
            r2_cond = (var_fixed + var_random) / (var_fixed + var_random + var_residual)

            rw.write(f'\nVariance explained (Nakagawa & Schielzeth 2013):')
            rw.write(f'  Marginal R²   = {r2_marg:.4f}  (fixed effects only)')
            rw.write(f'  Conditional R² = {r2_cond:.4f}  (fixed + random effects)')
            rw.write(f'  Random intercept variance = {var_random:.4f}')
            rw.write(f'  Residual variance         = {var_residual:.4f}')
        except Exception as e:
            rw.write(f'\nVariance components: {e}')
    else:
        rw.write(f'\nR² (OLS with clustered SEs): {result.rsquared:.4f}')


# ==============================================================================
# PHASE 6: ADDITIONAL EFFECT SIZE — RAW SCALE
# ==============================================================================

def raw_scale_effects(player_df: pd.DataFrame, rw: ResultsWriter) -> None:
    """
    Complement the log-scale model with raw-scale descriptives.
    Cohen's d in seconds is more interpretable to non-statistician readers.
    """
    rw.section('RAW-SCALE EFFECT SIZES (player-level, in seconds)')

    x   = player_df['mean_time_delta'].dropna()
    d   = cohens_d_one_sample(x)
    t, p = stats.ttest_1samp(x, 0)

    rw.write(f'\nOne-sample t-test: mean_time_delta vs 0')
    rw.write(f'  Mean Δt = {x.mean():.3f} s  (95% CI: '
             f'[{x.mean() - 1.96*x.sem():.3f}, {x.mean() + 1.96*x.sem():.3f}])')
    rw.write(f'  t({len(x)-1}) = {t:.3f},  p = {p:.2e}  {stars(p)}')
    rw.write(f'  Cohen\'s d = {d:.4f}')

    rw.write(f'\nRaw means by rating band:')
    for band in BAND_ORDER:
        sub = player_df[player_df['rating_band'] == band]['mean_time_delta']
        rw.write(f'  {band}: M = {sub.mean():.3f}s  '
                 f'Mdn = {sub.median():.3f}s  '
                 f'SD = {sub.std():.3f}s  '
                 f'N = {len(sub):,}')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    log.info('Script 05: Statistical Analysis')
    log.info(f'Random seed: {RANDOM_SEED}')
    log.info(f'Mixed-effects subsample: {MIXEDLM_PAIRS_PER_BAND} pairs per band')
    log.info('')

    for path in [SAMPLE_PATH, PLAYER_PATH]:
        if not path.exists():
            log.error(f'Input not found: {path}')
            log.error('Run Script 04 first.')
            sys.exit(1)

    rw = ResultsWriter(RESULTS_PATH)

    rw.write('STATISTICAL ANALYSIS RESULTS')
    rw.write('Chess Book-Exit Cognitive Switch Study')
    rw.write('Pre-registered hypotheses H1 and H2')
    rw.write('')
    rw.write('Significance codes:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.10')

    # ── Load ──────────────────────────────────────────────────────────────────
    player_df, sample_df = load_data()

    # ── Phase 2: Descriptives ─────────────────────────────────────────────────
    descriptive_statistics(player_df, rw)

    # ── Phase 3: H1 (Wilcoxon, player level) ─────────────────────────────────
    test_h1(player_df, rw)

    # ── Phase 4: H2 (Kruskal-Wallis, player level) ───────────────────────────
    test_h2_kruskal(player_df, rw)

    # ── Phase 5: Mixed-effects model ──────────────────────────────────────────
    run_mixed_effects_model(sample_df, rw)

    # ── Phase 6: Raw-scale effect sizes ──────────────────────────────────────
    raw_scale_effects(player_df, rw)

    # ── Save ──────────────────────────────────────────────────────────────────
    rw.write('')
    rw.write('End of statistical results.')
    rw.save()

    log.info('\nScript 05 complete.')
    log.info(f'Results: {RESULTS_PATH}')
    log.info('Next step: python scripts/07_figures.py')


if __name__ == '__main__':
    main()
