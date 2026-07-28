#!/usr/bin/env python3
"""
Script 07: Publication-Quality Figures
==============================================================================
PURPOSE
    Generate all figures for the paper from the statistical analysis outputs.
    All figures are saved as both PNG (300 dpi, for submission) and PDF
    (vector, for final print quality).

FIGURES PRODUCED
    Figure 1 — Main effect: violin plot of log thinking time, book vs OOB
    Figure 2 — Rating moderation: is_oob effect by rating band
    Figure 3 — VOC control: complexity covariate by book status
    Figure 4 — Coefficient plot: model estimates with 95% CIs

INPUT
    data/processed/analysis_sample.csv
    data/processed/player_aggregated.csv

OUTPUT
    outputs/figures/fig1_main_effect.png / .pdf
    outputs/figures/fig2_rating_moderation.png / .pdf
    outputs/figures/fig3_voc_control.png / .pdf
    outputs/figures/fig4_coefficients.png / .pdf

ESTIMATED RUNTIME
    2-5 minutes.
==============================================================================
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RATING_BANDS, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
FIGURES_DIR   = PROJECT_ROOT / 'outputs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH = PROCESSED_DIR / 'analysis_sample.csv'
PLAYER_PATH = PROCESSED_DIR / 'player_aggregated.csv'

BAND_ORDER  = sorted(RATING_BANDS.keys())

# Rating band labels for display on figures
BAND_LABELS = {
    'band_1': '<1200\n(Beginner)',
    'band_2': '1200–1599\n(Intermediate)',
    'band_3': '1600–1999\n(Club)',
    'band_4': '2000–2299\n(Advanced)',
    'band_5': '2300+\n(Expert)',
}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
})

BOOK_COLOR = '#4878CF'    # blue — in-book moves
OOB_COLOR  = '#D65F5F'    # red — out-of-book moves
BAND_PALETTE = sns.color_palette('Blues_d', n_colors=5)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as both PNG and PDF."""
    for ext in ('png', 'pdf'):
        path = FIGURES_DIR / f'{name}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        log.info(f'  Saved: {path.name}')
    plt.close(fig)


# ==============================================================================
# FIGURE 1 — MAIN EFFECT (violin plot)
# ==============================================================================

def figure1_main_effect(sample_df: pd.DataFrame) -> None:
    """
    Violin plot comparing log thinking time for last-book vs first-OOB moves,
    pooled across all players and rating bands.

    This is the visual representation of H1: the distribution of thinking
    times shifts right and spreads wider when players leave book.

    Uses a random subsample of 50,000 pairs for plotting speed.
    """
    log.info('Figure 1: Main effect violin plot...')

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(sample_df), size=min(50_000, len(sample_df)), replace=False)
    sub = sample_df.iloc[idx].copy()

    # Build long-format for the two conditions
    df_book = sub[['log_last_book_time', 'rating_band']].rename(
        columns={'log_last_book_time': 'log_time'})
    df_book['condition'] = 'Last book move'

    df_oob = sub[['log_first_oob_time', 'rating_band']].rename(
        columns={'log_first_oob_time': 'log_time'})
    df_oob['condition'] = 'First out-of-book move'

    df_long = pd.concat([df_book, df_oob], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.violinplot(
        data=df_long,
        x='condition',
        y='log_time',
        palette=[BOOK_COLOR, OOB_COLOR],
        inner='box',
        cut=0,
        ax=ax,
    )

    ax.set_xlabel('')
    ax.set_ylabel('Log thinking time (log seconds)')
    ax.set_title('Figure 1. Thinking time at the book-exit boundary\n'
                 '(n = 50,000 boundary pairs, all rating bands)')

    # Annotate median values
    for i, (cond, color) in enumerate([
        ('Last book move', BOOK_COLOR),
        ('First out-of-book move', OOB_COLOR),
    ]):
        med = df_long[df_long['condition'] == cond]['log_time'].median()
        ax.text(i, med + 0.05, f'Mdn = {med:.2f}',
                ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

    ax.set_xticklabels(['Last book move\n(in preparation)',
                        'First out-of-book move\n(leaving preparation)'])

    fig.tight_layout()
    save_figure(fig, 'fig1_main_effect')


# ==============================================================================
# FIGURE 2 — RATING MODERATION
# ==============================================================================

def figure2_rating_moderation(player_df: pd.DataFrame) -> None:
    """
    Bar chart with error bars showing mean log time delta by rating band.

    This visualises H2: the book-exit time cost decreases with player rating,
    consistent with template theory (stronger players generalise recognition
    beyond literal memorised sequences).

    Error bars are 95% confidence intervals of the mean.
    """
    log.info('Figure 2: Rating moderation by band...')

    rows = []
    for band in BAND_ORDER:
        sub = player_df[player_df['rating_band'] == band]['mean_log_time_delta'].dropna()
        mean = sub.mean()
        se   = sub.sem()
        rows.append({
            'band':  band,
            'label': BAND_LABELS[band],
            'mean':  mean,
            'ci_lo': mean - 1.96 * se,
            'ci_hi': mean + 1.96 * se,
            'n':     len(sub),
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        range(len(df)),
        df['mean'],
        color=BAND_PALETTE,
        edgecolor='white',
        linewidth=0.5,
        zorder=3,
    )

    ax.errorbar(
        range(len(df)),
        df['mean'],
        yerr=[df['mean'] - df['ci_lo'], df['ci_hi'] - df['mean']],
        fmt='none',
        color='#333333',
        capsize=4,
        linewidth=1.5,
        zorder=4,
    )

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['label'], fontsize=9)
    ax.set_ylabel('Mean log time delta\n(log sec; first OOB − last book)')
    ax.set_title('Figure 2. Book-exit time cost by player rating band\n'
                 '(error bars = 95% CI; n = 544,599 players)')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Annotate N per band
    for i, row in df.iterrows():
        ax.text(i, 0.02, f'N={row["n"]:,}',
                ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')

    fig.tight_layout()
    save_figure(fig, 'fig2_rating_moderation')


# ==============================================================================
# FIGURE 3 — VOC CONTROL
# ==============================================================================

def figure3_voc_control(sample_df: pd.DataFrame) -> None:
    """
    Scatter plot showing VOC (complexity) distribution for last-book versus
    first-OOB positions, with a linear fit for each condition.

    Demonstrates that:
    (a) First-OOB positions are modestly more complex (higher VOC) than
        last-book positions, partly explaining the raw time difference.
    (b) The positive relationship between VOC and thinking time exists for
        both conditions, justifying VOC as a control variable.

    Uses a subsample of 10,000 pairs for clarity.
    """
    log.info('Figure 3: VOC control plot...')

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(sample_df), size=min(10_000, len(sample_df)), replace=False)
    sub = sample_df.iloc[idx].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, (col_x, col_y, color, label) in zip(axes, [
        ('last_book_voc',  'log_last_book_time',  BOOK_COLOR, 'Last book move'),
        ('first_oob_voc',  'log_first_oob_time',  OOB_COLOR,  'First out-of-book move'),
    ]):
        x = sub[col_x].clip(0, 500)   # cap extreme outliers for display
        y = sub[col_y]

        ax.scatter(x, y, alpha=0.04, s=4, color=color, rasterized=True)

        # Linear fit
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.quantile(0.99), 100)
        ax.plot(x_line, m * x_line + b, color='black', linewidth=1.5, zorder=5)

        ax.set_xlabel('Value of computation (VOC, centipawns)')
        ax.set_title(label, color=color, fontweight='bold')
        ax.set_xlim(-5, 505)

    axes[0].set_ylabel('Log thinking time (log seconds)')

    fig.suptitle('Figure 3. Relationship between position complexity (VOC) and\n'
                 'thinking time for book vs. out-of-book positions (n = 10,000 pairs)',
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, 'fig3_voc_control')


# ==============================================================================
# FIGURE 4 — COEFFICIENT PLOT
# ==============================================================================

def figure4_coefficients() -> None:
    """
    Coefficient plot showing the total is_oob effect per rating band,
    derived from the mixed-effects model (is_oob + interaction terms).

    This is the clearest visualisation of H2: each bar is the estimated
    multiplier on thinking time (in log units) for the book-exit event,
    for players in that rating band, holding VOC constant.

    Values come directly from the model output in Script 05.
    """
    log.info('Figure 4: Coefficient plot...')

    # Values from Script 05 mixed-effects model output
    # is_oob base (band_3 reference) = 1.3449
    # Interaction deltas from the model output
    is_oob_base = 1.3449
    interactions = {
        'band_1': +0.4095,
        'band_2': +0.1833,
        'band_3':  0.0000,   # reference
        'band_4': -0.1531,
        'band_5': -0.0966,
    }
    # SEs of interaction terms (from model output)
    interaction_se = {
        'band_1': 0.0444,
        'band_2': 0.0445,
        'band_3': 0.0316,   # SE of is_oob itself
        'band_4': 0.0446,
        'band_5': 0.0445,
    }

    totals = {b: is_oob_base + d for b, d in interactions.items()}
    # CIs: total ± 1.96 * sqrt(SE_base² + SE_interaction²)
    # For band_3: ± 1.96 * SE_base
    base_se = 0.0316
    cis = {}
    for band in BAND_ORDER:
        se_int = interaction_se[band]
        if band == 'band_3':
            combined_se = base_se
        else:
            combined_se = np.sqrt(base_se**2 + se_int**2)
        cis[band] = 1.96 * combined_se

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x      = range(len(BAND_ORDER))
    totals_list = [totals[b] for b in BAND_ORDER]
    cis_list    = [cis[b]    for b in BAND_ORDER]

    bars = ax.bar(
        x,
        totals_list,
        color=BAND_PALETTE,
        edgecolor='white',
        linewidth=0.5,
        zorder=3,
    )

    ax.errorbar(
        x, totals_list, yerr=cis_list,
        fmt='none', color='#333333', capsize=5, linewidth=1.5, zorder=4,
    )

    # exp() scale reference line
    ax2 = ax.twinx()
    ax2.set_ylim([np.exp(l) for l in ax.get_ylim()])
    ax2.set_ylabel('Multiplicative effect (×)', color='gray')
    ax2.tick_params(axis='y', colors='gray')

    ax.set_xticks(list(x))
    ax.set_xticklabels([BAND_LABELS[b] for b in BAND_ORDER], fontsize=9)
    ax.set_ylabel('Total is_oob coefficient (log scale)')
    ax.set_title('Figure 4. VOC-controlled book-exit time cost by rating band\n'
                 '(mixed-effects model coefficients ± 95% CI)')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    # Annotate exp(total) above each bar
    for i, (total, band) in enumerate(zip(totals_list, BAND_ORDER)):
        ax.text(i, total + cis_list[i] + 0.03,
                f'{np.exp(total):.2f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.tight_layout()
    save_figure(fig, 'fig4_coefficients')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    log.info('Script 07: Publication-Quality Figures')
    log.info('')

    for path in [SAMPLE_PATH, PLAYER_PATH]:
        if not path.exists():
            log.error(f'Input not found: {path}')
            log.error('Run Scripts 04 and 05 first.')
            sys.exit(1)

    log.info('Loading data...')
    sample_df = pd.read_csv(SAMPLE_PATH, low_memory=False)
    player_df = pd.read_csv(PLAYER_PATH)
    log.info(f'Sample: {len(sample_df):,} pairs | Players: {len(player_df):,}')
    log.info('')

    figure1_main_effect(sample_df)
    figure2_rating_moderation(player_df)
    figure3_voc_control(sample_df)
    figure4_coefficients()

    log.info('')
    log.info('All figures saved to: outputs/figures/')
    log.info('')
    log.info('Script 07 complete.')
    log.info('Next step: write the paper.')


if __name__ == '__main__':
    main()
