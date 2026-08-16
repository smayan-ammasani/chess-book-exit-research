#!/usr/bin/env python3
"""
Script 09: Transition curve figure (EXPLORATORY)
==============================================================================
Plots the headline result: thinking time across the book boundary.

The visual argument is that thinking time is FLAT OR FALLING while a player
is still inside book, then steps up sharply at the first out-of-book move.
The design puts the in-book region and the out-of-book region on visibly
different footings so a reader can see, without reading the axis, that the
jump is not a continuation of a trend.

Reads outputs/tables/transition_curve.csv, written by Script 08.

STATUS: EXPLORATORY. This analysis was proposed after the confirmatory
results were observed. Label it as such wherever the figure appears.
See https://osf.io/n6tkx/ Section 7.

Produces:
    fig6_transition_curve.png / .pdf   all five bands, one panel
    fig7_curve_by_band.png  / .pdf     small multiples, one panel per band

USAGE
    python scripts/09_curve_figure.py
==============================================================================
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ── Design tokens ────────────────────────────────────────────────────────────
# Sequential blue->orange ramp: encodes rating band as an ordered variable, and
# stays distinguishable in greyscale (lightness increases monotonically).
# Chosen over a categorical palette because the bands ARE ordered -- a
# categorical scheme would throw that information away.
BAND_COLORS = {
    'band_1': '#0B3D5C',
    'band_2': '#2E7A9E',
    'band_3': '#7FB2C4',
    'band_4': '#E08A3C',
    'band_5': '#B33A1E',
}
BAND_LABELS = {
    'band_1': 'band 1  (<1200)',
    'band_2': 'band 2  (1200–1599)',
    'band_3': 'band 3  (1600–1999)',
    'band_4': 'band 4  (2000–2299)',
    'band_5': 'band 5  (2300+)',
}

INK = '#1A1A1A'
MUTED = '#6B6B6B'
BOOK_WASH = '#F0EDE6'      # in-book region
RULE = '#C8C8C8'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.edgecolor': RULE,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


def resolve():
    here = Path(__file__).resolve()
    for base in (here.parent.parent, Path.cwd()):
        p = base / 'outputs' / 'tables' / 'transition_curve.csv'
        if p.exists():
            out = base / 'outputs' / 'figures'
            out.mkdir(parents=True, exist_ok=True)
            return p, out
    root = here.parent.parent
    return (root / 'outputs' / 'tables' / 'transition_curve.csv',
            root / 'outputs' / 'figures')


def seconds_formatter(logval, _pos):
    """Right-hand axis: show what the log value means in seconds."""
    return f'{np.exp(logval):.1f}'


def draw_boundary(ax, ymin, ymax, label=True):
    """Shade the in-book region and mark the boundary."""
    ax.axvspan(-5.6, -0.5, color=BOOK_WASH, zorder=0, lw=0)
    ax.axvline(-0.5, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1, alpha=0.7)
    if label:
        ax.text(-0.7, ymax, 'in book ', ha='right', va='top',
                fontsize=8, color=MUTED, style='italic')
        ax.text(-0.3, ymax, ' out of book', ha='left', va='top',
                fontsize=8, color=MUTED, style='italic')


def main():
    csv_path, fig_dir = resolve()
    if not csv_path.exists():
        print(f'ERROR: {csv_path} not found. Run Script 08 first.')
        sys.exit(1)

    df = pd.read_csv(csv_path)
    bands = [b for b in BAND_COLORS if b in set(df.rating_band)]
    print(f'Loaded {len(df)} rows, {len(bands)} bands.')

    # ═══ FIGURE 6: all bands, one panel ══════════════════════════════════════
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ymin = df.ci_lo.min() - 0.06
    ymax = df.ci_hi.max() + 0.14
    draw_boundary(ax, ymin, ymax)

    for band in bands:
        c = df[df.rating_band == band].sort_values('relative_move')
        pre = c[c.relative_move < 0]
        post = c[c.relative_move >= 0]
        col = BAND_COLORS[band]

        # Break BOTH the line and the CI band at the boundary. The two regimes
        # are not continuous, and bridging them would draw the step as if it
        # were just another trend segment.
        for seg in (pre, post):
            ax.fill_between(seg.relative_move, seg.ci_lo, seg.ci_hi,
                            color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(seg.relative_move, seg.mean_log, color=col, lw=1.9,
                    zorder=3, solid_capstyle='round',
                    label=BAND_LABELS[band] if seg is pre else None)
        ax.plot(c.relative_move, c.mean_log, 'o', color=col, ms=2.6,
                zorder=4, mec='white', mew=0.5)

    ax.set_xlim(-5.6, 10.6)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(-5, 11))
    ax.set_xlabel("Move relative to book exit  (player's own moves)", labelpad=8)
    ax.set_ylabel('Mean log thinking time')

    # Legend sits in the empty lower-right; edge labels collided badly because
    # the band curves converge there.
    leg = ax.legend(loc='lower right', frameon=False, fontsize=8,
                    handlelength=1.6, labelspacing=0.45,
                    bbox_to_anchor=(1.0, 0.02))
    for text, band in zip(leg.get_texts(), bands):
        text.set_color(BAND_COLORS[band])

    ax2 = ax.twinx()
    ax2.set_ylim(ymin, ymax)
    ax2.yaxis.set_major_formatter(FuncFormatter(seconds_formatter))
    ax2.set_ylabel('seconds', color=MUTED, fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(labelsize=8)

    ax.set_title('Thinking time steps up at the book boundary',
                 loc='left', fontsize=12, pad=14, weight='medium')
    ax.text(0, 1.015,
            'Time is flat or falling while still in book, then jumps at the '
            'first out-of-book move.',
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va='bottom')

    fig.text(0.005, -0.02,
             'Exploratory analysis. Shaded bands are 95% bootstrap CIs. '
             'Cell n ranges from 3,770 (band 1, move −5) to 206,635; '
             'coverage at −5 is ~17% of player-games versus 100% at −1 and 0.',
             fontsize=6.8, color=MUTED, ha='left')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = fig_dir / f'fig6_transition_curve.{ext}'
        fig.savefig(p, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {p.name}')
    plt.close(fig)

    # ═══ FIGURE 7: small multiples ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, len(bands), figsize=(12.5, 3.0), sharey=True)
    if len(bands) == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        c = df[df.rating_band == band].sort_values('relative_move')
        pre = c[c.relative_move < 0]
        post = c[c.relative_move >= 0]
        col = BAND_COLORS[band]

        draw_boundary(ax, ymin, ymax, label=False)
        ax.fill_between(c.relative_move, c.ci_lo, c.ci_hi,
                        color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(pre.relative_move, pre.mean_log, color=col, lw=2.0, zorder=3)
        ax.plot(post.relative_move, post.mean_log, color=col, lw=2.0, zorder=3)
        ax.plot(c.relative_move, c.mean_log, 'o', color=col, ms=2.8,
                zorder=4, mec='white', mew=0.5)

        # Annotate the size of the step
        m1 = c.loc[c.relative_move == -1, 'mean_log']
        m0 = c.loc[c.relative_move == 0, 'mean_log']
        if not m1.empty and not m0.empty:
            jump = float(m0.iloc[0] - m1.iloc[0])
            ax.annotate('', xy=(0, float(m0.iloc[0])), xytext=(0, float(m1.iloc[0])),
                        arrowprops=dict(arrowstyle='->', color=INK, lw=1.1,
                                        shrinkA=0, shrinkB=0))
            ax.text(0.5, (float(m0.iloc[0]) + float(m1.iloc[0])) / 2,
                    f'+{jump:.2f}', fontsize=8, color=INK, va='center',
                    ha='left', weight='medium')

        ax.set_xlim(-5.6, 10.6)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([-5, 0, 5, 10])
        ax.set_title(BAND_LABELS[band], fontsize=9, color=col, pad=8)
        ax.set_xlabel('move rel. to exit', fontsize=8)

    axes[0].set_ylabel('Mean log thinking time')
    fig.suptitle('The boundary step grows with rating',
                 x=0.005, ha='left', fontsize=12, weight='medium', y=1.06)
    fig.text(0.005, 0.99,
             'Same axes throughout. Arrow marks the step from the last in-book '
             'move to the first out-of-book move.',
             fontsize=8.5, color=MUTED, ha='left', va='top')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = fig_dir / f'fig7_curve_by_band.{ext}'
        fig.savefig(p, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {p.name}')
    plt.close(fig)

    print('\nDone. Both figures written to outputs/figures/.')
    print('Reminder: label these as exploratory wherever they appear.')


if __name__ == '__main__':
    main()
