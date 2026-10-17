#!/usr/bin/env python3
"""
Script 09: Transition curve figure (EXPLORATORY)
==============================================================================
Plots thinking time across the book boundary.

The design puts the in-book region and the out-of-book region on visibly
different footings and breaks the line at the boundary, so the step reads
as a step. The curve shows cell means, which mix different games at
different positions; claims about trends should cite the within-game
numbers in transition_curve.txt, not the shape of this curve.

Reads outputs/tables/transition_curve.csv, written by Script 08 (version 3).
The footnote's numbers are computed from that file.

STATUS: EXPLORATORY. This analysis was proposed after the confirmatory
results were observed. Label it as such wherever the figure appears.
See https://osf.io/n6tkx/ Section 7.

Produces:
    fig6_transition_curve.png / .pdf   all five bands, one panel
    fig7_curve_by_band.png  / .pdf     small multiples, one panel per band

USAGE
    python scripts/09_curve_figure.py             figures with titles, for reading
    python scripts/09_curve_figure.py --journal   journal versions: no title,
                                                  subtitle or footnote, title-case
                                                  labels, 300 dpi PNG, saved as
                                                  outputs/figures/journal/figure1.png
                                                  (by band) and figure2.png (all bands)
==============================================================================
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
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

# Journal mode: the journal wants each figure as its own image with no title or
# caption (captions go in a separate file) and labels in title case.
JOURNAL = '--journal' in sys.argv
if JOURNAL:
    BAND_LABELS = {k: 'Band' + v[4:] for k, v in BAND_LABELS.items()}
# Journal figures are sized to the 6.5-inch text width so the type prints at
# its set size instead of being shrunk to fit the page.
LOG_LABEL = 'Mean Log Thinking Time (ln s)' if JOURNAL else 'Mean log thinking time'

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
        ax.text(-0.7, ymax, 'In Book ' if JOURNAL else 'in book ', ha='right',
                va='top', fontsize=8, color=MUTED, style='italic')
        ax.text(-0.3, ymax, ' Out of Book' if JOURNAL else ' out of book',
                ha='left', va='top', fontsize=8, color=MUTED, style='italic')


def draw_key(ax):
    """Key to the symbols in the small-multiple panels (journal mode).

    Everything is placed in axes-fraction coordinates, because this cell shares
    its y-axis with the band panels and their data limits would move it.
    """
    ax.set_axis_off()
    tr = ax.transAxes
    x0, x1, xt = 0.04, 0.20, 0.27
    rows = [0.82, 0.62, 0.42, 0.22]
    # In-book region
    ax.add_patch(Rectangle((x0, rows[0] - 0.06), x1 - x0, 0.12, transform=tr,
                           facecolor=BOOK_WASH, edgecolor=RULE, lw=0.6))
    ax.text(xt, rows[0], 'In-Book Moves', va='center', fontsize=8, transform=tr)
    # Book exit
    xm = (x0 + x1) / 2
    ax.plot([xm, xm], [rows[1] - 0.07, rows[1] + 0.07], transform=tr,
            color=INK, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    ax.text(xt, rows[1], 'Book Exit', va='center', fontsize=8, transform=tr)
    # Step arrow
    ax.annotate('', xy=(xm, rows[2] + 0.07), xytext=(xm, rows[2] - 0.07),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.1,
                                shrinkA=0, shrinkB=0))
    ax.text(xt, rows[2], 'Step From Move \u22121 to 0', va='center', fontsize=8,
            transform=tr)
    # Mean with its confidence band
    ax.add_patch(Rectangle((x0, rows[3] - 0.035), x1 - x0, 0.07, transform=tr,
                           facecolor=MUTED, alpha=0.18, lw=0))
    ax.plot([x0, x1], [rows[3], rows[3]], transform=tr, color=MUTED, lw=2.0)
    ax.text(xt, rows[3], 'Mean With 95% CI', va='center', fontsize=8, transform=tr)


def main():
    csv_path, fig_dir = resolve()
    if not csv_path.exists():
        print(f'ERROR: {csv_path} not found. Run Script 08 first.')
        sys.exit(1)

    df = pd.read_csv(csv_path)
    bands = [b for b in BAND_COLORS if b in set(df.rating_band)]
    print(f'Loaded {len(df)} rows, {len(bands)} bands.')
    if JOURNAL:
        fig_dir = fig_dir / 'journal'
        fig_dir.mkdir(parents=True, exist_ok=True)
        print('Journal mode: no titles or footnotes; 300 dpi PNG.')

    def save(fig, normal_name, journal_name):
        if JOURNAL:
            p = fig_dir / journal_name
            fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
            print(f'  Saved: {p}')
            return
        for ext in ('png', 'pdf'):
            p = fig_dir / f'{normal_name}.{ext}'
            fig.savefig(p, bbox_inches='tight', facecolor='white')
            print(f'  Saved: {p.name}')

    # ═══ FIGURE 6: all bands, one panel ══════════════════════════════════════
    fig, ax = plt.subplots(figsize=(6.5, 4.4) if JOURNAL else (7.2, 4.8))

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
    if JOURNAL:
        ax.set_xlabel("Move Relative to Book Exit (Player's Own Moves)", labelpad=8)
        ax.set_ylabel(LOG_LABEL)
    else:
        ax.set_xlabel("Move relative to book exit  (player's own moves)", labelpad=8)
        ax.set_ylabel('Mean log thinking time')

    # Legend sits in the empty lower-right; edge labels collided badly because
    # the band curves converge there.
    handles, labels = ax.get_legend_handles_labels()
    if JOURNAL:
        handles.append(Patch(facecolor=MUTED, alpha=0.18, lw=0))
        labels.append('95% CI')
    leg = ax.legend(handles, labels, loc='lower right', frameon=False, fontsize=8,
                    handlelength=1.6, labelspacing=0.45,
                    bbox_to_anchor=(1.0, 0.02))
    for text, band in zip(leg.get_texts(), bands):
        text.set_color(BAND_COLORS[band])

    ax2 = ax.twinx()
    ax2.set_ylim(ymin, ymax)
    ax2.yaxis.set_major_formatter(FuncFormatter(seconds_formatter))
    ax2.set_ylabel('Geometric Mean Thinking Time (s)' if JOURNAL else 'seconds',
                   color=MUTED, fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(labelsize=8)

    if not JOURNAL:
        ax.set_title('Thinking time around the book exit',
                     loc='left', fontsize=12, pad=14, weight='medium')
        ax.text(0, 1.015,
                "Mean log thinking time at each of the player's own moves, "
                'relative to the first out-of-book move.',
                transform=ax.transAxes, fontsize=8.5, color=MUTED, va='bottom')

    # Footnote numbers come from the data, so they stay correct after reruns.
    lo_row = df.loc[df.n.idxmin()]
    footnote = (f'Exploratory analysis. Shaded bands are 95% confidence intervals '
                f'(standard errors clustered by player). Cell n ranges from '
                f'{int(lo_row.n):,} ({lo_row.rating_band.replace("_", " ")}, move '
                f'{str(int(lo_row.relative_move)).replace("-", "−")}) to '
                f'{int(df.n.max()):,}.')
    if 'n_present' in df.columns:
        # Every player-game has its exit move (position 0), so coverage is
        # measured against position 0.
        at = df.groupby('relative_move').n_present.sum()
        if all(r in at.index for r in (-5, -1, 0)) and at[0] > 0:
            footnote += (f' Player-games with a timed move at position −1: '
                         f'{100 * at[-1] / at[0]:.0f}%; at −5: '
                         f'{100 * at[-5] / at[0]:.0f}%. First moves are excluded '
                         f'because Lichess does not time them.')
    if not JOURNAL:
        fig.text(0.005, -0.02, footnote, fontsize=6.8, color=MUTED, ha='left',
                 wrap=True)

    fig.tight_layout()
    save(fig, 'fig6_transition_curve', 'figure2.png')
    plt.close(fig)

    # ═══ FIGURE 7: small multiples ═══════════════════════════════════════════
    if JOURNAL:
        # Two rows of three at the page width; the unused sixth cell is removed.
        ncol = 3
        nrow = -(-len(bands) // ncol)
        fig, grid = plt.subplots(nrow, ncol, figsize=(6.5, 2.3 * nrow),
                                 sharey=True, squeeze=False)
        cells = list(grid.flat)
        axes = cells[:len(bands)]
        spare = cells[len(bands):]
        # The journal wants keys to symbols inside the figure, not in the
        # caption, so the first spare cell holds a key; any others are removed.
        if spare:
            draw_key(spare[0])
        for extra in spare[1:]:
            fig.delaxes(extra)
        first_col = [grid[r][0] for r in range(nrow)]
    else:
        fig, axes = plt.subplots(1, len(bands), figsize=(12.5, 3.0), sharey=True)
        if len(bands) == 1:
            axes = [axes]
        first_col = [axes[0]]

    for ax, band in zip(axes, bands):
        c = df[df.rating_band == band].sort_values('relative_move')
        pre = c[c.relative_move < 0]
        post = c[c.relative_move >= 0]
        col = BAND_COLORS[band]

        draw_boundary(ax, ymin, ymax, label=False)
        # Break the interval at the boundary, as in the one-panel figure.
        for seg in (pre, post):
            ax.fill_between(seg.relative_move, seg.ci_lo, seg.ci_hi,
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
            # The label is a difference of cell means, which mix different
            # games; the journal figure shows the arrow only and leaves the
            # within-game step to the table.
            if not JOURNAL:
                ax.text(0.5, (float(m0.iloc[0]) + float(m1.iloc[0])) / 2,
                        f'+{jump:.2f}', fontsize=8, color=INK, va='center',
                        ha='left', weight='medium')

        ax.set_xlim(-5.6, 10.6)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([-5, 0, 5, 10])
        ax.set_title(BAND_LABELS[band], fontsize=9, color=col, pad=8)
        ax.set_xlabel('Move Relative to Exit' if JOURNAL else 'move rel. to exit',
                      fontsize=8)

    for ax in first_col:
        ax.set_ylabel(LOG_LABEL, fontsize=8 if JOURNAL else None)
    if not JOURNAL:
        fig.suptitle('Thinking time around the book exit, by rating band',
                     x=0.005, ha='left', fontsize=12, weight='medium', y=1.06)
        fig.text(0.005, 0.99,
                 'Same axes throughout. Arrow marks the step from the last in-book '
                 'move to the first out-of-book move.',
                 fontsize=8.5, color=MUTED, ha='left', va='top')

    fig.tight_layout()
    save(fig, 'fig7_curve_by_band', 'figure1.png')
    plt.close(fig)

    print(f'\nDone. Both figures written to {fig_dir}.')
    print('Reminder: label these as exploratory wherever they appear.')


if __name__ == '__main__':
    main()
