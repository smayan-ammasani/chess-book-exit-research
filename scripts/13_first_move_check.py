#!/usr/bin/env python3
"""
Script 13: Untimed first moves (check and corrected analyses)
==============================================================================
WHAT WAS FOUND
    Lichess does not time either player's first move. The clock starts only
    after Black's first move, and neither first move earns an increment
    (scalachess: Game.applyClock starts the clock after ply 1, and
    Clock.step takes no time and adds no increment while the clock is not
    running). So the [%clk] value after each player's first move is the
    starting time, and Script 02's formula

        thinking_time = prev_clock - curr_clock + increment
        (prev_clock = base time for a player's first move)

    gives every first move a "thinking time" equal to the increment:
    0 s in games with no increment, 5 to 20 s in games with one. Neither
    number measures thinking.

WHERE IT MATTERS
    A boundary pair is affected when the player's last in-book move was
    their first move of the game, so they left book on their second move:
    ply_at_exit 3 (White) or 4 (Black).
      - H3-revised: in games with no increment, every affected pair has a
        book move recorded as 0 s, so it counts as near-instant by
        construction.
      - H1 and H2: affected pairs from games with no increment were already
        removed by the premove filter (book time 0 s). Affected pairs from
        games with an increment were kept, with the increment standing in
        for the book move's thinking time.
      - Transition curve: first moves sit at in-book positions. Script 08
        version 3 removes them.

WHAT THIS SCRIPT DOES
    1. Tests the claim on your data: for affected pairs, how often the
       recorded book-move time equals the game's increment, next to the
       same rate for all other pairs.
    2. Counts affected pairs by band and time-control type.
    3. Re-runs H3-revised, H1 (registered model, band model, checks 3 and
       4), H2 and the deviator/responder split without affected pairs,
       beside the registered results. The registered numbers are recomputed
       first and should match Script 10 exactly.

    The corrected analyses are a correction made after registration. Report
    them beside the registered results, not in place of them.

    Requires scripts/10_registered_analyses.py (its functions are reused,
    so both scripts fit models and compute intervals the same way).

USAGE
    python scripts/13_first_move_check.py
    Runtime: about the same as Script 10.
    Output: outputs/tables/first_move_check.txt
==============================================================================
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FIRST_MOVE_MAX_EXIT = 4   # ply_at_exit 3 (White) or 4 (Black): book move = first move
CONFIRM_SHARE = 0.99      # share of affected pairs whose book time must equal the increment

HERE = Path(__file__).resolve()
S10_PATH = HERE.parent / '10_registered_analyses.py'
if not S10_PATH.exists():
    sys.exit(f'ERROR: {S10_PATH} not found. Script 13 reuses its functions.')
_spec = importlib.util.spec_from_file_location('s10', S10_PATH)
s10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s10)

BANDS = s10.BANDS
CHUNK = s10.CHUNK
MIN_PAIRS = s10.MIN_PAIRS
MAX_EXIT_PLY = s10.MAX_EXIT_PLY
header = s10.header


def increments(tc: pd.Series) -> np.ndarray:
    """Increment in seconds from a TimeControl string such as '600+5'."""
    tc = tc.astype(str)
    lookup = {}
    for s in tc.unique():
        try:
            lookup[s] = float(s.split('+', 1)[1])
        except (IndexError, ValueError):
            lookup[s] = np.nan
    return tc.map(lookup).to_numpy(float)


# ==============================================================================
# Passes over the large files
# ==============================================================================

def pass_boundary(path, sample_games):
    cols = ['game_id', 'player_color', 'player_username', 'rating_band',
            'ply_at_exit', 'last_book_thinking_time', 'first_oob_thinking_time',
            'last_book_is_premove', 'first_oob_is_premove', 'time_control']
    h3_parts, chk_parts, role_parts, n_rows = [], [], [], 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        ch = ch.dropna(subset=['player_username', 'rating_band'])
        n_rows += len(ch)
        inc = increments(ch['time_control'])
        lb = pd.to_numeric(ch['last_book_thinking_time'], errors='coerce').to_numpy(float)
        ob = pd.to_numeric(ch['first_oob_thinking_time'], errors='coerce').to_numpy(float)
        first = (ch['ply_at_exit'] <= FIRST_MOVE_MAX_EXIT).to_numpy()
        cap = (ch['ply_at_exit'] > MAX_EXIT_PLY).to_numpy()
        x1 = s10.to_bool(ch['last_book_is_premove']).to_numpy(int)
        x2 = s10.to_bool(ch['first_oob_is_premove']).to_numpy(int)
        band = ch['rating_band'].to_numpy()
        g = pd.DataFrame({'band': band,
                          'player': ch['player_username'].astype(str).to_numpy(),
                          'cap': cap, 'first': first, 'n': 1, 'x1': x1, 'x2': x2})
        h3_parts.append(g.groupby(['band', 'player', 'cap', 'first'], sort=False).sum())
        if len(h3_parts) >= 4:
            h3_parts = [pd.concat(h3_parts).groupby(level=[0, 1, 2, 3], sort=False).sum()]
        c = pd.DataFrame({'band': band, 'first': first, 'no_inc': inc == 0, 'n': 1,
                          'lb_eq': np.abs(lb - inc) < 1e-6,
                          'ob_eq': np.abs(ob - inc) < 1e-6,
                          'x1': x1, 'x2': x2})
        chk_parts.append(c.groupby(['band', 'first', 'no_inc'], sort=False).sum())
        keep = ch['game_id'].isin(sample_games)
        if keep.any():
            role_parts.append(ch.loc[keep, ['game_id', 'player_color', 'ply_at_exit']])
    h3 = pd.concat(h3_parts).groupby(level=[0, 1, 2, 3], sort=False).sum().reset_index()
    chk = pd.concat(chk_parts).groupby(level=[0, 1, 2]).sum().reset_index()
    roles = (pd.concat(role_parts, ignore_index=True) if role_parts
             else pd.DataFrame(columns=['game_id', 'player_color', 'ply_at_exit']))
    return h3, chk, roles, n_rows


def pass_filtered(path):
    cols = ['player_username', 'rating_band', 'ply_at_exit', 'log_time_delta',
            'last_book_voc', 'first_oob_voc']
    parts, n_rows = [], 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        ch = ch.dropna(subset=['player_username', 'rating_band'])
        n_rows += len(ch)
        x = (ch['first_oob_voc'] - ch['last_book_voc']).to_numpy(float)
        y = ch['log_time_delta'].to_numpy(float)
        g = pd.DataFrame({
            'player': ch['player_username'].astype(str).to_numpy(),
            'band': ch['rating_band'].to_numpy(),
            'cap': (ch['ply_at_exit'] > MAX_EXIT_PLY).to_numpy(),
            'first': (ch['ply_at_exit'] <= FIRST_MOVE_MAX_EXIT).to_numpy(),
            'n': 1, 'sy': y, 'sx': x, 'sxx': x * x, 'sxy': x * y,
            'slb': ch['last_book_voc'].to_numpy(float),
            'soob': ch['first_oob_voc'].to_numpy(float),
        })
        parts.append(g.groupby(['player', 'band', 'cap', 'first'], sort=False).sum())
        if len(parts) >= 4:
            parts = [pd.concat(parts).groupby(level=[0, 1, 2, 3], sort=False).sum()]
    agg = pd.concat(parts).groupby(level=[0, 1, 2, 3], sort=False).sum().reset_index()
    return agg, n_rows


# ==============================================================================
# Summaries
# ==============================================================================

def h3_ratios(d):
    out = {}
    for bnd in BANDS:
        s = d[d['band'] == bnd].groupby('player')[['n', 'x1', 'x2']].sum()
        if s.empty:
            continue
        N, X1, X2, G = s['n'].sum(), s['x1'].sum(), s['x2'].sum(), len(s)
        if X2 == 0 or G < 2:
            out[bnd] = (X1 / N, X2 / N, np.inf, np.nan, np.nan)
            continue
        ratio = X1 / X2
        phi = s['x1'].to_numpy() / X1 - s['x2'].to_numpy() / X2
        se_log = np.sqrt(G / (G - 1) * np.sum(phi ** 2))
        out[bnd] = (X1 / N, X2 / N, ratio, ratio * np.exp(-1.96 * se_log),
                    ratio * np.exp(1.96 * se_log))
    return out


def kw_summary(Q, col):
    groups = [Q.loc[Q['rating_band'] == b, col].dropna().to_numpy(float) for b in BANDS]
    H, _ = stats.kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    return (H - k + 1) / (n - k), [float(np.median(g)) for g in groups]


def band_model(L, title):
    m = s10.fit_terms(L, ['is_oob', 'voc', 'bands', 'is_oob:bands'])
    sd_e = np.sqrt(m['s2e'])
    print(f'\n{title}')
    print(f'residual SD {sd_e:.4f}')
    print(f'{"band":<8}{"is_oob effect":>15}{"95% CI":>22}{"ratio":>8}{"d":>8}')
    i0 = m['names'].index('is_oob')
    for bnd in BANDS:
        if bnd == s10.REF_BAND:
            est, var = m['b'][i0], m['cov'][i0, i0]
        else:
            name = f'is_oob:band[{bnd}]'
            if name not in m['names']:
                continue
            k = m['names'].index(name)
            est = m['b'][i0] + m['b'][k]
            var = m['cov'][i0, i0] + m['cov'][k, k] + 2 * m['cov'][i0, k]
        se = np.sqrt(var)
        print(f'{bnd:<8}{est:>15.4f}'
              f'{f"[{est - 1.96 * se:.4f}, {est + 1.96 * se:.4f}]":>22}'
              f'{np.exp(est):>8.3f}{est / sd_e:>8.3f}')


def h2_players(P):
    Q = P[P['n'] >= MIN_PAIRS].copy()
    icpt, slope, n_ols = s10.ols_from_sums(Q)
    Q['mean_log_time_delta'] = Q['sy'] / Q['n']
    Q['resid'] = (Q['sy'] - icpt * Q['n'] - slope * Q['sx']) / Q['n']
    return Q, icpt, slope, n_ols


# ==============================================================================
# Main
# ==============================================================================

def main():
    root = s10.resolve_root()
    proc = root / 'data' / 'processed'
    out_dir = root / 'outputs' / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = s10.Tee(out_dir / 'first_move_check.txt')
    sys.stdout = tee
    warnings.filterwarnings('ignore')
    try:
        print('Script 13: Untimed first moves (check and corrected analyses)')
        print('Pre-registration: https://osf.io/n6tkx/')
        print(f'Affected pairs: ply_at_exit <= {FIRST_MOVE_MAX_EXIT}, meaning the '
              "player's last in-book move\nwas their first move of the game, "
              'which Lichess does not time.')

        S = pd.read_csv(proc / 'analysis_sample.csv', low_memory=False)
        S['player_username'] = S['player_username'].astype(str)
        S = S.drop_duplicates(['game_id', 'player_color'])

        # ── Pass over boundary_pairs.csv ──────────────────────────────────
        print('\nReading boundary_pairs.csv (one pass) ...')
        h3, chk, roles, n_bp = pass_boundary(proc / 'boundary_pairs.csv',
                                             set(S['game_id'].unique()))
        print(f'rows read: {n_bp:,}')

        # ── 1. The check ──────────────────────────────────────────────────
        header("1. CHECK: is a first move's recorded time just the increment?")
        print('If Lichess does not time first moves, every affected pair should '
              'show a book-move\ntime exactly equal to the increment (0 s in games '
              'without one). Other pairs show\nthat value only when a move really '
              'took exactly that long.')
        print(f'\n{"pairs":<38}{"count":>12}{"book time = increment":>24}'
              f'{"out-of-book time = increment":>31}')
        shares = {}
        for flag, label in ((True, "book move = player's first move"),
                            (False, 'all other pairs')):
            s = chk[chk['first'] == flag]
            n, e1, e2 = s['n'].sum(), s['lb_eq'].sum(), s['ob_eq'].sum()
            shares[flag] = e1 / n if n else np.nan
            print(f'{label:<38}{n:>12,}{100 * e1 / max(n, 1):>23.2f}%'
                  f'{100 * e2 / max(n, 1):>30.2f}%')
        print('\nAffected pairs by time-control type:')
        for no_inc, label in ((True, 'no increment (x+0)'), (False, 'with increment')):
            s = chk[chk['first'] & (chk['no_inc'] == no_inc)]
            n = s['n'].sum()
            if n == 0:
                continue
            print(f'  {label:<20} {n:>10,} pairs; book time = increment '
                  f'{100 * s["lb_eq"].sum() / n:6.2f}%; book move counted as '
                  f'near-instant {100 * s["x1"].sum() / n:6.2f}%')
        n_affected = int(chk.loc[chk['first'], 'n'].sum())
        confirmed = bool(n_affected > 0 and shares.get(True, 0) >= CONFIRM_SHARE)
        if n_affected == 0:
            print('\nNo affected pairs: no pair left book on the player\'s second move, '
                  'so nothing\nbelow should differ from Script 10.')
        elif confirmed:
            print(f'\nCONFIRMED: {100 * shares[True]:.2f}% of affected pairs carry the '
                  'increment as the book move\'s time. These times are not '
                  'measurements.')
        else:
            print(f'\nNOT CONFIRMED: only {100 * shares.get(True, 0):.2f}% of affected '
                  'pairs carry the increment.\nSend this output before using the '
                  'corrected results below.')

        # ── 2. How many pairs ─────────────────────────────────────────────
        header('2. HOW MANY PAIRS ARE AFFECTED (all boundary pairs)')
        print(f'{"band":<8}{"pairs":>12}{"affected":>11}{"share":>8}'
              f'{"in x+0 games":>14}{"book near-instant from affected":>34}')
        for bnd in BANDS:
            s = chk[chk['band'] == bnd]
            if s.empty:
                continue
            N = s['n'].sum()
            A = s.loc[s['first'], 'n'].sum()
            A0 = s.loc[s['first'] & s['no_inc'], 'n'].sum()
            X1, X1a = s['x1'].sum(), s.loc[s['first'], 'x1'].sum()
            print(f'{bnd:<8}{N:>12,}{A:>11,}{100 * A / N:>7.1f}%{A0:>14,}'
                  f'{f"{X1a:,} of {X1:,} ({100 * X1a / max(X1, 1):.1f}%)":>34}')

        # ── 3. H3-revised ─────────────────────────────────────────────────
        header('3. H3-REVISED: registered versus corrected')
        h3_all = h3.groupby(['band', 'player', 'cap'], sort=False)[
            ['n', 'x1', 'x2']].sum().reset_index()
        h3_cor = h3[~h3['first']].groupby(['band', 'player', 'cap'], sort=False)[
            ['n', 'x1', 'x2']].sum().reset_index()
        met_h3_reg = s10.h3_block(h3_all, False, 'Registered (should match Script 10):')
        met_h3_cor = s10.h3_block(h3_cor, False,
                                  'Corrected (pairs whose book move was the first '
                                  'move removed):')
        r_reg = h3_ratios(h3_all[~h3_all['cap']])
        r_cor = h3_ratios(h3_cor[~h3_cor['cap']])

        # ── Pass over filtered_pairs.csv ──────────────────────────────────
        header('READING filtered_pairs.csv (players, H2, 3-pair rule)')
        agg, n_f = pass_filtered(proc / 'filtered_pairs.csv')
        n_first_f = int(agg.loc[agg['first'], 'n'].sum())
        print(f'filtered pairs: {n_f:,}; affected: {n_first_f:,} '
              f'({100 * n_first_f / n_f:.2f}%). Affected pairs from x+0 games are not '
              'here,\nbecause the premove filter already removed them.')
        P_reg, ties_reg = s10.build_players(agg, include_cap=False)
        P_cor, ties_cor = s10.build_players(agg[~agg['first']], include_cap=False)
        q_reg = set(P_reg.index[P_reg['n'] >= MIN_PAIRS])
        q_cor = set(P_cor.index[P_cor['n'] >= MIN_PAIRS])
        print(f'players with >= {MIN_PAIRS} pairs: registered {len(q_reg):,}; '
              f'corrected {len(q_cor):,}')

        # ── 4. H1 ─────────────────────────────────────────────────────────
        header('4. H1: registered versus corrected')
        cap_s = S['ply_at_exit'] > MAX_EXIT_PLY
        first_s = S['ply_at_exit'] <= FIRST_MOVE_MAX_EXIT
        R = S[~cap_s & S['player_username'].isin(q_reg)].copy()
        Rc = S[~cap_s & ~first_s & S['player_username'].isin(q_cor)].copy()
        aff = R[R['ply_at_exit'] <= FIRST_MOVE_MAX_EXIT]
        print(f'registered sample {len(R):,} pairs; affected {len(aff):,} '
              f'({100 * len(aff) / max(len(R), 1):.2f}%)')
        if len(aff):
            inc_a = increments(aff['time_control']) if 'time_control' in aff else None
            if inc_a is not None:
                eq = np.abs(aff['last_book_thinking_time'].to_numpy(float) - inc_a) < 1e-6
                print(f'  affected sample pairs whose book time equals the increment: '
                      f'{100 * eq.mean():.2f}%')
            print('  affected pairs by band: ' + ', '.join(
                f'{b} {int((aff["rating_band"] == b).sum()):,}' for b in BANDS))
        print(f'corrected sample {len(Rc):,} pairs, {Rc["player_username"].nunique():,} '
              f'players (affected pairs removed and the {MIN_PAIRS}-pair rule '
              're-applied)')
        print('Band counts, corrected: ' + ', '.join(
            f'{b} {int((Rc["rating_band"] == b).sum()):,}' for b in BANDS))

        L_reg, L_cor = s10.log_long(R), s10.log_long(Rc)
        m_reg = s10.fit_terms(L_reg, ['is_oob', 'voc'])
        b_reg, d_reg, met_h1_reg = s10.h1_rule(m_reg, '\nH1 registered (should match '
                                                      'Script 10)')
        m_cor = s10.fit_terms(L_cor, ['is_oob', 'voc'])
        s10.print_fit(m_cor, 'Corrected sample: log time ~ is_oob + voc + (1 | player)')
        b_cor, d_cor, met_h1_cor = s10.h1_rule(m_cor, 'H1 corrected')
        vb, _ = s10.coef(m_cor, 'voc')
        gap = (L_cor.loc[L_cor.is_oob == 1, 'voc'].mean()
               - L_cor.loc[L_cor.is_oob == 0, 'voc'].mean())
        print(f'VOC: {vb:.6f} per centipawn; out-of-book positions average {gap:.2f} '
              f'cp more VOC,\nwhich accounts for {100 * vb * gap / b_cor:.2f}% of the '
              'is_oob effect.')
        m0 = s10.fit_terms(L_cor, ['is_oob'])
        b0, _, _ = s10.h1_rule(m0, 'H1 corrected, without VOC')
        print(f'Adding VOC moves beta from {b0:.4f} to {b_cor:.4f} '
              f'({100 * (b_cor - b0) / b0:+.2f}%).')

        band_model(L_cor, 'Band model (supplementary), corrected sample:')

        print('\nRobustness checks on the corrected sample')
        keep3 = ((Rc['last_book_thinking_time'] > 1.0)
                 & (Rc['first_oob_thinking_time'] > 1.0)).to_numpy()
        print(f'  check 3: pairs kept {keep3.sum():,} of {len(Rc):,} ({keep3.mean():.1%})')
        _, _, met3 = s10.h1_rule(s10.fit_terms(s10.log_long(Rc[keep3]), ['is_oob', 'voc']),
                                 '  check 3')
        met4 = True
        for label, f in (('raw seconds', lambda v: v), ('square root', np.sqrt)):
            L4 = s10.long_format(Rc, f(Rc['last_book_thinking_time'].to_numpy(float)),
                                 f(Rc['first_oob_thinking_time'].to_numpy(float)))
            m4 = s10.fit_terms(L4, ['is_oob', 'voc'])
            b4, se4 = s10.coef(m4, 'is_oob')
            d4 = b4 / np.sqrt(m4['s2e'])
            met4 &= bool(b4 > 0 and d4 >= s10.D_THRESHOLD)
            print(f'  check 4, {label:<12} beta {b4:+.4f} (SE {se4:.4f}), d {d4:.4f}')

        # ── 5. H2 ─────────────────────────────────────────────────────────
        header('5. H2: registered versus corrected')
        Q_reg, i_r, s_r, n_r = h2_players(P_reg)
        Q_cor, i_c, s_c, n_c = h2_players(P_cor)
        print(f'Registered residualization over {n_r:,} pairs: '
              f'diff = {i_r:.5f} + {s_r:.7f} x VOC diff')
        print(f'Corrected residualization over {n_c:,} pairs: '
              f'diff = {i_c:.5f} + {s_c:.7f} x VOC diff')
        print(f'tied players assigned to the lower band: registered {ties_reg:,}, '
              f'corrected {ties_cor:,}')
        met_h2_reg = s10.h2_block(Q_reg, 'resid', 'Registered (should match Script 10)',
                                  with_ci=False)
        met_h2_cor = s10.h2_block(Q_cor, 'resid', 'Corrected: residualized player means')
        s10.h2_block(Q_cor, 'mean_log_time_delta',
                     'Corrected: unresidualized means, same players', with_ci=False)
        e_reg, md_reg = kw_summary(Q_reg, 'resid')
        e_cor, md_cor = kw_summary(Q_cor, 'resid')

        # ── 6. Roles ──────────────────────────────────────────────────────
        header('6. EXPLORATORY (not registered): deviator versus responder, corrected')
        gs = roles.drop_duplicates(['game_id', 'player_color']).groupby('game_id')[
            'ply_at_exit'].agg(['min', 'max', 'size'])
        ok = gs[(gs['size'] == 2) & (gs['max'] - gs['min'] == 1)]
        Rr = Rc.merge(ok[['min']], left_on='game_id', right_index=True, how='inner')
        Rr['responder'] = (Rr['ply_at_exit'] > Rr['min']).astype(float)
        print(f'pairs with a known role: {len(Rr):,} of {len(Rc):,} (deviators '
              f'{int((Rr["responder"] == 0).sum()):,}, responders '
              f'{int((Rr["responder"] == 1).sum()):,})')
        if len(Rr) > 1000 and Rr['responder'].nunique() == 2:
            mB = s10.fit_terms(s10.log_long(Rr), ['is_oob', 'voc', 'responder',
                                                  'is_oob:responder'])
            s10.print_fit(mB, 'H1 with a role interaction, corrected sample',
                          show={'is_oob', 'voc', 'responder', 'is_oob:responder'})
            i0 = mB['names'].index('is_oob')
            k = mB['names'].index('is_oob:responder')
            sd_e = np.sqrt(mB['s2e'])
            bd, sd_d = mB['b'][i0], mB['se'][i0]
            br = bd + mB['b'][k]
            se_r = np.sqrt(mB['cov'][i0, i0] + mB['cov'][k, k] + 2 * mB['cov'][i0, k])
            print(f'deviators:  is_oob {bd:.4f} ({np.exp(bd):.3f}x), d {bd / sd_e:.4f}, '
                  f'95% CI [{bd - 1.96 * sd_d:.4f}, {bd + 1.96 * sd_d:.4f}]')
            print(f'responders: is_oob {br:.4f} ({np.exp(br):.3f}x), d {br / sd_e:.4f}, '
                  f'95% CI [{br - 1.96 * se_r:.4f}, {br + 1.96 * se_r:.4f}]')

        # ── Summary ───────────────────────────────────────────────────────
        header('SUMMARY: registered versus corrected')
        yn = {True: 'MET', False: 'NOT MET'}
        print(f'{"":<30}{"registered":>26}{"corrected":>26}')
        print(f'{"H1 beta (d)":<30}{f"{b_reg:.4f} ({d_reg:.3f})":>26}'
              f'{f"{b_cor:.4f} ({d_cor:.3f})":>26}')
        print(f'{"H1 rule":<30}{yn[met_h1_reg]:>26}{yn[met_h1_cor]:>26}')
        print(f'{"H1 pairs":<30}{len(R):>26,}{len(Rc):>26,}')
        print(f'{"H2 epsilon-squared":<30}{e_reg:>26.4f}{e_cor:>26.4f}')
        for j, bnd in enumerate(BANDS):
            print(f'{"H2 median, " + bnd:<30}{md_reg[j]:>26.4f}{md_cor[j]:>26.4f}')
        print(f'{"H2 rule":<30}{yn[met_h2_reg]:>26}{yn[met_h2_cor]:>26}')
        for bnd in BANDS:
            a, c = r_reg.get(bnd), r_cor.get(bnd)
            if a is None or c is None:
                continue
            print(f'{"H3 ratio, " + bnd:<30}'
                  f'{f"{a[2]:.2f} [{a[3]:.2f}, {a[4]:.2f}]":>26}'
                  f'{f"{c[2]:.2f} [{c[3]:.2f}, {c[4]:.2f}]":>26}')
            print(f'{"   book / OOB rate, " + bnd:<30}'
                  f'{f"{100 * a[0]:.2f}% / {100 * a[1]:.2f}%":>26}'
                  f'{f"{100 * c[0]:.2f}% / {100 * c[1]:.2f}%":>26}')
        print(f'{"H3 rule":<30}{yn[met_h3_reg]:>26}{yn[met_h3_cor]:>26}')
        print(f'{"Check 3 (corrected)":<30}{"":>26}{yn[met3]:>26}')
        print(f'{"Check 4 (corrected)":<30}{"":>26}{yn[met4]:>26}')
        print(f'\nFirst-move check: {"CONFIRMED" if confirmed else "NOT CONFIRMED"}')
        print('\nSaved: outputs/tables/first_move_check.txt')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == '__main__':
    main()
