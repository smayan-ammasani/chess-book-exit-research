#!/usr/bin/env python3
"""
Script 10: Registered analyses
==============================================================================
PURPOSE
    Runs every confirmatory analysis in the OSF registration
    (https://osf.io/n6tkx/) as written, including the parts the pipeline had
    not yet delivered. Earlier scripts are not changed; this one reads their
    outputs.

    Gaps in the earlier scripts, and how this script addresses them:

    1. H1 was never fitted as registered. Script 05b fitted a model with band
       terms and band x is_oob interactions on 1,500 resampled players per
       band, so its is_oob coefficient described band_3 only. This script
       fits the registered formula
           log_thinking_time ~ is_oob + voc + (1 | player)
       on the full analysis sample, with and without VOC, as registered.

    2. One registered inclusion criterion was never applied. The registration
       says "the player must actually exit the book within the first 40
       plies". Book sequences were counted only up to 40 plies, so a game
       still in book at ply 40 shows an "exit" at ply 41 or 42 by
       construction. Script 02 kept those pairs. Here they are removed, the
       3-pair minimum is re-applied, and every registered test is run on that
       sample. Results on the sample as originally built are printed beside
       them for comparison.

    3. Robustness checks 3, 4 and 5 had not been run. Check 1 (book
       thresholds 5 and 20) needs a re-parse and is Script 12. Check 2 was
       done by voc_robustness_check.py.

    4. H2 was run on unresidualized player means with Bonferroni correction.
       The registration specifies VOC-residualized values, Holm correction,
       and bootstrap confidence intervals for the band medians.

    5. H3-revised was registered with confidence intervals for each band's
       ratio. Script 06b did not compute them.

    6. The registration says positions where Stockfish fails are excluded and
       counted. Script 03 instead codes failures (and terminal positions) as
       VOC = 0 with both evaluations 0. This script counts positions carrying
       that code and, if there are any, re-fits H1 without them.

    Also included and labeled EXPLORATORY (not registered): H1 split by
    deviator (the player whose own move left theory) versus responder.

MODEL FITTING
    Every mixed model here is a linear model with a random intercept for
    player, fitted by restricted maximum likelihood (REML) using the exact
    profiled likelihood for a random-intercept model. That is the same
    likelihood statsmodels MixedLM maximizes, computed in vectorized form so
    it runs on about 1.8 million rows in seconds. Before anything else the
    script fits the same model with statsmodels MixedLM on a 2,000-player
    subsample and prints both sets of estimates, so the equivalence can be
    checked in the output.

USAGE
    python scripts/10_registered_analyses.py
    Runtime: roughly 15-40 minutes (one pass over filtered_pairs.csv and one
    over boundary_pairs.csv).
    Output: outputs/tables/registered_analyses.txt
==============================================================================
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

RANDOM_SEED = 42
D_THRESHOLD = 0.20        # registered, H1
EPS2_THRESHOLD = 0.01     # registered, H2
RATIO_THRESHOLD = 2.0     # registered, H3-revised
BANDS = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5']
REF_BAND = 'band_3'       # registered reference level
CHUNK = 1_000_000
N_BOOT = 2000

ROOT_GUESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_GUESS))
try:
    from config import MIN_TRANSITIONS_PER_PLAYER as MIN_PAIRS
except Exception:
    MIN_PAIRS = 3
try:
    from config import MAX_PLIES_FOR_BOOK as MAX_EXIT_PLY
except Exception:
    MAX_EXIT_PLY = 40


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


def resolve_root() -> Path:
    for base in (ROOT_GUESS, Path.cwd()):
        if (base / 'data' / 'processed' / 'analysis_sample.csv').exists():
            return base
    return ROOT_GUESS


def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])


def normalize_fen(fen) -> str:
    return ' '.join(str(fen).split()[:4])


def header(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# ==============================================================================
# Random-intercept REML
# ==============================================================================

def fit_ri_reml(y, X, groups, names):
    """
    Fit y = X b + u[group] + e by REML, with u ~ N(0, s2u), e ~ N(0, s2e).

    With lam = s2u / s2e, the GLS transform subtracts theta_g times each
    group's mean, theta_g = 1 - 1 / sqrt(1 + n_g * lam). OLS on the
    transformed data gives b(lam). The REML criterion, profiled over s2e, is
        (N - p) log(RSS / (N - p)) + sum_g log(1 + n_g lam) + log det(X*'X*)
    and is minimized over lam >= 0. Fixed-effect covariance is
    s2e * inverse(X*'X*), the same as statsmodels.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    codes = np.asarray(groups)
    N, p = X.shape
    n_g = np.bincount(codes).astype(float)
    gy = np.bincount(codes, weights=y) / n_g
    gX = np.column_stack([np.bincount(codes, weights=X[:, j]) / n_g
                          for j in range(p)])
    gyr, gXr = gy[codes], gX[codes]

    def pieces(lam):
        th = (1.0 - 1.0 / np.sqrt(1.0 + n_g * lam))[codes]
        ys = y - th * gyr
        Xs = X - th[:, None] * gXr
        XtX = Xs.T @ Xs
        b = np.linalg.solve(XtX, Xs.T @ ys)
        r = ys - Xs @ b
        return b, float(r @ r), XtX

    def crit(t):
        lam = np.exp(t)
        b, rss, XtX = pieces(lam)
        return ((N - p) * np.log(rss / (N - p))
                + np.sum(np.log1p(n_g * lam)) + np.linalg.slogdet(XtX)[1])

    res = optimize.minimize_scalar(crit, bounds=(np.log(1e-8), np.log(1e3)),
                                   method='bounded',
                                   options={'xatol': 1e-10, 'maxiter': 500})
    lam = float(np.exp(res.x))
    b, rss, XtX = pieces(lam)
    s2e = rss / (N - p)
    cov = s2e * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(cov))
    z = b / se
    # Restricted log-likelihood with all constants, on the same scale as
    # statsmodels' llf, so the two can be compared directly.
    llf = -0.5 * ((N - p) * np.log(2 * np.pi * s2e) + (N - p)
                  + np.sum(np.log1p(n_g * lam)) + np.linalg.slogdet(XtX)[1])
    return {
        'names': list(names), 'b': b, 'se': se, 'z': z,
        'p': 2 * stats.norm.sf(np.abs(z)),
        's2e': s2e, 's2u': lam * s2e, 'N': N, 'G': len(n_g),
        'at_boundary': lam < 1e-6, 'cov': cov, 'llf': float(llf),
    }


def coef(fit, name):
    i = fit['names'].index(name)
    return fit['b'][i], fit['se'][i]


def print_fit(fit, title, show=None):
    print(f'\n--- {title} ---')
    print(f'rows {fit["N"]:,}   players {fit["G"]:,}   residual var '
          f'{fit["s2e"]:.4f} (SD {np.sqrt(fit["s2e"]):.4f})   between-player var '
          f'{fit["s2u"]:.4f}')
    if fit['at_boundary']:
        print('WARNING: between-player variance estimated at zero.')
    print(f'{"term":<24}{"estimate":>12}{"SE":>10}{"z":>10}{"p (2-sided)":>13}')
    for j, nm in enumerate(fit['names']):
        if show is not None and nm not in show:
            continue
        print(f'{nm:<24}{fit["b"][j]:>12.5f}{fit["se"][j]:>10.5f}'
              f'{fit["z"][j]:>10.2f}{fit["p"][j]:>13.2e}')


def h1_rule(fit, label):
    b, se = coef(fit, 'is_oob')
    sd_e = np.sqrt(fit['s2e'])
    sd_t = np.sqrt(fit['s2e'] + fit['s2u'])
    d = b / sd_e
    met = bool(b > 0 and d >= D_THRESHOLD)
    print(f'{label}: beta = {b:+.4f} (95% CI {b - 1.96 * se:.4f} to '
          f'{b + 1.96 * se:.4f}; ratio {np.exp(b):.3f}x), d = {d:.4f} '
          f'[total-SD d {b / sd_t:.4f}], one-tailed p = '
          f'{stats.norm.sf(b / se):.2e}  -> rule {"MET" if met else "NOT MET"}')
    return b, d, met


def long_format(df, y_book, y_oob):
    n = len(df)
    out = pd.DataFrame({
        'y': np.concatenate([np.asarray(y_book, float), np.asarray(y_oob, float)]),
        'is_oob': np.r_[np.zeros(n), np.ones(n)],
        'band': np.tile(df['rating_band'].to_numpy(), 2),
        'player': np.tile(df['player_username'].astype(str).to_numpy(), 2),
        'voc': np.concatenate([df['last_book_voc'].to_numpy(float),
                               df['first_oob_voc'].to_numpy(float)]),
    })
    if 'responder' in df.columns:
        out['responder'] = np.tile(df['responder'].to_numpy(float), 2)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out['g'] = pd.factorize(out['player'])[0]
    return out


def design(L, terms):
    cols, names = [np.ones(len(L))], ['Intercept']
    for t in terms:
        if t == 'bands':
            for bnd in BANDS:
                if bnd != REF_BAND:
                    cols.append((L['band'] == bnd).to_numpy(float))
                    names.append(f'band[{bnd}]')
        elif t == 'is_oob:bands':
            for bnd in BANDS:
                if bnd != REF_BAND:
                    cols.append((L['band'] == bnd).to_numpy(float)
                                * L['is_oob'].to_numpy())
                    names.append(f'is_oob:band[{bnd}]')
        elif t == 'is_oob:responder':
            cols.append(L['is_oob'].to_numpy() * L['responder'].to_numpy())
            names.append(t)
        elif t == 'voc_c':
            v = L['voc'].to_numpy(float)
            cols.append(v - v.mean())
            names.append(t)
        else:
            cols.append(L[t].to_numpy(float))
            names.append(t)
    return np.column_stack(cols), names


def fit_terms(L, terms):
    X, names = design(L, terms)
    return fit_ri_reml(L['y'].to_numpy(), X, L['g'].to_numpy(), names)


def log_long(df):
    return long_format(df, df['log_last_book_time'], df['log_first_oob_time'])


# ==============================================================================
# Pass over filtered_pairs.csv: per player sums (for H2 and the 3-pair rule)
# ==============================================================================

SUM_COLS = ['n', 'sy', 'sx', 'sxx', 'sxy', 'slb', 'soob']


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
            'n': 1, 'sy': y, 'sx': x, 'sxx': x * x, 'sxy': x * y,
            'slb': ch['last_book_voc'].to_numpy(float),
            'soob': ch['first_oob_voc'].to_numpy(float),
        })
        parts.append(g.groupby(['player', 'band', 'cap'], sort=False).sum())
        if len(parts) >= 4:
            parts = [pd.concat(parts).groupby(level=[0, 1, 2], sort=False).sum()]
    agg = pd.concat(parts).groupby(level=[0, 1, 2], sort=False).sum().reset_index()
    return agg, n_rows


def build_players(agg, include_cap):
    a = agg if include_cap else agg[~agg['cap']]
    by_pb = a.groupby(['player', 'band'], sort=False)[SUM_COLS].sum().reset_index()
    tot = by_pb.groupby('player', sort=False)[SUM_COLS].sum()
    by_pb = by_pb.sort_values(['player', 'n', 'band'], ascending=[True, False, True])
    top = by_pb.groupby('player', sort=False)['n'].transform('max')
    n_ties = int((by_pb[by_pb['n'] == top].groupby('player').size() > 1).sum())
    tot['rating_band'] = by_pb.drop_duplicates('player').set_index('player')['band']
    return tot, n_ties


def ols_from_sums(t):
    N, Sx, Sy = t['n'].sum(), t['sx'].sum(), t['sy'].sum()
    slope = (t['sxy'].sum() - Sx * Sy / N) / (t['sxx'].sum() - Sx * Sx / N)
    return (Sy - slope * Sx) / N, slope, int(N)


# ==============================================================================
# Pass over boundary_pairs.csv: H3 sums, roles, and failure-code flags
# ==============================================================================

def pass_boundary(path, sample_games, zero_fens):
    cols = ['game_id', 'player_color', 'player_username', 'rating_band',
            'ply_at_exit', 'last_book_is_premove', 'first_oob_is_premove']
    if zero_fens:
        cols += ['last_book_fen', 'first_oob_fen']
    h3_parts, role_parts, n_rows = [], [], 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        ch = ch.dropna(subset=['player_username', 'rating_band'])
        n_rows += len(ch)
        x1 = to_bool(ch['last_book_is_premove']).to_numpy(int)
        x2 = to_bool(ch['first_oob_is_premove']).to_numpy(int)
        g = pd.DataFrame({
            'band': ch['rating_band'].to_numpy(),
            'player': ch['player_username'].astype(str).to_numpy(),
            'cap': (ch['ply_at_exit'] > MAX_EXIT_PLY).to_numpy(),
            'n': 1, 'x1': x1, 'x2': x2,
        })
        h3_parts.append(g.groupby(['band', 'player', 'cap'], sort=False).sum())
        if len(h3_parts) >= 4:
            h3_parts = [pd.concat(h3_parts).groupby(level=[0, 1, 2], sort=False).sum()]
        keep = ch['game_id'].isin(sample_games)
        if keep.any():
            r = ch.loc[keep, ['game_id', 'player_color', 'ply_at_exit']].copy()
            if zero_fens:
                lb = ch.loc[keep, 'last_book_fen'].map(normalize_fen)
                ob = ch.loc[keep, 'first_oob_fen'].map(normalize_fen)
                r['zero_coded'] = (lb.isin(zero_fens) | ob.isin(zero_fens)).to_numpy()
            role_parts.append(r)
    h3 = pd.concat(h3_parts).groupby(level=[0, 1, 2], sort=False).sum().reset_index()
    roles = (pd.concat(role_parts, ignore_index=True) if role_parts
             else pd.DataFrame(columns=['game_id', 'player_color', 'ply_at_exit']))
    return h3, roles, n_rows


def h3_block(h3, include_cap, label):
    d = h3 if include_cap else h3[~h3['cap']]
    print(f'\n{label}')
    print(f'{"band":<8}{"pairs":>12}{"book rate":>11}{"OOB rate":>10}{"ratio":>8}'
          f'{"95% CI (clustered by player)":>32}')
    ok_dir, ok_ratio = True, True
    for bnd in BANDS:
        s = d[d['band'] == bnd].groupby('player')[['n', 'x1', 'x2']].sum()
        if s.empty:
            continue
        N, X1, X2, G = s['n'].sum(), s['x1'].sum(), s['x2'].sum(), len(s)
        r1, r2 = X1 / N, X2 / N
        ok_dir &= bool(r1 > r2)
        if X2 == 0:
            print(f'{bnd:<8}{N:>12,}{100 * r1:>10.2f}%{100 * r2:>9.2f}%     inf')
            continue
        ratio = X1 / X2
        phi = s['x1'].to_numpy() / X1 - s['x2'].to_numpy() / X2
        se_log = np.sqrt(G / (G - 1) * np.sum(phi ** 2))
        lo, hi = ratio * np.exp(-1.96 * se_log), ratio * np.exp(1.96 * se_log)
        ok_ratio &= bool(ratio >= RATIO_THRESHOLD)
        print(f'{bnd:<8}{N:>12,}{100 * r1:>10.2f}%{100 * r2:>9.2f}%{ratio:>8.2f}'
              f'{f"[{lo:.2f}, {hi:.2f}]":>32}')
    met = ok_dir and ok_ratio
    print(f'registered rule (book > OOB in all bands and ratio >= '
          f'{RATIO_THRESHOLD} in every band): {"MET" if met else "NOT MET"}')
    return met


# ==============================================================================
# H2 helpers
# ==============================================================================

def boot_median_ci(x, rng):
    n = len(x)
    meds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        meds[i] = np.median(x[rng.integers(0, n, n)])
    return np.percentile(meds, [2.5, 97.5])


def h2_block(P, col, label, with_ci=True):
    groups = [P.loc[P['rating_band'] == bnd, col].dropna().to_numpy(float)
              for bnd in BANDS]
    H, p_kw = stats.kruskal(*groups)
    n_all, k = sum(len(g) for g in groups), len(groups)
    eps2 = (H - k + 1) / (n_all - k)
    meds = [float(np.median(g)) for g in groups]
    decreasing = all(a > b for a, b in zip(meds, meds[1:]))
    met = bool(decreasing and eps2 >= EPS2_THRESHOLD)
    print(f'\n{label}')
    print(f'  players {n_all:,};  H({k - 1}) = {H:.3f}, p = {p_kw:.2e}, '
          f'epsilon-squared = {eps2:.5f}')
    rng = np.random.default_rng(RANDOM_SEED)
    for bnd, g, m in zip(BANDS, groups, meds):
        ci_s = ''
        if with_ci:
            lo, hi = boot_median_ci(g, rng)
            ci_s = f'95% bootstrap CI [{lo:.4f}, {hi:.4f}]'
        print(f'  {bnd}: n = {len(g):>8,}   median {m:.4f}   {ci_s}')
    print(f'  medians decrease with rating (the registered direction): {decreasing}')
    print(f'  registered rule (decreasing AND epsilon-squared >= {EPS2_THRESHOLD}): '
          f'{"MET" if met else "NOT MET"}')
    if with_ci:
        try:
            import scikit_posthocs as sp
            dunn = sp.posthoc_dunn(P[P['rating_band'].isin(BANDS)], val_col=col,
                                   group_col='rating_band', p_adjust='holm')
            print("  Dunn's pairwise tests, Holm-adjusted p:")
            print('  ' + dunn.apply(lambda c: c.map(lambda v: f'{v:.2e}'))
                  .to_string().replace('\n', '\n  '))
        except ImportError:
            print('  (scikit-posthocs not installed; Dunn tests skipped)')
    return met


# ==============================================================================
# Main
# ==============================================================================

def main():
    root = resolve_root()
    proc = root / 'data' / 'processed'
    out_dir = root / 'outputs' / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(out_dir / 'registered_analyses.txt')
    sys.stdout = tee
    warnings.filterwarnings('ignore')
    verdicts = []
    try:
        print('Script 10: Registered analyses')
        print('Pre-registration: https://osf.io/n6tkx/')
        print(f'Seed {RANDOM_SEED}. Minimum pairs per player {MIN_PAIRS}. '
              f'Registered exit limit: ply {MAX_EXIT_PLY}.')

        # ── Load the analysis sample ──────────────────────────────────────
        S = pd.read_csv(proc / 'analysis_sample.csv', low_memory=False)
        S['player_username'] = S['player_username'].astype(str)
        n_dup = int(S.duplicated(['game_id', 'player_color']).sum())
        print(f'\nanalysis_sample.csv: {len(S):,} pairs, '
              f'{S["player_username"].nunique():,} players; '
              f'duplicate (game, colour) keys: {n_dup}')
        if n_dup:
            S = S.drop_duplicates(['game_id', 'player_color'])
            print(f'  duplicates dropped; {len(S):,} pairs remain')

        # ── Pass over filtered_pairs.csv ──────────────────────────────────
        header('BUILDING THE REGISTERED SAMPLE')
        print('Reading filtered_pairs.csv (one pass) ...')
        agg, n_filtered = pass_filtered(proc / 'filtered_pairs.csv')
        n_cap_f = int(agg.loc[agg['cap'], 'n'].sum())
        print(f'filtered pairs: {n_filtered:,}; exits after ply {MAX_EXIT_PLY}: '
              f'{n_cap_f:,} ({100 * n_cap_f / n_filtered:.3f}%)')

        P_run, ties_run = build_players(agg, include_cap=True)
        P_reg, ties_reg = build_players(agg, include_cap=False)
        q_run = set(P_run.index[P_run['n'] >= MIN_PAIRS])
        q_reg = set(P_reg.index[P_reg['n'] >= MIN_PAIRS])
        print(f'players with >= {MIN_PAIRS} pairs: as built {len(q_run):,}; '
              f'with late exits removed {len(q_reg):,}')

        cap_s = S['ply_at_exit'] > MAX_EXIT_PLY
        R = S[~cap_s & S['player_username'].isin(q_reg)].copy()
        print(f'analysis sample: {len(S):,} as built -> {len(R):,} registered '
              f'({int(cap_s.sum()):,} late exits removed, '
              f'{len(S) - len(R) - int(cap_s.sum()):,} pairs of players who fell '
              f'below {MIN_PAIRS} pairs)')
        print('Band counts in the registered sample:')
        for bnd in BANDS:
            print(f'  {bnd}: {(R["rating_band"] == bnd).sum():>9,}')
        ppp = R.groupby('player_username').size()
        print(f'pairs per player in the sample: mean {ppp.mean():.2f}, median '
              f'{ppp.median():.0f}; 1 pair {(ppp == 1).mean():.1%}, 2 pairs '
              f'{(ppp == 2).mean():.1%}, 3+ pairs {(ppp >= 3).mean():.1%}')
        print("(The 3-pair minimum counts all of a player's filtered pairs; the "
              'sample holds a\nsystematic subset of them, so many players have 1 '
              'or 2 pairs here.)')

        L = log_long(R)

        # ── Cross-check against statsmodels ───────────────────────────────
        header('CROSS-CHECK: this REML code versus statsmodels MixedLM')
        try:
            import statsmodels.formula.api as smf
            rng = np.random.default_rng(RANDOM_SEED)
            players = L['player'].unique()
            pick = set(rng.choice(players, size=min(2000, len(players)),
                                  replace=False))
            Sb = L[L['player'].isin(pick)].copy()
            Sb['g'] = pd.factorize(Sb['player'])[0]
            fast = fit_terms(Sb, ['is_oob', 'voc'])
            print(f'subsample: {Sb["player"].nunique():,} players, {len(Sb):,} rows')
            print(f'\n{"statsmodels method":<20}{"converged":>10}{"REML loglik":>14}'
                  f'{"is_oob":>11}{"player var":>12}')
            fits = {}
            for method in ('lbfgs', 'bfgs', 'powell', 'nm'):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        m = smf.mixedlm('y ~ is_oob + voc', Sb,
                                        groups=Sb['player']).fit(
                            reml=True, method=method, maxiter=5000, disp=False)
                    fits[method] = m
                    print(f'{method:<20}{str(bool(m.converged)):>10}{m.llf:>14.4f}'
                          f'{float(m.params["is_oob"]):>11.6f}'
                          f'{float(m.cov_re.iloc[0, 0]):>12.6f}')
                except Exception as e:
                    print(f'{method:<20}  failed: {type(e).__name__}')
            print(f'{"this script (REML)":<20}{"":>10}{fast["llf"]:>14.4f}'
                  f'{coef(fast, "is_oob")[0]:>11.6f}{fast["s2u"]:>12.6f}')
            valid = {k: m for k, m in fits.items()
                     if np.isfinite(m.llf) and m.cov_re.iloc[0, 0] > 1e-8}
            if valid:
                best_k = max(valid, key=lambda k: valid[k].llf)
                sm = valid[best_k]
                print(f'\nBest statsmodels fit ({best_k}) next to this script:')
                print(f'{"":22}{"this script":>14}{"statsmodels":>14}')
                for nm in ('is_oob', 'voc'):
                    b, se = coef(fast, nm)
                    print(f'{nm + " estimate":<22}{b:>14.6f}{float(sm.params[nm]):>14.6f}')
                    print(f'{nm + " SE":<22}{se:>14.6f}{float(sm.bse[nm]):>14.6f}')
                print(f'{"residual variance":<22}{fast["s2e"]:>14.6f}'
                      f'{float(sm.scale):>14.6f}')
                print(f'{"player variance":<22}{fast["s2u"]:>14.6f}'
                      f'{float(sm.cov_re.iloc[0, 0]):>14.6f}')
                print('Matching columns mean the fits below are the model '
                      'statsmodels fits at its best.')
            lb = fits.get('lbfgs')
            if lb is not None and (not np.isfinite(lb.llf)
                                   or lb.cov_re.iloc[0, 0] < 1e-8):
                print('\nNOTE: lbfgs, the first optimizer in the registered '
                      'fallback chain, stopped at an\ninvalid boundary point '
                      '(player variance 0, log-likelihood not finite) while\n'
                      'reporting convergence. The registered rule "take the first '
                      'that converges"\nwould accept it. This is the likely cause of '
                      'the singular fit in Script 05,\nand the reason the fits below '
                      'maximize the same likelihood directly. This is a\n'
                      'deviation from the registered fitting procedure.')
        except Exception as e:
            print(f'Cross-check skipped: {type(e).__name__}: {e}')

        # ── H1 ────────────────────────────────────────────────────────────
        header('H1 AS REGISTERED: log time ~ is_oob + voc + (1 | player)')
        m1 = fit_terms(L, ['is_oob', 'voc'])
        print_fit(m1, 'Registered sample')
        b1, d1, met1 = h1_rule(m1, 'H1 (registered)')
        verdicts.append(('H1', met1, f'beta {b1:.4f}, d {d1:.4f}'))
        vb, _ = coef(m1, 'voc')
        gap = L.loc[L.is_oob == 1, 'voc'].mean() - L.loc[L.is_oob == 0, 'voc'].mean()
        print(f'VOC: {vb:.6f} log units per centipawn. Out-of-book positions average '
              f'{gap:.2f} cp more VOC,\nwhich accounts for {100 * vb * gap / b1:.2f}% '
              f'of the is_oob effect.')

        m0 = fit_terms(L, ['is_oob'])
        print_fit(m0, 'Registered comparison: same model without VOC')
        b0, _, _ = h1_rule(m0, 'H1 without VOC')
        print(f'Adding VOC moves beta from {b0:.4f} to {b1:.4f} '
              f'({100 * (b1 - b0) / b0:+.2f}%).')

        if len(R) != len(S):
            print()
            h1_rule(fit_terms(log_long(S), ['is_oob', 'voc']),
                    'For comparison, the sample as built (late exits kept)')

        # ── Band model ────────────────────────────────────────────────────
        header('SUPPLEMENTARY: band model (same formula as 05b), registered sample')
        m2 = fit_terms(L, ['is_oob', 'voc', 'bands', 'is_oob:bands'])
        sd_e = np.sqrt(m2['s2e'])
        print(f'residual SD {sd_e:.4f}')
        print(f'{"band":<8}{"is_oob effect":>15}{"95% CI":>22}{"ratio":>8}{"d":>8}')
        i0 = m2['names'].index('is_oob')
        for bnd in BANDS:
            if bnd == REF_BAND:
                est, var = m2['b'][i0], m2['cov'][i0, i0]
            else:
                k = m2['names'].index(f'is_oob:band[{bnd}]')
                est = m2['b'][i0] + m2['b'][k]
                var = m2['cov'][i0, i0] + m2['cov'][k, k] + 2 * m2['cov'][i0, k]
            se = np.sqrt(var)
            print(f'{bnd:<8}{est:>15.4f}'
                  f'{f"[{est - 1.96 * se:.4f}, {est + 1.96 * se:.4f}]":>22}'
                  f'{np.exp(est):>8.3f}{est / sd_e:>8.3f}')

        # ── Robustness checks ─────────────────────────────────────────────
        header('REGISTERED ROBUSTNESS CHECKS 3, 4 AND 5 (registered sample)')
        print('\nCheck 3: premove-adjacent pairs removed. The registration does not '
              'define\n"premove-adjacent". Operationalized after registration as: '
              'either move recorded\nat 1 s. Clock values are whole seconds, so a '
              'move that took a fraction of a\nsecond can be recorded as 0 s or as '
              '1 s; 1 s is the value next to the premove\ncode. This '
              'definition was chosen after registration.')
        keep3 = ((R['last_book_thinking_time'] > 1.0)
                 & (R['first_oob_thinking_time'] > 1.0)).to_numpy()
        print(f'  pairs kept: {keep3.sum():,} of {len(R):,} ({keep3.mean():.1%})')
        _, _, met3 = h1_rule(fit_terms(log_long(R[keep3]), ['is_oob', 'voc']),
                             '  check 3')

        print('\nCheck 4: transform sensitivity (same model, other scales)')
        met4 = True
        for label, f in (('raw seconds', lambda v: v), ('square root', np.sqrt)):
            L4 = long_format(R, f(R['last_book_thinking_time'].to_numpy(float)),
                             f(R['first_oob_thinking_time'].to_numpy(float)))
            m4 = fit_terms(L4, ['is_oob', 'voc'])
            b4, se4 = coef(m4, 'is_oob')
            d4 = b4 / np.sqrt(m4['s2e'])
            met4 &= bool(b4 > 0 and d4 >= D_THRESHOLD)
            print(f'  {label:<14} beta {b4:+.4f} (SE {se4:.4f}), d {d4:.4f}')
        print('  (On these scales beta is in seconds or root-seconds, so only the '
              'sign and d\n  are comparable. Raw seconds have a very long right '
              'tail, which inflates the residual SD.)')

        print('\nCheck 5: VOC mean-centered')
        _, _, met5 = h1_rule(fit_terms(L, ['is_oob', 'voc_c']), '  H1, centered VOC')
        print('  Identical by construction: centering an additive covariate moves '
              'only the\n  intercept. For H2, residuals from a regression on VOC '
              'are also unchanged by\n  centering, so H2 with centered VOC equals '
              'the registered H2 below.')
        verdicts += [('Check 3', met3, 'H1 rule on reduced sample'),
                     ('Check 4', met4, 'H1 rule on raw and square-root scales'),
                     ('Check 5', met5, 'H1 rule with centered VOC')]

        # ── Stockfish failure code ────────────────────────────────────────
        header('STOCKFISH FAILURE CODE (registered: failures excluded and counted)')
        cov_path = proc / 'stockfish_covariate.csv'
        zero_fens = set()
        if cov_path.exists():
            C = pd.read_csv(cov_path)
            z = C[(C['eval_depth1'] == 0) & (C['eval_depth15'] == 0)]
            zero_fens = set(z['fen_normalized'])
            print(f'positions in stockfish_covariate.csv: {len(C):,}')
            print(f'positions with both evaluations exactly 0 (the code Script 03 '
                  f'writes for a\nfailure or a finished game): {len(zero_fens):,}')
            if zero_fens:
                print('Some of these may be real evaluations of 0. They are flagged '
                      'and tested below.')
        else:
            print('stockfish_covariate.csv not found; check skipped.')

        # ── Pass over boundary_pairs.csv ──────────────────────────────────
        header('READING boundary_pairs.csv (H3 intervals, roles, failure flags)')
        sample_games = set(R['game_id'].unique())
        h3, roles, n_bp = pass_boundary(proc / 'boundary_pairs.csv',
                                        sample_games, zero_fens)
        print(f'rows read: {n_bp:,}; exits after ply {MAX_EXIT_PLY} among them: '
              f'{int(h3.loc[h3["cap"], "n"].sum()):,}')

        if zero_fens:
            flags = roles[['game_id', 'player_color', 'zero_coded']].drop_duplicates(
                ['game_id', 'player_color'])
            Rz = R.merge(flags, on=['game_id', 'player_color'], how='left')
            zc = Rz['zero_coded'].fillna(False).astype(bool)
            print(f'\nsample pairs touching a zero-coded position: {int(zc.sum()):,}')
            if zc.any():
                h1_rule(fit_terms(log_long(Rz[~zc]), ['is_oob', 'voc']),
                        'H1 without them')

        # ── H3 ────────────────────────────────────────────────────────────
        header('H3-REVISED with the registered ratio confidence intervals')
        print('Rates use every boundary pair before premove filtering, as in '
              'Script 06b.\nIntervals are for the ratio of rates (delta method, '
              'players as clusters,\nsince each player contributes many pairs).')
        met_h3 = h3_block(h3, False,
                          f'Registered (exits after ply {MAX_EXIT_PLY} removed):')
        h3_block(h3, True, 'As run in Script 06b (all pairs):')
        verdicts.append(('H3-revised', met_h3, 'see table'))

        # ── H2 ────────────────────────────────────────────────────────────
        header('H2 AS REGISTERED: VOC-residualized, Kruskal-Wallis, Dunn + Holm')
        Q = P_reg[P_reg['n'] >= MIN_PAIRS].copy()
        icpt, slope, n_ols = ols_from_sums(Q)
        print(f'Residualization: OLS over the {n_ols:,} pairs that make up the '
              f'player means:\n  log-time difference = {icpt:.5f} + {slope:.7f} x '
              f'(VOC difference)')
        Q['mean_log_time_delta'] = Q['sy'] / Q['n']
        Q['resid'] = (Q['sy'] - icpt * Q['n'] - slope * Q['sx']) / Q['n']
        print(f'players whose pairs split evenly between bands (assigned to the '
              f'lower band): {ties_reg:,}')
        met_h2 = h2_block(Q, 'resid', 'Registered: residualized player means')
        verdicts.append(('H2', met_h2, 'see table'))
        print("\n(Adding the intercept back gives each band's cost at equal VOC. "
              'That shift\nchanges no test.)')
        h2_block(Q, 'mean_log_time_delta',
                 'For comparison: unresidualized means, same players', with_ci=False)

        pa = proc / 'player_aggregated.csv'
        if pa.exists():
            PA = pd.read_csv(pa)
            PA['player_username'] = PA['player_username'].astype(str)
            Qrun = P_run[P_run['n'] >= MIN_PAIRS]
            i2, s2, _ = ols_from_sums(Qrun)
            PA['resid'] = (PA['mean_log_time_delta'] - i2
                           - s2 * (PA['mean_first_oob_voc'] - PA['mean_last_book_voc']))
            chk = PA.set_index('player_username').join(
                (Qrun['sy'] / Qrun['n']).rename('recomputed'), how='inner')
            diff = np.abs(chk['mean_log_time_delta'] - chk['recomputed']).max()
            print(f'\nCheck against player_aggregated.csv: {len(PA):,} players in '
                  f'the file, {len(Qrun):,} recomputed,\n{len(chk):,} matched; '
                  f'largest difference in mean log delta {diff:.5f} (the file '
                  f'rounds to 4 decimals)')
            h2_block(PA, 'mean_log_time_delta',
                     'As run in Script 05 (unresidualized)', with_ci=False)
            h2_block(PA, 'resid', 'Script 05 players, residualized', with_ci=False)

        # ── Exploratory: deviator versus responder ────────────────────────
        header('EXPLORATORY (not registered): deviator versus responder')
        print('Book status belongs to the game, so in each game one player made '
              'the move\nthat left theory (the deviator) and the other answered '
              'it (the responder).\nThe deviator\'s exit ply is one less than the '
              'responder\'s.')
        gs = roles.drop_duplicates(['game_id', 'player_color']).groupby('game_id')[
            'ply_at_exit'].agg(['min', 'max', 'size'])
        ok = gs[(gs['size'] == 2) & (gs['max'] - gs['min'] == 1)]
        Rr = R.merge(ok[['min']], left_on='game_id', right_index=True, how='inner')
        Rr['responder'] = (Rr['ply_at_exit'] > Rr['min']).astype(float)
        print(f'pairs with a known role: {len(Rr):,} of {len(R):,} '
              f'(deviators {int((Rr["responder"] == 0).sum()):,}, responders '
              f'{int((Rr["responder"] == 1).sum()):,})')
        print('A role is unknown when the opponent has no usable pair (for example '
              'a missing\nclock time, or the opponent left book on their first '
              'move).')
        if len(Rr) > 1000 and Rr['responder'].nunique() == 2:
            mB = fit_terms(log_long(Rr), ['is_oob', 'voc', 'responder',
                                          'is_oob:responder'])
            print_fit(mB, 'H1 with a role interaction',
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
        header('SUMMARY OF REGISTERED DECISIONS (registered sample)')
        for name, met, note in verdicts:
            print(f'  {name:<12} {"MET" if met else "NOT MET":<9} {note}')
        print('\nNot covered here: robustness check 1 (Script 12) and check 2 '
              '(voc_robustness_check.py).')
        print('\nSaved: outputs/tables/registered_analyses.txt')
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == '__main__':
    main()
