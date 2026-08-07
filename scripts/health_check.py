#!/usr/bin/env python3
"""
Pipeline health check
==============================================================================
Read-only. Verifies that every fix made to this pipeline over the last three
days is actually present in the code, on disk, and in git history -- rather
than inferred from run output alone.

Checks four things:
  1. CODE   - does each script contain the fix it's supposed to contain?
  2. DATA   - do the expected files exist, and is their modification order
              consistent with the pipeline (e.g. filtered_pairs.csv should
              be newer than the corrected stockfish_covariate.csv)?
  3. GIT    - do the commits that should exist, exist?
  4. IGNORE - does .gitignore still block outputs/tables or outputs/figures?

This does not re-run any analysis and does not modify any file.

USAGE
    python scripts/health_check.py
    (run from the project root, or anywhere inside the repo)
==============================================================================
"""

import subprocess
import sys
from pathlib import Path

PASS, WARN, FAIL, INFO = 'PASS', 'WARN', 'FAIL', 'INFO'
results = []


def check(label, status, detail=''):
    results.append((label, status, detail))
    mark = {'PASS': ' OK ', 'WARN': 'WARN', 'FAIL': 'FAIL', 'INFO': 'INFO'}[status]
    print(f'  [{mark}] {label}')
    if detail:
        for line in detail.splitlines():
            print(f'         {line}')


def find_root() -> Path:
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / '.git').exists() and (p / 'scripts').exists():
            return p
    return here


def read(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''


def git(root, *args):
    try:
        out = subprocess.run(['git', '-C', str(root), *args],
                             capture_output=True, text=True, timeout=30)
        return out.stdout
    except Exception as e:
        return f'__GIT_ERROR__ {e}'


def section(title):
    print(f'\n{"=" * 74}\n{title}\n{"=" * 74}')


def count_lines_fast(path: Path) -> int:
    """Line count without loading the file into memory. For large CSVs."""
    n = 0
    with open(path, 'rb') as f:
        for _ in f:
            n += 1
    return n


def main():
    root = find_root()
    print(f'Project root: {root}\n')
    if not (root / '.git').exists():
        print('WARNING: no .git directory found here or in any parent. '
              'Git checks will be skipped.')

    data = root / 'data' / 'processed'
    scripts = root / 'scripts'
    outputs = root / 'outputs'

    # ── 1. CODE ──────────────────────────────────────────────────────────────
    section('1. CODE — is each fix actually present in the script?')

    cfg = read(root / 'config.py')
    if 'STOCKFISH_THREADS' in cfg:
        line = next((l for l in cfg.splitlines() if 'STOCKFISH_THREADS' in l
                     and '=' in l), '')
        if '= 1' in line.replace(' ', '') or '=1' in line.replace(' ', ''):
            check('config.py: STOCKFISH_THREADS = 1', PASS)
        else:
            check('config.py: STOCKFISH_THREADS = 1', FAIL,
                 f'found: {line.strip()}\nMulti-threaded Stockfish is non-deterministic.')
    else:
        check('config.py: STOCKFISH_THREADS present', WARN, 'setting not found at all')

    if 'pre-registered' in cfg.lower():
        bad = [l for l in cfg.splitlines() if 'pre-registered' in l.lower()]
        check('config.py: no false pre-registration claim', FAIL,
             'still contains:\n' + '\n'.join(bad))
    else:
        check('config.py: no false pre-registration claim', PASS)

    s03 = read(scripts / '03_stockfish_covariate.py')
    game_fen_count = s03.count('game=board.fen()') + s03.count("game=fen")
    if game_fen_count >= 2:
        check('Script 03: game= passed to force ucinewgame', PASS,
             f'found {game_fen_count} occurrence(s)')
    elif game_fen_count == 1:
        check('Script 03: game= passed to force ucinewgame', WARN,
             'found only 1 occurrence, expected 2 (one per analyse() call)')
    else:
        check('Script 03: game= passed to force ucinewgame', FAIL,
             'not found. Transposition table will not be cleared between '
             'positions -- this is Defect 3, unfixed.')

    if 'log.debug' in s03 and 'Stockfish error' in s03:
        check('Script 03: engine errors are visible (log.warning)', WARN,
             'log.debug still used for the Stockfish error path; failures '
             'are silent at default log level')
    elif 'log.warning' in s03:
        check('Script 03: engine errors are visible (log.warning)', PASS)
    else:
        check('Script 03: engine errors are visible (log.warning)', INFO,
             'could not locate the error-handling line to check')

    s04 = read(scripts / '04_assemble_dataset.py')

    has_symmetric_mask = ('last_book_is_premove' in s04
                          and 'first_oob_is_premove' in s04
                          and 'premove_mask' in s04)
    applies_filter = ('chunk[~premove_mask]' in s04.replace(' ', '')
                      or 'chunk = chunk[~premove_mask]' in s04)
    if has_symmetric_mask and applies_filter:
        check('Script 04: symmetric premove filter, and it is applied', PASS)
    elif has_symmetric_mask and not applies_filter:
        check('Script 04: symmetric premove filter, and it is applied', FAIL,
             'the mask references both is_premove columns, but no line '
             'reassigns chunk to the filtered rows. This is the exact bug '
             'found on 2026-08-05: the mask is computed and counted but '
             'never applied.')
    else:
        check('Script 04: symmetric premove filter, and it is applied', FAIL,
             'mask does not reference both last_book_is_premove and '
             'first_oob_is_premove -- filter is still one-sided (Defect 2).')

    log_offset_line = next((l for l in s04.splitlines()
                            if l.strip().startswith('LOG_OFFSET')), '')
    if '0.0' in log_offset_line or log_offset_line.strip().endswith('= 0'):
        check('Script 04: LOG_OFFSET = 0.0', PASS)
    elif '0.001' in log_offset_line:
        check('Script 04: LOG_OFFSET = 0.0', FAIL,
             f'found: {log_offset_line.strip()}\n'
             'With any retained zero-second time this produces -inf, not '
             'a merely-wrong number.')
    else:
        check('Script 04: LOG_OFFSET = 0.0', WARN,
             f'could not parse the line: {log_offset_line!r}')

    if 'Pass 1/3' in s04 and 'MIN_TRANSITIONS_PER_PLAYER' in s04:
        # confirm it's used in the sampling function, not only aggregation
        idx = s04.find('def create_stratified_sample')
        seg = s04[idx:idx + 3000] if idx != -1 else ''
        if 'MIN_TRANSITIONS_PER_PLAYER' in seg or 'qualified' in seg:
            check('Script 04: MIN_TRANSITIONS_PER_PLAYER reaches the sample', PASS)
        else:
            check('Script 04: MIN_TRANSITIONS_PER_PLAYER reaches the sample', WARN,
                 'Pass 1/3 found, but could not confirm the player-count '
                 'filter is used inside create_stratified_sample specifically')
    else:
        check('Script 04: MIN_TRANSITIONS_PER_PLAYER reaches the sample', FAIL,
             'systematic sampling function not found -- may still be the '
             'original reservoir block that silently did nothing')

    s05 = read(scripts / '05_statistical_analysis.py')
    if 'H1 CONFIRMED' in s05 or "'CONFIRMED'" in s05:
        check('Script 05: interpretation text', INFO,
             'still prints a hardcoded CONFIRMED verdict from p-values, not '
             'from the registered d / epsilon-squared thresholds. Known, '
             'not fixed -- 05b applies the correct rule separately and its '
             'output supersedes this file\'s printed verdict. Consider a '
             'one-line note at the top of statistical_results.txt pointing '
             'to mixed_model_refit.txt so this is never read in isolation.')

    # ── 2. DATA ──────────────────────────────────────────────────────────────
    section('2. DATA — do the expected files exist, in a sane order?')

    expect = [
        ('boundary_pairs.csv', True),
        ('stockfish_covariate_PRE_FIX.csv', False),
        ('stockfish_covariate.csv', True),
        ('filtered_pairs.csv', True),
        ('analysis_sample.csv', True),
        ('player_aggregated.csv', True),
    ]
    mtimes = {}
    for name, required in expect:
        p = data / name
        if p.exists():
            mtimes[name] = p.stat().st_mtime
            size_mb = p.stat().st_size / 1e6
            check(f'data/processed/{name} exists', PASS, f'{size_mb:,.1f} MB')
        elif required:
            check(f'data/processed/{name} exists', FAIL, 'missing')
        else:
            check(f'data/processed/{name} exists', WARN,
                 'missing -- if this was deleted, robustness check 2 in the '
                 'pre-registration can no longer be independently re-verified '
                 '(the result you already computed is still valid and committed)')

    if 'stockfish_covariate.csv' in mtimes and 'stockfish_covariate_PRE_FIX.csv' in mtimes:
        if mtimes['stockfish_covariate.csv'] > mtimes['stockfish_covariate_PRE_FIX.csv']:
            check('Corrected VOC file is newer than the PRE_FIX file', PASS)
        else:
            check('Corrected VOC file is newer than the PRE_FIX file', WARN,
                 'unexpected timestamp order -- confirm which file is which')

    if 'filtered_pairs.csv' in mtimes and 'stockfish_covariate.csv' in mtimes:
        if mtimes['filtered_pairs.csv'] > mtimes['stockfish_covariate.csv']:
            check('filtered_pairs.csv postdates the corrected VOC table', PASS)
        else:
            check('filtered_pairs.csv postdates the corrected VOC table', WARN,
                 'filtered_pairs.csv looks older than the corrected VOC table -- '
                 'it may have been built from the contaminated version')

    for name in ('mixed_model_refit.txt',):
        p = outputs / 'tables' / name
        if p.exists():
            check(f'outputs/tables/{name} exists', PASS)
        else:
            check(f'outputs/tables/{name} exists', WARN,
                 'not found -- has the refit output been saved here yet?')

    for name in ('voc_robustness.txt', 'h3_diagnostic_output.txt'):
        p = outputs / 'diagnostics' / name
        if p.exists():
            check(f'outputs/diagnostics/{name} exists', PASS)
        else:
            check(f'outputs/diagnostics/{name} exists', INFO, 'not found')

    # ── 3. GIT ───────────────────────────────────────────────────────────────
    section('3. GIT — are the fix commits and result commits actually there?')

    if (root / '.git').exists():
        status = git(root, 'status', '--porcelain')
        if status.strip():
            check('Working tree clean (everything committed)', WARN,
                 'uncommitted changes present:\n' + status.strip())
        else:
            check('Working tree clean (everything committed)', PASS)

        log_text = git(root, 'log', '--oneline', '--all')
        expected_commits = [
            ('premove filter', 'Script 04 filter/sampling fix'),
            ('stockfish hash', 'Script 03 hash-contamination fix'),
            ('osf.io', 'OSF pre-registration commit'),
            ('robustness check', 'VOC robustness-check commit'),
            ('refit', 'Mixed-model refit commit'),
        ]
        low = log_text.lower()
        for needle, label in expected_commits:
            if needle in low:
                check(f'Commit found: {label}', PASS)
            else:
                check(f'Commit found: {label}', WARN,
                     f'no commit message contains "{needle}" -- '
                     'may not be committed yet, or worded differently')
    else:
        check('Git checks', INFO, 'no .git directory found; skipped')

    # ── 4. GITIGNORE ─────────────────────────────────────────────────────────
    section('4. GITIGNORE — are results still trackable?')

    gi = read(root / '.gitignore')
    bad_patterns = ['outputs/tables', 'outputs/tables/', 'outputs/figures',
                    'outputs/figures/', 'outputs/']
    gi_lines = [l.strip() for l in gi.splitlines()
               if l.strip() and not l.strip().startswith('#')]
    blocked = [l for l in gi_lines if l in bad_patterns]
    if blocked:
        check('outputs/tables and outputs/figures are trackable', FAIL,
             'still ignored by:\n' + '\n'.join(blocked))
    else:
        check('outputs/tables and outputs/figures are trackable', PASS)

    for essential in ('data/processed', 'data/raw'):
        hits = [l for l in gi_lines if essential in l]
        if hits:
            check(f'{essential}/ still ignored (should be, it is large)', PASS)
        else:
            check(f'{essential}/ still ignored (should be, it is large)', WARN,
                 f'no rule found for {essential} -- risk of trying to commit '
                 'a multi-GB file')

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    section('SUMMARY')
    counts = {PASS: 0, WARN: 0, FAIL: 0, INFO: 0}
    for _, status, _ in results:
        counts[status] += 1
    print(f'  {counts[PASS]} passed, {counts[WARN]} warnings, '
         f'{counts[FAIL]} failed, {counts[INFO]} informational\n')

    fails = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]
    if fails:
        print('  FAILED checks -- address these before trusting any result:')
        for label, _, detail in fails:
            print(f'    - {label}')
    if warns:
        print('  WARNINGS -- worth a look, not necessarily wrong:')
        for label, _, detail in warns:
            print(f'    - {label}')
    if not fails and not warns:
        print('  Nothing failed and nothing needs a second look.')


if __name__ == '__main__':
    main()
