#!/usr/bin/env python3
"""
Print the software versions the paper must report, including the Stockfish
build named in config.py. Saves them to outputs/tables/software_versions.txt.

USAGE
    python scripts/print_versions.py
"""

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PACKAGES = [('pandas', 'pandas'), ('numpy', 'NumPy'), ('scipy', 'SciPy'),
            ('statsmodels', 'statsmodels'), ('scikit_posthocs', 'scikit-posthocs'),
            ('chess', 'python-chess'), ('zstandard', 'zstandard'),
            ('tqdm', 'tqdm'), ('matplotlib', 'matplotlib')]

lines = [f'Python {platform.python_version()} on {platform.system()} {platform.release()}']
for module, label in PACKAGES:
    try:
        m = importlib.import_module(module)
        lines.append(f'{label} {getattr(m, "__version__", "(version not exposed)")}')
    except ImportError:
        lines.append(f'{label}: not installed')

try:
    import chess.engine
    from config import STOCKFISH_PATH
    path = Path(STOCKFISH_PATH)
    if not path.exists() and path.with_suffix('.exe').exists():
        path = path.with_suffix('.exe')
    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    lines.append(f'Stockfish: {engine.id.get("name", "unknown")} (binary: {path.name})')
    engine.quit()
except Exception as e:
    lines.append(f'Stockfish: could not be started ({type(e).__name__}: {e})')

out = ROOT / 'outputs' / 'tables' / 'software_versions.txt'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
print(f'\nSaved: {out}')
