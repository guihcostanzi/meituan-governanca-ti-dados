"""Pytest bootstrap: make the repository root importable.

``pytest`` is invoked from the repository root (``.venv/bin/python -m pytest
tests``), but it only guarantees the test directory itself on ``sys.path``.
Adding the repository root here lets every test module ``import meituan_eta``
without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
