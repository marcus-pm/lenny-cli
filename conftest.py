"""Root conftest — ensure src/ is on sys.path for editable installs.

Setuptools editable installs use a .pth file to add src/ to sys.path,
but .pth processing can fail when the project directory contains spaces.
This conftest ensures the import path is correct regardless.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
