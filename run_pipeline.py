#!/usr/bin/env python
from __future__ import annotations

"""Entry-point for running the LiA experiment pipeline.

This repository uses a `src/` layout. To prevent accidentally importing an older *installed*
`lia` package (e.g., from a previous pipeline version), we explicitly prepend the local
`src/` directory to `sys.path`.

If you see results that don't match the current code, first check the banner printed by
this script (it shows the pipeline version and root path).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from lia import __version__ as lia_version  # noqa: E402
from lia.experiment.runner import main  # noqa: E402


if __name__ == "__main__":
    print(f"[LiA Pipeline] version={lia_version} root={ROOT}")
    main()
