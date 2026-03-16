#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

BANNED_PATTERNS = [
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bTBD\b", re.IGNORECASE),
            # Match explicit placeholder markers, but do not flag the generic word 'placeholder' in docs/comments.
    re.compile(r"__PLACEHOLDER__|<PLACEHOLDER>|\bPLACEHOLDER\b"),
    re.compile(r"fake\s+data", re.IGNORECASE),
    re.compile(r"dummy\s+data", re.IGNORECASE),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.root)
    offenders = []

    for path in root.rglob("*.py"):
        if path.name == "validate_no_placeholders.py":
            continue
        if ".venv" in path.parts:
            continue
        text = path.read_text(errors="ignore")
        for pat in BANNED_PATTERNS:
            if pat.search(text):
                offenders.append((str(path), pat.pattern))

    if offenders:
        print("Found banned placeholder markers:")
        for p, pat in offenders:
            print(f"  - {p}  (pattern: {pat})")
        raise SystemExit(2)

    # Ensure we are not shipping precomputed runs
    runs_dir = root / "runs"
    if runs_dir.exists():
        # Allow empty directory, but not populated runs
        nontrivial = [p for p in runs_dir.iterdir() if p.name not in {".gitkeep"}]
        if nontrivial:
            print("The runs/ directory is non-empty. Remove precomputed outputs before release.")
            for p in nontrivial[:20]:
                print(f"  - {p}")
            raise SystemExit(3)

    print("OK: no placeholder markers found and no precomputed outputs detected.")


if __name__ == "__main__":
    main()
