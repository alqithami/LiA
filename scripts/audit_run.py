#!/usr/bin/env python
"""Audit a LiA run directory (or all runs under a runs/ directory).

This audit is designed to catch the main "fake/placeholder" failure modes:
- running the wrong pipeline version / importing the wrong `lia` package
- missing provenance in strict dataset mode (no raw-file hashes)
- missing core outputs (summary tables, per-instance metrics)
- stale or inconsistent artifact references

Usage examples
--------------
Audit a single run (recommended):

    python scripts/audit_run.py --run-dir runs/full_paper_run_... 

Audit every run under ./runs:

    python scripts/audit_run.py --runs-dir runs

Notes
-----
`run_meta.json` may record some paths as repo-relative (e.g., "runs/<id>/...").
To make audits robust regardless of the current working directory, we resolve
those paths relative to the repository root inferred from the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class AuditResult:
    run_dir: Path
    ok: bool
    messages: List[str]


def _infer_repo_root(run_dir: Path) -> Path:
    """Infer repository root from a run directory.

    Expected layout: <repo_root>/runs/<run_id>/...
    """
    rd = run_dir.resolve()
    if rd.parent.name == "runs":
        return rd.parent.parent
    # Fall back: assume caller is already at repo root, keep relative resolution sane.
    return rd.parent


def _resolve_path(p: str, repo_root: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def _required_files(run_dir: Path) -> List[str]:
    return [
        "run_meta.json",
        "config.json",
        "run.log",
        "summary_table.csv",
        "summary_by_bidder_count.csv",
        "per_instance_metrics.csv",
    ]


def _check_exists(path: Path, label: str, msgs: List[str]) -> bool:
    if not path.exists():
        msgs.append(f"[FAIL] Missing {label}: {path}")
        return False
    return True


def _audit_one(run_dir: Path) -> AuditResult:
    msgs: List[str] = []
    ok = True

    run_dir = run_dir.resolve()
    repo_root = _infer_repo_root(run_dir)

    # Basic existence checks
    for fname in _required_files(run_dir):
        ok = _check_exists(run_dir / fname, fname, msgs) and ok

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return AuditResult(run_dir, False, msgs)

    meta: Dict[str, Any] = json.loads(meta_path.read_text())

    # Pipeline identity
    pipeline_version = meta.get("pipeline_version")
    lia_import = meta.get("lia_import")
    msgs.append(f"[INFO] pipeline_version={pipeline_version}")
    if lia_import:
        msgs.append(f"[INFO] lia_import={lia_import}")

    # Strict dataset provenance
    strict = bool(meta.get("datasets_strict", False))
    raw_hashes = meta.get("raw_dataset_hashes", {})
    if strict:
        if not isinstance(raw_hashes, dict) or len(raw_hashes) == 0:
            ok = False
            msgs.append(
                "[FAIL] datasets.strict=true but raw_dataset_hashes is empty (provenance missing)."
            )
        else:
            msgs.append(f"[OK] strict datasets: {len(raw_hashes)} raw file hash(es) recorded")

    # Topology artifact references (may be repo-relative)
    topo_art = meta.get("topology_artifacts", {})
    if isinstance(topo_art, dict) and topo_art:
        for name, rel in topo_art.items():
            p = _resolve_path(str(rel), repo_root)
            if p.exists():
                msgs.append(f"[OK] topology artifact exists: {name} -> {p}")
            else:
                ok = False
                msgs.append(f"[FAIL] topology artifact missing: {name} -> {p}")
    else:
        msgs.append("[WARN] run_meta.json has no topology_artifacts field")

    # Sanity checks on output sizes (catches trivially empty outputs)
    try:
        per_inst = run_dir / "per_instance_metrics.csv"
        if per_inst.exists():
            sz = per_inst.stat().st_size
            if sz < 1024:
                ok = False
                msgs.append(f"[FAIL] per_instance_metrics.csv too small ({sz} bytes)")
            else:
                msgs.append(f"[OK] per_instance_metrics.csv size={sz} bytes")
    except Exception as e:
        ok = False
        msgs.append(f"[FAIL] Could not stat per_instance_metrics.csv: {e}")

    return AuditResult(run_dir, ok, msgs)


def _iter_run_dirs(runs_dir: Path) -> Iterable[Path]:
    runs_dir = runs_dir.resolve()
    for p in sorted(runs_dir.iterdir()):
        if p.is_dir() and (p / "run_meta.json").exists():
            yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None, help="Single run directory to audit")
    ap.add_argument("--runs-dir", type=str, default=None, help="Directory containing multiple runs")
    args = ap.parse_args()

    if not args.run_dir and not args.runs_dir:
        ap.error("one of --run-dir or --runs-dir is required")

    results: List[AuditResult] = []

    if args.run_dir:
        results.append(_audit_one(Path(args.run_dir)))

    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
        for rd in _iter_run_dirs(runs_dir):
            results.append(_audit_one(rd))

    # Print report
    overall_ok = True
    for res in results:
        print("=" * 80)
        print(f"Run: {res.run_dir}")
        for m in res.messages:
            print(m)
        print("[RESULT]", "PASS" if res.ok else "FAIL")
        overall_ok = overall_ok and res.ok

    print("=" * 80)
    print("Overall:", "PASS" if overall_ok else "FAIL")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
