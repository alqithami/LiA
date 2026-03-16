#!/usr/bin/env python
"""Generate LaTeX tables from a LiA run directory.

Rationale
---------
During revisions it is easy to accidentally copy the wrong column (e.g., confusing
decision/commit times with compute-time). This script emits LaTeX-ready tables
directly from the run artifacts to avoid manual transcription errors.

Inputs (produced by `run_pipeline.py`)
------------------------------------
- <run_dir>/summary_table.csv
- <run_dir>/lai_estimates.csv (optional)

Outputs
-------
Writes one table per (topology, epsilon) into:

    <run_dir>/tables/

The tables include (as available):
- Welfare ratio (SW/OPT)
- Revenue ratio (Rev/OPT)
- Clearing latency (ms)
- Effective welfare
- Compute time (ms)
- LAI sup_g (if lai_estimates.csv is present)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _latest_run_dir(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    denom = np.nansum(w)
    if denom <= 0:
        return float("nan")
    return float(np.nansum(v * w) / denom)


def _aggregate_lai(lai: pd.DataFrame) -> pd.DataFrame:
    if lai.empty:
        return pd.DataFrame(columns=["eps_ms", "topology", "mechanism", "sup_g"])

    rows = []
    for (eps_ms, topo, mech), g in lai.groupby(["eps_ms", "topology", "mechanism"], dropna=False):
        w = g.get("instance_count", pd.Series(1.0, index=g.index)).fillna(1.0)
        rows.append(
            {
                "eps_ms": float(eps_ms),
                "topology": topo,
                "mechanism": mech,
                "sup_g": _weighted_mean(g["sup_g"], w),
            }
        )
    return pd.DataFrame(rows)


def _fmt(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def _table_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = [
        "mechanism",
        "welfare_ratio_mean",
        "revenue_ratio_mean",
        "clearing_latency_ms_mean",
        "effective_welfare_mean",
        "compute_time_ms",
    ]
    if "sup_g" in df.columns:
        cols.append("sup_g")

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{6pt}")

    # Header
    headers = [
        "Mechanism",
        "SW/OPT",
        "Rev/OPT",
        "Latency (ms)",
        "Eff. welfare",
        "Runtime (ms)",
    ]
    if "sup_g" in df.columns:
        headers.append("LAI")

    out.append("\\begin{tabular}{l" + "c" * (len(headers) - 1) + "}")
    out.append("\\toprule")
    out.append(" & ".join(headers) + " \\\\")
    out.append("\\midrule")

    for _, r in df[cols].iterrows():
        row = [
            str(r["mechanism"]),
            _fmt(r["welfare_ratio_mean"], 3),
            _fmt(r["revenue_ratio_mean"], 3),
            _fmt(r["clearing_latency_ms_mean"], 2),
            _fmt(r["effective_welfare_mean"], 2),
            _fmt(r["compute_time_ms"], 3),
        ]
        if "sup_g" in df.columns:
            row.append(_fmt(r["sup_g"], 3))
        out.append(" & ".join(row) + " \\\\")

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate LaTeX tables from a LiA run directory")
    ap.add_argument("--run-dir", type=str, default="", help="Path to a specific run directory under runs/")
    ap.add_argument("--runs-dir", type=str, default="runs", help="Parent runs directory (used if --run-dir omitted)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None:
        run_dir = _latest_run_dir(Path(args.runs_dir))
    if run_dir is None or not run_dir.exists():
        raise SystemExit("Could not locate a run directory. Provide --run-dir or ensure runs/ exists.")

    summary_path = run_dir / "summary_table.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")
    summary = pd.read_csv(summary_path)

    lai_path = run_dir / "lai_estimates.csv"
    lai_agg: Optional[pd.DataFrame] = None
    if lai_path.exists():
        try:
            lai = pd.read_csv(lai_path)
            lai_agg = _aggregate_lai(lai)
        except Exception:
            lai_agg = None

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for (topo, eps_ms), g in summary.groupby(["topology", "eps_ms"], dropna=False):
        gg = g.copy()
        gg["compute_time_ms"] = gg["compute_time_s_mean"].astype(float) * 1000.0

        if lai_agg is not None and not lai_agg.empty:
            sub_lai = lai_agg[(lai_agg["topology"] == topo) & (lai_agg["eps_ms"] == float(eps_ms))]
            gg = gg.merge(sub_lai[["mechanism", "sup_g"]], on="mechanism", how="left")

        gg = gg.sort_values("mechanism")

        caption = f"Summary metrics for {topo} ($\\epsilon$={eps_ms:g}\\,ms)."
        label = f"tab:{topo.lower().replace('-', '')}_eps{str(eps_ms).replace('.', 'p')}"
        tex = _table_to_latex(gg, caption=caption, label=label)

        out_path = tables_dir / f"{topo}_eps{eps_ms:g}ms.tex"
        out_path.write_text(tex)

    print(f"LaTeX tables written to: {tables_dir}")


if __name__ == "__main__":
    main()
