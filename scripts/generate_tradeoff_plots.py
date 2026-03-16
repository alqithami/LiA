#!/usr/bin/env python
"""Generate paper-style trade-off plots from a completed run directory.

This script complements `scripts/generate_figures.py` by producing scatter/line plots
that directly visualize the welfare/latency/fairness trade-offs that motivated LIA.

Inputs (produced by `run_pipeline.py`)
------------------------------------
- <run_dir>/summary_table.csv
- <run_dir>/lai_estimates.csv   (optional but strongly recommended)

Outputs
-------
Writes PNG figures into:

    <run_dir>/figures/tradeoffs/

Notes
-----
- `summary_table.csv` aggregates across bidder_counts.
- `lai_estimates.csv` is produced per bidder_count; this script aggregates LAI
  across bidder_counts using `instance_count` as weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _latest_run_dir(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _pretty_mech_label(m: str) -> str:
    if m == "FastVCG":
        return "Fast-VCG"
    if m == "SyncVCG":
        return "Sync-VCG"
    if m == "HoldBack":
        return "HoldBack"
    if m.startswith("BatchVCG_B"):
        suffix = m.replace("BatchVCG_B", "")
        return f"Batch-VCG ({suffix})"
    if m.startswith("LIA_lambda"):
        try:
            rest = m[len("LIA_lambda"):]
            lam_part = rest.split("_", 1)[0]
            return f"LIA (λ={lam_part})"
        except Exception:
            return m
    return m


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    denom = np.nansum(w)
    if denom <= 0:
        return float("nan")
    return float(np.nansum(v * w) / denom)


def _aggregate_lai(lai: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LAI across bidder_counts to match summary_table granularity."""

    if lai.empty:
        return pd.DataFrame(columns=["eps_ms", "topology", "mechanism", "sup_g", "g1ms"])

    rows = []
    for (eps_ms, topo, mech), g in lai.groupby(["eps_ms", "topology", "mechanism"], dropna=False):
        w = g.get("instance_count", pd.Series(1.0, index=g.index)).fillna(1.0)
        rows.append(
            {
                "eps_ms": float(eps_ms),
                "topology": topo,
                "mechanism": mech,
                "sup_g": _weighted_mean(g["sup_g"], w),
                "g1ms": _weighted_mean(g["g1ms"], w) if "g1ms" in g.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _scatter_tradeoff(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, xlab: str, ylab: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return

    plt.figure(figsize=(7.2, 4.2))
    for mech, g in df.groupby("mechanism"):
        xx = g[x].to_numpy(dtype=float)
        yy = g[y].to_numpy(dtype=float)
        # Use a marker so multiple parameter settings of the same mechanism are visible.
        plt.scatter(xx, yy, label=_pretty_mech_label(mech), alpha=0.9)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _line_by_eps(df: pd.DataFrame, metric: str, title: str, out_path: Path, ylab: str) -> None:
    if df.empty or metric not in df.columns:
        return

    plt.figure(figsize=(7.2, 4.2))
    for mech, g in df.groupby("mechanism"):
        g = g.sort_values("eps_ms")
        plt.plot(g["eps_ms"], g[metric], marker="o", label=_pretty_mech_label(mech))

    plt.xlabel("Measurement error $\\epsilon$ (ms)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate trade-off plots from a LiA run directory")
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

    out_dir = run_dir / "figures" / "tradeoffs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-topology plots
    for (topo, eps_ms), g in summary.groupby(["topology", "eps_ms"], dropna=False):
        gg = g.copy()

        # Merge in LAI if available
        if lai_agg is not None and not lai_agg.empty:
            sub_lai = lai_agg[(lai_agg["topology"] == topo) & (lai_agg["eps_ms"] == float(eps_ms))]
            gg = gg.merge(sub_lai[["mechanism", "sup_g", "g1ms"]], on="mechanism", how="left")

        tag = f"{topo}_eps{eps_ms:g}ms"

        # Key trade-offs: fairness (LAI) vs latency and vs effective welfare.
        if "sup_g" in gg.columns:
            _scatter_tradeoff(
                gg,
                x="clearing_latency_ms_mean",
                y="sup_g",
                title=f"Fairness–latency trade-off ({topo}, $\\epsilon$={eps_ms:g}ms)",
                out_path=out_dir / f"{tag}_lai_vs_latency.png",
                xlab="Clearing latency (ms)",
                ylab="LAI = sup$_\\Delta$ g($\\Delta$)",
            )
            _scatter_tradeoff(
                gg,
                x="effective_welfare_mean",
                y="sup_g",
                title=f"Fairness–efficiency trade-off ({topo}, $\\epsilon$={eps_ms:g}ms)",
                out_path=out_dir / f"{tag}_lai_vs_effective_welfare.png",
                xlab="Effective welfare",
                ylab="LAI = sup$_\\Delta$ g($\\Delta$)",
            )

        # Always-available trade-off: welfare vs clearing latency.
        _scatter_tradeoff(
            gg,
            x="clearing_latency_ms_mean",
            y="welfare_ratio_mean",
            title=f"Welfare–latency trade-off ({topo}, $\\epsilon$={eps_ms:g}ms)",
            out_path=out_dir / f"{tag}_welfare_vs_latency.png",
            xlab="Clearing latency (ms)",
            ylab="Welfare ratio (SW/OPT)",
        )

    # Robustness lines (requires LAI data)
    if lai_agg is not None and not lai_agg.empty:
        for topo, g in lai_agg.groupby("topology"):
            _line_by_eps(
                g,
                metric="sup_g",
                title=f"LAI robustness vs measurement error ($\\epsilon$) - {topo}",
                out_path=out_dir / f"{topo}_lai_vs_eps.png",
                ylab="LAI = sup$_\\Delta$ g($\\Delta$)",
            )

    print(f"Trade-off figures written to: {out_dir}")


if __name__ == "__main__":
    main()
