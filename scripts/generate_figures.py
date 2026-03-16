#!/usr/bin/env python
from __future__ import annotations

"""Generate paper-style figures from a completed run directory.

This script is intentionally lightweight: it only reads the CSV outputs produced by
`run_pipeline.py` and writes PNG figures into `<run_dir>/figures/`.

It does not require internet access and does not re-run any experiments.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _latest_run_dir(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _bar_with_ci(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    # df expected columns: mechanism, <metric>_mean, <metric>_ci_low, <metric>_ci_high
    m = f"{metric}_mean"
    lo = f"{metric}_ci_low"
    hi = f"{metric}_ci_high"
    if m not in df.columns:
        return

    plot_df = df.sort_values(m, ascending=False).reset_index(drop=True)

    x = range(len(plot_df))
    y = plot_df[m].to_numpy(dtype=float)

    # error bars if present (ignore NaNs)
    yerr = None
    if lo in plot_df.columns and hi in plot_df.columns:
        lower = y - plot_df[lo].to_numpy(dtype=float)
        upper = plot_df[hi].to_numpy(dtype=float) - y
        lower = np.nan_to_num(lower, nan=0.0)
        upper = np.nan_to_num(upper, nan=0.0)
        # Percentile bootstrap intervals do not guarantee that the mean lies inside
        # the reported CI (especially for skewed / truncated distributions).
        # Matplotlib requires non-negative error bars, so we defensively clamp.
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
        if not (np.all(lower == 0.0) and np.all(upper == 0.0)):
            yerr = np.vstack([lower, upper])

    plt.figure()
    plt.bar(x, y, yerr=yerr, capsize=3)
    plt.xticks(list(x), plot_df["mechanism"].tolist(), rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _runtime_scaling(df: pd.DataFrame, title: str, out_path: Path) -> None:
    # df expected columns: bidder_count, mechanism, compute_time_s_mean
    if "compute_time_s_mean" not in df.columns:
        return

    plt.figure()
    for mech, g in df.groupby("mechanism"):
        g = g.sort_values("bidder_count")
        plt.plot(g["bidder_count"], g["compute_time_s_mean"], marker="o", label=mech)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bidder count (log)")
    plt.ylabel("Compute time (s, log)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _parse_lia_lambda(m: str) -> str | None:
    if not m.startswith("LIA_lambda"):
        return None
    try:
        rest = m[len("LIA_lambda"):]
        return rest.split("_", 1)[0]
    except Exception:
        return None


def _pretty_mech_label(m: str) -> str:
    """Compact mechanism labels for paper figures."""
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
        lam_part = _parse_lia_lambda(m)
        return f"LIA (λ={lam_part})" if lam_part is not None else m
    return m


def _lia_sort_key(m: str) -> float:
    lam_part = _parse_lia_lambda(m)
    if lam_part is None:
        return 1e9
    try:
        return float(lam_part)
    except Exception:
        return 1e9


def _default_mechanism_order(mechanisms: List[str]) -> List[str]:
    """Return a stable, paper-friendly ordering for mechanisms."""
    base = []
    for name in ["FastVCG", "BatchVCG_B1ms", "BatchVCG_B5ms", "BatchVCG_B10ms", "BatchVCG_B20ms", "BatchVCG_B50ms", "HoldBack", "SyncVCG"]:
        if name in mechanisms:
            base.append(name)
    lia = sorted([m for m in mechanisms if m.startswith("LIA_lambda")], key=_lia_sort_key)
    others = sorted([m for m in mechanisms if m not in set(base) and m not in set(lia)])
    return base + lia + others


def _row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _complementary_heatmap(summary: pd.DataFrame, title: str, out_path: Path, mechanisms: Optional[List[str]] = None) -> None:
    """Create the complementary-indicators heatmap (SWR, RVR, inverted runtime)."""
    required = ["mechanism", "welfare_ratio_mean", "revenue_ratio_mean", "compute_time_s_mean"]
    if any(c not in summary.columns for c in required):
        return

    df = summary.copy()
    df = df.set_index("mechanism")

    if mechanisms is None:
        mechanisms = _default_mechanism_order(list(df.index))

    mechs = [m for m in mechanisms if m in df.index]
    df = df.reindex(mechs)

    swr = df["welfare_ratio_mean"].to_numpy(dtype=float)
    rvr = df["revenue_ratio_mean"].to_numpy(dtype=float)

    # runtime is inverted as 1 - rt/max(rt), then row-normalized
    rt_ms = (df["compute_time_s_mean"].to_numpy(dtype=float) * 1000.0)
    rt_max = np.nanmax(rt_ms) if len(rt_ms) else np.nan
    if not np.isfinite(rt_max) or rt_max <= 0:
        rt_inv = np.zeros_like(rt_ms)
    else:
        rt_inv = 1.0 - (rt_ms / rt_max)

    mat = np.vstack([
        _row_normalize(swr),
        _row_normalize(rvr),
        _row_normalize(rt_inv),
    ])

    # Size scales with number of mechanisms.
    fig_w = max(6.0, 0.45 * len(mechs) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 2.6))
    im = ax.imshow(mat, aspect="auto")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["SWR", "RVR", "RT (inv)"], fontsize=9)

    ax.set_xticks(list(range(len(mechs))))
    ax.set_xticklabels([_pretty_mech_label(m) for m in mechs], rotation=45, ha="right", fontsize=8)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Row-normalized [0,1]")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    denom = np.nansum(w)
    if denom <= 0:
        return float("nan")
    return float(np.nansum(v * w) / denom)


def _paper_proper_time_lai_figure(summary: pd.DataFrame, lai: Optional[pd.DataFrame], out_path: Path) -> None:
    """Generate a single "Figure 2" style graphic: LAI bar plot + complementary heatmap.

    The plot aggregates eps=0 results across topologies.
    """
    if "eps_ms" not in summary.columns:
        return

    # Focus on eps=0
    s0 = summary[summary["eps_ms"] == 0.0].copy()
    if s0.empty:
        return

    # Aggregate summary across topologies (weight by n_instances if present)
    if "n_instances" in s0.columns:
        w = s0["n_instances"].fillna(1.0)
    else:
        w = pd.Series(1.0, index=s0.index)

    agg = []
    for mech, g in s0.groupby("mechanism"):
        ww = w.loc[g.index]
        agg.append({
            "mechanism": mech,
            "welfare_ratio_mean": _weighted_mean(g["welfare_ratio_mean"], ww),
            "revenue_ratio_mean": _weighted_mean(g["revenue_ratio_mean"], ww),
            "compute_time_s_mean": _weighted_mean(g["compute_time_s_mean"], ww),
        })
    agg_df = pd.DataFrame(agg)

    mechs = _default_mechanism_order(list(agg_df["mechanism"].unique()))
    agg_df = agg_df.set_index("mechanism").reindex(mechs).reset_index()

    # Aggregate LAI across topologies and bidder counts (weight by instance_count when available)
    lai_means = None
    if lai is not None and ("sup_g" in lai.columns):
        l0 = lai[lai["eps_ms"] == 0.0].copy()
        if not l0.empty:
            w_lai = l0.get("instance_count", pd.Series(1.0, index=l0.index)).fillna(1.0)
            rows = []
            for mech, g in l0.groupby("mechanism"):
                ww = w_lai.loc[g.index]
                rows.append({
                    "mechanism": mech,
                    "sup_g": _weighted_mean(g["sup_g"], ww),
                })
            lai_means = pd.DataFrame(rows)

    # Figure layout: left = LAI, right = heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1.15, 1.35]})

    # Left: LAI bar plot (symlog for dynamic range; supports zeros)
    if lai_means is not None and not lai_means.empty:
        lai_means = lai_means.set_index("mechanism").reindex(mechs)
        y = lai_means["sup_g"].to_numpy(dtype=float)
        x = np.arange(len(mechs))
        ax1.bar(x, y)
        ax1.set_xticks(x)
        ax1.set_xticklabels([_pretty_mech_label(m) for m in mechs], rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("LAI = sup$_Δ$ g($Δ$)")
        ax1.set_title("Latency-arbitrage incentive")
        ax1.set_yscale("symlog", linthresh=1.0)
    else:
        ax1.text(0.5, 0.5, "LAI estimates not found", ha="center", va="center")
        ax1.set_axis_off()

    # Right: complementary heatmap
    # Build matrix on ax2 directly
    df = agg_df.set_index("mechanism")
    swr = df["welfare_ratio_mean"].to_numpy(dtype=float)
    rvr = df["revenue_ratio_mean"].to_numpy(dtype=float)
    rt_ms = (df["compute_time_s_mean"].to_numpy(dtype=float) * 1000.0)
    rt_max = np.nanmax(rt_ms) if len(rt_ms) else np.nan
    rt_inv = np.zeros_like(rt_ms) if not np.isfinite(rt_max) or rt_max <= 0 else (1.0 - (rt_ms / rt_max))

    mat = np.vstack([
        _row_normalize(swr),
        _row_normalize(rvr),
        _row_normalize(rt_inv),
    ])

    im = ax2.imshow(mat, aspect="auto")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["SWR", "RVR", "RT (inv)"], fontsize=9)
    ax2.set_xticks(list(range(len(mechs))))
    ax2.set_xticklabels([_pretty_mech_label(m) for m in mechs], rotation=45, ha="right", fontsize=8)
    ax2.set_title("Complementary indicators")

    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate figures from a LiA run directory")
    ap.add_argument("--run-dir", type=str, default="", help="Path to a specific run directory under runs/")
    ap.add_argument("--runs-dir", type=str, default="runs", help="Parent runs directory (used if --run-dir omitted)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None:
        run_dir = _latest_run_dir(Path(args.runs_dir))
    if run_dir is None or not run_dir.exists():
        raise SystemExit("Could not locate a run directory. Provide --run-dir or ensure runs/ exists.")

    summary_path = run_dir / "summary_table.csv"
    summary_n_path = run_dir / "summary_by_bidder_count.csv"
    lai_path = run_dir / "lai_estimates.csv"

    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")

    summary = pd.read_csv(summary_path)
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Figures by topology (and epsilon if present)
    for (eps_ms, topo), g in summary.groupby(["eps_ms", "topology"], dropna=False):
        sub = g.copy()
        eps_tag = f"eps{eps_ms:g}ms" if pd.notna(eps_ms) else "epsNA"
        base = figs_dir / f"{topo}_{eps_tag}"
        base.mkdir(parents=True, exist_ok=True)

        _bar_with_ci(sub, "welfare_ratio", f"SW/OPT by mechanism - {topo} (eps={eps_ms}ms)", base / "welfare_ratio.png")
        _bar_with_ci(sub, "revenue_ratio", f"Rev/OPT by mechanism - {topo} (eps={eps_ms}ms)", base / "revenue_ratio.png")
        _bar_with_ci(sub, "effective_welfare", f"Effective welfare by mechanism - {topo} (eps={eps_ms}ms)", base / "effective_welfare.png")
        _bar_with_ci(sub, "clearing_latency_ms", f"Clearing latency (ms) by mechanism - {topo} (eps={eps_ms}ms)", base / "clearing_latency_ms.png")
        _bar_with_ci(sub, "compute_time_s", f"Compute time (s) by mechanism - {topo} (eps={eps_ms}ms)", base / "compute_time_s.png")

    # Runtime scaling (if available)
    if summary_n_path.exists():
        summary_n = pd.read_csv(summary_n_path)
        for (eps_ms, topo), g in summary_n.groupby(["eps_ms", "topology"], dropna=False):
            eps_tag = f"eps{eps_ms:g}ms" if pd.notna(eps_ms) else "epsNA"
            base = figs_dir / f"{topo}_{eps_tag}"
            _runtime_scaling(g, f"Runtime scaling - {topo} (eps={eps_ms}ms)", base / "runtime_scaling.png")

    # LAI plots (if available)
    if lai_path.exists():
        lai = pd.read_csv(lai_path)
        for (eps_ms, topo), g in lai.groupby(["eps_ms", "topology"], dropna=False):
            eps_tag = f"eps{eps_ms:g}ms" if pd.notna(eps_ms) else "epsNA"
            base = figs_dir / f"{topo}_{eps_tag}"
            # map to summary-like format for sup_g
            if "sup_g" in g.columns:
                plot_df = g[["mechanism", "sup_g", "sup_g_ci_low", "sup_g_ci_high"]].copy()
                plot_df = plot_df.rename(columns={
                    "sup_g": "sup_g_mean",
                    "sup_g_ci_low": "sup_g_ci_low",
                    "sup_g_ci_high": "sup_g_ci_high",
                })
                _bar_with_ci(plot_df, "sup_g", f"LAI sup_g by mechanism - {topo} (eps={eps_ms}ms)", base / "lai_sup_g.png")

    # Paper figures (eps=0): complementary heatmaps + composite Figure 2
    paper_dir = figs_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    if "eps_ms" in summary.columns:
        s0 = summary[summary["eps_ms"] == 0.0].copy()
        if not s0.empty:
            mechs_order = _default_mechanism_order(list(summary["mechanism"].unique()))
            for topo, g in s0.groupby("topology"):
                _complementary_heatmap(g, f"Complementary indicators - {topo} (eps=0ms)", paper_dir / f"{topo}_heatmap.png", mechanisms=mechs_order)

    lai_df = None
    if lai_path.exists():
        try:
            lai_df = pd.read_csv(lai_path)
        except Exception:
            lai_df = None
    _paper_proper_time_lai_figure(summary, lai_df, paper_dir / "proper_time_lai_figure.png")

    print(f"Figures written to: {figs_dir}")


if __name__ == "__main__":
    main()
