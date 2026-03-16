#!/usr/bin/env python
from __future__ import annotations

"""Summarize horizon-slack dispersion for cached topologies.

This utility helps interpret the revised welfare bound by reporting empirical slack
spread quantiles for a chosen bidder count and number of Monte Carlo instances.
It uses the same instance generator as the main pipeline but does not run any
auction mechanisms.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from lia.experiment.instance_generator import generate_instances
from lia.network.graph import load_topology


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize slack-spread quantiles for cached topologies")
    ap.add_argument("--data-dir", type=str, default="data", help="Repository data directory containing topologies/")
    ap.add_argument("--topologies", nargs="*", default=["STARLINK-200", "INTERNET-100", "DSN-30"])
    ap.add_argument("--bidder-count", type=int, default=50)
    ap.add_argument("--instances", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--value-low", type=float, default=0.0)
    ap.add_argument("--value-high", type=float, default=1000.0)
    ap.add_argument("--horizon-percentile", type=float, default=0.95)
    ap.add_argument("--horizon-extra-ms", type=float, default=0.0)
    ap.add_argument("--csv", type=str, default="", help="Optional CSV output path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    topo_dir = data_dir / "topologies"

    rows = []
    for name in args.topologies:
        topo_path = topo_dir / f"{name}.json"
        if not topo_path.exists():
            raise SystemExit(f"Missing cached topology: {topo_path}")
        topo = load_topology(topo_path)
        instances = generate_instances(
            topology=topo,
            bidder_count=int(args.bidder_count),
            n_instances=int(args.instances),
            value_low=float(args.value_low),
            value_high=float(args.value_high),
            horizon_policy={"percentile": float(args.horizon_percentile), "extra_ms": float(args.horizon_extra_ms)},
            eps_ms=0.0,
            rng=rng,
            measurement_error_model="iid_uniform",
        )

        spreads = []
        feasible_fracs = []
        for inst in instances:
            feasible = [b for b in inst.bids if b.arrival_ms <= inst.horizon_ms]
            feasible_fracs.append(len(feasible) / len(inst.bids) if inst.bids else float("nan"))
            if feasible:
                slacks = np.array([b.slack_true_ms for b in feasible], dtype=float)
                spreads.append(float(np.max(slacks) - np.min(slacks)))
            else:
                spreads.append(float("nan"))

        arr = np.array(spreads, dtype=float)
        arr = arr[np.isfinite(arr)]
        ff = np.array(feasible_fracs, dtype=float)
        rows.append(
            {
                "topology": name,
                "bidder_count": int(args.bidder_count),
                "instances": int(args.instances),
                "slack_spread_p50_ms": float(np.nanpercentile(arr, 50)) if arr.size else float("nan"),
                "slack_spread_p90_ms": float(np.nanpercentile(arr, 90)) if arr.size else float("nan"),
                "slack_spread_p95_ms": float(np.nanpercentile(arr, 95)) if arr.size else float("nan"),
                "mean_feasible_fraction": float(np.nanmean(ff)) if ff.size else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nWrote CSV: {out}")


if __name__ == "__main__":
    main()
