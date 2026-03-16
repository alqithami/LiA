#!/usr/bin/env python
from __future__ import annotations

"""Build and cache topology graphs from real public datasets.

Important: the repository uses a `src/` layout. We prepend the local `src/` directory to
`sys.path` to prevent accidentally importing an older installed `lia` package.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from lia.datasets.dsn import DsnBuilderConfig, build_dsn_topology  # noqa: E402
from lia.datasets.starlink import StarlinkBuilderConfig, build_starlink_topology  # noqa: E402
from lia.datasets.topology_zoo import TopologyZooBuilderConfig, build_internet_topology  # noqa: E402
from lia.network.graph import save_topology  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and cache topology graphs from real datasets.")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--force", action="store_true", help="Rebuild even if JSON already exists")
    args = parser.parse_args()

    base = Path(args.data_dir)
    raw_dir = base / "raw"
    topo_dir = base / "topologies"

    topo_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # STARLINK-200
    starlink_out = topo_dir / "STARLINK-200.json"
    if args.force or not starlink_out.exists():
        t = build_starlink_topology(raw_dir, StarlinkBuilderConfig())
        save_topology(t, starlink_out)
        print(f"Wrote {starlink_out}")
    else:
        print(f"Skipping {starlink_out} (exists)")

    # INTERNET-100
    internet_out = topo_dir / "INTERNET-100.json"
    if args.force or not internet_out.exists():
        t = build_internet_topology(raw_dir, TopologyZooBuilderConfig())
        save_topology(t, internet_out)
        print(f"Wrote {internet_out}")
    else:
        print(f"Skipping {internet_out} (exists)")

    # DSN-30
    dsn_out = topo_dir / "DSN-30.json"
    if args.force or not dsn_out.exists():
        t = build_dsn_topology(raw_dir, DsnBuilderConfig())
        save_topology(t, dsn_out)
        print(f"Wrote {dsn_out}")
    else:
        print(f"Skipping {dsn_out} (exists)")


if __name__ == "__main__":
    main()
