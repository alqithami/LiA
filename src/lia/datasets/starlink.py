from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from sgp4.api import Satrec
from sgp4.conveniences import jday_datetime

from lia.network.graph import Edge, Node, Topology
from lia.utils.hashing import sha256_file

C_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class StarlinkBuilderConfig:
    tle_url: str = "https://celestrak.org/NORAD/elements/gp.php?FORMAT=tle&GROUP=starlink"
    num_nodes: int = 200
    k_nearest: int = 4
    snapshot_utc: str = "2026-01-18T00:00:00Z"
    seed: int = 123


def download_tle(url: str, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    p.write_bytes(resp.content)
    return p


def _parse_tle_file(tle_path: Path) -> List[Tuple[str, Satrec]]:
    lines = tle_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    sats: List[Tuple[str, Satrec]] = []
    i = 0
    while i < len(lines) - 2:
        name = lines[i].strip()
        l1 = lines[i + 1].strip()
        l2 = lines[i + 2].strip()
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                sat = Satrec.twoline2rv(l1, l2)
                sats.append((name if name else f"SAT_{len(sats)}", sat))
            except Exception:
                pass
            i += 3
        else:
            i += 1
    return sats


def _propagate_positions_km(sats: List[Tuple[str, Satrec]], snapshot_dt: datetime) -> List[Tuple[str, np.ndarray]]:
    jd, fr = jday_datetime(snapshot_dt)
    out: List[Tuple[str, np.ndarray]] = []
    for name, sat in sats:
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            continue
        # r is in km (TEME)
        out.append((name, np.array(r, dtype=float)))
    return out


def _pairwise_knn_edges(positions: List[np.ndarray], k: int) -> List[Tuple[int, int, float]]:
    n = len(positions)
    if n == 0:
        return []
    X = np.vstack(positions)
    # Compute squared distances matrix efficiently for n=200 (fine)
    # d^2 = ||x||^2 + ||y||^2 - 2 x·y
    norms = (X ** 2).sum(axis=1)
    d2 = norms[:, None] + norms[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)

    edges: set[Tuple[int, int]] = set()
    weights_ms: Dict[Tuple[int, int], float] = {}

    for i in range(n):
        nn = np.argpartition(d2[i], k)[:k]
        for j in nn:
            u, v = (i, int(j))
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edges:
                continue
            dist_km = float(math.sqrt(d2[u, v]))
            # propagation time in ms
            weight_ms = dist_km * 1000.0 / C_M_PER_S * 1000.0
            edges.add((a, b))
            weights_ms[(a, b)] = weight_ms

    return [(u, v, weights_ms[(u, v)]) for (u, v) in sorted(edges)]


def build_starlink_topology(
    raw_dir: str | Path,
    config: StarlinkBuilderConfig,
) -> Topology:
    """Build STARLINK-200 using a CelesTrak Starlink TLE snapshot.

    The resulting topology contains `num_nodes` satellites, with ISL edges between
    each satellite and its k nearest neighbors (in 3D snapshot geometry).

    Auctioneer is node 0.
    """

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tle_path = raw_dir / "celestrak_starlink.tle"
    if not tle_path.exists():
        download_tle(config.tle_url, tle_path)

    all_sats = _parse_tle_file(tle_path)
    if len(all_sats) < config.num_nodes:
        raise RuntimeError(f"TLE file only contains {len(all_sats)} satellites; need {config.num_nodes}.")

    # Deterministic selection for reproducibility
    rng = random.Random(config.seed)
    # Sort by name, then take first N (stable)
    all_sats_sorted = sorted(all_sats, key=lambda x: x[0])
    selected = all_sats_sorted[: config.num_nodes]

    snapshot_dt = datetime.fromisoformat(config.snapshot_utc.replace("Z", "+00:00"))
    if snapshot_dt.tzinfo is None:
        snapshot_dt = snapshot_dt.replace(tzinfo=timezone.utc)

    pos_pairs = _propagate_positions_km(selected, snapshot_dt)
    if len(pos_pairs) < config.num_nodes:
        # If some TLEs cannot be propagated, backfill with more satellites
        remaining = [p for p in all_sats_sorted[config.num_nodes :] if p not in selected]
        while len(pos_pairs) < config.num_nodes and remaining:
            name, sat = remaining.pop(0)
            extra = _propagate_positions_km([(name, sat)], snapshot_dt)
            pos_pairs.extend(extra)
        pos_pairs = pos_pairs[: config.num_nodes]

    names = [n for n, _ in pos_pairs]
    positions = [p for _, p in pos_pairs]

    knn_edges = _pairwise_knn_edges(positions, k=config.k_nearest)

    nodes = [
        Node(
            node_id=i,
            label=names[i],
            meta={
                "frame": "TEME",
                "x_km": float(positions[i][0]),
                "y_km": float(positions[i][1]),
                "z_km": float(positions[i][2]),
            },
        )
        for i in range(config.num_nodes)
    ]

    edges = [
        Edge(
            u=int(u),
            v=int(v),
            weight_ms=float(w),
            meta={"type": "ISL", "model": "kNN", "k": config.k_nearest},
        )
        for u, v, w in knn_edges
    ]

    return Topology(
        name="STARLINK-200",
        num_nodes=config.num_nodes,
        auctioneer_node=0,
        directed=False,
        nodes=nodes,
        edges=edges,
        source={
            "tle_url": config.tle_url,
            "tle_file": str(tle_path.name),
            "tle_sha256": sha256_file(tle_path),
            "snapshot_utc": config.snapshot_utc,
            "k_nearest": config.k_nearest,
        },
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
