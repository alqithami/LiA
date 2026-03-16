from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from skyfield.api import Loader, wgs84

from lia.network.graph import Edge, Node, Topology
from lia.utils.hashing import sha256_file

C_KM_PER_S = 299_792.458


@dataclass(frozen=True)
class DsnBuilderConfig:
    num_nodes: int = 30
    seed: int = 123
    epoch_utc: str = "2026-01-18T00:00:00Z"
    k_nearest: int = 4

    # Approx relay altitudes
    earth_orbit_alt_km: float = 550.0
    mars_orbit_alt_km: float = 400.0

    # How many nodes in each bucket (must sum to num_nodes)
    n_dsn_stations: int = 3
    n_earth_orbit_relays: int = 10
    n_mars_orbit_relays: int = 10
    n_interplanetary: int = 7


def build_dsn_topology(raw_dir: str | Path, config: DsnBuilderConfig) -> Topology:
    """Build DSN-30 using real planetary ephemerides (Skyfield DE421) and real DSN station locations.

    Notes:
      * DSN stations are modeled as ground points on Earth.
      * Additional relay and interplanetary nodes are *synthetic*, but their positions are
        anchored to the real Earth/Mars state vectors at a common epoch.

    The result is a time-stamped, weighted graph with propagation delays computed from Euclidean distance / c.
    """

    if config.n_dsn_stations + config.n_earth_orbit_relays + config.n_mars_orbit_relays + config.n_interplanetary != config.num_nodes:
        raise ValueError("Bucket sizes must sum to num_nodes")

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    load = Loader(str(raw_dir))
    ts = load.timescale()
    t = ts.from_datetime(datetime.fromisoformat(config.epoch_utc.replace("Z", "+00:00")))

    planets = load("de421.bsp")
    ephem_path = raw_dir / "de421.bsp"
    ephemeris_sha256 = sha256_file(ephem_path) if ephem_path.exists() else None
    earth = planets["earth"]
    mars = planets["mars"]

    # DSN complex reference locations (approx): Goldstone (USA), Madrid (Spain), Canberra (Australia)
    stations = [
        ("Goldstone", 35.425, -116.889),
        ("Madrid", 40.431, -4.248),
        ("Canberra", -35.401, 148.981),
    ]

    positions_km: List[np.ndarray] = []
    nodes: List[Node] = []

    # 0..2: DSN stations
    for idx, (name, lat, lon) in enumerate(stations[: config.n_dsn_stations]):
        geo = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
        # Position in ICRF frame (km)
        pos = (earth + geo).at(t).position.km
        positions_km.append(np.array(pos, dtype=float))
        nodes.append(Node(node_id=idx, label=name, meta={"lat": lat, "lon": lon, "type": "dsn_station"}))

    rng = random.Random(config.seed)

    # Earth orbit relays
    earth_center = earth.at(t).position.km
    earth_center = np.array(earth_center, dtype=float)
    earth_r_km = 6371.0

    for i in range(config.n_earth_orbit_relays):
        idx = len(nodes)
        # Random unit vector
        vec = np.array([rng.gauss(0, 1) for _ in range(3)], dtype=float)
        vec /= np.linalg.norm(vec)
        r = earth_r_km + config.earth_orbit_alt_km
        pos = earth_center + r * vec
        positions_km.append(pos)
        nodes.append(Node(node_id=idx, label=f"EarthRelay{i}", meta={"type": "earth_relay", "alt_km": config.earth_orbit_alt_km}))

    # Mars orbit relays
    mars_center = mars.at(t).position.km
    mars_center = np.array(mars_center, dtype=float)
    mars_r_km = 3389.5

    for i in range(config.n_mars_orbit_relays):
        idx = len(nodes)
        vec = np.array([rng.gauss(0, 1) for _ in range(3)], dtype=float)
        vec /= np.linalg.norm(vec)
        r = mars_r_km + config.mars_orbit_alt_km
        pos = mars_center + r * vec
        positions_km.append(pos)
        nodes.append(Node(node_id=idx, label=f"MarsRelay{i}", meta={"type": "mars_relay", "alt_km": config.mars_orbit_alt_km}))

    # Interplanetary nodes along Earth->Mars line
    em_vec = mars_center - earth_center
    for i in range(config.n_interplanetary):
        idx = len(nodes)
        frac = (i + 1) / (config.n_interplanetary + 1)
        # Small random lateral jitter (up to 1% of distance)
        jitter = np.array([rng.gauss(0, 1) for _ in range(3)], dtype=float)
        jitter /= np.linalg.norm(jitter)
        jitter_scale = 0.01 * frac * np.linalg.norm(em_vec)
        pos = earth_center + frac * em_vec + jitter_scale * jitter
        positions_km.append(pos)
        nodes.append(Node(node_id=idx, label=f"Probe{i}", meta={"type": "interplanetary", "frac_earth_mars": frac}))

    # Build kNN edges in 3D geometry
    X = np.vstack(positions_km)
    norms = (X ** 2).sum(axis=1)
    d2 = norms[:, None] + norms[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)

    edges_set: set[Tuple[int, int]] = set()
    edges: List[Edge] = []

    for i in range(config.num_nodes):
        nn = np.argpartition(d2[i], config.k_nearest)[: config.k_nearest]
        for j in nn:
            u = i
            v = int(j)
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edges_set:
                continue
            dist_km = float(math.sqrt(d2[u, v]))
            # One-way light-time in ms
            weight_ms = (dist_km / C_KM_PER_S) * 1000.0
            edges_set.add((a, b))
            edges.append(Edge(u=a, v=b, weight_ms=float(weight_ms), meta={"type": "space_link", "dist_km": dist_km}))

    # Auctioneer: node 0 (Goldstone) for DSN-like environment
    return Topology(
        name="DSN-30",
        num_nodes=config.num_nodes,
        auctioneer_node=0,
        directed=False,
        nodes=nodes,
        edges=edges,
        source={
            "ephemeris": "de421.bsp",
            "ephemeris_sha256": ephemeris_sha256,
            "epoch_utc": config.epoch_utc,
            "k_nearest": config.k_nearest,
            "seed": config.seed,
            "notes": "DSN stations real; relays/probes synthetic but anchored to ephemerides",
        },
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
