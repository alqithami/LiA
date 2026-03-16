from __future__ import annotations

import hashlib
import io
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import requests

from lia.network.graph import Edge, Node, Topology
from lia.utils.hashing import sha256_file


@dataclass(frozen=True)
class TopologyZooBuilderConfig:
    archive_url: str = "https://topology-zoo.org/files/archive.zip"
    archive_filename: str = "topology_zoo_archive.zip"
    graphml_name: str = "Interoute.graphml"
    target_nodes: int = 100

    # Weight model: speed of light in fiber (~2e5 km/s) and an optional path stretch
    fiber_speed_km_per_s: float = 200_000.0
    path_stretch: float = 1.30
    per_hop_processing_ms: float = 0.10


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Earth radius in km
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _impute_missing_coords(G: nx.Graph, max_iter: int = 25) -> None:
    """Impute missing Latitude/Longitude by averaging neighbor coordinates.

    Many Topology Zoo graphs are geocoded, but a small number of nodes can be missing
    coordinates. This routine fills them in so that edge propagation delays can still
    be computed from geography.
    """

    coords: Dict[Any, Tuple[float, float] | None] = {}
    for n, attrs in G.nodes(data=True):
        lat = attrs.get("Latitude")
        lon = attrs.get("Longitude")
        try:
            lat_f = float(lat) if lat is not None else None
            lon_f = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            lat_f, lon_f = None, None
        coords[n] = (lat_f, lon_f) if (lat_f is not None and lon_f is not None) else None

    for _ in range(max_iter):
        changed = False
        for n in G.nodes:
            if coords[n] is not None:
                continue
            neigh = [coords[m] for m in G.neighbors(n) if coords.get(m) is not None]
            if not neigh:
                continue
            lat_mean = float(np.mean([c[0] for c in neigh]))
            lon_mean = float(np.mean([c[1] for c in neigh]))
            coords[n] = (lat_mean, lon_mean)
            changed = True
        if not changed:
            break

    # Write back
    for n, c in coords.items():
        if c is None:
            # As a last resort, leave missing (edges incident to this node will fall back to a safe default).
            continue
        G.nodes[n]["Latitude"] = c[0]
        G.nodes[n]["Longitude"] = c[1]


def download_topology_zoo_archive(config: TopologyZooBuilderConfig, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / config.archive_filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    resp = requests.get(config.archive_url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def build_internet_topology(raw_dir: Path, config: TopologyZooBuilderConfig) -> Topology:
    """Build INTERNET-100 from a Topology Zoo GraphML topology."""

    archive_path = download_topology_zoo_archive(config, raw_dir)

    with zipfile.ZipFile(archive_path) as z:
        available = set(z.namelist())
        if config.graphml_name not in available:
            # Try a case-insensitive match
            candidates = [n for n in available if n.lower() == config.graphml_name.lower()]
            if not candidates:
                raise FileNotFoundError(
                    f"{config.graphml_name} not found in Topology Zoo archive. Example entries: {sorted(list(available))[:10]}"
                )
            graphml_name = candidates[0]
        else:
            graphml_name = config.graphml_name

        data = z.read(graphml_name)

    G = nx.read_graphml(io.BytesIO(data))

    # Fill missing coordinates if needed
    _impute_missing_coords(G)

    # Choose an auctioneer node as highest-degree node
    degrees = dict(G.degree())
    auctioneer_node = max(degrees.keys(), key=lambda n: degrees[n])

    # BFS order ensures the induced subgraph remains connected
    bfs_order = [auctioneer_node]
    seen = {auctioneer_node}
    queue = [auctioneer_node]
    while queue and len(bfs_order) < config.target_nodes:
        u = queue.pop(0)
        for v in G.neighbors(u):
            if v in seen:
                continue
            seen.add(v)
            bfs_order.append(v)
            queue.append(v)
            if len(bfs_order) >= config.target_nodes:
                break

    if len(bfs_order) < config.target_nodes:
        raise ValueError(
            f"Requested {config.target_nodes} nodes but only reached {len(bfs_order)}. Is {graphml_name} disconnected?"
        )

    H = G.subgraph(bfs_order).copy()

    # Relabel nodes to 0..N-1 for downstream speed and determinism
    old_to_new = {old: i for i, old in enumerate(H.nodes())}
    H = nx.relabel_nodes(H, old_to_new, copy=True)

    new_auctioneer = old_to_new[auctioneer_node]

    nodes: List[Node] = []
    for i, attrs in H.nodes(data=True):
        lat = attrs.get("Latitude")
        lon = attrs.get("Longitude")
        try:
            lat_f = float(lat) if lat is not None else None
            lon_f = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            lat_f, lon_f = None, None

        nodes.append(
            Node(
                node_id=int(i),
                label=str(i),
                meta={"lat": lat_f, "lon": lon_f},
            )
        )

    edges: List[Edge] = []
    for u, v, _attrs in H.edges(data=True):
        u = int(u)
        v = int(v)
        lat1 = H.nodes[u].get("Latitude")
        lon1 = H.nodes[u].get("Longitude")
        lat2 = H.nodes[v].get("Latitude")
        lon2 = H.nodes[v].get("Longitude")

        if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
            # Fallback: treat as an intra-metro link
            dist_km = 50.0
        else:
            dist_km = _haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))

        # Propagation in fiber plus mild path stretch + fixed per-hop processing
        prop_s = (config.path_stretch * dist_km) / config.fiber_speed_km_per_s
        prop_ms = prop_s * 1000.0
        weight_ms = prop_ms + config.per_hop_processing_ms

        edges.append(Edge(u=u, v=v, weight_ms=float(weight_ms), meta={"dist_km": float(dist_km)}))

    now = datetime.now(timezone.utc).isoformat()
    topo = Topology(
        name="INTERNET-100",
        num_nodes=config.target_nodes,
        directed=False,
        auctioneer_node=int(new_auctioneer),
        nodes=nodes,
        edges=edges,
        created_at_utc=now,
        source={
            "dataset": "Topology Zoo",
            "archive_url": config.archive_url,
            "archive_filename": config.archive_filename,
            "archive_sha256": sha256_file(archive_path),
            "graphml_sha256": hashlib.sha256(data).hexdigest(),
            "graphml_name": graphml_name,
            "weight_model": {
                "fiber_speed_km_per_s": config.fiber_speed_km_per_s,
                "path_stretch": config.path_stretch,
                "per_hop_processing_ms": config.per_hop_processing_ms,
            },
        },
    )
    topo.adjacency()
    return topo
