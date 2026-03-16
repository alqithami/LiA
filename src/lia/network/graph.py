from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Node:
    node_id: int
    label: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    weight_ms: float
    meta: Dict[str, Any]


@dataclass
class Topology:
    name: str
    num_nodes: int
    auctioneer_node: int
    directed: bool
    nodes: List[Node]
    edges: List[Edge]
    source: Dict[str, Any]
    created_at_utc: str

    def adjacency(self) -> List[List[Tuple[int, float]]]:
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(self.num_nodes)]
        for e in self.edges:
            adj[e.u].append((e.v, float(e.weight_ms)))
            if not self.directed:
                adj[e.v].append((e.u, float(e.weight_ms)))
        return adj


def save_topology(topology: Topology, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": topology.name,
        "num_nodes": topology.num_nodes,
        "auctioneer_node": topology.auctioneer_node,
        "directed": topology.directed,
        "created_at_utc": topology.created_at_utc,
        "source": topology.source,
        "nodes": [
            {"node_id": n.node_id, "label": n.label, "meta": n.meta} for n in topology.nodes
        ],
        "edges": [
            {"u": e.u, "v": e.v, "weight_ms": e.weight_ms, "meta": e.meta} for e in topology.edges
        ],
    }

    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_topology(path: str | Path) -> Topology:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    nodes = [Node(int(n["node_id"]), str(n.get("label", "")), dict(n.get("meta", {}))) for n in payload["nodes"]]
    edges = [
        Edge(int(e["u"]), int(e["v"]), float(e["weight_ms"]), dict(e.get("meta", {})))
        for e in payload["edges"]
    ]

    return Topology(
        name=str(payload["name"]),
        num_nodes=int(payload["num_nodes"]),
        auctioneer_node=int(payload["auctioneer_node"]),
        directed=bool(payload["directed"]),
        nodes=nodes,
        edges=edges,
        source=dict(payload.get("source", {})),
        created_at_utc=str(payload.get("created_at_utc", "")),
    )
