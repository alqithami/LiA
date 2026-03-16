from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from lia.network.graph import Topology
from lia.network.shortest_path import dijkstra_all_to_one


@dataclass(frozen=True)
class Bid:
    bidder_id: int
    node_id: int
    value: float
    emission_ms: float
    travel_ms: float
    arrival_ms: float
    slack_true_ms: float
    slack_est_ms: float


@dataclass(frozen=True)
class AuctionInstance:
    topology_name: str
    instance_id: int
    horizon_ms: float
    auctioneer_node: int
    bids: List[Bid]


def compute_horizon_ms(
    dist_to_auctioneer_ms: np.ndarray,
    auctioneer_node: int,
    percentile: float,
    extra_ms: float = 0.0,
) -> float:
    if not (0.0 < percentile < 1.0):
        raise ValueError("percentile must be in (0,1)")

    mask = np.ones_like(dist_to_auctioneer_ms, dtype=bool)
    mask[auctioneer_node] = False
    finite = dist_to_auctioneer_ms[mask]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("No finite distances; topology may be disconnected")

    h = float(np.quantile(finite, percentile)) + float(extra_ms)
    return h


def _clip_eps(x: float, eps_ms: float) -> float:
    return float(max(-eps_ms, min(eps_ms, x)))


def _distance_bin_assignments(dist: np.ndarray, candidate_nodes: np.ndarray, n_bins: int = 4) -> Dict[int, int]:
    finite = dist[candidate_nodes]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {int(node): 0 for node in candidate_nodes}

    # Quantile bins are a simple proxy for correlated subnetwork classes.
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, qs)
    if np.allclose(edges, edges[0]):
        return {int(node): 0 for node in candidate_nodes}

    bins: Dict[int, int] = {}
    inner = edges[1:-1]
    for node in candidate_nodes:
        d = float(dist[int(node)])
        if not np.isfinite(d):
            bins[int(node)] = n_bins - 1
        else:
            bins[int(node)] = int(np.searchsorted(inner, d, side="right"))
    return bins


def generate_instances(
    topology: Topology,
    bidder_count: int,
    n_instances: int,
    value_low: float,
    value_high: float,
    horizon_policy: Dict,
    eps_ms: float,
    rng: np.random.Generator,
    measurement_error_model: str = "iid_uniform",
    measurement_error_common_fraction: float = 0.0,
) -> List[AuctionInstance]:
    """Generate synthetic auction instances on top of a real topology graph.

    Supported measurement-error models
    ----------------------------------
    iid_uniform
        Independent bounded perturbation per bidder.
    common_plus_iid_uniform
        Shared common-mode offset plus independent bounded noise.
    distance_biased_uniform
        Farther bidders receive a systematically pessimistic slack estimate
        (bounded one-sided bias plus residual jitter).
    subnetwork_correlated_uniform
        Bidders in the same coarse distance band share a correlated bias plus
        small local noise.
    """

    adj = topology.adjacency()
    dist_list = dijkstra_all_to_one(adj, target=topology.auctioneer_node)
    dist = np.array(dist_list, dtype=float)

    horizon_ms = compute_horizon_ms(
        dist_to_auctioneer_ms=dist,
        auctioneer_node=topology.auctioneer_node,
        percentile=float(horizon_policy.get("percentile", 0.95)),
        extra_ms=float(horizon_policy.get("extra_ms", 0.0)),
    )

    all_nodes = np.arange(topology.num_nodes)
    candidate_nodes = all_nodes[all_nodes != topology.auctioneer_node]
    finite_candidates = dist[candidate_nodes]
    finite_candidates = finite_candidates[np.isfinite(finite_candidates)]
    max_finite_dist = float(np.max(finite_candidates)) if finite_candidates.size > 0 else max(1.0, horizon_ms)
    node_bins = _distance_bin_assignments(dist, candidate_nodes, n_bins=4)

    instances: List[AuctionInstance] = []

    model = str(measurement_error_model or "iid_uniform").strip().lower()
    common_frac = max(0.0, min(1.0, float(measurement_error_common_fraction)))

    iid_aliases = {"iid_uniform", "slack_uniform_iid", "uniform_iid"}
    common_aliases = {
        "common_plus_iid_uniform",
        "slack_uniform_common_plus_iid",
        "uniform_common_plus_iid",
    }
    distance_bias_aliases = {
        "distance_biased_uniform",
        "distance_bias_uniform",
        "distance_biased",
    }
    subnetwork_aliases = {
        "subnetwork_correlated_uniform",
        "subnetwork_correlated",
        "cluster_correlated_uniform",
    }

    for inst_id in range(n_instances):
        eta_common = 0.0
        if eps_ms > 0.0 and model in common_aliases:
            eta_common = float(rng.uniform(-eps_ms * common_frac, eps_ms * common_frac))

        # Coarse correlated bias per subnetwork class (sampled once per instance).
        cluster_bias = {}
        if eps_ms > 0.0 and model in subnetwork_aliases:
            for k in set(node_bins.values()):
                cluster_bias[int(k)] = float(rng.uniform(-eps_ms, eps_ms))

        nodes = rng.choice(candidate_nodes, size=bidder_count, replace=True)
        values = rng.uniform(low=value_low, high=value_high, size=bidder_count)

        bids: List[Bid] = []
        for i in range(bidder_count):
            node_id = int(nodes[i])
            v = float(values[i])
            travel_ms = float(dist[node_id])

            if not np.isfinite(travel_ms) or travel_ms > horizon_ms:
                emission_ms = float(rng.uniform(0.0, horizon_ms))
                arrival_ms = emission_ms + (travel_ms if np.isfinite(travel_ms) else horizon_ms * 10)
                slack_true = horizon_ms - arrival_ms
            else:
                max_emit = max(0.0, horizon_ms - travel_ms)
                emission_ms = float(rng.uniform(0.0, max_emit))
                arrival_ms = emission_ms + travel_ms
                slack_true = horizon_ms - arrival_ms

            if eps_ms <= 0.0:
                eta = 0.0
            elif model in iid_aliases:
                eta = float(rng.uniform(-eps_ms, eps_ms))
            elif model in common_aliases:
                eps_ind = eps_ms * (1.0 - common_frac)
                eta_ind = float(rng.uniform(-eps_ind, eps_ind))
                eta = float(eta_common + eta_ind)
                eta = _clip_eps(eta, eps_ms)
            elif model in distance_bias_aliases:
                # Normalized distance proxy in [0,1]; distant bidders are systematically
                # biased downward (their slack is underestimated), plus bounded jitter.
                if np.isfinite(travel_ms) and max_finite_dist > 0.0:
                    norm = max(0.0, min(1.0, travel_ms / max_finite_dist))
                else:
                    norm = 1.0
                bias = -eps_ms * norm
                jitter = (eps_ms * (1.0 - norm)) * float(rng.uniform(-1.0, 1.0))
                eta = _clip_eps(bias + jitter, eps_ms)
            elif model in subnetwork_aliases:
                band = int(node_bins.get(node_id, 0))
                bias = float(cluster_bias.get(band, 0.0))
                local_jitter = float(rng.uniform(-0.25 * eps_ms, 0.25 * eps_ms))
                eta = _clip_eps(bias + local_jitter, eps_ms)
            else:
                raise ValueError(f"Unknown measurement_error_model: {measurement_error_model}")

            slack_est = slack_true + eta

            bids.append(
                Bid(
                    bidder_id=i,
                    node_id=node_id,
                    value=v,
                    emission_ms=emission_ms,
                    travel_ms=travel_ms,
                    arrival_ms=arrival_ms,
                    slack_true_ms=slack_true,
                    slack_est_ms=slack_est,
                )
            )

        instances.append(
            AuctionInstance(
                topology_name=topology.name,
                instance_id=inst_id,
                horizon_ms=horizon_ms,
                auctioneer_node=topology.auctioneer_node,
                bids=bids,
            )
        )

    return instances
