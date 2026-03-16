from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from lia.experiment.instance_generator import AuctionInstance, Bid
from lia.mechanisms.common import MechanismOutcome, utility
from lia.metrics.bootstrap import bootstrap_mean_ci


@dataclass(frozen=True)
class LaiEstimate:
    """Estimated LAI under the paper's ordering: expectation first, then supremum.

    We estimate g(Δ) = E[ u(d-Δ) - u(d) ] for a grid of Δ values, and then
    report both g(1ms) and sup_Δ g(Δ) over that grid.

    Optional bootstrap confidence intervals are computed via resampling over
    instances (paired across Δ).
    """

    delta_grid_ms: List[float]
    g_delta: List[float]
    g1ms: float
    sup_g: float
    instance_count: int
    bidder_sample_count: int

    # Optional (95%) bootstrap CIs
    g_delta_ci_low: Optional[List[float]] = None
    g_delta_ci_high: Optional[List[float]] = None
    g1ms_ci_low: Optional[float] = None
    g1ms_ci_high: Optional[float] = None
    sup_g_ci_low: Optional[float] = None
    sup_g_ci_high: Optional[float] = None


def apply_delay_reduction(inst: AuctionInstance, bidder_id: int, delta_ms: float) -> AuctionInstance:
    """Return a new instance where a single bidder's propagation delay is reduced by `delta_ms`.

    The bidder's emission time is unchanged. Arrival time and slack are updated consistently.
    The slack estimation error (eta = slack_est - slack_true) is kept fixed.
    """

    bids_new: List[Bid] = []
    for b in inst.bids:
        if b.bidder_id != bidder_id:
            bids_new.append(b)
            continue

        eta = b.slack_est_ms - b.slack_true_ms
        new_travel = max(0.0, float(b.travel_ms) - float(delta_ms))
        new_arrival = float(b.emission_ms) + new_travel
        new_slack_true = float(inst.horizon_ms) - new_arrival
        new_slack_est = new_slack_true + eta

        bids_new.append(
            Bid(
                bidder_id=b.bidder_id,
                node_id=b.node_id,
                value=b.value,
                emission_ms=b.emission_ms,
                travel_ms=new_travel,
                arrival_ms=new_arrival,
                slack_true_ms=new_slack_true,
                slack_est_ms=new_slack_est,
            )
        )

    return AuctionInstance(
        topology_name=inst.topology_name,
        instance_id=inst.instance_id,
        horizon_ms=inst.horizon_ms,
        auctioneer_node=inst.auctioneer_node,
        bids=bids_new,
    )


def _bootstrap_sup_ci(
    per_instance_gains: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> Optional[Tuple[float, float]]:
    """Bootstrap CI for sup_Δ mean(gain(Δ)) over the Δ grid.

    Resampling is done over instances, preserving the joint dependence across Δ.
    """

    if n_resamples <= 0:
        return None

    if per_instance_gains.ndim != 2:
        raise ValueError("per_instance_gains must be 2D")

    mask = np.all(np.isfinite(per_instance_gains), axis=1)
    x = per_instance_gains[mask]
    n, d = x.shape
    if n == 0:
        return None

    chunk = 256
    sups: List[np.ndarray] = []
    for start in range(0, n_resamples, chunk):
        bs = min(chunk, n_resamples - start)
        idx = rng.integers(0, n, size=(bs, n), dtype=np.int64)
        means = x[idx].mean(axis=1)  # (bs, d)
        # Paper definition: sup_\Delta g(\Delta) is lower-bounded by 0.
        sups.append(np.maximum(np.max(means, axis=1), 0.0))

    sup_samples = np.concatenate(sups)
    low, high = np.quantile(sup_samples, [0.025, 0.975])
    return float(low), float(high)


def estimate_lai(
    instances: List[AuctionInstance],
    mechanism_fn: Callable[[AuctionInstance], MechanismOutcome],
    delta_grid_ms: List[float],
    sample_bidders_per_instance: int,
    rng: np.random.Generator,
    bootstrap_resamples: int = 0,
    bootstrap_rng: Optional[np.random.Generator] = None,
) -> LaiEstimate:
    """Estimate LAI metrics for a set of instances.

    Returns g(1ms) and sup over the provided delta grid.
    """

    if not delta_grid_ms:
        raise ValueError("delta_grid_ms must not be empty")

    grid = sorted(set(float(d) for d in delta_grid_ms))
    d = len(grid)

    per_inst: List[np.ndarray] = []
    bidder_samples_total = 0

    for inst in instances:
        n = len(inst.bids)
        k = min(int(sample_bidders_per_instance), n)
        if k <= 0:
            continue

        base_out = mechanism_fn(inst)

        bidder_ids = np.arange(n)
        sampled = rng.choice(bidder_ids, size=k, replace=False)
        bidder_samples_total += int(len(sampled))

        # Cache baseline utilities for sampled bidders
        base_util: Dict[int, float] = {}
        for bid_idx in sampled:
            bid = inst.bids[int(bid_idx)]
            won = base_out.winner_id == bid.bidder_id
            base_util[bid.bidder_id] = utility(bid.value, won, base_out.payment)

        gains_sum = np.zeros(d, dtype=float)

        for bid_idx in sampled:
            bidder_id = int(bid_idx)
            u0 = float(base_util[bidder_id])

            for gi, delta in enumerate(grid):
                inst_mod = apply_delay_reduction(inst, bidder_id=bidder_id, delta_ms=delta)
                out_mod = mechanism_fn(inst_mod)
                bid_mod = inst_mod.bids[bidder_id]
                u1 = utility(bid_mod.value, out_mod.winner_id == bidder_id, out_mod.payment)
                gains_sum[gi] += float(u1 - u0)

        gains_mean = gains_sum / max(1, int(len(sampled)))
        per_inst.append(gains_mean)

    if not per_inst:
        return LaiEstimate(
            delta_grid_ms=grid,
            g_delta=[float("nan")] * d,
            g1ms=float("nan"),
            sup_g=float("nan"),
            instance_count=0,
            bidder_sample_count=0,
        )

    per_inst_arr = np.vstack(per_inst)  # (N, d)
    g_delta = np.nanmean(per_inst_arr, axis=0)

    # g(1ms): exact match if present; otherwise nearest grid point
    if 1.0 in grid:
        g1 = float(g_delta[grid.index(1.0)])
        idx1 = int(grid.index(1.0))
    else:
        idx1 = int(np.argmin(np.abs(np.array(grid) - 1.0)))
        g1 = float(g_delta[idx1])

    sup_g = float(np.nanmax(np.concatenate([g_delta, np.array([0.0])], dtype=float)))

    g_ci_low: Optional[List[float]] = None
    g_ci_high: Optional[List[float]] = None
    g1_low: Optional[float] = None
    g1_high: Optional[float] = None
    sup_low: Optional[float] = None
    sup_high: Optional[float] = None

    if bootstrap_resamples and int(bootstrap_resamples) > 0:
        boot_rng = bootstrap_rng if bootstrap_rng is not None else rng

        g_ci_low = [float("nan")] * d
        g_ci_high = [float("nan")] * d
        for gi in range(d):
            ci = bootstrap_mean_ci(per_inst_arr[:, gi], n_resamples=int(bootstrap_resamples), rng=boot_rng)
            if ci is None:
                continue
            g_ci_low[gi] = float(ci.low)
            g_ci_high[gi] = float(ci.high)

        # g(1ms) CI
        ci1 = bootstrap_mean_ci(per_inst_arr[:, idx1], n_resamples=int(bootstrap_resamples), rng=boot_rng)
        if ci1 is not None:
            g1_low, g1_high = float(ci1.low), float(ci1.high)

        # sup CI (paired across deltas)
        sup_ci = _bootstrap_sup_ci(per_instance_gains=per_inst_arr, n_resamples=int(bootstrap_resamples), rng=boot_rng)
        if sup_ci is not None:
            sup_low, sup_high = sup_ci

    return LaiEstimate(
        delta_grid_ms=grid,
        g_delta=[float(v) for v in g_delta],
        g1ms=g1,
        sup_g=sup_g,
        instance_count=int(per_inst_arr.shape[0]),
        bidder_sample_count=int(bidder_samples_total),
        g_delta_ci_low=g_ci_low,
        g_delta_ci_high=g_ci_high,
        g1ms_ci_low=g1_low,
        g1ms_ci_high=g1_high,
        sup_g_ci_low=sup_low,
        sup_g_ci_high=sup_high,
    )
