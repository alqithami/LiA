from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from lia.experiment.instance_generator import AuctionInstance
from lia.mechanisms.common import MechanismOutcome
from lia.metrics.interval_graph import clique_number


@dataclass(frozen=True)
class InstanceMetrics:
    topology: str
    instance_id: int
    bidder_count: int
    feasible_count: int
    infeasible_count: int
    feasible_fraction: float
    mechanism: str
    horizon_ms: float

    # Benchmark values
    opt_value: float  # backward-compatible alias for opt_all_value
    opt_all_value: float
    opt_feasible_value: float
    feasible_opt_coverage: float

    winner_id: Optional[int]
    winner_value: float
    payment: float

    # Backward-compatible paper columns (overall benchmark = best value in instance)
    welfare_ratio: float
    revenue_ratio: float

    # Additional benchmark decompositions used in the revised manuscript
    welfare_ratio_all: float
    welfare_ratio_feasible: float
    revenue_ratio_all: float
    revenue_ratio_feasible: float

    decision_time_ms: float
    clearing_latency_ms: float
    effective_welfare: float

    compute_time_s: float

    clique_number_w: int


def _safe_ratio(num: float, den: float) -> float:
    if den > 0.0 and math.isfinite(num) and math.isfinite(den):
        return float(num / den)
    return float("nan")


def compute_instance_metrics(
    inst: AuctionInstance,
    outcome: MechanismOutcome,
    discount_rate_r_per_ms: float,
) -> InstanceMetrics:
    feasible = [b for b in inst.bids if b.arrival_ms <= inst.horizon_ms]
    bidder_count = len(inst.bids)
    feasible_count = len(feasible)
    infeasible_count = bidder_count - feasible_count
    feasible_fraction = (feasible_count / bidder_count) if bidder_count > 0 else float("nan")

    # Revised-benchmark bookkeeping:
    # - opt_all_value      : best value anywhere in the instance (legacy paper denominator)
    # - opt_feasible_value : best value among causally feasible bids only
    opt_all_value = max((float(b.value) for b in inst.bids), default=0.0)
    opt_feasible_value = max((float(b.value) for b in feasible), default=0.0)
    feasible_opt_coverage = _safe_ratio(opt_feasible_value, opt_all_value)

    winner_value = 0.0
    if outcome.winner_id is not None:
        for b in inst.bids:
            if b.bidder_id == outcome.winner_id:
                winner_value = float(b.value)
                break

    welfare = float(winner_value)
    payment = float(outcome.payment)

    welfare_ratio_all = _safe_ratio(welfare, opt_all_value)
    welfare_ratio_feasible = _safe_ratio(welfare, opt_feasible_value)
    revenue_ratio_all = _safe_ratio(payment, opt_all_value)
    revenue_ratio_feasible = _safe_ratio(payment, opt_feasible_value)

    # Backward-compatible aliases used by existing scripts.
    welfare_ratio = welfare_ratio_all
    revenue_ratio = revenue_ratio_all

    # Commit time (ms) returned by each mechanism already includes the mechanism's own
    # compute time (see mechanisms/*). Clearing latency follows the paper definition:
    # the time from the earliest bid emission to commitment.
    decision_time = float(outcome.decision_time_ms)
    min_emission = float(min((b.emission_ms for b in inst.bids), default=float("nan")))
    clearing_latency = float(decision_time - min_emission) if math.isfinite(min_emission) else float("nan")
    if math.isfinite(clearing_latency) and clearing_latency < 0.0:
        clearing_latency = 0.0

    effective_welfare = welfare * math.exp(-float(discount_rate_r_per_ms) * float(clearing_latency)) if math.isfinite(clearing_latency) else float("nan")

    intervals = [(b.arrival_ms, inst.horizon_ms) for b in feasible]
    w = clique_number(intervals) if intervals else 0

    return InstanceMetrics(
        topology=inst.topology_name,
        instance_id=inst.instance_id,
        bidder_count=bidder_count,
        feasible_count=int(feasible_count),
        infeasible_count=int(infeasible_count),
        feasible_fraction=float(feasible_fraction),
        mechanism=outcome.mechanism,
        horizon_ms=float(inst.horizon_ms),
        opt_value=float(opt_all_value),
        opt_all_value=float(opt_all_value),
        opt_feasible_value=float(opt_feasible_value),
        feasible_opt_coverage=float(feasible_opt_coverage),
        winner_id=outcome.winner_id,
        winner_value=float(winner_value),
        payment=payment,
        welfare_ratio=float(welfare_ratio),
        revenue_ratio=float(revenue_ratio),
        welfare_ratio_all=float(welfare_ratio_all),
        welfare_ratio_feasible=float(welfare_ratio_feasible),
        revenue_ratio_all=float(revenue_ratio_all),
        revenue_ratio_feasible=float(revenue_ratio_feasible),
        decision_time_ms=decision_time,
        clearing_latency_ms=float(clearing_latency),
        effective_welfare=float(effective_welfare),
        compute_time_s=float(outcome.compute_time_s),
        clique_number_w=int(w),
    )
