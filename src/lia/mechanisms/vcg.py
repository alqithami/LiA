from __future__ import annotations

import time
from typing import List, Optional, Tuple

from lia.experiment.instance_generator import AuctionInstance, Bid
from lia.mechanisms.common import MechanismOutcome


def _second_price_winner(eligible: List[Bid]) -> Tuple[Optional[int], float]:
    """Single-item VCG equals a second-price auction.

    Implemented in O(n) time (no full sort) for scalability experiments.

    Tie-break: smallest bidder_id among the top-value bidders.
    """

    if not eligible:
        return None, 0.0

    tol = 1e-12
    best_id: Optional[int] = None
    best_val: float = -1.0
    second_val: float = -1.0

    for b in eligible:
        v = float(b.value)

        if best_id is None:
            best_id = int(b.bidder_id)
            best_val = v
            continue

        if v > best_val + tol:
            second_val = best_val
            best_val = v
            best_id = int(b.bidder_id)
        elif abs(v - best_val) <= tol:
            # Tie at the top: the second-highest equals the top.
            second_val = best_val
            if int(b.bidder_id) < int(best_id):
                best_id = int(b.bidder_id)
        elif v > second_val + tol:
            second_val = v

    payment = float(max(0.0, second_val))
    return best_id, payment


def run_sync_vcg(inst: AuctionInstance) -> MechanismOutcome:
    start = time.perf_counter()
    eligible = [b for b in inst.bids if b.arrival_ms <= inst.horizon_ms]
    winner, payment = _second_price_winner(eligible)
    end = time.perf_counter()
    compute_s = float(end - start)
    base_decision_ms = float(inst.horizon_ms)
    return MechanismOutcome(
        mechanism="SyncVCG",
        winner_id=winner,
        payment=payment,
        decision_time_ms=float(base_decision_ms + 1000.0 * compute_s),
        compute_time_s=compute_s,
    )


def run_holdback(inst: AuctionInstance) -> MechanismOutcome:
    """HoldBack baseline: buffer bids until the horizon, then clear via second-price."""

    start = time.perf_counter()
    eligible = [b for b in inst.bids if b.arrival_ms <= inst.horizon_ms]
    winner, payment = _second_price_winner(eligible)
    end = time.perf_counter()
    compute_s = float(end - start)
    base_decision_ms = float(inst.horizon_ms)
    return MechanismOutcome(
        mechanism="HoldBack",
        winner_id=winner,
        payment=payment,
        decision_time_ms=float(base_decision_ms + 1000.0 * compute_s),
        compute_time_s=compute_s,
    )


def run_fast_vcg(inst: AuctionInstance) -> MechanismOutcome:
    """FastVCG baseline: clear immediately when the first bid arrives."""

    start = time.perf_counter()
    if inst.bids:
        min_arrival = min(b.arrival_ms for b in inst.bids)
    else:
        min_arrival = float(inst.horizon_ms)

    decision = float(min(inst.horizon_ms, min_arrival))
    eligible = [b for b in inst.bids if b.arrival_ms <= decision]
    winner, payment = _second_price_winner(eligible)
    end = time.perf_counter()
    compute_s = float(end - start)
    return MechanismOutcome(
        mechanism="FastVCG",
        winner_id=winner,
        payment=payment,
        decision_time_ms=float(decision + 1000.0 * compute_s),
        compute_time_s=compute_s,
    )


def run_batch_vcg(inst: AuctionInstance, batch_ms: float) -> MechanismOutcome:
    """BatchVCG baseline: clear after a fixed interval (ms) from the first arrival."""

    start = time.perf_counter()
    if inst.bids:
        min_arrival = min(b.arrival_ms for b in inst.bids)
    else:
        min_arrival = float(inst.horizon_ms)

    decision = float(min(inst.horizon_ms, min_arrival + float(batch_ms)))
    eligible = [b for b in inst.bids if b.arrival_ms <= decision]
    winner, payment = _second_price_winner(eligible)
    end = time.perf_counter()
    compute_s = float(end - start)
    return MechanismOutcome(
        mechanism=f"BatchVCG_B{float(batch_ms):g}ms",
        winner_id=winner,
        payment=payment,
        decision_time_ms=float(decision + 1000.0 * compute_s),
        compute_time_s=compute_s,
    )
