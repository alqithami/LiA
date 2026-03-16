from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

from lia.experiment.instance_generator import AuctionInstance, Bid
from lia.mechanisms.common import MechanismOutcome


@dataclass(frozen=True)
class LiaConfig:
    """Configuration for the Lorentz-Invariant Auction (single-item)."""

    # Interpreted according to `lambda_unit`.
    lambda_value: float

    # "per_ms": lambda has units ms^-1
    # "per_s":  lambda has units s^-1  (internally converted to ms^-1)
    # "normalized_by_horizon": lambda_eff = lambda_value / horizon_ms
    lambda_unit: str = "per_s"

    # If True, the mechanism uses the perturbed/estimated slack (delta-hat) for discounting.
    use_estimated_slack: bool = True

    # If True and `use_estimated_slack`, clamp delta-hat to [0, horizon_ms] to avoid
    # physically impossible negative slacks from bounded noise.
    clamp_estimated_slack: bool = True


def _lambda_per_ms(cfg: LiaConfig, horizon_ms: float) -> float:
    """Return lambda in ms^-1."""

    if cfg.lambda_unit == "per_ms":
        return float(cfg.lambda_value)
    if cfg.lambda_unit == "per_s":
        return float(cfg.lambda_value) / 1000.0
    if cfg.lambda_unit == "normalized_by_horizon":
        return float(cfg.lambda_value) / max(1e-9, float(horizon_ms))
    raise ValueError(f"Unknown lambda_unit: {cfg.lambda_unit}")


def _clamp_slack_ms(slack_ms: float, horizon_ms: float) -> float:
    s = float(slack_ms)
    if s < 0.0:
        return 0.0
    if s > float(horizon_ms):
        return float(horizon_ms)
    return s


def run_lia(inst: AuctionInstance, cfg: LiaConfig) -> MechanismOutcome:
    """Lorentz-Invariant Auction (single-item).\n\n    Discounted bid:  b_i * exp(-lambda * delta_i)\n    Winner:          argmax_i discounted_bid\n    Payment:         critical value, equivalent to\n                    (second-highest discounted bid) / exp(-lambda * delta_w)\n\n    This implementation is numerically stable by operating in log space.
    """

    start = time.perf_counter()

    # Only bids that physically arrive by the horizon are eligible.
    eligible: List[Bid] = [b for b in inst.bids if b.arrival_ms <= inst.horizon_ms]

    # Decision time policy (paper): LIA does **not** add artificial buffering;
    # it can clear as soon as the last *eligible* bid has arrived. We then add
    # the mechanism's own compute time to obtain the commit time.
    base_decision_ms = float(max((b.arrival_ms for b in eligible), default=float(inst.horizon_ms)))

    if not eligible:
        end = time.perf_counter()
        compute_s = float(end - start)
        return MechanismOutcome(
            mechanism=f"LIA_lambda{cfg.lambda_value:g}_{cfg.lambda_unit}",
            winner_id=None,
            payment=0.0,
            decision_time_ms=float(base_decision_ms + 1000.0 * compute_s),
            compute_time_s=compute_s,
        )

    lam_ms = _lambda_per_ms(cfg, horizon_ms=inst.horizon_ms)

    best_id: Optional[int] = None
    best_score = -math.inf
    best_slack = 0.0
    second_score = -math.inf

    # Tight tie tolerance for floating-point comparisons.
    tol = 1e-12

    for b in eligible:
        slack = b.slack_est_ms if cfg.use_estimated_slack else b.slack_true_ms
        if cfg.use_estimated_slack and cfg.clamp_estimated_slack:
            slack = _clamp_slack_ms(slack, horizon_ms=inst.horizon_ms)

        v = float(b.value)
        if v <= 0.0:
            score = -math.inf
        else:
            # log( v * exp(-lam*slack) ) = log(v) - lam*slack
            score = math.log(v) - lam_ms * float(slack)

        if score > best_score + tol:
            second_score = best_score
            best_score = score
            best_id = int(b.bidder_id)
            best_slack = float(slack)
        elif abs(score - best_score) <= tol:
            # Tie at the top: apply fixed ID-based tie-break (paper uses fixed IDs).
            # In a tie, the best "other" discounted bid equals the best score.
            second_score = best_score
            if best_id is None or int(b.bidder_id) < best_id:
                best_id = int(b.bidder_id)
                best_slack = float(slack)
        elif score > second_score + tol:
            second_score = score

    if best_id is None or best_score == -math.inf:
        end = time.perf_counter()
        compute_s = float(end - start)
        return MechanismOutcome(
            mechanism=f"LIA_lambda{cfg.lambda_value:g}_{cfg.lambda_unit}",
            winner_id=None,
            payment=0.0,
            decision_time_ms=float(base_decision_ms + 1000.0 * compute_s),
            compute_time_s=compute_s,
        )

    if second_score == -math.inf:
        payment = 0.0
    else:
        # payment = exp(second_score) / exp(-lam*delta_w)
        #         = exp(second_score + lam*delta_w)
        log_payment = float(second_score + lam_ms * best_slack)
        # Hard clamp for numerical safety.
        log_payment = max(-700.0, min(700.0, log_payment))
        payment = float(math.exp(log_payment))

    end = time.perf_counter()
    compute_s = float(end - start)
    return MechanismOutcome(
        mechanism=f"LIA_lambda{cfg.lambda_value:g}_{cfg.lambda_unit}",
        winner_id=best_id,
        payment=payment,
        decision_time_ms=float(base_decision_ms + 1000.0 * compute_s),
        compute_time_s=compute_s,
    )
