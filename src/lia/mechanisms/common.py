from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MechanismOutcome:
    mechanism: str
    winner_id: Optional[int]
    payment: float
    decision_time_ms: float
    compute_time_s: float


def utility(bid_value: float, won: bool, payment: float) -> float:
    if not won:
        return 0.0
    return float(bid_value - payment)
