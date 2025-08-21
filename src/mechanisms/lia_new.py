from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import numpy as np

class LIA(object):
    """
    Lorentz-Invariant Auction (single-item default).

    - Winner: arg max of discounted bids  θ_i * exp(-λ Δτ_i).
    - Payment (face value at winner's emission): (max_{j≠w} θ_j e^{-λ Δτ_j}) / exp(-λ Δτ_w).
      This is the Clarke pivot expressed in emission (face-value) units.

    Bidders dicts are expected to include:
      - 'valuation' : float   (θ_i)
      - timing fields sufficient to compute Δτ_i (see _horizon_slack()).

    We try multiple key fallbacks so this works with your current generator.
    """
    def __init__(self, lambda_param: float = 1.0, single_item: bool = True):
        self.lambda_param: float = float(lambda_param)
        self.single_item: bool = bool(single_item)
        self.name: str = f"LIA(λ={self.lambda_param:g})"

    # ---------- public API ----------
    def run(self,
            bidders: List[Dict[str, Any]],
            clearing_horizon: float) -> Dict[str, Any]:
        t0 = time.perf_counter()

        if not bidders:
            return {
                "mechanism": self.name, "winners": [], "payments": {},
                "welfare": 0.0, "revenue": 0.0,
                "optimal_welfare": 0.0,
                "efficiency_ratio": 0.0,
                "runtime": 0.0
            }

        # Compute discounted bids and select winner(s)
        if self.single_item:
            discounted = [self._discounted_bid(b, clearing_horizon) for b in bidders]
            if len(discounted) == 0:
                winners = []
            else:
                w = int(max(range(len(discounted)), key=lambda i: discounted[i]))
                winners = [w]
        else:
            # Fallback to interval-graph based routine if you ever enable k>1
            winners = self._fallback_interval_graph_winners(bidders, clearing_horizon)

        payments: Dict[int, float] = {}
        revenue = 0.0
        welfare = 0.0

        if winners:
            if self.single_item:
                w = winners[0]
                # Welfare under truthful values is just θ_w
                theta_w = float(bidders[w].get("valuation", 0.0))
                welfare = theta_w

                # Horizon threshold among others:
                discounted = [self._discounted_bid(b, clearing_horizon) for b in bidders]
                other_max = max([discounted[j] for j in range(len(bidders)) if j != w], default=0.0)

                # Winner's slack:
                slack_w = self._horizon_slack(bidders[w], clearing_horizon)
                disc_w  = self._safe_exp(-self.lambda_param * slack_w) if slack_w is not None else 1.0

                # Face-value payment at emission (divide threshold by discount)
                pay_w = other_max / disc_w
                pay_w = max(0.0, float(pay_w))
                payments[w] = pay_w
                revenue = pay_w
            else:
                # Multi-winner fallback (rarely used in this project)
                for w in winners:
                    payments[w] = 0.0  # define per your multi-winner policy
                revenue = sum(payments.values())
                welfare = float(sum(bidders[w].get("valuation", 0.0) for w in winners))

        optimal_welfare = float(max((b.get("valuation", 0.0) for b in bidders), default=0.0))
        eff_ratio = (welfare / optimal_welfare) if optimal_welfare > 0 else 0.0

        t1 = time.perf_counter()
        return {
            "mechanism": self.name,
            "winners": winners,
            "payments": payments,
            "welfare": welfare,
            "revenue": revenue,
            "optimal_welfare": optimal_welfare,
            "efficiency_ratio": eff_ratio,
            "runtime": (t1 - t0)
        }

    # ---------- helpers ----------
    def _discounted_bid(self, bidder: Dict[str, Any], clearing_horizon: float) -> float:
        theta = float(bidder.get("valuation", 0.0))
        slack = self._horizon_slack(bidder, clearing_horizon)
        disc  = self._safe_exp(-self.lambda_param * slack) if slack is not None else 1.0
        return theta * disc

    def _horizon_slack(self, bidder: Dict[str, Any], clearing_horizon: float) -> float:
        """
        Compute Δτ = τ_H - T_H(bidder). We support multiple field patterns to
        accommodate your current data generation.

        Accepted keys (best effort):
          - 'slack' (already Δτ)
          - 'arrival_time' (then Δτ = τ_H - arrival_time)
          - 'emission_time' + one of {'delay_to_horizon','prop_delay','latency','latency_ms'}
            (we convert ms to the same unit as clearing_horizon if needed)
        Fallback: Δτ = max(0, τ_H - emission_time).
        """
        if "slack" in bidder:
            return max(0.0, float(bidder["slack"]))

        tau_H = float(clearing_horizon)

        if "arrival_time" in bidder:
            arr = float(bidder["arrival_time"])
            return max(0.0, tau_H - arr)

        emit = float(bidder.get("emission_time", 0.0))

        # Preferred field names for one-way delay to horizon
        if "delay_to_horizon" in bidder:
            d = float(bidder["delay_to_horizon"])
            return max(0.0, tau_H - (emit + d))

        if "prop_delay" in bidder:
            d = float(bidder["prop_delay"])
            return max(0.0, tau_H - (emit + d))

        if "latency" in bidder:
            d = float(bidder["latency"])
            return max(0.0, tau_H - (emit + d))

        if "latency_ms" in bidder:
            d_ms = float(bidder["latency_ms"])
            # Assume clearing_horizon/emission_time are seconds; convert ms → s
            d_s  = d_ms / 1000.0
            return max(0.0, tau_H - (emit + d_s))

        # Check for 'propagation_delay' field (your current format)
        if "propagation_delay" in bidder:
            d = float(bidder["propagation_delay"])
            return max(0.0, tau_H - (emit + d))

        # Last resort
        return max(0.0, tau_H - emit)

    def _safe_exp(self, x: float) -> float:
        # Clamp exponent to avoid overflow/underflow
        return float(np.exp(np.clip(x, -700.0, 700.0)))

    # Optional: keep your multi-winner DP if ever needed
    def _fallback_interval_graph_winners(self,
                                         bidders: List[Dict[str, Any]],
                                         clearing_horizon: float) -> List[int]:
        # For single-item experiments, this path is not used.
        discounted = [self._discounted_bid(b, clearing_horizon) for b in bidders]
        if not discounted:
            return []
        return [int(max(range(len(discounted)), key=lambda i: discounted[i]))]
