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
    
    def calculate_horizon_slack(self, 
                               bidder: Dict[str, Any], 
                               clearing_horizon: float) -> float:
        """
        Calculate the horizon slack for a bidder.
        
        The horizon slack is the proper time between the bid's arrival at the
        auctioneer and the clearing horizon.
        
        Args:
            bidder: Bidder dictionary with emission_time and propagation_delay
            clearing_horizon: Time at which the auction clears
            
        Returns:
            Horizon slack in the same time units as emission_time and propagation_delay
        """
        arrival_time = bidder['emission_time'] + bidder['propagation_delay']
        horizon_slack = clearing_horizon - arrival_time
        
        # Ensure non-negative slack
        return max(0, horizon_slack)
    
    def calculate_discounted_bid(self, 
                                bidder: Dict[str, Any], 
                                clearing_horizon: float) -> float:
        """
        Calculate the horizon-discounted bid value.
        
        The discounted bid is the original valuation multiplied by an exponential
        discount factor based on the horizon slack.
        
        Args:
            bidder: Bidder dictionary with valuation, emission_time, and propagation_delay
            clearing_horizon: Time at which the auction clears
            
        Returns:
            Discounted bid value
        """
        horizon_slack = self.calculate_horizon_slack(bidder, clearing_horizon)
        discount_factor = np.exp(-self.lambda_param * horizon_slack)
        return bidder['valuation'] * discount_factor
    
    def find_maximum_weight_independent_set(self, 
                                          bidders: List[Dict[str, Any]], 
                                          clearing_horizon: float) -> List[int]:
        """
        Find the maximum weight independent set in the interval graph.
        
        This implements the interval graph algorithm for finding the maximum
        weight independent set, which is used to determine the auction winners.
        
        Args:
            bidders: List of bidder dictionaries
            clearing_horizon: Time at which the auction clears
            
        Returns:
            List of bidder indices in the maximum weight independent set
        """
        # Calculate discounted bids and arrival times
        discounted_bids = []
        arrival_times = []
        
        for bidder in bidders:
            discounted_bid = self.calculate_discounted_bid(bidder, clearing_horizon)
            arrival_time = bidder['emission_time'] + bidder['propagation_delay']
            
            discounted_bids.append(discounted_bid)
            arrival_times.append(arrival_time)
        
        # Sort bidders by arrival time
        sorted_indices = np.argsort(arrival_times)
        
        # Dynamic programming approach to find maximum weight independent set
        n = len(bidders)
        dp = [0] * (n + 1)  # dp[i] = max weight of independent set ending at i
        prev = [-1] * (n + 1)  # prev[i] = previous bidder in the optimal solution
        
        # Base case: empty set
        dp[0] = 0
        
        # Fill dp table
        for i in range(1, n + 1):
            bidder_idx = sorted_indices[i - 1]
            current_bid = discounted_bids[bidder_idx]
            current_arrival = arrival_times[bidder_idx]
            
            # Option 1: Don't include current bidder
            option1 = dp[i - 1]
            
            # Option 2: Include current bidder
            option2 = current_bid
            
            # Find the latest non-overlapping bidder
            j = i - 1
            while j > 0:
                prev_idx = sorted_indices[j - 1]
                prev_arrival = arrival_times[prev_idx]
                
                # Check if bidders are compatible (non-overlapping)
                if prev_arrival <= current_arrival - bidders[prev_idx]['propagation_delay']:
                    option2 += dp[j]
                    prev[i] = j
                    break
                
                j -= 1
            
            # Choose the better option
            if option1 >= option2:
                dp[i] = option1
                prev[i] = prev[i - 1]
            else:
                dp[i] = option2
            
        # Reconstruct the solution
        winners = []
        i = n
        while i > 0:
            if prev[i] != prev[i - 1]:
                winners.append(sorted_indices[i - 1])
            i = prev[i]
        
        return winners
    
    def calculate_payments(self, 
                          bidders: List[Dict[str, Any]], 
                          winners: List[int], 
                          clearing_horizon: float) -> Dict[int, float]:
        """
        Calculate payments for the auction winners using critical values.
        
        This implements the payment rule for the LIA mechanism, which ensures
        truthfulness by charging each winner their critical value.
        
        Args:
            bidders: List of bidder dictionaries
            winners: List of indices of winning bidders
            clearing_horizon: Time at which the auction clears
            
        Returns:
            Dictionary mapping winner indices to their payments
        """
        payments = {}
        
        # For each winner, find their critical value
        for winner_idx in winners:
            winner = bidders[winner_idx]
            original_valuation = winner['valuation']
            
            # Binary search to find the critical value
            low = 0
            high = original_valuation * 2  # Upper bound on critical value
            
            # Precision for binary search
            epsilon = 1e-6
            
            while high - low > epsilon:
                mid = (low + high) / 2
                
                # Create a modified bidder with the test valuation
                test_bidder = winner.copy()
                test_bidder['valuation'] = mid
                
                # Replace the winner with the test bidder
                test_bidders = [bidder.copy() for bidder in bidders]
                test_bidders[winner_idx] = test_bidder
                
                # Run the auction with the test bidder
                test_winners = self.find_maximum_weight_independent_set(test_bidders, clearing_horizon)
                
                # Check if the bidder still wins
                if winner_idx in test_winners:
                    # If still winning, critical value might be lower
                    high = mid
                else:
                    # If not winning, critical value must be higher
                    low = mid
            
            # The critical value is the converged value from binary search
            critical_value = high
            
            # Calculate the payment based on the critical value and discount factor
            horizon_slack = self.calculate_horizon_slack(winner, clearing_horizon)
            discount_factor = np.exp(-self.lambda_param * horizon_slack)
            payment = critical_value * discount_factor
            
            payments[winner_idx] = payment
        
        return payments
    
    def calculate_welfare(self, 
                         bidders: List[Dict[str, Any]], 
                         winners: List[int]) -> float:
        """
        Calculate the social welfare of the auction outcome.
        
        The social welfare is the sum of the valuations of the winning bidders.
        
        Args:
            bidders: List of bidder dictionaries
            winners: List of indices of winning bidders
            
        Returns:
            Social welfare value
        """
        return sum(bidders[idx]['valuation'] for idx in winners)
    
    def calculate_revenue(self, payments: Dict[int, float]) -> float:
        """
        Calculate the revenue of the auction outcome.
        
        The revenue is the sum of the payments from the winning bidders.
        
        Args:
            payments: Dictionary mapping winner indices to their payments
            
        Returns:
            Revenue value
        """
        return sum(payments.values())
    
    def calculate_optimal_welfare(self, bidders: List[Dict[str, Any]]) -> float:
        """
        Calculate the optimal social welfare (ignoring delays).
        
        The optimal welfare is the maximum possible sum of valuations
        that could be achieved if all bids were received simultaneously.
        
        Args:
            bidders: List of bidder dictionaries
            
        Returns:
            Optimal welfare value
        """
        # Sort bidders by valuation in descending order
        sorted_bidders = sorted(bidders, key=lambda b: b['valuation'], reverse=True)
        
        # Take the highest valuation
        return sorted_bidders[0]['valuation'] if sorted_bidders else 0
    
    def calculate_latency_arbitrage_index(self, 
                                         bidders: List[Dict[str, Any]], 
                                         clearing_horizon: float,
                                         advance_ms: float = 1.0) -> float:
        """
        Calculate the Latency Arbitrage Index (LAI).
        
        The LAI measures the expected utility gain from reducing delay by 1 ms.
        
        Args:
            bidders: List of bidder dictionaries
            clearing_horizon: Time at which the auction clears
            advance_ms: Amount of advance in milliseconds to simulate
            
        Returns:
            LAI value
        """
        # Run the original auction
        original_winners = self.find_maximum_weight_independent_set(bidders, clearing_horizon)
        original_payments = self.calculate_payments(bidders, original_winners, clearing_horizon)
        original_welfare = self.calculate_welfare(bidders, original_winners)
        
        # Calculate utility for each bidder in the original auction
        original_utilities = {}
        for idx in range(len(bidders)):
            if idx in original_winners:
                original_utilities[idx] = bidders[idx]['valuation'] - original_payments[idx]
            else:
                original_utilities[idx] = 0
        
        # Simulate advancing each bidder by advance_ms
        total_utility_gain = 0
        
        for i in range(len(bidders)):
            # Create a copy of bidders with bidder i advanced
            advanced_bidders = [b.copy() for b in bidders]
            advanced_bidders[i]['propagation_delay'] -= advance_ms / 1000  # Convert ms to same units as propagation_delay
            
            # Ensure non-negative propagation delay
            advanced_bidders[i]['propagation_delay'] = max(0, advanced_bidders[i]['propagation_delay'])
            
            # Run the auction with the advanced bidder
            advanced_winners = self.find_maximum_weight_independent_set(advanced_bidders, clearing_horizon)
            advanced_payments = self.calculate_payments(advanced_bidders, advanced_winners, clearing_horizon)
            
            # Calculate utility for bidder i in the advanced auction
            if i in advanced_winners:
                advanced_utility = bidders[i]['valuation'] - advanced_payments[i]
            else:
                advanced_utility = 0
            
            # Calculate utility gain
            utility_gain = advanced_utility - original_utilities[i]
            total_utility_gain += utility_gain
        
        # Normalize by the optimal welfare and the number of bidders
        normalized_lai = total_utility_gain / (self.calculate_optimal_welfare(bidders) * len(bidders))
        
        return normalized_lai
    
    def run(self, 
           bidders: List[Dict[str, Any]], 
           clearing_horizon: float,
           compute_lai: bool = False,
           advance_ms: float = 1.0,
           error_level: float = 0.0) -> Dict[str, Any]:
        """
        Run the LIA mechanism on the given bidders.
        
        Args:
            bidders: List of bidder dictionaries
            clearing_horizon: Time at which the auction clears
            compute_lai: Whether to compute the Latency Arbitrage Index
            advance_ms: Amount of advance in milliseconds for LAI calculation
            error_level: Error level in milliseconds to add to propagation delays
            
        Returns:
            Dictionary with auction results
        """
        start_time = time.time()
        
        # Add error to propagation delays if specified
        if error_level > 0:
            bidders_with_error = []
            for bidder in bidders:
                bidder_with_error = bidder.copy()
                # Add random error in range [-error_level, error_level]
                error = np.random.uniform(-error_level, error_level)
                bidder_with_error['propagation_delay'] += error / 1000  # Convert ms to same units as propagation_delay
                # Ensure non-negative propagation delay
                bidder_with_error['propagation_delay'] = max(0, bidder_with_error['propagation_delay'])
                bidders_with_error.append(bidder_with_error)
            bidders = bidders_with_error
        
        # Find winners
        winners = self.find_maximum_weight_independent_set(bidders, clearing_horizon)
        
        # Calculate payments
        payments = self.calculate_payments(bidders, winners, clearing_horizon)
        
        # Calculate welfare and revenue
        welfare = self.calculate_welfare(bidders, winners)
        revenue = self.calculate_revenue(payments)
        
        # Calculate optimal welfare
        optimal_welfare = self.calculate_optimal_welfare(bidders)
        
        # Calculate efficiency and revenue ratios
        efficiency_ratio = welfare / optimal_welfare if optimal_welfare > 0 else 0
        revenue_ratio = revenue / optimal_welfare if optimal_welfare > 0 else 0
        
        # Calculate LAI if requested
        lai = None
        if compute_lai:
            lai = self.calculate_latency_arbitrage_index(bidders, clearing_horizon, advance_ms)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Prepare result dictionary
        result = {
            'mechanism': 'LIA',
            'lambda': self.lambda_param,
            'winners': winners,
            'payments': payments,
            'welfare': welfare,
            'revenue': revenue,
            'optimal_welfare': optimal_welfare,
            'efficiency_ratio': efficiency_ratio,
            'revenue_ratio': revenue_ratio,
            'runtime': runtime
        }
        
        if lai is not None:
            result['lai'] = lai
        
        return result

