"""
HoldBack mechanism implementation.

This module implements the HoldBack mechanism, which introduces artificial delays
to equalize arrival times before running a standard VCG auction.
"""

import time
from typing import List, Dict, Any

class HoldBack:
    """
    HoldBack mechanism for single-item auctions.
    
    This mechanism holds all bids until the latest feasible bid arrives,
    then runs a standard VCG auction.
    """
    
    def __init__(self):
        """Initialize the HoldBack mechanism."""
        self.name = "HoldBack"
    
    def run(self, 
           bidders: List[Dict[str, Any]], 
           clearing_horizon: float,
           error_level: float = 0.0) -> Dict[str, Any]:
        """
        Run the HoldBack mechanism on the given auction instance.
        
        Args:
            bidders: List of bidder dictionaries
            clearing_horizon: The clearing horizon time
            error_level: Optional error level to add to measurements
            
        Returns:
            Dictionary with allocation results and metrics
        """
        start_time = time.time()
        
        # Step 1: Calculate arrival times for all bids
        arrival_times = []
        for bidder in bidders:
            # Add random error if specified (for robustness testing)
            if error_level > 0:
                error = error_level  # Worst-case error for HoldBack
            else:
                error = 0
                
            arrival_time = bidder['emission_time'] + bidder['propagation_delay'] + error
            arrival_times.append(arrival_time)
        
        # Step 2: Find the latest arrival time before the clearing horizon
        latest_arrival = max(arrival_times)
        if latest_arrival > clearing_horizon:
            # If the latest arrival is after the clearing horizon, no allocation
            return {
                'mechanism': self.name,
                'winners': [],
                'payments': [],
                'welfare': 0,
                'revenue': 0,
                'optimal_welfare': 0,
                'efficiency_ratio': 0,
                'revenue_ratio': 0,
                'runtime': time.time() - start_time
            }
        
        # Step 3: Determine which bids are feasible (arrive before the clearing horizon)
        feasible_bidders = []
        for i, bidder in enumerate(bidders):
            if arrival_times[i] <= clearing_horizon:
                feasible_bidders.append(bidder)
        
        # Extract valuations of feasible bidders
        valuations = [bidder['valuation'] for bidder in feasible_bidders]
        
        # Find the highest bidder among feasible bidders
        if not valuations:
            return {
                'mechanism': self.name,
                'winners': [],
                'payments': [],
                'welfare': 0,
                'revenue': 0,
                'optimal_welfare': 0,
                'efficiency_ratio': 0,
                'revenue_ratio': 0,
                'runtime': time.time() - start_time
            }
        
        winner_index = max(range(len(valuations)), key=valuations.__getitem__)
        winner_global_index = bidders.index(feasible_bidders[winner_index])
        winners = [winner_global_index]
        
        # Calculate payment (second-highest bid among feasible bidders)
        if len(valuations) > 1:
            # Create a copy of valuations without the winner's bid
            other_valuations = valuations.copy()
            other_valuations.pop(winner_index)
            second_price = max(other_valuations) if other_valuations else 0
        else:
            second_price = 0
        
        # Set up payments list (0 for everyone except the winner)
        payments = [0] * len(bidders)
        payments[winner_global_index] = second_price
        
        # Calculate welfare and revenue
        welfare = bidders[winner_global_index]['valuation']
        revenue = second_price
        
        # Calculate optimal welfare (for efficiency ratio)
        optimal_welfare = max(bidder['valuation'] for bidder in bidders)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        return {
            'mechanism': self.name,
            'winners': winners,
            'payments': payments,
            'welfare': welfare,
            'revenue': revenue,
            'optimal_welfare': optimal_welfare,
            'efficiency_ratio': welfare / optimal_welfare if optimal_welfare > 0 else 0,
            'revenue_ratio': revenue / optimal_welfare if optimal_welfare > 0 else 0,
            'runtime': runtime,
            'holdback_time': latest_arrival
        }

