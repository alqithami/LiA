"""
Vickrey-Clarke-Groves (VCG) mechanism implementation.

This module implements the standard VCG mechanism for single-item auctions,
which allocates to the highest bidder and charges the second-highest bid.
"""

import time
from typing import List, Dict, Any

class VCG:
    """
    Vickrey-Clarke-Groves (VCG) mechanism for single-item auctions.
    
    This is a standard second-price auction that ignores delay heterogeneity.
    """
    
    def __init__(self):
        """Initialize the VCG mechanism."""
        self.name = "VCG"
    
    def run(self, 
           bidders: List[Dict[str, Any]], 
           clearing_horizon: float,
           error_level: float = 0.0) -> Dict[str, Any]:
        """
        Run the VCG mechanism on the given auction instance.
        
        Args:
            bidders: List of bidder dictionaries
            clearing_horizon: The clearing horizon time (ignored in VCG)
            error_level: Optional error level (ignored in VCG)
            
        Returns:
            Dictionary with allocation results and metrics
        """
        start_time = time.time()
        
        # Extract valuations
        valuations = [bidder['valuation'] for bidder in bidders]
        
        # Find the highest bidder
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
        winners = [winner_index]
        
        # Calculate payment (second-highest bid)
        if len(valuations) > 1:
            # Create a copy of valuations without the winner's bid
            other_valuations = valuations.copy()
            other_valuations.pop(winner_index)
            second_price = max(other_valuations) if other_valuations else 0
        else:
            second_price = 0
        
        # Set up payments list (0 for everyone except the winner)
        payments = [0] * len(bidders)
        payments[winner_index] = second_price
        
        # Calculate welfare and revenue
        welfare = valuations[winner_index]
        revenue = second_price
        
        # Calculate optimal welfare (for efficiency ratio)
        optimal_welfare = max(valuations)
        
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
            'runtime': runtime
        }

