"""
Internet topology generator for LIA experiments.

This module generates a realistic Internet backbone topology
with appropriate delay characteristics.
"""

import numpy as np
from typing import Dict, Any

def generate_internet_topology(num_nodes: int = 100,
                              num_regions: int = 5,
                              nodes_per_region: int = 20) -> Dict[str, Any]:
    """
    Generate an Internet backbone topology.
    
    Args:
        num_nodes: Total number of nodes in the network
        num_regions: Number of geographic regions
        nodes_per_region: Number of nodes per region
        
    Returns:
        Dictionary with topology information
    """
    # Constants
    speed_of_light = 299792.458  # km/s
    speed_in_fiber = speed_of_light * 0.7  # km/s (70% of c due to fiber medium)
    
    # Region centers (approximate coordinates for major global regions)
    region_centers = [
        (37.7749, -122.4194),  # North America (San Francisco)
        (51.5074, -0.1278),    # Europe (London)
        (35.6762, 139.6503),   # Asia (Tokyo)
        (-33.8688, 151.2093),  # Australia (Sydney)
        (-23.5505, -46.6333)   # South America (São Paulo)
    ]
    
    # Ensure we have enough region centers
    if num_regions > len(region_centers):
        raise ValueError(f"Only {len(region_centers)} region centers defined, but {num_regions} requested")
    
    # Create nodes
    nodes = []
    for region in range(num_regions):
        region_center = region_centers[region]
        
        for i in range(nodes_per_region):
            node_id = region * nodes_per_region + i
            
            # Add some random variation around the region center
            lat_variation = np.random.uniform(-5, 5)
            lon_variation = np.random.uniform(-5, 5)
            
            latitude = region_center[0] + lat_variation
            longitude = region_center[1] + lon_variation
            
            nodes.append({
                'id': node_id,
                'region': region,
                'coordinates': (latitude, longitude)
            })
    
    # Calculate delays between regions
    region_delays = {}
    for i in range(num_regions):
        region_delays[i] = {}
        for j in range(num_regions):
            if i == j:
                # Intra-region delay (low)
                region_delays[i][j] = np.random.uniform(0.3, 10)  # ms
            else:
                # Inter-region delay (high)
                # Calculate approximate distance between region centers
                lat1, lon1 = region_centers[i]
                lat2, lon2 = region_centers[j]
                
                # Simple approximation of distance using Euclidean distance
                # (for more accuracy, would use Haversine formula)
                distance_km = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111  # 1 degree ≈ 111 km
                
                # Convert distance to delay (ms)
                delay_ms = distance_km / speed_in_fiber * 1000
                
                # Add some random variation for routing inefficiencies
                delay_ms *= np.random.uniform(1.0, 1.3)
                
                region_delays[i][j] = delay_ms
    
    # Create a simplified delay list for the experiment
    simplified_delays = []
    for node in nodes:
        region = node['region']
        
        # Assign a delay based on node position
        if node['id'] % 3 == 0:
            # Close to auctioneer (same region, nearby)
            delay = np.random.uniform(0.3, 5)  # ms
        elif node['id'] % 3 == 1:
            # Medium distance (same region, farther)
            delay = np.random.uniform(5, 20)  # ms
        else:
            # Far from auctioneer (different region)
            target_region = (region + 1) % num_regions
            delay = region_delays[region][target_region]
        
        simplified_delays.append(delay)
    
    return {
        'name': 'INTERNET-100',
        'num_nodes': num_nodes,
        'num_regions': num_regions,
        'nodes_per_region': nodes_per_region,
        'nodes': nodes,
        'region_delays': region_delays,
        'delays': simplified_delays,
        'min_delay': min(simplified_delays),
        'max_delay': max(simplified_delays),
        'avg_delay': np.mean(simplified_delays),
        'std_delay': np.std(simplified_delays)
    }

