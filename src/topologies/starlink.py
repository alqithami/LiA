"""
Starlink topology generator for LIA experiments.

This module generates a realistic Starlink-like LEO satellite constellation
topology with appropriate delay characteristics.
"""

import numpy as np
from typing import Dict, Any

def generate_starlink_topology(num_nodes: int = 200,
                              num_orbital_planes: int = 10,
                              nodes_per_plane: int = 20,
                              altitude_km: float = 550) -> Dict[str, Any]:
    """
    Generate a Starlink-like LEO satellite constellation topology.
    
    Args:
        num_nodes: Total number of nodes in the constellation
        num_orbital_planes: Number of orbital planes
        nodes_per_plane: Number of satellites per orbital plane
        altitude_km: Altitude of the constellation in kilometers
        
    Returns:
        Dictionary with topology information
    """
    # Constants
    earth_radius_km = 6371  # km
    speed_of_light = 299792.458  # km/s
    
    # Create nodes
    nodes = []
    for plane in range(num_orbital_planes):
        for pos in range(nodes_per_plane):
            node_id = plane * nodes_per_plane + pos
            
            # Calculate position in orbital plane
            theta = 2 * np.pi * pos / nodes_per_plane  # angle in orbital plane
            phi = np.pi * plane / num_orbital_planes  # orbital plane angle
            
            # Convert to Cartesian coordinates
            r = earth_radius_km + altitude_km
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            nodes.append({
                'id': node_id,
                'plane': plane,
                'position': pos,
                'coordinates': (x, y, z)
            })
    
    # Calculate delays between nodes
    delays = {}
    for node in nodes:
        # Calculate delay from this node to all other nodes
        node_delays = {}
        
        for other_node in nodes:
            if node['id'] == other_node['id']:
                # Self-delay is 0
                node_delays[other_node['id']] = 0
            else:
                # Calculate Euclidean distance
                x1, y1, z1 = node['coordinates']
                x2, y2, z2 = other_node['coordinates']
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                
                # Convert distance to delay (ms)
                delay_ms = distance / speed_of_light * 1000
                node_delays[other_node['id']] = delay_ms
        
        delays[node['id']] = node_delays
    
    # Create a simplified delay list for the experiment
    simplified_delays = []
    for node in nodes:
        # Use the average delay to other nodes in the same plane
        same_plane = [n for n in nodes if n['plane'] == node['plane'] and n['id'] != node['id']]
        if same_plane:
            same_plane_delays = [delays[node['id']][n['id']] for n in same_plane]
            avg_same_plane_delay = np.mean(same_plane_delays)
        else:
            avg_same_plane_delay = 0
        
        # Use the average delay to nodes in different planes
        diff_plane = [n for n in nodes if n['plane'] != node['plane']]
        if diff_plane:
            diff_plane_delays = [delays[node['id']][n['id']] for n in diff_plane]
            avg_diff_plane_delay = np.mean(diff_plane_delays)
        else:
            avg_diff_plane_delay = 0
        
        # Assign a delay based on node position
        if node['id'] % 3 == 0:
            # Close to auctioneer
            delay = 1.8  # ms
        elif node['id'] % 3 == 1:
            # Medium distance
            delay = avg_same_plane_delay
        else:
            # Far from auctioneer
            delay = avg_diff_plane_delay
        
        simplified_delays.append(delay)
    
    return {
        'name': 'STARLINK-200',
        'num_nodes': num_nodes,
        'num_orbital_planes': num_orbital_planes,
        'nodes_per_plane': nodes_per_plane,
        'altitude_km': altitude_km,
        'nodes': nodes,
        'delays': simplified_delays,
        'min_delay': min(simplified_delays),
        'max_delay': max(simplified_delays),
        'avg_delay': np.mean(simplified_delays),
        'std_delay': np.std(simplified_delays)
    }

