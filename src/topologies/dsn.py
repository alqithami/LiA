"""
Deep Space Network topology generator for LIA experiments.

This module generates a realistic Deep Space Network topology
with appropriate delay characteristics for interplanetary communication.
"""

import numpy as np
from typing import Dict, Any

def generate_dsn_topology(num_nodes: int = 30,
                         earth_stations: int = 5,
                         relay_satellites: int = 10,
                         deep_space_probes: int = 15) -> Dict[str, Any]:
    """
    Generate a Deep Space Network topology.
    
    Args:
        num_nodes: Total number of nodes in the network
        earth_stations: Number of Earth ground stations
        relay_satellites: Number of relay satellites
        deep_space_probes: Number of deep space probes
        
    Returns:
        Dictionary with topology information
    """
    # Constants
    speed_of_light = 299792.458  # km/s
    
    # Astronomical distances (in km)
    earth_orbit_radius = 42164  # Geostationary orbit
    moon_distance = 384400
    mars_min_distance = 54.6e6
    mars_max_distance = 401e6
    jupiter_min_distance = 588e6
    jupiter_max_distance = 968e6
    
    # Create nodes
    nodes = []
    
    # Earth stations
    for i in range(earth_stations):
        nodes.append({
            'id': i,
            'type': 'earth_station',
            'location': f'Earth Station {i+1}',
            'distance_km': 0  # On Earth
        })
    
    # Relay satellites
    for i in range(relay_satellites):
        node_id = earth_stations + i
        
        # Determine satellite type/orbit
        if i < 3:
            # Geostationary satellites
            distance = earth_orbit_radius
            location = f'GEO Relay {i+1}'
        elif i < 6:
            # Lunar orbit satellites
            distance = moon_distance
            location = f'Lunar Relay {i-2}'
        else:
            # Lagrange point satellites
            distance = earth_orbit_radius * 1.5
            location = f'L-Point Relay {i-5}'
        
        nodes.append({
            'id': node_id,
            'type': 'relay_satellite',
            'location': location,
            'distance_km': distance
        })
    
    # Deep space probes
    for i in range(deep_space_probes):
        node_id = earth_stations + relay_satellites + i
        
        # Determine probe location
        if i < 5:
            # Mars vicinity
            distance = np.random.uniform(mars_min_distance, mars_max_distance)
            location = f'Mars Probe {i+1}'
        elif i < 10:
            # Jupiter vicinity
            distance = np.random.uniform(jupiter_min_distance, jupiter_max_distance)
            location = f'Jupiter Probe {i-4}'
        else:
            # Other deep space locations
            distance = np.random.uniform(mars_max_distance, jupiter_min_distance)
            location = f'Deep Space Probe {i-9}'
        
        nodes.append({
            'id': node_id,
            'type': 'deep_space_probe',
            'location': location,
            'distance_km': distance
        })
    
    # Calculate one-way light-time delays
    delays = []
    for node in nodes:
        # Convert distance to delay (ms)
        delay_ms = node['distance_km'] / speed_of_light * 1000
        delays.append(delay_ms)
    
    return {
        'name': 'DSN-30',
        'num_nodes': num_nodes,
        'earth_stations': earth_stations,
        'relay_satellites': relay_satellites,
        'deep_space_probes': deep_space_probes,
        'nodes': nodes,
        'delays': delays,
        'min_delay': min(delays),
        'max_delay': max(delays),
        'avg_delay': np.mean(delays),
        'std_delay': np.std(delays)
    }

