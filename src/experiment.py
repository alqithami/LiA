"""
Main experiment runner for the LIA mechanism evaluation.

This module orchestrates the experiments described in the paper:
"Latency-Aware Resource Allocation over Heterogeneous Networks: 
A Lorentz-Invariant Market Mechanism with Interval-Graph Algorithms"
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

# Import mechanisms
from src.mechanisms.lia import LIA
from src.mechanisms.vcg import VCG
from src.mechanisms.holdback import HoldBack

# Import topology generators
from src.topologies.starlink import generate_starlink_topology
from src.topologies.internet import generate_internet_topology
from src.topologies.dsn import generate_dsn_topology


def compute_lai_for_mechanism(mechanism, bidders, clearing_horizon, advance_ms=1.0):
    """
    Proper-time LAI for a mechanism: for each bidder i, advance emission by 'advance_ms'
    (same time units as your delays) and re-run; record max(0, u_i(advanced) - u_i(orig)).
    Returns the average LAI across bidders.
    """
    import copy
    import numpy as np

    lai_values = []
    for i in range(len(bidders)):
        # Baseline
        res0 = mechanism.run(bidders=bidders, clearing_horizon=clearing_horizon)
        winners0 = set(res0.get('winners', []))
        payments0 = res0.get('payments', {})
        theta_i = float(bidders[i].get('valuation', 0.0))
        u0 = (theta_i - payments0.get(i, 0.0)) if (i in winners0) else 0.0

        # Advance emission of i
        bidders_adv = copy.deepcopy(bidders)
        bidders_adv[i]['emission_time'] = max(
            0.0, float(bidders[i].get('emission_time', 0.0)) - float(advance_ms)
        )
        res1 = mechanism.run(bidders=bidders_adv, clearing_horizon=clearing_horizon)
        winners1 = set(res1.get('winners', []))
        payments1 = res1.get('payments', {})
        u1 = (theta_i - payments1.get(i, 0.0)) if (i in winners1) else 0.0

        lai_values.append(max(0.0, u1 - u0))
    return float(np.mean(lai_values)) if lai_values else 0.0

class ExperimentRunner:
    """Main class for running experiments."""
    
    def __init__(self, config_data):
        """
        Initialize the experiment runner.
        
        Args:
            config_data: Configuration dictionary or path to config file
        """
        # Load configuration
        if isinstance(config_data, str):
            with open(config_data, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config_data
        
        # Create output directories
        output_dir = self.config['general']['output_dir']
        if not output_dir.endswith('/'):
            output_dir += '/'
            
        os.makedirs(output_dir + 'data', exist_ok=True)
        os.makedirs(output_dir + 'figures', exist_ok=True)
        os.makedirs(output_dir + 'tables', exist_ok=True)
        
        # Initialize mechanisms
        default_lambda = self.config['basic_experiments']['default_lambda']
        self.mechanisms = {
            'LIA': LIA(lambda_param=default_lambda, single_item=True),
            'VCG': VCG(),
            'HoldBack': HoldBack()
        }
        
        # Set random seed for reproducibility
        np.random.seed(self.config['general']['random_seeds'][0])
    
    def generate_auction_instance(self, 
                                 topology_name: str, 
                                 num_bidders: int) -> Dict[str, Any]:
        """
        Generate a single auction instance for the given topology.
        
        Args:
            topology_name: Name of the topology to use
            num_bidders: Number of bidders to generate
            
        Returns:
            Dictionary with auction instance data
        """
        # Get topology parameters
        topology_params = self.config['topologies'][topology_name]
        
        # Generate network topology based on topology name
        if topology_name == 'STARLINK-200':
            topology = generate_starlink_topology(
                num_nodes=200,
                num_orbital_planes=10,
                nodes_per_plane=20,
                altitude_km=550
            )
        elif topology_name == 'INTERNET-100':
            topology = generate_internet_topology(
                num_nodes=100,
                num_regions=5,
                nodes_per_region=20
            )
        elif topology_name == 'DSN-30':
            topology = generate_dsn_topology(
                num_nodes=30,
                earth_stations=5,
                relay_satellites=10,
                deep_space_probes=15
            )
        else:
            raise ValueError(f"Unknown topology: {topology_name}")
        
        # Select random nodes for bidders
        num_nodes = len(topology['nodes'])
        bidder_node_indices = np.random.choice(num_nodes, size=num_bidders, replace=True)
        
        # Generate random valuations (uniform distribution)
        val_min, val_max = 1.0, 100.0  # Default valuation range
        valuations = np.random.uniform(val_min, val_max, size=num_bidders)
        
        # Generate random emission times
        emission_time_range = topology_params.get('emission_time_range', [0, 100])
        emission_times = np.random.uniform(emission_time_range[0], emission_time_range[1], size=num_bidders)
        
        # Get propagation delays from topology
        propagation_delays = [topology['delays'][idx] for idx in bidder_node_indices]
        
        # Create bidders list
        bidders = []
        for i in range(num_bidders):
            bidders.append({
                'id': i,
                'node': bidder_node_indices[i],
                'valuation': valuations[i],
                'emission_time': emission_times[i],
                'propagation_delay': propagation_delays[i] / 1000.0  # Convert ms to seconds
            })
        
        # Calculate clearing horizon
        max_delay = max(propagation_delays) / 1000.0  # Convert ms to seconds
        clearing_horizon_factor = self.config['basic_experiments']['clearing_horizon_factor']
        clearing_horizon = max(emission_times) + max_delay + clearing_horizon_factor
        
        return {
            'topology_name': topology_name,
            'bidders': bidders,
            'clearing_horizon': clearing_horizon,
            'topology': topology
        }
    
    def run_basic_experiments(self):
        """
        Run basic experiments across all topologies and mechanisms.
        """
        results = []
        
        # For each topology
        for topology_name in self.config['topologies'].keys():
            print(f"Running basic experiments for {topology_name}...")
            
            # For each random seed
            for seed in self.config['general']['random_seeds']:
                np.random.seed(seed)
                
                # Generate instances for this seed
                num_instances = self.config['general']['instances_per_seed']
                num_bidders = self.config['topologies'][topology_name]['num_bidders']
                
                for instance_id in range(num_instances):
                    if instance_id % 100 == 0:
                        print(f"  Seed {seed}, Instance {instance_id}/{num_instances}")
                    
                    # Generate auction instance
                    instance = self.generate_auction_instance(topology_name, num_bidders)
                    
                    # Run each mechanism on this instance
                    for mech_name, mechanism in self.mechanisms.items():
                        result = mechanism.run(
                            bidders=instance['bidders'],
                            clearing_horizon=instance['clearing_horizon']
                        )
                        
                        # Compute LAI if requested by config
                        if 'lai' in self.config.get('metrics', []):
                            topology = topology_name  
                            adv = self.config.get('lai_computation', {}).get('advance_ms', {}).get(topology, 1.0)
                            try:
                                result['lai'] = compute_lai_for_mechanism(
                                    mechanism, instance['bidders'], instance['clearing_horizon'], advance_ms=adv
                                )
                            except Exception:
                                result['lai'] = None
                        
                        # Add metadata
                        result['topology_name'] = topology_name
                        result['seed'] = seed
                        result['instance_id'] = instance_id
                        result['num_bidders'] = num_bidders
                        
                        results.append(result)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.config['general']['output_dir'], 'data', 'basic_experiments.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Basic experiment results saved to {output_path}")
        
        return results_df
    
    def run_sensitivity_experiments(self):
        """
        Run parameter sensitivity experiments for the LIA mechanism.
        """
        results = []
        
        # For each lambda value
        lambda_values = self.config['sensitivity_experiments']['lambda_values']
        
        print(f"Running sensitivity experiments for lambda values: {lambda_values}...")
        
        # Use a subset of seeds for sensitivity analysis
        seeds = self.config['general']['random_seeds'][:3]
        num_instances = 500  # Fewer instances per lambda value
        
        for lambda_val in lambda_values:
            print(f"  Testing lambda = {lambda_val}")
            
            # Create LIA mechanism with this lambda
            lia = LIA(lambda_param=lambda_val, single_item=True)
            
            # Use the topology specified for sensitivity experiments
            topology_name = self.config['sensitivity_experiments']['topology']
            
            # For each seed
            for seed in seeds:
                np.random.seed(seed)
                
                # Generate instances
                num_bidders = self.config['topologies'][topology_name]['num_bidders']
                
                for instance_id in range(num_instances):
                    # Generate auction instance
                    instance = self.generate_auction_instance(topology_name, num_bidders)
                    
                    # Run LIA on this instance
                    result = lia.run(
                        bidders=instance['bidders'],
                        clearing_horizon=instance['clearing_horizon']
                    )
                    
                    # Add metadata
                    result['topology_name'] = topology_name
                    result['seed'] = seed
                    result['instance_id'] = instance_id
                    result['num_bidders'] = num_bidders
                    result['lambda'] = lambda_val
                    
                    results.append(result)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.config['general']['output_dir'], 'data', 'sensitivity_experiments.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Sensitivity experiment results saved to {output_path}")
        
        return results_df
    
    def run_scalability_experiments(self):
        """
        Run computational scalability experiments.
        """
        results = []
        
        # For each bidder count
        bidder_counts = self.config['scalability_experiments']['bidder_counts']
        
        print(f"Running scalability experiments for bidder counts: {bidder_counts}...")
        
        # Use a subset of seeds for scalability analysis
        seeds = self.config['general']['random_seeds'][:3]
        num_instances = self.config['scalability_experiments']['instances_per_count']
        
        for num_bidders in bidder_counts:
            print(f"  Testing {num_bidders} bidders")
            
            # Use the base topology for scalability experiments
            topology_name = self.config['scalability_experiments']['topology_base']
            
            # For each seed
            for seed in seeds:
                np.random.seed(seed)
                
                # Generate instances
                for instance_id in range(num_instances):
                    # Generate auction instance
                    instance = self.generate_auction_instance(topology_name, num_bidders)
                    
                    # Run each mechanism on this instance
                    for mech_name, mechanism in self.mechanisms.items():
                        result = mechanism.run(
                            bidders=instance['bidders'],
                            clearing_horizon=instance['clearing_horizon']
                        )
                        
                        # Add metadata
                        result['topology_name'] = topology_name
                        result['seed'] = seed
                        result['instance_id'] = instance_id
                        result['num_bidders'] = num_bidders
                        
                        results.append(result)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.config['general']['output_dir'], 'data', 'scalability_experiments.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Scalability experiment results saved to {output_path}")
        
        return results_df
    
    def run_robustness_experiments(self):
        """
        Run error robustness experiments.
        """
        results = []
        
        # For each error level
        error_levels = self.config['robustness_experiments']['error_levels']
        
        print(f"Running robustness experiments for error levels: {error_levels}...")
        
        # Use a subset of seeds for robustness analysis
        seeds = self.config['general']['random_seeds'][:3]
        num_instances = self.config['robustness_experiments']['instances_per_error']
        
        for error_level in error_levels:
            print(f"  Testing error level = {error_level} ms")
            
            # Use the topology specified for robustness experiments
            topology_name = self.config['robustness_experiments']['topology']
            
            # For each seed
            for seed in seeds:
                np.random.seed(seed)
                
                # Generate instances
                num_bidders = self.config['topologies'][topology_name]['num_bidders']
                
                for instance_id in range(num_instances):
                    # Generate auction instance
                    instance = self.generate_auction_instance(topology_name, num_bidders)
                    
                    # Run each mechanism on this instance with the error level
                    for mech_name, mechanism in self.mechanisms.items():
                        result = mechanism.run(
                            bidders=instance['bidders'],
                            clearing_horizon=instance['clearing_horizon'],
                            error_level=error_level
                        )
                        
                        # Add metadata
                        result['topology_name'] = topology_name
                        result['seed'] = seed
                        result['instance_id'] = instance_id
                        result['num_bidders'] = num_bidders
                        result['error_level'] = error_level
                        
                        results.append(result)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.config['general']['output_dir'], 'data', 'robustness_experiments.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Robustness experiment results saved to {output_path}")
        
        return results_df

def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(description='LIA Experiment Runner')
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'sensitivity', 'scalability', 'robustness', 'all'],
                       help='Experiment mode to run')
    parser.add_argument('--config', type=str, default='config/experiment_config.json',
                       help='Path to experiment configuration file')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(args.config)
    
    # Run experiments based on mode
    if args.mode == 'basic' or args.mode == 'all':
        runner.run_basic_experiments()
    
    if args.mode == 'sensitivity' or args.mode == 'all':
        runner.run_sensitivity_experiments()
    
    if args.mode == 'scalability' or args.mode == 'all':
        runner.run_scalability_experiments()
    
    if args.mode == 'robustness' or args.mode == 'all':
        runner.run_robustness_experiments()

if __name__ == '__main__':
    main()

