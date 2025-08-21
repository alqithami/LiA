"""
Table generation utilities for LIA experiments.

This module contains functions for generating tables from the experimental results.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def generate_efficiency_table(results_df: pd.DataFrame, output_dir: str):
    """
    Generate efficiency table by mechanism and topology.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the table
    """
    # Group by mechanism and topology
    grouped = results_df.groupby(['mechanism', 'topology_name'])
    
    # Calculate statistics
    stats = grouped['efficiency_ratio'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate 95% confidence interval
    stats['ci_95'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    
    # Format for table
    stats['efficiency'] = stats.apply(
        lambda row: f"{row['mean']:.3f} ± {row['ci_95']:.3f}",
        axis=1
    )
    
    # Pivot to create table
    table = stats.pivot(index='topology_name', columns='mechanism', values='efficiency')
    
    # Add average row
    avg_stats = results_df.groupby('mechanism')['efficiency_ratio'].agg(['mean', 'std', 'count']).reset_index()
    avg_stats['ci_95'] = 1.96 * avg_stats['std'] / np.sqrt(avg_stats['count'])
    avg_stats['efficiency'] = avg_stats.apply(
        lambda row: f"{row['mean']:.3f} ± {row['ci_95']:.3f}",
        axis=1
    )
    
    avg_row = pd.DataFrame(
        {mech: [avg_stats[avg_stats['mechanism'] == mech]['efficiency'].values[0]] 
         for mech in table.columns},
        index=['Average']
    )
    
    table = pd.concat([table, avg_row])
    
    # Save table
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    table.to_csv(os.path.join(output_dir, 'efficiency_table.csv'))
    
    # Save as LaTeX
    with open(os.path.join(output_dir, 'efficiency_table.tex'), 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Efficiency ratio by network topology}\n")
        f.write("\\label{tab:efficiency_by_topology}\n")
        f.write("\\begin{tabular}{l" + "c" * len(table.columns) + "}\n")
        f.write("\\toprule\n")
        f.write("Topology & " + " & ".join(table.columns) + " \\\\\n")
        f.write("\\midrule\n")
        
        for idx, row in table.iterrows():
            f.write(f"{idx} & " + " & ".join(row.values) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def generate_runtime_table(results_df: pd.DataFrame, output_dir: str):
    """
    Generate runtime table by mechanism and number of bidders.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the table
    """
    # Group by mechanism and number of bidders
    grouped = results_df.groupby(['mechanism', 'num_bidders'])
    
    # Calculate statistics
    stats = grouped['runtime'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate 95% confidence interval
    stats['ci_95'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    
    # Format for table
    stats['runtime'] = stats.apply(
        lambda row: f"{row['mean']:.6f} ± {row['ci_95']:.6f}",
        axis=1
    )
    
    # Pivot to create table
    table = stats.pivot(index='num_bidders', columns='mechanism', values='runtime')
    
    # Calculate scaling exponents
    from src.utils.metrics import calculate_scalability_exponent
    exponents = calculate_scalability_exponent(results_df)
    
    # Add exponent row
    exponent_row = pd.DataFrame(
        {mech: [f"{exp['exponent']:.3f} ± {exp['std_err']:.3f} (R²={exp['r_squared']:.3f})"] 
         for mech, exp in exponents.items()},
        index=['Scaling Exponent']
    )
    
    table = pd.concat([table, exponent_row])
    
    # Save table
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    table.to_csv(os.path.join(output_dir, 'runtime_table.csv'))
    
    # Save as LaTeX
    with open(os.path.join(output_dir, 'runtime_table.tex'), 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Runtime (seconds) by number of bidders}\n")
        f.write("\\label{tab:runtime_by_bidders}\n")
        f.write("\\begin{tabular}{l" + "c" * len(table.columns) + "}\n")
        f.write("\\toprule\n")
        f.write("Bidders & " + " & ".join(table.columns) + " \\\\\n")
        f.write("\\midrule\n")
        
        for idx, row in table.iterrows():
            f.write(f"{idx} & " + " & ".join(row.values) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def generate_robustness_table(results_df: pd.DataFrame, output_dir: str):
    """
    Generate error robustness table.
    
    Args:
        results_df: DataFrame with robustness experiment results
        output_dir: Directory to save the table
    """
    # Group by mechanism and error level
    grouped = results_df.groupby(['mechanism', 'error_level'])
    
    # Calculate statistics
    stats = grouped['efficiency_ratio'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate 95% confidence interval
    stats['ci_95'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    
    # Format for table
    stats['efficiency'] = stats.apply(
        lambda row: f"{row['mean']:.3f} ± {row['ci_95']:.3f}",
        axis=1
    )
    
    # Pivot to create table
    table = stats.pivot(index='error_level', columns='mechanism', values='efficiency')
    
    # Calculate theoretical bounds
    lambda_val = 0.5  # Default lambda value
    error_levels = sorted(results_df['error_level'].unique())
    
    theoretical_bounds = []
    for error in error_levels:
        # Theoretical bound: SW/OPT ≥ e^(-λ(Δ+2ε))
        # Assuming Δ = 25 ms (typical for STARLINK-200)
        delta = 25  # ms
        bound = np.exp(-lambda_val * (delta + 2 * error))
        theoretical_bounds.append(bound)
    
    # Add theoretical bound column
    table['Theoretical Bound'] = [f"{bound:.3f}" for bound in theoretical_bounds]
    
    # Save table
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    table.to_csv(os.path.join(output_dir, 'robustness_table.csv'))
    
    # Save as LaTeX
    with open(os.path.join(output_dir, 'robustness_table.tex'), 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Efficiency ratio under measurement errors}\n")
        f.write("\\label{tab:error_robustness}\n")
        f.write("\\begin{tabular}{l" + "c" * len(table.columns) + "}\n")
        f.write("\\toprule\n")
        f.write("Error (ms) & " + " & ".join(table.columns) + " \\\\\n")
        f.write("\\midrule\n")
        
        for idx, row in table.iterrows():
            f.write(f"{idx} & " + " & ".join(row.values) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def generate_summary_table(results_df: pd.DataFrame, output_dir: str):
    """
    Generate summary statistics table.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the table
    """
    # Group by mechanism
    grouped = results_df.groupby('mechanism')
    
    # Calculate statistics
    summary = pd.DataFrame()
    
    # Efficiency
    efficiency_stats = grouped['efficiency_ratio'].agg(['mean', 'std']).reset_index()
    efficiency_stats['ci_95'] = 1.96 * efficiency_stats['std'] / np.sqrt(grouped.size())
    efficiency_stats['efficiency'] = efficiency_stats.apply(
        lambda row: f"{row['mean']:.3f} ± {row['ci_95']:.3f}",
        axis=1
    )
    summary['Efficiency'] = efficiency_stats.set_index('mechanism')['efficiency']
    
    # Revenue
    revenue_stats = grouped['revenue_ratio'].agg(['mean', 'std']).reset_index()
    revenue_stats['ci_95'] = 1.96 * revenue_stats['std'] / np.sqrt(grouped.size())
    revenue_stats['revenue'] = revenue_stats.apply(
        lambda row: f"{row['mean']:.3f} ± {row['ci_95']:.3f}",
        axis=1
    )
    summary['Revenue Ratio'] = revenue_stats.set_index('mechanism')['revenue']
    
    # Runtime
    runtime_stats = grouped['runtime'].agg(['mean', 'median', 'std']).reset_index()
    runtime_stats['runtime'] = runtime_stats.apply(
        lambda row: f"{row['mean']:.6f} s (±{row['std']:.6f})",
        axis=1
    )
    summary['Runtime'] = runtime_stats.set_index('mechanism')['runtime']
    
    # LAI
    from src.utils.metrics import calculate_latency_arbitrage_index
    lai_values = calculate_latency_arbitrage_index(results_df)
    summary['LAI'] = pd.Series(lai_values)
    
    # Save table
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    summary.to_csv(os.path.join(output_dir, 'summary_table.csv'))
    
    # Save as LaTeX
    with open(os.path.join(output_dir, 'summary_table.tex'), 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary statistics by mechanism}\n")
        f.write("\\label{tab:summary_statistics}\n")
        f.write("\\begin{tabular}{l" + "c" * len(summary.columns) + "}\n")
        f.write("\\toprule\n")
        f.write("Mechanism & " + " & ".join(summary.columns) + " \\\\\n")
        f.write("\\midrule\n")
        
        for idx, row in summary.iterrows():
            f.write(f"{idx} & " + " & ".join(row.values) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def generate_all_tables(config_path: str = "config/experiment_config.json"):
    """
    Generate all tables from experiment results.
    
    Args:
        config_path: Path to experiment configuration file
    """
    import json
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = os.path.join(config['general']['output_dir'], 'tables')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment results
    basic_results_path = os.path.join(config['general']['output_dir'], 'data', 'basic_experiments.csv')
    scalability_results_path = os.path.join(config['general']['output_dir'], 'data', 'scalability_experiments.csv')
    robustness_results_path = os.path.join(config['general']['output_dir'], 'data', 'robustness_experiments.csv')
    
    # Check if files exist
    if not os.path.exists(basic_results_path):
        print(f"Warning: {basic_results_path} not found. Run basic experiments first.")
        return
    
    # Load data
    basic_results = pd.read_csv(basic_results_path)
    
    # Generate tables
    print("Generating summary statistics table...")
    generate_summary_table(basic_results, output_dir)
    
    print("Generating efficiency table by topology...")
    generate_efficiency_table(basic_results, output_dir)
    
    # Check if scalability results exist
    if os.path.exists(scalability_results_path):
        scalability_results = pd.read_csv(scalability_results_path)
        print("Generating runtime table by number of bidders...")
        generate_runtime_table(scalability_results, output_dir)
    
    # Check if robustness results exist
    if os.path.exists(robustness_results_path):
        robustness_results = pd.read_csv(robustness_results_path)
        print("Generating error robustness table...")
        generate_robustness_table(robustness_results, output_dir)
    
    print(f"All tables generated and saved to {output_dir}")

def main():
    """Generate tables from experiment results."""
    
    # Check if results exist
    results_file = 'results/data/basic_experiments.csv'
    if not os.path.exists(results_file):
        print("No results file found. Skipping table generation.")
        return
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Create summary table
    summary_table = df.groupby(['mechanism', 'topology_name']).agg({
        'efficiency_ratio': ['mean', 'std'],
        'revenue_ratio': ['mean', 'std'],
        'runtime': ['mean', 'std']
    }).round(4)
    
    # Save to CSV
    os.makedirs('results/tables', exist_ok=True)
    summary_table.to_csv('results/tables/summary_table.csv')
    print("Tables saved to results/tables/")

if __name__ == '__main__':
    main()

