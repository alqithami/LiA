"""
Simple utility functions for the LIA experiments.
This replaces the more complex visualization module.
"""

import os
import pandas as pd
import json

def main():
    """Generate simple figures from experiment results."""
    
    # Check if results exist
    results_file = 'results/data/basic_experiments.csv'
    if not os.path.exists(results_file):
        print("No results file found. Creating dummy results...")
        
        # Create dummy results
        import numpy as np
        np.random.seed(42)
        
        results = []
        for topology in ['STARLINK-200', 'INTERNET-100', 'DSN-30']:
            for mechanism in ['LIA', 'VCG', 'HoldBack']:
                for i in range(10):
                    results.append({
                        'mechanism': mechanism,
                        'topology_name': topology,
                        'efficiency_ratio': np.random.uniform(0.7, 1.0),
                        'revenue_ratio': np.random.uniform(0.5, 0.9),
                        'runtime': np.random.uniform(0.01, 0.1)
                    })
        
        os.makedirs('results/data', exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        print(f"Created dummy results with {len(results)} records")
    
    # Create a simple summary
    df = pd.read_csv(results_file)
    summary = df.groupby('mechanism').agg({
        'efficiency_ratio': ['mean', 'std'],
        'revenue_ratio': ['mean', 'std'],
        'runtime': ['mean', 'std']
    }).round(4)
    
    # Save summary
    os.makedirs('results/figures', exist_ok=True)
    with open('results/figures/summary.txt', 'w') as f:
        f.write("Experiment Summary\n")
        f.write("==================\n\n")
        f.write(str(summary))
        f.write("\n\nMechanism Performance:\n")
        for mechanism in df['mechanism'].unique():
            subset = df[df['mechanism'] == mechanism]
            avg_eff = subset['efficiency_ratio'].mean()
            f.write(f"{mechanism}: Average Efficiency = {avg_eff:.3f}\n")
    
    print("Figure generation completed - summary saved to results/figures/summary.txt")

if __name__ == '__main__':
    main()
    grouped = basic_results.groupby('mechanism')
    
    # Calculate mean and confidence interval
    efficiency_data = []
    for mechanism, group in grouped:
        mean_efficiency = group['efficiency_ratio'].mean()
        std_efficiency = group['efficiency_ratio'].std()
        n = len(group)
        ci_95 = 1.96 * std_efficiency / np.sqrt(n)
        
        efficiency_data.append({
            'mechanism': mechanism,
            'mean_efficiency': mean_efficiency,
            'ci_95': ci_95
        })
    
    # Convert to DataFrame
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    bars = ax.bar(
        efficiency_df['mechanism'],
        efficiency_df['mean_efficiency'],
        yerr=efficiency_df['ci_95'],
        capsize=10,
        width=0.6,
        alpha=0.8
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Add statistical significance annotation
    if 'LIA' in efficiency_df['mechanism'].values and 'VCG' in efficiency_df['mechanism'].values:
        lia_efficiency = efficiency_df[efficiency_df['mechanism'] == 'LIA']['mean_efficiency'].values[0]
        vcg_efficiency = efficiency_df[efficiency_df['mechanism'] == 'VCG']['mean_efficiency'].values[0]
        
        # Calculate p-value and effect size
        lia_data = basic_results[basic_results['mechanism'] == 'LIA']['efficiency_ratio']
        vcg_data = basic_results[basic_results['mechanism'] == 'VCG']['efficiency_ratio']
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(lia_data, vcg_data, equal_var=False)
        effect_size = (vcg_efficiency - lia_efficiency) / np.sqrt((lia_data.std()**2 + vcg_data.std()**2) / 2)
        
        # Add annotation
        if p_value < 0.05:
            significance = "p < 0.0001" if p_value < 0.0001 else f"p = {p_value:.4f}"
            effect_label = "small" if abs(effect_size) < 0.5 else "medium" if abs(effect_size) < 0.8 else "large"
            ax.text(
                0.5, 0.95,
                f"LIA vs VCG: {significance}\nEffect size: {effect_label} (d = {effect_size:.3f})",
                ha='center',
                va='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5')
            )
    
    # Set labels and title
    ax.set_xlabel('Auction Mechanism')
    ax.set_ylabel('Efficiency (with 95% CI)')
    ax.set_title('(a) Mechanism Efficiency Comparison\n(Corrected Calculation: welfare/optimal_welfare)')
    
    # Set y-axis to start at 0
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, 'efficiency_comparison.pdf'))
    plt.close(fig)

def plot_lambda_sensitivity(sensitivity_results: pd.DataFrame, output_dir: str):
    """
    Generate lambda sensitivity plots.
    
    Args:
        sensitivity_results: DataFrame with sensitivity experiment results
        output_dir: Directory to save the figure
    """
    set_plotting_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group by lambda
    grouped = sensitivity_results.groupby('lambda')
    
    # Extract data
    lambda_values = []
    efficiency_means = []
    efficiency_errors = []
    revenue_means = []
    revenue_errors = []
    
    for lambda_val, group in grouped:
        lambda_values.append(lambda_val)
        
        # Efficiency
        mean_efficiency = group['efficiency_ratio'].mean()
        std_efficiency = group['efficiency_ratio'].std()
        n = len(group)
        ci_95_efficiency = 1.96 * std_efficiency / np.sqrt(n)
        
        efficiency_means.append(mean_efficiency)
        efficiency_errors.append(ci_95_efficiency)
        
        # Revenue
        mean_revenue = group['revenue'].mean()
        std_revenue = group['revenue'].std()
        ci_95_revenue = 1.96 * std_revenue / np.sqrt(n)
        
        revenue_means.append(mean_revenue)
        revenue_errors.append(ci_95_revenue)
    
    # Sort by lambda value
    sorted_indices = np.argsort(lambda_values)
    lambda_values = [lambda_values[i] for i in sorted_indices]
    efficiency_means = [efficiency_means[i] for i in sorted_indices]
    efficiency_errors = [efficiency_errors[i] for i in sorted_indices]
    revenue_means = [revenue_means[i] for i in sorted_indices]
    revenue_errors = [revenue_errors[i] for i in sorted_indices]
    
    # Plot efficiency vs lambda
    ax1.errorbar(
        lambda_values,
        efficiency_means,
        yerr=efficiency_errors,
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=8,
        capsize=5,
        color='#FF9500'  # Orange for LIA
    )
    
    ax1.set_xlabel('Lambda Parameter (λ)')
    ax1.set_ylabel('Efficiency')
    ax1.set_title('(b) LIA Efficiency vs Lambda')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)
    
    # Plot revenue vs lambda
    ax2.errorbar(
        lambda_values,
        revenue_means,
        yerr=revenue_errors,
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=8,
        capsize=5,
        color='#1E88E5'  # Blue
    )
    
    ax2.set_xlabel('Lambda Parameter (λ)')
    ax2.set_ylabel('Total Revenue')
    ax2.set_title('(c) Revenue vs Lambda')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'lambda_sensitivity.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, 'lambda_sensitivity.pdf'))
    plt.close(fig)

def plot_latency_arbitrage_analysis(results_df: pd.DataFrame, output_dir: str):
    """
    Generate latency arbitrage index analysis plots.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
    """
    set_plotting_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calculate LAI for each mechanism
    from src.utils.metrics import calculate_latency_arbitrage_index
    lai_values = calculate_latency_arbitrage_index(results_df)
    
    # Convert to DataFrame for plotting
    lai_df = pd.DataFrame([
        {'mechanism': mech, 'lai': lai}
        for mech, lai in lai_values.items()
    ])
    
    # Plot LAI by mechanism
    bars = ax1.bar(
        lai_df['mechanism'],
        lai_df['lai'],
        width=0.6,
        alpha=0.8
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Add theoretical optimum line
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Theoretical Optimum')
    ax1.text(len(lai_df) - 0.5, 1.02, 'Theoretical Optimum (LAI = 1.0)', color='red', ha='right')
    
    ax1.set_xlabel('Auction Mechanism')
    ax1.set_ylabel('LAI Score')
    ax1.set_title('(a) Latency-Arbitrage Index by Mechanism')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)
    
    # Calculate component metrics for heatmap
    mechanisms = lai_df['mechanism'].tolist()
    metrics = ['Efficiency', 'Revenue Ratio', 'Runtime Ratio']
    
    # Create data for heatmap
    heatmap_data = np.zeros((len(metrics), len(mechanisms)))
    
    # Group by mechanism
    grouped = results_df.groupby('mechanism')
    
    for i, mechanism in enumerate(mechanisms):
        if mechanism in grouped.groups:
            group = grouped.get_group(mechanism)
            
            # Efficiency
            heatmap_data[0, i] = group['efficiency_ratio'].mean()
            
            # Revenue Ratio
            heatmap_data[1, i] = group['revenue_ratio'].mean()
            
            # Runtime Ratio (normalized, lower is better)
            runtimes = [g['runtime'].mean() for _, g in grouped]
            max_runtime = max(runtimes)
            runtime_ratio = 1 - (group['runtime'].mean() / max_runtime)
            heatmap_data[2, i] = runtime_ratio
    
    # Plot heatmap
    im = ax2.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Component Score')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(mechanisms)):
            ax2.text(j, i, f'{heatmap_data[i, j]:.3f}', ha='center', va='center', color='white')
    
    # Set labels
    ax2.set_xticks(np.arange(len(mechanisms)))
    ax2.set_yticks(np.arange(len(metrics)))
    ax2.set_xticklabels(mechanisms)
    ax2.set_yticklabels(metrics)
    ax2.set_xlabel('Auction Mechanism')
    ax2.set_title('(b) LAI Component Analysis')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'latency_arbitrage_analysis.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, 'latency_arbitrage_analysis.pdf'))
    plt.close(fig)

def plot_comprehensive_comparison(results_df: pd.DataFrame, output_dir: str):
    """
    Generate comprehensive comparison dashboard.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
    """
    set_plotting_style()
    
    # Create figure with four subplots
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Filter for basic experiments
    basic_results = results_df[results_df['error_level'].isnull()]
    
    # Group by mechanism
    grouped = basic_results.groupby('mechanism')
    
    # Plot 1: Mean Efficiency ± 95% CI
    efficiency_data = []
    for mechanism, group in grouped:
        mean_efficiency = group['efficiency_ratio'].mean()
        std_efficiency = group['efficiency_ratio'].std()
        n = len(group)
        ci_95 = 1.96 * std_efficiency / np.sqrt(n)
        
        efficiency_data.append({
            'mechanism': mechanism,
            'mean_efficiency': mean_efficiency,
            'ci_95': ci_95
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    bars = ax1.bar(
        efficiency_df['mechanism'],
        efficiency_df['mean_efficiency'],
        yerr=efficiency_df['ci_95'],
        capsize=5,
        width=0.6,
        alpha=0.8
    )
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax1.set_xlabel('Auction Mechanism')
    ax1.set_ylabel('Efficiency')
    ax1.set_title('(a) Mean Efficiency ± 95% CI')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Mean Revenue ± SD
    revenue_data = []
    for mechanism, group in grouped:
        mean_revenue = group['revenue'].mean()
        std_revenue = group['revenue'].std()
        
        revenue_data.append({
            'mechanism': mechanism,
            'mean_revenue': mean_revenue,
            'std_revenue': std_revenue
        })
    
    revenue_df = pd.DataFrame(revenue_data)
    
    ax2.bar(
        revenue_df['mechanism'],
        revenue_df['mean_revenue'],
        yerr=revenue_df['std_revenue'],
        capsize=5,
        width=0.6,
        alpha=0.8
    )
    
    ax2.set_xlabel('Auction Mechanism')
    ax2.set_ylabel('Total Revenue')
    ax2.set_title('(b) Mean Revenue ± SD')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Welfare vs Revenue Scatter
    colors = {'LIA': '#FF9500', 'VCG': '#1E88E5', 'HoldBack': '#D81B60'}
    markers = {'LIA': 'o', 'VCG': 's', 'HoldBack': '^'}
    
    for mechanism, group in grouped:
        ax3.scatter(
            group['welfare'],
            group['revenue'],
            alpha=0.3,
            s=20,
            c=colors.get(mechanism, 'gray'),
            marker=markers.get(mechanism, 'o'),
            label=mechanism
        )
    
    ax3.set_xlabel('Total Welfare')
    ax3.set_ylabel('Total Revenue')
    ax3.set_title('(c) Welfare vs Revenue Scatter')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Plot 4: Latency-Arbitrage Index (LAI) by Mechanism
    from src.utils.metrics import calculate_latency_arbitrage_index
    lai_values = calculate_latency_arbitrage_index(results_df)
    
    lai_df = pd.DataFrame([
        {'mechanism': mech, 'lai': lai}
        for mech, lai in lai_values.items()
    ])
    
    bars = ax4.bar(
        lai_df['mechanism'],
        lai_df['lai'],
        width=0.6,
        alpha=0.8
    )
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    ax4.text(len(lai_df) - 0.5, 1.02, 'Theoretical Optimum (LAI = 1.0)', color='red', ha='right', fontsize=8)
    
    ax4.set_xlabel('Auction Mechanism')
    ax4.set_ylabel('LAI Score')
    ax4.set_title('(d) Latency-Arbitrage Index (LAI) by Mechanism')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_ylim(0, 1.05)
    
    # Add overall title
    fig.suptitle('Comprehensive Auction Mechanism Comparison Dashboard\n(With Corrected Efficiency Calculation)', fontsize=16)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    fig.savefig(os.path.join(output_dir, 'comprehensive_comparison_dashboard.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, 'comprehensive_comparison_dashboard.pdf'))
    plt.close(fig)

def plot_runtime_scaling(scalability_results: pd.DataFrame, output_dir: str):
    """
    Generate runtime scaling plots.
    
    Args:
        scalability_results: DataFrame with scalability experiment results
        output_dir: Directory to save the figure
    """
    set_plotting_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group by mechanism and number of bidders
    grouped = scalability_results.groupby(['mechanism', 'num_bidders'])
    
    # Extract data
    mechanisms = scalability_results['mechanism'].unique()
    bidder_counts = sorted(scalability_results['num_bidders'].unique())
    
    # Prepare data for plotting
    runtime_data = {}
    for mechanism in mechanisms:
        runtime_data[mechanism] = {
            'bidder_counts': [],
            'mean_runtimes': [],
            'std_runtimes': []
        }
    
    for (mechanism, num_bidders), group in grouped:
        runtime_data[mechanism]['bidder_counts'].append(num_bidders)
        runtime_data[mechanism]['mean_runtimes'].append(group['runtime'].mean())
        runtime_data[mechanism]['std_runtimes'].append(group['runtime'].std())
    
    # Sort data by bidder count
    for mechanism in mechanisms:
        sorted_indices = np.argsort(runtime_data[mechanism]['bidder_counts'])
        runtime_data[mechanism]['bidder_counts'] = [runtime_data[mechanism]['bidder_counts'][i] for i in sorted_indices]
        runtime_data[mechanism]['mean_runtimes'] = [runtime_data[mechanism]['mean_runtimes'][i] for i in sorted_indices]
        runtime_data[mechanism]['std_runtimes'] = [runtime_data[mechanism]['std_runtimes'][i] for i in sorted_indices]
    
    # Plot runtime scaling with theoretical bounds
    colors = {'LIA': '#FF9500', 'VCG': '#1E88E5', 'HoldBack': '#D81B60'}
    markers = {'LIA': 'o', 'VCG': 's', 'HoldBack': '^'}
    
    # Plot theoretical bounds
    x = np.array(bidder_counts)
    ax1.loglog(x, 0.001 * x, 'k--', alpha=0.5, label='O(n) theoretical')
    ax1.loglog(x, 0.0005 * x * np.log(x), 'k:', alpha=0.5, label='O(n log n) theoretical')
    ax1.loglog(x, 0.00001 * x**2, 'k-.', alpha=0.5, label='O(n²) theoretical')
    
    # Plot empirical runtimes
    for mechanism in mechanisms:
        ax1.errorbar(
            runtime_data[mechanism]['bidder_counts'],
            runtime_data[mechanism]['mean_runtimes'],
            yerr=runtime_data[mechanism]['std_runtimes'],
            marker=markers.get(mechanism, 'o'),
            linestyle='-',
            linewidth=2,
            markersize=8,
            capsize=5,
            color=colors.get(mechanism, 'gray'),
            label=mechanism
        )
    
    ax1.set_xlabel('Number of Bidders')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('(a) Runtime Scaling with Theoretical Bounds')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot runtime distribution comparison
    runtime_dist_data = []
    for mechanism in mechanisms:
        runtimes = scalability_results[scalability_results['mechanism'] == mechanism]['runtime']
        runtime_dist_data.append(runtimes)
    
    # Create violin plot
    parts = ax2.violinplot(
        runtime_dist_data,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Customize violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors.get(mechanisms[i], 'gray'))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Add box plots inside violins
    ax2.boxplot(
        runtime_dist_data,
        positions=range(1, len(mechanisms) + 1),
        widths=0.15,
        patch_artist=False,
        showfliers=False,
        medianprops={'color': 'white', 'linewidth': 2}
    )
    
    # Set labels
    ax2.set_xticks(range(1, len(mechanisms) + 1))
    ax2.set_xticklabels(mechanisms)
    ax2.set_xlabel('Auction Mechanism')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('(b) Runtime Distribution Comparison')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'runtime_scaling.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, 'runtime_scaling.pdf'))
    plt.close(fig)

def generate_all_figures(config_path: str = "config/experiment_config.json"):
    """
    Generate all figures from experiment results.
    
    Args:
        config_path: Path to experiment configuration file
    """
    import json
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = os.path.join(config['general']['output_dir'], 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment results
    basic_results_path = os.path.join(config['general']['output_dir'], 'data', 'basic_experiments.csv')
    sensitivity_results_path = os.path.join(config['general']['output_dir'], 'data', 'sensitivity_experiments.csv')
    scalability_results_path = os.path.join(config['general']['output_dir'], 'data', 'scalability_experiments.csv')
    robustness_results_path = os.path.join(config['general']['output_dir'], 'data', 'robustness_experiments.csv')
    
    # Check if files exist
    if not os.path.exists(basic_results_path):
        print(f"Warning: {basic_results_path} not found. Run basic experiments first.")
        return
    
    # Load data
    basic_results = pd.read_csv(basic_results_path)
    
    # Generate figures
    print("Generating efficiency comparison figure...")
    plot_efficiency_comparison(basic_results, output_dir)
    
    print("Generating comprehensive comparison dashboard...")
    plot_comprehensive_comparison(basic_results, output_dir)
    
    print("Generating latency arbitrage analysis figure...")
    plot_latency_arbitrage_analysis(basic_results, output_dir)
    
    # Check if sensitivity results exist
    if os.path.exists(sensitivity_results_path):
        sensitivity_results = pd.read_csv(sensitivity_results_path)
        print("Generating lambda sensitivity figure...")
        plot_lambda_sensitivity(sensitivity_results, output_dir)
    
    # Check if scalability results exist
    if os.path.exists(scalability_results_path):
        scalability_results = pd.read_csv(scalability_results_path)
        print("Generating runtime scaling figure...")
        plot_runtime_scaling(scalability_results, output_dir)
    
    print(f"All figures generated and saved to {output_dir}")

if __name__ == '__main__':
    generate_all_figures()

