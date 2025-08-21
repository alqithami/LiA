#!/usr/bin/env python3
import os, json
from src.experiment import ExperimentRunner

def main():
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'config', 'experiment_config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    runner = ExperimentRunner(cfg)

    # Run the three batches described in your config
    df_basic = runner.run_basic_experiments()
    df_sens  = runner.run_sensitivity_experiments()
    df_scal  = runner.run_scalability_experiments()

    # Create basic summaries
    figs_dir = os.path.join(cfg['general']['output_dir'], 'figures')
    os.makedirs(figs_dir, exist_ok=True)

    # Save basic summaries for each topology
    for topo in df_basic['topology_name'].unique():
        topo_data = df_basic[df_basic['topology_name']==topo]
        summary = topo_data.groupby('mechanism').agg({
            'efficiency_ratio': ['mean', 'std'],
            'revenue': ['mean', 'std'],
            'runtime': ['mean', 'std']
        }).round(4)
        
        with open(os.path.join(figs_dir, f'summary_{topo}.txt'), 'w') as f:
            f.write(f'Experiment Summary for {topo}\n')
            f.write('=' * 40 + '\n\n')
            f.write(str(summary))

    print('All done. Results saved under', cfg['general']['output_dir'])

if __name__ == '__main__':
    main()
