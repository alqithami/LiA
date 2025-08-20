# LIA Experiment Pipeline

This repository contains the complete code pipeline for reproducing the experimental results in the paper "Latency-Aware Resource Allocation over Heterogeneous Networks: A Lorentz-Invariant Market Mechanism with Interval-Graph Algorithms".

## Requirements

- Python 3.9+
- NumPy 1.21.0+
- NetworkX 2.6.3+
- SciPy 1.7.0+
- Matplotlib 3.4.2+
- Pandas 1.3.0+
- Seaborn 0.11.1+

## Directory Structure

```
lia_experiment_pipeline/
├── README.md
├── run_all.sh                  # Main script to run the entire pipeline
├── src/
│   ├── __init__.py
│   ├── mechanisms/             # Implementation of auction mechanisms
│   │   ├── __init__.py
│   │   ├── lia.py              # LIA mechanism implementation
│   │   ├── vcg.py              # VCG mechanism implementation
│   │   └── holdback.py         # HoldBack mechanism implementation
│   ├── topologies/             # Network topology generators
│   │   ├── __init__.py
│   │   ├── starlink.py         # STARLINK-200 topology generator
│   │   ├── internet.py         # INTERNET-100 topology generator
│   │   └── dsn.py              # DSN-30 topology generator
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── visualization.py    # Plotting functions
│   └── experiment.py           # Main experiment runner
├── config/
│   └── experiment_config.json  # Configuration for experiments
└── results/                    # Directory for storing results
    ├── data/                   # Raw data from experiments
    ├── figures/                # Generated figures
    └── tables/                 # Generated tables
```

## Running the Experiments

To run the entire pipeline and reproduce all results from the paper:

```bash
./run_all.sh
```

This script will:
1. Generate the network topologies
2. Run all auction instances across mechanisms and topologies
3. Compute evaluation metrics
4. Generate all figures and tables
5. Export results to the `results/` directory

## Configuration

The experiment parameters can be modified in `config/experiment_config.json`:

- `num_instances`: Number of auction instances to generate per topology
- `num_bidders`: Range of bidders to test
- `lambda_values`: Values of λ parameter to test for LIA
- `error_levels`: Measurement error levels to test for robustness
- `random_seeds`: Random seeds for reproducibility
- `topology_params`: Parameters for each network topology

## Extending the Experiments

To add new mechanisms or topologies:
1. Create a new file in the appropriate directory
2. Implement the required interface
3. Register the new component in the corresponding `__init__.py` file
4. Update the configuration file as needed

## Results

The results will be saved in the following formats:
- Raw data: CSV files in `results/data/`
- Figures: PNG and PDF files in `results/figures/`
- Tables: LaTeX and CSV files in `results/tables/`

