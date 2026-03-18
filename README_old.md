# LiA: Lorentz-Invariant Auction Mechanism

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation and experimental framework for the **Lorentz-Invariant Auction (LIA)** mechanism described in the paper:

> **Latency-Aware Resource Allocation over Heterogeneous Networks: A Lorentz-Invariant Market Mechanism with Interval-Graph Algorithms**
<img width="3169" height="1347" alt="lia_banner" src="https://github.com/user-attachments/assets/194dde21-42cb-4716-b9c1-c52f86f86df9" />

## Overview

LIA is a novel auction mechanism designed for resource allocation in telecommunication networks with heterogeneous delays. It addresses the challenges of fairness and efficiency in environments where bidders experience different propagation delays, such as satellite networks, global internet infrastructure, and deep space communication systems.

Key features of LIA:
- **Lorentz Invariance**: Maintains fairness across reference frames with different proper-time measurements
- **Exponential Discounting**: Discounts bids based on horizon slack using an exponential function
- **Interval Graph Algorithms**: Efficiently determines winners using interval graph representations
- **Theoretical Guarantees**: Provides provable welfare bounds and incentive properties
- **Computational Efficiency**: Achieves near-linear runtime scaling

## Repository Structure

```
lia_experiment_pipeline/
├── README.md                  # This file
├── run_all.sh                 # Main script to run the entire pipeline
├── src/
│   ├── __init__.py
│   ├── mechanisms/            # Implementation of auction mechanisms
│   │   ├── __init__.py
│   │   ├── lia.py             # LIA mechanism implementation
│   │   ├── vcg.py             # VCG mechanism implementation
│   │   └── holdback.py        # HoldBack mechanism implementation
│   ├── topologies/            # Network topology generators
│   │   ├── __init__.py
│   │   ├── starlink.py        # STARLINK-200 topology generator
│   │   ├── internet.py        # INTERNET-100 topology generator
│   │   └── dsn.py             # DSN-30 topology generator
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── visualization.py   # Plotting functions
│   └── experiment.py          # Main experiment runner
├── config/
│   └── experiment_config.json # Configuration for experiments
└── results/                   # Directory for storing results
    ├── data/                  # Raw data from experiments
    ├── figures/               # Generated figures
    └── tables/                # Generated tables
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alqithami/LiA.git
cd LiA
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Experiments

To run the entire experimental pipeline:

```bash
chmod +x run_all.sh
./run_all.sh
```

This script will:
1. Generate the network topologies
2. Run all auction instances across mechanisms and topologies
3. Compute evaluation metrics
4. Generate all figures and tables
5. Export results to the `results/` directory

### Running Specific Experiments

You can also run specific parts of the pipeline:

```bash
# Run basic experiments
python -m src.experiment --mode basic

# Run parameter sensitivity experiments
python -m src.experiment --mode sensitivity

# Run computational scalability experiments
python -m src.experiment --mode scalability

# Run error robustness experiments
python -m src.experiment --mode robustness
```

## Configuring Experiments

The experiment parameters can be modified in `config/experiment_config.json`:

- `num_instances`: Number of auction instances to generate per topology
- `num_bidders`: Range of bidders to test
- `lambda_values`: Values of λ parameter to test for LIA
- `error_levels`: Measurement error levels to test for robustness
- `random_seeds`: Random seeds for reproducibility
- `topology_params`: Parameters for each network topology

## Visualizing Results

After running the experiments, you can generate visualizations:

```bash
python -m src.utils.visualization
```

This will create figures in the `results/figures/` directory, including:
- Efficiency comparison across mechanisms
- Lambda parameter sensitivity analysis
- Latency arbitrage index analysis
- Runtime scaling with theoretical bounds
- Comprehensive comparison dashboard

## Generating Tables

To generate tables summarizing the results:

```bash
python -m src.utils.generate_tables
```

This will create tables in the `results/tables/` directory in both CSV and LaTeX formats.

## Extending the Framework

### Adding New Mechanisms

To add a new auction mechanism:

1. Create a new file in `src/mechanisms/` (e.g., `new_mechanism.py`)
2. Implement the mechanism with a `run()` method that takes `bidders`, `clearing_horizon`, and optional `error_level` parameters
3. Register the mechanism in `src/mechanisms/__init__.py`
4. Add the mechanism to the `mechanisms` dictionary in `src/experiment.py`

### Adding New Topologies

To add a new network topology:

1. Create a new file in `src/topologies/` (e.g., `new_topology.py`)
2. Implement a generator function that returns a dictionary with topology information
3. Register the topology in `src/topologies/__init__.py`
4. Add the topology to the `topology_generators` dictionary in `src/experiment.py`
5. Update the configuration file with parameters for the new topology

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lia2025,
  title={Latency-Aware Resource Allocation over Heterogeneous Networks: A Lorentz-Invariant Market Mechanism with Interval-Graph Algorithms},
  author={Alqithami, S.},
  journal={TBD -- e.g., Telecommunication Systems},
  year={2026},
  publisher={TBD}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

