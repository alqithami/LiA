# LiA: Lorentz-Invariant Auction Experiment Pipeline

This repository contains the reference implementation and experiment pipeline for the Lorentz-Invariant Auction (LIA) mechanism studied in the manuscript:

> **Latency-Aware Resource Allocation over Heterogeneous Networks: A Lorentz-Invariant Market Mechanism**

LIA is a latency-aware auction framework for heterogeneous communication settings in which bidders face materially different propagation delays, including LEO satellite systems, terrestrial backbone networks, and deep-space relay networks. The codebase supports reproducible topology construction, instance generation, mechanism evaluation, summary statistics, and paper-ready figures and tables.

## Repository status

The canonical, maintained pipeline is:

- `run_pipeline.py`
- `config/*.json`
- `scripts/*.py`
- `src/lia/`

If older or archival files remain elsewhere in the repository while the project is being consolidated, the entry points and commands in this README take precedence.

## What this repository provides

- LIA, FastVCG, SyncVCG, HoldBack, and BatchVCG implementations
- Topology builders and cached topologies for:
  - `STARLINK-200`
  - `INTERNET-100`
  - `DSN-30`
- Paper-scale experiment configurations
- Robustness configurations, including structured slack-estimation bias models
- Figure, tradeoff-plot, LaTeX-table, audit, and slack-spread utilities
- Run provenance via per-run hashes and metadata

## Requirements

- Python **3.10+**
- A POSIX-like shell for the convenience scripts (`bash`, `zsh`, Linux/macOS terminal, WSL, etc.)

Core dependencies are listed in `requirements.txt`.

## Quick start

Clone the repository and enter it:

```bash
git clone https://github.com/alqithami/LiA.git
cd LiA
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You do **not** need to run `pip install -e .` for normal use because `run_pipeline.py` prepends `./src` to the import path. If you prefer an editable install, that also works:

```bash
python -m pip install -e .
```

## Recommended first commands

Validate that the repository does not contain placeholder content:

```bash
python scripts/validate_no_placeholders.py --root .
```

Build or refresh the cached topologies:

```bash
python scripts/build_topologies.py --data-dir data
```

Then run a quick smoke test:

```bash
python run_pipeline.py --config config/quick.json
```

## Main experiment runs

### Paper-scale run

```bash
python run_pipeline.py --config config/full.json
```

### Larger scaling run

```bash
python run_pipeline.py --config config/full_strong.json
```

### Robustness sweeps

Generic bounded-error sweep:

```bash
python run_pipeline.py --config config/robustness.json
```

Distance-biased slack-estimation error:

```bash
python run_pipeline.py --config config/robustness_distancebias.json
```

Subnetwork-correlated slack-estimation error:

```bash
python run_pipeline.py --config config/robustness_subnetwork.json
```

Clock/common-mode error models:

```bash
python run_pipeline.py --config config/robustness_clockbias.json
python run_pipeline.py --config config/robustness_strong.json
```

### Table replication / small checks

```bash
python run_pipeline.py --config config/table3_replication.json
python run_pipeline.py --config config/mini_test.json
```

## Post-processing and paper artifacts

Each run writes to a fresh directory under `runs/`. After a run completes, generate figures and tables directly from that run directory.

Example:

```bash
python scripts/generate_figures.py --run-dir runs/<run_id>
python scripts/generate_tradeoff_plots.py --run-dir runs/<run_id>
python scripts/generate_latex_tables.py --run-dir runs/<run_id>
python scripts/audit_run.py --run-dir runs/<run_id>
```

To summarize empirical slack-spread statistics used in the revised welfare discussion:

```bash
python scripts/slack_spread_summary.py --data-dir data --bidder-count 50 --instances 1000
```

## Output structure

A typical run directory contains:

- `per_instance_metrics.csv` — one row per `(instance, mechanism)`
- `summary_table.csv` — bootstrap means and confidence intervals by topology / mechanism / error level
- `summary_by_bidder_count.csv` — summaries split by bidder count
- `lai_estimates.csv` — latency-arbitrage and fairness diagnostics
- `paired_differences.csv` — paired bootstrap differences across mechanisms
- `run_meta.json` — provenance metadata, hashes, and environment information
- `run.log` — detailed execution log
- `figures/` — generated paper figures
- `tables/` — generated LaTeX/CSV tables

## Metric conventions

The pipeline exports both benchmark conventions used in the manuscript:

- `opt_all_value`, `welfare_ratio_all`, `revenue_ratio_all` compare outcomes against the best value anywhere in the instance.
- `opt_feasible_value`, `welfare_ratio_feasible`, `revenue_ratio_feasible` compare outcomes against the best causally feasible bid.
- `feasible_opt_coverage = opt_feasible_value / opt_all_value` separates reachability loss from conditional auction efficiency.

For backward compatibility, `opt_value`, `welfare_ratio`, and `revenue_ratio` remain aliases for the overall benchmark columns.

Other exported timing metrics follow the manuscript definitions:

- `decision_time_ms` — mechanism commit time, including any waiting policy plus measured compute time
- `clearing_latency_ms` — `decision_time_ms - min_i tau_i`

## Topology sources

Topologies are built from public raw inputs and cached locally.

- `STARLINK-200`: Starlink constellation snapshot from a CelesTrak Starlink TLE file
- `INTERNET-100`: a 100-node backbone graph extracted from the Topology Zoo archive
- `DSN-30`: a 30-node deep-space communication graph built from a JPL planetary ephemeris file

Raw inputs are stored under `data/raw/`, and cached topology JSONs are stored under `data/topologies/`.

## Configuration guide

All experiment settings live in `config/*.json`.

Common knobs include:

- `bidder_counts`
- `instances_per_setting`
- `measurement_error.eps_grid_ms`
- `measurement_error.model`
- `measurement_error.common_fraction`
- `mechanisms.batch_vcg.batch_ms_grid`
- `mechanisms.lia.lambda_grid`
- `mechanisms.lia.lambda_unit`
- `datasets.strict`
- `datasets.rebuild_topologies`

Supported measurement-error models include:

- `iid_uniform`
- `common_plus_iid_uniform`
- `distance_biased_uniform`
- `subnetwork_correlated_uniform`

## Reproducibility and provenance

Every run writes a `run_meta.json` file containing:

- pipeline version
- import path for the `lia` package actually executed
- topology hashes
- raw dataset hashes
- environment metadata

A quick provenance check:

```bash
python - <<'PY'
import glob, json, os
run = sorted(glob.glob('runs/*'), key=os.path.getmtime)[-1]
meta = json.load(open(os.path.join(run, 'run_meta.json')))
print('run:', run)
print('pipeline_version:', meta.get('pipeline_version'))
print('lia_import:', meta.get('lia_import'))
print('topologies:', list((meta.get('topology_hashes') or {}).keys()))
print('raw_files:', list((meta.get('raw_dataset_hashes') or {}).keys()))
PY
```

## Canonical repository layout

```text
LiA/
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_pipeline.py
├── run_all.sh
├── config/
├── data/
│   ├── raw/
│   └── topologies/
├── scripts/
└── src/
    └── lia/
        ├── datasets/
        ├── experiment/
        ├── mechanisms/
        ├── metrics/
        ├── network/
        └── utils/
```

## Citation

If you use this repository, please cite the accompanying manuscript. If final publication metadata is not yet available, a neutral placeholder is:

```bibtex
@misc{alqithami2026lia,
  title        = {Latency-Aware Resource Allocation over Heterogeneous Networks: A Lorentz-Invariant Market Mechanism},
  author       = {Alqithami, S.},
  year         = {2026},
  note         = {Manuscript and code repository}
}
```

Replace this with the final journal or conference citation once publication details are available.

## License

Add a top-level `LICENSE` file and keep this section consistent with that file. If you intend to release the code under MIT, include the standard MIT license text in `LICENSE` before making the repository public as the archival artifact.
