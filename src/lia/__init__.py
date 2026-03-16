"""LiA experiment pipeline.

This package contains:
- dataset/topology builders for STARLINK-200, INTERNET-100, DSN-30
- graph + temporal earliest-arrival algorithms
- mechanisms (SyncVCG, FastVCG, BatchVCG, HoldBack, LIA)
- metrics (welfare/revenue ratios, clearing latency, effective welfare, LAI)
- bootstrap confidence intervals

The pipeline prints its version at runtime (see `run_pipeline.py`) and records it inside
each run's `run_meta.json`, so it is easy to verify that results were produced by the
expected code.
"""

from __future__ import annotations

__version__ = "4.1.0"

__all__ = ["__version__"]
