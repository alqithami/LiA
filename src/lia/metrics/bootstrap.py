from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    low: float
    high: float
    n: int


def bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    chunk: int = 200,
) -> Optional[BootstrapCI]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return None

    mean = float(x.mean())
    if n_resamples <= 0:
        return BootstrapCI(mean=mean, low=float("nan"), high=float("nan"), n=n)

    boot_means: list[float] = []
    # Chunked resampling to avoid large intermediate arrays
    remaining = int(n_resamples)
    while remaining > 0:
        k = min(chunk, remaining)
        idx = rng.integers(0, n, size=(k, n), endpoint=False)
        samp_means = x[idx].mean(axis=1)
        boot_means.extend([float(v) for v in samp_means])
        remaining -= k

    arr = np.array(boot_means, dtype=float)
    low = float(np.quantile(arr, alpha / 2.0))
    high = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return BootstrapCI(mean=mean, low=low, high=high, n=n)
