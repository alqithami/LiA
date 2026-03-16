from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import lia
from lia import __version__ as pipeline_version

from lia.experiment.instance_generator import AuctionInstance, generate_instances
from lia.mechanisms.lia import LiaConfig, run_lia
from lia.mechanisms.vcg import run_batch_vcg, run_fast_vcg, run_holdback, run_sync_vcg
from lia.metrics.bootstrap import bootstrap_mean_ci
from lia.metrics.lai import estimate_lai
from lia.metrics.summary import compute_instance_metrics
from lia.network.graph import load_topology, save_topology
from lia.utils.hashing import sha256_bytes, sha256_file, sha256_tree
from lia.utils.logging import get_logger


def derive_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *[str(p) for p in parts]]).encode("utf-8")
    # Use 8 bytes to fit into 64-bit seed
    h = sha256_bytes(payload)
    return int(h[:16], 16) % (2**32 - 1)


def ensure_topologies(data_dir: Path, names: List[str], logger, strict: bool = False, force_rebuild: bool = False) -> Dict[str, Any]:
    """Ensure requested topology JSONs exist in data/topologies.

    Parameters
    ----------
    data_dir:
        Base data directory (contains raw/ and topologies/).
    names:
        List of topology names to build/load.

    Returns
    -------
    Dict[str, Topology]
        Mapping from topology name to loaded/built Topology object.
    """

    topo_dir = data_dir / "topologies"
    raw_dir = data_dir / "raw"
    topo_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    built: Dict[str, Any] = {}

    def _verify_topology_sources(name: str, topo: Any) -> None:
        """Validate that a cached topology is backed by the expected raw dataset files.

        This is a *safety* check to prevent accidental fallback to synthetic/cached graphs
        without the corresponding raw artifacts being present.
        """
        if not hasattr(topo, "source") or topo.source is None:
            raise RuntimeError(f"Topology {name} is missing `source` metadata (strict mode)")
        src = topo.source or {}

        # STARLINK-200: built from a CelesTrak Starlink TLE snapshot
        if name == "STARLINK-200":
            raw_file = raw_dir / "celestrak_starlink.tle"
            exp = src.get("tle_sha256")
            if exp is None:
                raise RuntimeError("STARLINK-200 topology missing source.tle_sha256 (strict mode)")
            if not raw_file.exists():
                raise RuntimeError(f"Missing raw TLE file: {raw_file} (strict mode)")
            got = sha256_file(raw_file)
            if got != exp:
                raise RuntimeError(f"STARLINK-200 raw TLE sha256 mismatch: expected={exp} got={got}")

        # INTERNET-100: built from Topology Zoo archive
        if name == "INTERNET-100":
            raw_file = raw_dir / "topology_zoo_archive.zip"
            exp = src.get("archive_sha256")
            if exp is None:
                raise RuntimeError("INTERNET-100 topology missing source.archive_sha256 (strict mode)")
            if not raw_file.exists():
                raise RuntimeError(f"Missing raw TopologyZoo archive: {raw_file} (strict mode)")
            got = sha256_file(raw_file)
            if got != exp:
                raise RuntimeError(f"INTERNET-100 raw archive sha256 mismatch: expected={exp} got={got}")

        # DSN-30: built from JPL ephemeris (downloaded by Skyfield Loader)
        if name == "DSN-30":
            raw_file = raw_dir / "de421.bsp"
            exp = src.get("ephemeris_sha256")
            if exp is None:
                raise RuntimeError("DSN-30 topology missing source.ephemeris_sha256 (strict mode)")
            if not raw_file.exists():
                raise RuntimeError(f"Missing raw ephemeris file: {raw_file} (strict mode)")
            got = sha256_file(raw_file)
            if got != exp:
                raise RuntimeError(f"DSN-30 ephemeris sha256 mismatch: expected={exp} got={got}")

    def _ensure(name: str, builder_fn, out_path: Path):
        if out_path.exists() and not force_rebuild:
            logger.info(f"Loading cached topology {name} <- {out_path}")
            t = load_topology(out_path)
            if strict:
                _verify_topology_sources(name, t)
            built[name] = t
            return
        logger.info(f"Building topology {name} -> {out_path}")
        t = builder_fn(raw_dir)
        save_topology(t, out_path)
        if strict:
            _verify_topology_sources(name, t)
        built[name] = t

    def _build_starlink(rd):
        from lia.datasets.starlink import StarlinkBuilderConfig, build_starlink_topology
        return build_starlink_topology(rd, StarlinkBuilderConfig())

    def _build_internet(rd):
        from lia.datasets.topology_zoo import TopologyZooBuilderConfig, build_internet_topology
        return build_internet_topology(rd, TopologyZooBuilderConfig())

    def _build_dsn(rd):
        from lia.datasets.dsn import DsnBuilderConfig, build_dsn_topology
        return build_dsn_topology(rd, DsnBuilderConfig())

    builders = {
        "STARLINK-200": (_build_starlink, topo_dir / "STARLINK-200.json"),
        "INTERNET-100": (_build_internet, topo_dir / "INTERNET-100.json"),
        "DSN-30": (_build_dsn, topo_dir / "DSN-30.json"),
    }

    for name in names:
        if name not in builders:
            raise ValueError(f"Unknown topology name: {name}")
        builder_fn, out_path = builders[name]
        _ensure(name, builder_fn, out_path)

    return built
def build_mechanism_fns(cfg: Dict[str, Any]) -> Dict[str, Callable[[AuctionInstance], Any]]:
    mechs: Dict[str, Callable[[AuctionInstance], Any]] = {}

    mech_cfg = cfg.get("mechanisms", {})

    if "sync_vcg" in mech_cfg:
        mechs["SyncVCG"] = run_sync_vcg

    if "fast_vcg" in mech_cfg:
        mechs["FastVCG"] = run_fast_vcg

    if "batch_vcg" in mech_cfg:
        batch_grid = mech_cfg["batch_vcg"].get("batch_ms_grid", [10.0])
        for b in batch_grid:
            b = float(b)
            name = f"BatchVCG_B{b:g}ms"
            mechs[name] = (lambda inst, bb=b: run_batch_vcg(inst, batch_ms=bb))

    if "holdback" in mech_cfg:
        mechs["HoldBack"] = run_holdback

    if "lia" in mech_cfg:
        lia_grid = mech_cfg["lia"].get("lambda_grid", [1.0])

        # Backwards compatibility: older configs used `lambda_mode`.
        lambda_unit = mech_cfg["lia"].get("lambda_unit")
        if lambda_unit is None:
            lambda_unit = mech_cfg["lia"].get("lambda_mode", "per_s")

        use_est = bool(mech_cfg["lia"].get("use_estimated_slack", True))
        clamp_est = bool(mech_cfg["lia"].get("clamp_estimated_slack", True))

        for lam in lia_grid:
            lam = float(lam)
            name = f"LIA_lambda{lam:g}_{lambda_unit}"
            mechs[name] = (
                lambda inst, ll=lam, lu=lambda_unit, ue=use_est, ce=clamp_est: run_lia(
                    inst,
                    LiaConfig(
                        lambda_value=ll,
                        lambda_unit=str(lu),
                        use_estimated_slack=ue,
                        clamp_estimated_slack=ce,
                    ),
                )
            )

    return mechs


def _new_run_dir(runs_dir: Path, run_name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_name}_{ts}_pid{os.getpid()}"
    out = runs_dir / run_id
    out.mkdir(parents=True, exist_ok=False)
    return out


def run_experiment(config_path: str) -> Path:
    t0 = time.perf_counter()
    cfg = json.loads(Path(config_path).read_text())

    runs_dir = Path(cfg.get("output", {}).get("runs_dir", "runs"))
    run_name = cfg.get("run_name", "lia_run")
    out_dir = _new_run_dir(runs_dir, run_name)

    logger = get_logger("lia", log_path=out_dir / "run.log")

    data_dir = Path("data")

    logger.info(f"Run directory: {out_dir}")
    logger.info(f"Pipeline version: {pipeline_version}")
    try:
        logger.info(f"lia import: {Path(lia.__file__).resolve()}")
    except Exception:
        pass
    logger.info(f"Config: {config_path}")

    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    allowed_topos = [t["name"] for t in cfg.get("topologies", [])]
    ds_cfg = cfg.get("datasets", {})
    datasets_strict = bool(ds_cfg.get("strict", False))
    force_rebuild = bool(ds_cfg.get("rebuild_topologies", False))

    topologies = ensure_topologies(data_dir=data_dir, names=allowed_topos, logger=logger, strict=datasets_strict, force_rebuild=force_rebuild)

    if not topologies:
        raise RuntimeError("No topologies selected")

    base_seed = int(cfg.get("random_seed", 123))

    # Hash topology JSONs for provenance
    hashes: Dict[str, str] = {}
    for name in topologies.keys():
        p = data_dir / "topologies" / f"{name}.json"
        hashes[name] = sha256_file(p)
        logger.info(f"Topology hash {name}: {hashes[name]}")

    # Copy the exact topology artifacts used into the run directory so that a
    # results bundle is self-contained and auditable.
    topo_art_dir = out_dir / "artifacts" / "topologies"
    topo_art_dir.mkdir(parents=True, exist_ok=True)
    topo_artifact_paths: Dict[str, str] = {}
    for name in topologies.keys():
        src_path = data_dir / "topologies" / f"{name}.json"
        dst_path = topo_art_dir / f"{name}.json"
        shutil.copy2(src_path, dst_path)
        topo_artifact_paths[name] = str(dst_path)

    # Capture a code hash for provenance (so results are attributable to a specific
    # code state, even outside of git).
    lia_pkg_dir = Path(lia.__file__).resolve().parent
    code_sha256 = sha256_tree(lia_pkg_dir, include_suffixes=[".py"])
    

    # Capture hashes of downloaded raw datasets (TLEs, ephemerides, TopologyZoo tar, ...)
    raw_hashes: Dict[str, Dict[str, Any]] = {}
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        for rp in raw_dir.rglob("*"):
            if not rp.is_file():
                continue
            rel = str(rp.relative_to(data_dir))
            raw_hashes[rel] = {"sha256": sha256_file(rp), "bytes": rp.stat().st_size}

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
    }

    mech_fns = build_mechanism_fns(cfg)
    if not mech_fns:
        raise RuntimeError("No mechanisms configured")

    value_cfg = cfg.get("value_distribution", {})
    value_low = float(value_cfg.get("low", 0.0))
    value_high = float(value_cfg.get("high", 1000.0))

    horizon_policy = cfg.get(
        "horizon_policy",
        {"type": "pctl_of_shortest_path", "percentile": 0.95, "extra_ms": 0.0},
    )

    meas_cfg = cfg.get("measurement_error", {})
    eps_grid = [float(e) for e in meas_cfg.get("eps_grid_ms", [0.0])]
    meas_model = str(meas_cfg.get("model", "iid_uniform"))
    meas_common_frac = float(meas_cfg.get("common_fraction", 0.0))
    bidder_counts = [int(x) for x in cfg.get("bidder_counts", [10, 20, 30, 40, 50])]
    n_per_setting = int(cfg.get("instances_per_setting", 100))

    discount_rate = float(cfg.get("metrics", {}).get("discount_rate_r_per_ms", 0.0001))

    # Generate instances once and reuse for metrics + LAI
    instances_by_group: Dict[Tuple[float, str, int], List[AuctionInstance]] = {}

    rows: List[Dict[str, Any]] = []

    for eps_ms in eps_grid:
        logger.info(
            f"=== Measurement error eps_ms={eps_ms:g} (model={meas_model}, common_fraction={meas_common_frac:g}) ==="
        )

        for topo_name, topo in topologies.items():
            for n_bidders in bidder_counts:
                seed = derive_seed(base_seed, "instances", eps_ms, topo_name, n_bidders)
                rng = np.random.default_rng(seed)

                logger.info(f"Generating instances: topo={topo_name}, n={n_bidders}, m={n_per_setting}, seed={seed}")

                instances = generate_instances(
                    topology=topo,
                    bidder_count=n_bidders,
                    n_instances=n_per_setting,
                    value_low=value_low,
                    value_high=value_high,
                    horizon_policy=horizon_policy,
                    eps_ms=eps_ms,
                    rng=rng,
                    measurement_error_model=meas_model,
                    measurement_error_common_fraction=meas_common_frac,
                )
                instances_by_group[(eps_ms, topo_name, n_bidders)] = instances

                for inst in tqdm(instances, desc=f"{topo_name}-n{n_bidders}-eps{eps_ms:g}"):
                    for mech_name, mech_fn in mech_fns.items():
                        out = mech_fn(inst)
                        m = compute_instance_metrics(inst, out, discount_rate_r_per_ms=discount_rate)
                        row = asdict(m)
                        row["eps_ms"] = eps_ms
                        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_instance_metrics.csv", index=False)

    # LAI estimates (optional)
    lai_cfg = cfg.get("lai", {})
    lai_enabled = bool(lai_cfg.get("enabled", False))

    lai_rows: List[Dict[str, Any]] = []
    if lai_enabled:
        delta_grid = [float(d) for d in lai_cfg.get("delta_grid_ms", [1.0, 2.0, 5.0, 10.0, 20.0])]
        k = int(lai_cfg.get("sample_bidders_per_instance", 5))

        logger.info("=== Estimating LAI (expectation-first; then supremum over delta-grid) ===")

        for eps_ms in eps_grid:
            for topo_name in topologies.keys():
                for n_bidders in bidder_counts:
                    instances_group = list(instances_by_group[(eps_ms, topo_name, n_bidders)])

                    # Optional subsampling to keep LAI computation tractable.
                    max_inst = int(lai_cfg.get("max_instances_per_group", 0) or 0)
                    if max_inst > 0 and len(instances_group) > max_inst:
                        seed = derive_seed(base_seed, "lai_subsample", eps_ms, topo_name, n_bidders)
                        sub_rng = np.random.default_rng(seed)
                        idx = sub_rng.choice(len(instances_group), size=max_inst, replace=False)
                        instances_group = [instances_group[i] for i in idx]

                    for mech_name, mech_fn in mech_fns.items():
                        seed_sample = derive_seed(base_seed, "lai_sample", eps_ms, topo_name, n_bidders, mech_name)
                        seed_boot = derive_seed(base_seed, "lai_boot", eps_ms, topo_name, n_bidders, mech_name)
                        lai_rng = np.random.default_rng(seed_sample)
                        boot_rng = np.random.default_rng(seed_boot)

                        boot_cfg = cfg.get("bootstrap", {})
                        global_boot_enabled = bool(boot_cfg.get("enabled", False))
                        global_boot_n = int(boot_cfg.get("n_resamples", 0) or 0)
                        boot_resamples = int(lai_cfg.get("bootstrap_resamples", (global_boot_n if global_boot_enabled else 0)) or 0)
                        est = estimate_lai(
                            instances=instances_group,
                            mechanism_fn=mech_fn,
                            delta_grid_ms=delta_grid,
                            sample_bidders_per_instance=k,
                            rng=lai_rng,
                            bootstrap_resamples=boot_resamples,
                            bootstrap_rng=boot_rng,
                        )

                        lai_rows.append(
                            {
                                "eps_ms": eps_ms,
                                "topology": topo_name,
                                "bidder_count": int(n_bidders),
                                "mechanism": mech_name,
                                "g1ms": est.g1ms,
                                "g1ms_ci_low": est.g1ms_ci_low,
                                "g1ms_ci_high": est.g1ms_ci_high,
                                "sup_g": est.sup_g,
                                "sup_g_ci_low": est.sup_g_ci_low,
                                "sup_g_ci_high": est.sup_g_ci_high,
                                "instance_count": est.instance_count,
                                "bidder_sample_count": est.bidder_sample_count,
                                "delta_grid_ms": json.dumps(est.delta_grid_ms),
                                "g_delta": json.dumps(est.g_delta),
                                "g_delta_ci_low": json.dumps(est.g_delta_ci_low) if est.g_delta_ci_low is not None else None,
                                "g_delta_ci_high": json.dumps(est.g_delta_ci_high) if est.g_delta_ci_high is not None else None,
                            }
                        )

        pd.DataFrame(lai_rows).to_csv(out_dir / "lai_estimates.csv", index=False)

    # Summary with bootstrap CIs
    boot_cfg = cfg.get("bootstrap", {})
    boot_enabled = bool(boot_cfg.get("enabled", False))
    n_boot = int(boot_cfg.get("n_resamples", 10000))

    summary_rows: List[Dict[str, Any]] = []

    group_cols = ["eps_ms", "topology", "mechanism"]
    for (eps_ms, topo_name, mech), sub in df.groupby(group_cols):
        row: Dict[str, Any] = {
            "eps_ms": float(eps_ms),
            "topology": topo_name,
            "mechanism": mech,
            "n_instances": int(len(sub)),
        }

        for metric in [
            "feasible_fraction",
            "feasible_opt_coverage",
            "welfare_ratio",
            "welfare_ratio_all",
            "welfare_ratio_feasible",
            "revenue_ratio",
            "revenue_ratio_all",
            "revenue_ratio_feasible",
            "compute_time_s",
            "clearing_latency_ms",
            "effective_welfare",
            "clique_number_w",
        ]:
            vals = sub[metric].to_numpy(dtype=float)
            if boot_enabled:
                seed = derive_seed(base_seed, "bootstrap", eps_ms, topo_name, mech, metric)
                boot_rng = np.random.default_rng(seed)
                ci = bootstrap_mean_ci(vals, n_resamples=n_boot, rng=boot_rng)
                if ci is None:
                    continue
                row[f"{metric}_mean"] = ci.mean
                row[f"{metric}_ci_low"] = ci.low
                row[f"{metric}_ci_high"] = ci.high
                row[f"{metric}_n"] = ci.n
            else:
                row[f"{metric}_mean"] = float(np.nanmean(vals))

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_table.csv", index=False)

    # Also provide a bidder-count disaggregated summary for scalability plots.
    summary_n_rows: List[Dict[str, Any]] = []
    group_cols_n = ["eps_ms", "topology", "bidder_count", "mechanism"]
    for (eps_ms, topo_name, n_bidders, mech), sub in df.groupby(group_cols_n):
        row: Dict[str, Any] = {
            "eps_ms": float(eps_ms),
            "topology": topo_name,
            "bidder_count": int(n_bidders),
            "mechanism": mech,
            "n_instances": int(len(sub)),
        }
        for metric in [
            "feasible_fraction",
            "feasible_opt_coverage",
            "welfare_ratio",
            "welfare_ratio_all",
            "welfare_ratio_feasible",
            "revenue_ratio",
            "revenue_ratio_all",
            "revenue_ratio_feasible",
            "compute_time_s",
            "clearing_latency_ms",
            "effective_welfare",
            "clique_number_w",
        ]:
            vals = sub[metric].to_numpy(dtype=float)
            if boot_enabled:
                seed = derive_seed(base_seed, "bootstrap_n", eps_ms, topo_name, n_bidders, mech, metric)
                boot_rng = np.random.default_rng(seed)
                ci = bootstrap_mean_ci(vals, n_resamples=n_boot, rng=boot_rng)
                if ci is None:
                    continue
                row[f"{metric}_mean"] = ci.mean
                row[f"{metric}_ci_low"] = ci.low
                row[f"{metric}_ci_high"] = ci.high
                row[f"{metric}_n"] = ci.n
            else:
                row[f"{metric}_mean"] = float(np.nanmean(vals))
        summary_n_rows.append(row)

    if summary_n_rows:
        pd.DataFrame(summary_n_rows).to_csv(out_dir / "summary_by_bidder_count.csv", index=False)


    # Paired bootstrap differences (instance-level pairing across mechanisms)
    paired_rows: List[Dict[str, Any]] = []
    diff_metrics = [
        "welfare_ratio",
        "welfare_ratio_feasible",
        "revenue_ratio",
        "revenue_ratio_feasible",
        "clearing_latency_ms",
        "effective_welfare",
        "compute_time_s",
    ]

    for (eps_ms, topo_name), g in df.groupby(["eps_ms", "topology"], sort=False):
        g = g.copy()
        g["instance_key"] = g["bidder_count"].astype(str) + ":" + g["instance_id"].astype(str)

        mechs_present = sorted(g["mechanism"].unique().tolist())
        lia_mechs = [m for m in mechs_present if m.startswith("LIA_lambda")]
        other_mechs = [m for m in mechs_present if not m.startswith("LIA_lambda")]

        for metric in diff_metrics:
            for lia_mech in lia_mechs:
                for baseline_mech in other_mechs:
                    pivot = g.pivot(index="instance_key", columns="mechanism", values=metric)
                    if lia_mech not in pivot.columns or baseline_mech not in pivot.columns:
                        continue

                    diff = (pivot[lia_mech] - pivot[baseline_mech]).to_numpy()
                    diff = diff[np.isfinite(diff)]
                    if diff.size == 0:
                        continue

                    rng_pair = np.random.default_rng(
                        derive_seed(base_seed, "paired", eps_ms, topo_name, metric, lia_mech, baseline_mech)
                    )
                    ci = bootstrap_mean_ci(diff, n_resamples=n_boot, rng=rng_pair)
                    if ci is None:
                        continue

                    paired_rows.append(
                        {
                            "eps_ms": eps_ms,
                            "topology": topo_name,
                            "metric": metric,
                            "lia_mechanism": lia_mech,
                            "baseline_mechanism": baseline_mech,
                            "mean_diff": ci.mean,
                            "ci_low": ci.low,
                            "ci_high": ci.high,
                            "n": ci.n,
                        }
                    )

    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(out_dir / "paired_differences.csv", index=False)
        logger.info("Saved paired bootstrap differences to paired_differences.csv")

    run_seconds = float(time.perf_counter() - t0)
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": pipeline_version,
        "lia_import": str(Path(lia.__file__).resolve()) if getattr(lia, "__file__", None) else None,
        "datasets_strict": bool(datasets_strict),
        "rebuild_topologies": bool(force_rebuild),
        "config_path": str(config_path),
        "config_sha256": sha256_file(out_dir / "config.json"),
        "code_sha256": code_sha256,
        "env": env,
        "run_seconds": run_seconds,
        "topology_hashes": hashes,
        "topology_artifacts": topo_artifact_paths,
        "raw_dataset_hashes": raw_hashes,
        "n_metric_rows": int(len(df)),
        "lai_enabled": bool(lai_enabled),
        "bootstrap_enabled": bool(boot_enabled),
        "notes": "No precomputed outputs; all results generated by this run.",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("Done")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiA experiments on real topology datasets")
    parser.add_argument("--config", type=str, default="config/quick.json")
    args = parser.parse_args()

    out_dir = run_experiment(args.config)
    print(f"Run completed. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
