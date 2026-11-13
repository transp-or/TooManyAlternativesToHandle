#!/usr/bin/env python3
"""Create one validated, 95%-interval metrics table for the estimator comparison.

The program deliberately refuses to write a master table until all 9 datasets
for every estimator are present.  This prevents an incomplete estimator from
silently changing aggregate coverage or interval-width statistics.
"""

from __future__ import annotations

import re
import os
import subprocess
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
TURBOLFI_ROOT = REPO_ROOT / "TUBOLFI" / "final_result" / "SA"
BI_ROOT = (
    REPO_ROOT.parents[1]
    / "projects/Bayesian/bayesian-choice-modeling-w-alt-sampling"
    / "results/results_pymc/full_C/p10"
)

PARAMETERS = [
    "beta_rating", "beta_cost", "beta_log_dist", "beta_chinese",
    "beta_japanese", "beta_korean", "beta_indian", "beta_french",
    "beta_mexican", "beta_lebanese",
]
TRUE_VALUES = {
    "beta_rating": 0.75, "beta_cost": -0.40, "beta_log_dist": -0.60,
    "beta_chinese": 0.83, "beta_japanese": 1.25, "beta_korean": 0.75,
    "beta_indian": 1.00, "beta_french": 0.95, "beta_mexican": 1.30,
    "beta_lebanese": 0.90,
}
EXPECTED_CONFIGS = {(j, n, 10) for j in (100, 200, 500) for n in (10_000, 20_000, 50_000)}
BI_POSTERIOR_DRAWS_PER_CHAIN = 500
OUTPUT_COLUMNS = [
    "J", "N", "P", "parameter", "estimator", "posterior_mean",
    "posterior_sd", "hdi_low", "hdi_high", "true_value", "in_hdi",
    "interval_width", "relative_width", "r_hat", "ess_bulk", "ess_tail",
    "abs_error", "sq_error", "file",
    # Preserve complete original timing metadata after BI draw standardization.
    "n_chains", "source_draws_per_chain", "posterior_draws_per_chain",
    "posterior_total_draws", "draw_policy", "tuning_steps", "sampling_time_s", "run_time_s",
]


def parse_config(text: str) -> tuple[int, int, int] | None:
    """Extract (J, N, P) from either folder or trace-file naming convention."""
    j = re.search(r"J(\d+)", text)
    p = re.search(r"P(\d+)", text)
    n = re.search(r"N(\d+)(k)?", text)
    if not (j and n and p):
        return None
    n_value = int(n.group(1)) * (1000 if n.group(2) else 1)
    return int(j.group(1)), n_value, int(p.group(1))


def trace_index(root: Path, pattern: str) -> dict[tuple[int, int, int], Path]:
    indexed: dict[tuple[int, int, int], Path] = {}
    for path in sorted(root.glob(pattern)):
        config = parse_config(str(path))
        if config is None or config[2] != 10:
            continue
        if config in indexed:
            raise ValueError(f"Duplicate trace for {config}: {indexed[config]} and {path}")
        indexed[config] = path
    return indexed


def input_audit() -> pd.DataFrame:
    sources = {
        "TURBOLFI": trace_index(TURBOLFI_ROOT, "**/MCMC_trace_early_stop.nc"),
        "AdaptiveSampler": trace_index(RESULTS / "adaptive_sampler", "trace_*.nc"),
        "BI": trace_index(BI_ROOT, "restaurants_mnl_*.nc"),
    }
    rows = []
    for estimator, traces in sources.items():
        for config in sorted(EXPECTED_CONFIGS):
            path = traces.get(config)
            rows.append({
                "estimator": estimator, "J": config[0], "N": config[1], "P": config[2],
                "status": "available" if path else "missing",
                "source_file": str(path) if path else "",
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS / "metrics_95ci_input_audit.csv", index=False)
    return audit


def rows_from_trace(path: Path, estimator: str) -> pd.DataFrame:
    config = parse_config(str(path))
    if config is None:
        raise ValueError(f"Cannot parse configuration from {path}")
    idata = az.from_netcdf(path)
    full_posterior = idata.posterior
    available = set(full_posterior.data_vars)
    missing = set(PARAMETERS) - available
    unexpected = available - set(PARAMETERS)
    if missing:
        raise ValueError(f"{path} is missing posterior parameters: {sorted(missing)}")

    n_chains = int(full_posterior.sizes.get("chain", 0))
    source_draws = int(full_posterior.sizes.get("draw", 0))
    if n_chains != 4:
        raise ValueError(f"Expected four chains in {path}, found {n_chains}")
    target_draws = BI_POSTERIOR_DRAWS_PER_CHAIN if estimator == "BI" else source_draws
    if source_draws < target_draws:
        raise ValueError(
            f"{path} has only {source_draws} posterior draws per chain; "
            f"need at least {target_draws}"
        )

    # Only BI is harmonized to 500 post-warmup draws per chain because two BI
    # traces have 1,000 draws while the other BI traces have 500.  Adaptive
    # Sampler and TURBOLFI retain all draws configured for their own runs.
    # Timing metadata remains the original complete run cost.
    posterior = full_posterior[PARAMETERS].isel(draw=slice(0, target_draws))
    trace_attrs = full_posterior.attrs
    summary = az.summary(posterior, var_names=PARAMETERS, hdi_prob=0.95, round_to=None)
    hdi_columns = sorted(
        (column for column in summary.columns if column.startswith("hdi_")),
        key=lambda column: float(column.removeprefix("hdi_").removesuffix("%")),
    )
    if hdi_columns != ["hdi_2.5%", "hdi_97.5%"]:
        raise ValueError(f"Expected 95% HDI columns, got {hdi_columns} for {path}")

    rows = []
    for parameter in PARAMETERS:
        values = summary.loc[parameter]
        low, high = float(values[hdi_columns[0]]), float(values[hdi_columns[1]])
        truth = TRUE_VALUES[parameter]
        mean = float(values["mean"])
        rows.append({
            "J": config[0], "N": config[1], "P": config[2], "parameter": parameter,
            "estimator": estimator, "posterior_mean": mean,
            "posterior_sd": float(values["sd"]), "hdi_low": low, "hdi_high": high,
            "true_value": truth, "in_hdi": low <= truth <= high,
            "interval_width": high - low, "relative_width": (high - low) / abs(truth),
            "r_hat": values.get("r_hat", np.nan), "ess_bulk": values.get("ess_bulk", np.nan),
            "ess_tail": values.get("ess_tail", np.nan), "abs_error": abs(mean - truth),
            "sq_error": (mean - truth) ** 2, "file": path.name,
            "n_chains": n_chains,
            "source_draws_per_chain": source_draws,
            "posterior_draws_per_chain": target_draws,
            "posterior_total_draws": n_chains * target_draws,
            "draw_policy": "BI: first 500 post-warmup draws per chain; others: all draws",
            "tuning_steps": trace_attrs.get("tuning_steps", np.nan),
            "sampling_time_s": trace_attrs.get("sampling_time", np.nan),
            "run_time_s": trace_attrs.get("run_time_seconds", np.nan),
        })
    if unexpected:
        print(f"Ignoring non-parameter posterior variables in {path.name}: {sorted(unexpected)}")
    return pd.DataFrame(rows)


def load_mle() -> pd.DataFrame:
    path = RESULTS / "mle_sa_comparison_ready_no_starting_betas.csv"
    mle = pd.read_csv(path).copy()
    mle["file"] = path.name
    mle = mle[mle["parameter"].isin(PARAMETERS)].copy()
    if set(mle[["J", "N", "P"]].itertuples(index=False, name=None)) != EXPECTED_CONFIGS:
        raise ValueError("MLE source does not contain exactly the nine required configurations")
    ratio = mle["interval_width"] / (2 * mle["posterior_sd"])
    if not np.allclose(ratio, 1.96, rtol=1e-8, atol=1e-8):
        raise ValueError("MLE intervals are not the expected 95% Wald intervals (±1.96 SE)")
    mle["draw_policy"] = "not applicable: frequentist Wald interval"
    return mle.reindex(columns=OUTPUT_COLUMNS)


def recover_missing_turbolfi_traces(audit: pd.DataFrame) -> pd.DataFrame:
    """Restore missing SA GP checkpoints and rerun their MCMC stage.

    The checkpoint is *not* treated as posterior samples.  The companion
    recovery program reconstructs the GP and writes NetCDF posterior draws;
    this module then calculates the 95% HDIs from those draws.
    """
    missing = audit.query("estimator == 'TURBOLFI' and status == 'missing'")
    if missing.empty:
        return audit
    recovery_script = REPO_ROOT / "TUBOLFI" / "recover_mcmc_traces.py"
    python = os.environ.get("TURBOLFI_PYTHON", sys.executable)
    command = [python, str(recovery_script), "--all"]
    print("Recovering missing TURBOLFI posterior traces:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return input_audit()


def run(recover_turbolfi: bool = False) -> Path:
    audit = input_audit()
    if recover_turbolfi:
        audit = recover_missing_turbolfi_traces(audit)
    missing = audit[audit["status"] == "missing"]
    if not missing.empty:
        print(missing.to_string(index=False))
        raise RuntimeError(
            "Not writing 95% metrics: posterior traces are missing. "
            "Restore or regenerate the listed MCMC_trace_early_stop.nc files first."
        )

    trace_sources = {
        "TURBOLFI": trace_index(TURBOLFI_ROOT, "**/MCMC_trace_early_stop.nc"),
        "AdaptiveSampler": trace_index(RESULTS / "adaptive_sampler", "trace_*.nc"),
        "BI": trace_index(BI_ROOT, "restaurants_mnl_*.nc"),
    }
    frames = [load_mle()]
    per_estimator = {"MLE": frames[0]}
    for estimator, sources in trace_sources.items():
        frame = pd.concat([rows_from_trace(sources[config], estimator) for config in sorted(EXPECTED_CONFIGS)])
        frame = frame.reindex(columns=OUTPUT_COLUMNS)
        per_estimator[estimator] = frame
        frames.append(frame)

    for estimator, frame in per_estimator.items():
        frame.to_csv(RESULTS / f"{estimator.lower()}_95ci_metrics.csv", index=False)
    master = pd.concat(frames, ignore_index=True).sort_values(
        ["J", "N", "P", "estimator", "parameter"]
    )
    expected_rows = len(EXPECTED_CONFIGS) * len(PARAMETERS) * len(per_estimator)
    if len(master) != expected_rows or master.duplicated(["J", "N", "P", "estimator", "parameter"]).any():
        raise ValueError("Master metrics table is incomplete or contains duplicate parameter rows")
    output = RESULTS / "all_estimators_95ci_metrics.csv"
    master.to_csv(output, index=False)
    print(f"Wrote {len(master)} validated rows to {output}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recover-turbolfi", action="store_true",
        help="restore missing TURBOLFI GP checkpoints and rerun MCMC before creating metrics",
    )
    args = parser.parse_args()
    run(recover_turbolfi=args.recover_turbolfi)
