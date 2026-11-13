#!/usr/bin/env python3
"""Build the 95% estimator-comparison table at a shared MCMC stopping rule.

The MCMC estimators are evaluated every 10 post-warmup draws per chain.  A
configuration stops at the first checked prefix for which beta_cost,
beta_rating, and beta_log_dist all pass raw split R-hat <= 1.01 and both
bulk and tail ESS >= 400.  The selected prefix is then used for the 95% HDIs
of all ten model coefficients.
"""

from __future__ import annotations

import re
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
ADAPTIVE_ROOT = RESULTS / "adaptive_sampler"
TURBOLFI_ROOT = REPO_ROOT / "TUBOLFI" / "final_result" / "SA"
BI_ROOT = Path(
    "/Users/preto/Documents/EPFL/PhD/projects/Bayesian/"
    "bayesian-choice-modeling-w-alt-sampling/results/results_pymc/full_C/p10"
)

PARAMETERS = [
    "beta_rating", "beta_cost", "beta_log_dist", "beta_chinese",
    "beta_japanese", "beta_korean", "beta_indian", "beta_french",
    "beta_mexican", "beta_lebanese",
]
STOPPING_PARAMETERS = ["beta_cost", "beta_rating", "beta_log_dist"]
TRUE_VALUES = {
    "beta_rating": 0.75, "beta_cost": -0.40, "beta_log_dist": -0.60,
    "beta_chinese": 0.83, "beta_japanese": 1.25, "beta_korean": 0.75,
    "beta_indian": 1.00, "beta_french": 0.95, "beta_mexican": 1.30,
    "beta_lebanese": 0.90,
}
EXPECTED_CONFIGS = {
    (j, n, 10) for j in (100, 200, 500) for n in (10_000, 20_000, 50_000)
}
CHECK_EVERY_DRAWS = 10
ADAPTIVE_TUNING_DRAWS = 1_000
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400
OUTPUT = RESULTS / "all_estimators_95ci_metrics_stopping_criterion.csv"

BASE_COLUMNS = [
    "J", "N", "P", "parameter", "estimator", "posterior_mean",
    "posterior_sd", "hdi_low", "hdi_high", "true_value", "in_hdi",
    "interval_width", "relative_width", "r_hat", "ess_bulk", "ess_tail",
    "abs_error", "sq_error", "file", "n_chains",
    "source_draws_per_chain", "posterior_draws_per_chain",
    "posterior_total_draws", "draw_policy", "tuning_steps",
    "sampling_time_s", "run_time_s",
]
STOPPING_COLUMNS = [
    "stored_draws_per_chain", "tuning_draws_discarded",
    "stopping_draws_per_chain", "stopping_total_draws", "check_every_draws",
    "stopping_criterion", "stopping_criterion_met",
]
DIAGNOSTIC_COLUMNS = [
    f"{parameter}_{metric}"
    for parameter in STOPPING_PARAMETERS
    for metric in ("raw_r_hat", "ess_bulk", "ess_tail")
]
OUTPUT_COLUMNS = BASE_COLUMNS + STOPPING_COLUMNS + DIAGNOSTIC_COLUMNS


def parse_config(text: str) -> tuple[int, int, int] | None:
    """Return (J, N, P), accepting both N10k and N10000 naming."""
    j, n, p = (re.search(pattern, text) for pattern in (r"J(\d+)", r"N(\d+)(k)?", r"P(\d+)"))
    if not (j and n and p):
        return None
    n_value = int(n.group(1)) * (1_000 if n.group(2) else 1)
    return int(j.group(1)), n_value, int(p.group(1))


def index_traces(root: Path, pattern: str) -> dict[tuple[int, int, int], Path]:
    traces: dict[tuple[int, int, int], Path] = {}
    for path in sorted(root.glob(pattern)):
        config = parse_config(str(path))
        if config is None or config[2] != 10:
            continue
        if config in traces:
            raise ValueError(f"Duplicate trace for {config}: {traces[config]} and {path}")
        traces[config] = path
    missing = EXPECTED_CONFIGS - traces.keys()
    extra = traces.keys() - EXPECTED_CONFIGS
    if missing or extra:
        raise ValueError(f"Trace inventory mismatch for {root}: missing={sorted(missing)}, extra={sorted(extra)}")
    return traces


def diagnostics_at_prefix(posterior: "az.data.inference_data.xr.Dataset", draws: int) -> dict[str, float]:
    """Raw split R-hat and bulk/tail ESS for the required stopping variables."""
    prefix = posterior[STOPPING_PARAMETERS].isel(draw=slice(0, draws))
    # ``identity`` is the ordinary, unranked split R-hat requested for stopping.
    rhat = az.rhat(prefix, method="identity")
    ess_bulk = az.ess(prefix, method="bulk")
    ess_tail = az.ess(prefix, method="tail")
    result: dict[str, float] = {}
    for parameter in STOPPING_PARAMETERS:
        result[f"{parameter}_raw_r_hat"] = float(rhat[parameter].item())
        result[f"{parameter}_ess_bulk"] = float(ess_bulk[parameter].item())
        result[f"{parameter}_ess_tail"] = float(ess_tail[parameter].item())
    return result


def meets_stopping_criterion(diagnostics: dict[str, float]) -> bool:
    return all(
        diagnostics[f"{parameter}_raw_r_hat"] <= RHAT_THRESHOLD
        and diagnostics[f"{parameter}_ess_bulk"] >= ESS_THRESHOLD
        and diagnostics[f"{parameter}_ess_tail"] >= ESS_THRESHOLD
        for parameter in STOPPING_PARAMETERS
    )


def find_stopping_prefix(posterior: "az.data.inference_data.xr.Dataset") -> tuple[int, bool, dict[str, float]]:
    """Find the first qualifying 10-draw checkpoint, or retain the final prefix."""
    total_draws = int(posterior.sizes["draw"])
    if total_draws < CHECK_EVERY_DRAWS:
        raise ValueError(f"Need at least {CHECK_EVERY_DRAWS} post-warmup draws, found {total_draws}")
    last_diagnostics: dict[str, float] | None = None
    for draws in range(CHECK_EVERY_DRAWS, total_draws + 1, CHECK_EVERY_DRAWS):
        diagnostics = diagnostics_at_prefix(posterior, draws)
        if meets_stopping_criterion(diagnostics):
            return draws, True, diagnostics
        last_diagnostics = diagnostics
    # A final non-multiple checkpoint is still reported, but is not an extra check.
    if total_draws % CHECK_EVERY_DRAWS:
        last_diagnostics = diagnostics_at_prefix(posterior, total_draws)
    assert last_diagnostics is not None
    return total_draws, False, last_diagnostics


def rows_from_trace(path: Path, estimator: str) -> pd.DataFrame:
    config = parse_config(str(path))
    if config is None:
        raise ValueError(f"Cannot parse configuration from {path}")
    idata = az.from_netcdf(path)
    stored = idata.posterior
    missing = set(PARAMETERS) - set(stored.data_vars)
    if missing:
        raise ValueError(f"{path} is missing: {sorted(missing)}")
    n_chains = int(stored.sizes["chain"])
    if n_chains != 4:
        raise ValueError(f"Expected 4 chains in {path}, found {n_chains}")

    # The Adaptive Sampler NetCDF stores its 1,000 tuning draws as posterior
    # samples without recording them in attrs; the user-specified offset is
    # therefore applied explicitly.  BI and TURBOLFI posterior groups already
    # contain post-warmup samples.
    discarded = ADAPTIVE_TUNING_DRAWS if estimator == "AdaptiveSampler" else 0
    if int(stored.sizes["draw"]) <= discarded:
        raise ValueError(f"{path} has no post-warmup draws after discarding {discarded}")
    post_warmup = stored[PARAMETERS].isel(draw=slice(discarded, None))
    stopping_draws, criterion_met, diagnostics = find_stopping_prefix(post_warmup)
    selected = post_warmup.isel(draw=slice(0, stopping_draws))
    summary = az.summary(selected, var_names=PARAMETERS, hdi_prob=0.95, round_to=None)
    hdi_columns = ["hdi_2.5%", "hdi_97.5%"]
    if not set(hdi_columns).issubset(summary.columns):
        raise ValueError(f"Expected 95% HDI columns in {path}, found {list(summary.columns)}")
    attrs = stored.attrs
    common = {
        "J": config[0], "N": config[1], "P": config[2], "estimator": estimator,
        "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "n_chains": n_chains, "source_draws_per_chain": int(stored.sizes["draw"]),
        "posterior_draws_per_chain": stopping_draws,
        "posterior_total_draws": n_chains * stopping_draws,
        "draw_policy": "first common post-warmup convergence prefix; 95% HDI",
        "tuning_steps": attrs.get("tuning_steps", np.nan),
        "sampling_time_s": attrs.get("sampling_time", np.nan),
        "run_time_s": attrs.get("run_time_seconds", np.nan),
        "stored_draws_per_chain": int(stored.sizes["draw"]),
        "tuning_draws_discarded": discarded,
        "stopping_draws_per_chain": stopping_draws,
        "stopping_total_draws": n_chains * stopping_draws,
        "check_every_draws": CHECK_EVERY_DRAWS,
        "stopping_criterion": "raw split R-hat <= 1.01; bulk ESS >= 400; tail ESS >= 400 for beta_cost, beta_rating, beta_log_dist",
        "stopping_criterion_met": criterion_met,
        **diagnostics,
    }
    rows = []
    for parameter in PARAMETERS:
        values = summary.loc[parameter]
        low, high = float(values[hdi_columns[0]]), float(values[hdi_columns[1]])
        mean = float(values["mean"])
        truth = TRUE_VALUES[parameter]
        rows.append({
            **common, "parameter": parameter, "posterior_mean": mean,
            "posterior_sd": float(values["sd"]), "hdi_low": low, "hdi_high": high,
            "true_value": truth, "in_hdi": low <= truth <= high,
            "interval_width": high - low, "relative_width": (high - low) / abs(truth),
            # Keep the usual rank-normalized ArviZ summary diagnostics as
            # descriptive metrics; only stopping uses the raw R-hat above.
            "r_hat": values.get("r_hat", np.nan), "ess_bulk": values.get("ess_bulk", np.nan),
            "ess_tail": values.get("ess_tail", np.nan), "abs_error": abs(mean - truth),
            "sq_error": (mean - truth) ** 2,
        })
    return pd.DataFrame(rows)


def mle_rows() -> pd.DataFrame:
    source = next((path for path in (
        RESULTS / "mle_sa_comparison_ready_no_starting_betas.csv",
        RESULTS / "csv" / "mle_sa_comparison_ready_no_starting_betas.csv",
    ) if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Missing MLE 95% Wald-interval input CSV")
    frame = pd.read_csv(source).copy()
    frame = frame[frame["parameter"].isin(PARAMETERS)].copy()
    if set(frame[["J", "N", "P"]].itertuples(index=False, name=None)) != EXPECTED_CONFIGS:
        raise ValueError("MLE input does not contain exactly the nine expected configurations")
    if not np.allclose(frame["interval_width"] / (2 * frame["posterior_sd"]), 1.96):
        raise ValueError("MLE input intervals are not 95% Wald intervals (±1.96 SE)")
    frame["file"] = str(source.relative_to(REPO_ROOT))
    frame["n_chains"] = np.nan
    frame["source_draws_per_chain"] = np.nan
    frame["posterior_draws_per_chain"] = np.nan
    frame["posterior_total_draws"] = np.nan
    frame["draw_policy"] = "not applicable: 95% frequentist Wald interval"
    frame["tuning_steps"] = np.nan
    frame["sampling_time_s"] = np.nan
    frame["run_time_s"] = np.nan
    frame["stored_draws_per_chain"] = np.nan
    frame["tuning_draws_discarded"] = np.nan
    frame["stopping_draws_per_chain"] = np.nan
    frame["stopping_total_draws"] = np.nan
    frame["check_every_draws"] = np.nan
    frame["stopping_criterion"] = "not applicable: no MCMC trace"
    frame["stopping_criterion_met"] = np.nan
    for column in DIAGNOSTIC_COLUMNS:
        frame[column] = np.nan
    return frame.reindex(columns=OUTPUT_COLUMNS)


def main() -> Path:
    sources = {
        "BI": index_traces(BI_ROOT, "restaurants_mnl_*.nc"),
        "AdaptiveSampler": index_traces(ADAPTIVE_ROOT, "trace_*.nc"),
        "TURBOLFI": index_traces(TURBOLFI_ROOT, "**/MCMC_trace_early_stop.nc"),
    }
    frames = [mle_rows()]
    for estimator, traces in sources.items():
        frames.append(pd.concat(
            [rows_from_trace(traces[config], estimator) for config in sorted(EXPECTED_CONFIGS)],
            ignore_index=True,
        ))
    master = pd.concat(frames, ignore_index=True).reindex(columns=OUTPUT_COLUMNS)
    master = master.sort_values(["J", "N", "P", "estimator", "parameter"])
    expected_rows = 4 * len(EXPECTED_CONFIGS) * len(PARAMETERS)
    if len(master) != expected_rows or master.duplicated(["J", "N", "P", "estimator", "parameter"]).any():
        raise ValueError("Output does not contain 360 unique estimator/configuration/parameter rows")
    master.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(master)} rows to {OUTPUT}")
    print(master.groupby(["estimator", "stopping_criterion_met"], dropna=False).size())
    return OUTPUT


if __name__ == "__main__":
    main()
