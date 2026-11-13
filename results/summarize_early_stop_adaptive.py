#!/usr/bin/env python3
"""Summarize early-stop Adaptive Sampler traces without overwriting baseline metrics.

Outputs two deliberately new files in ``results/early_stop``:

* ``adaptivesampler_earlystop_95ci_metrics_261030.csv`` — one 95% HDI row per
  parameter and configuration.
* ``adaptivesampler_earlystop_vs_baseline_261030.csv`` — parameter-level
  deltas against the archived baseline Adaptive Sampler metrics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recreate_metrics_95ci import OUTPUT_COLUMNS, rows_from_trace


RESULTS = Path(__file__).resolve().parent
EARLY_STOP_ROOT = RESULTS / "early_stop" / "adaptive_sampler"
BASELINE = RESULTS / "adaptivesampler_95ci_metrics.csv"
METRICS_OUTPUT = RESULTS / "early_stop" / "adaptivesampler_earlystop_95ci_metrics_261030.csv"
COMPARISON_OUTPUT = RESULTS / "early_stop" / "adaptivesampler_earlystop_vs_baseline_261030.csv"
SUMMARY_OUTPUT = RESULTS / "early_stop" / "adaptivesampler_earlystop_vs_baseline_summary_261030.csv"
KEYS = ["J", "N", "P", "parameter"]


def main() -> None:
    traces = sorted(EARLY_STOP_ROOT.glob("trace_*.nc"))
    if not traces:
        raise FileNotFoundError(f"No NetCDF traces found in {EARLY_STOP_ROOT}")

    early_stop = pd.concat(
        [rows_from_trace(path, "AdaptiveSamplerEarlyStop") for path in traces],
        ignore_index=True,
    ).reindex(columns=OUTPUT_COLUMNS)
    if early_stop.duplicated(KEYS).any():
        raise ValueError("Duplicate early-stop trace configurations found")
    early_stop.to_csv(METRICS_OUTPUT, index=False)

    baseline = pd.read_csv(BASELINE)
    baseline = baseline[baseline["estimator"].eq("AdaptiveSampler")].copy()
    compare = early_stop.merge(
        baseline[KEYS + ["posterior_mean", "abs_error", "sq_error", "interval_width", "in_hdi", "r_hat", "ess_bulk", "run_time_s"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("_early_stop", "_baseline"),
    )
    if compare["posterior_mean_baseline"].isna().any():
        raise ValueError("Some early-stop rows have no corresponding baseline metrics")

    compare["mean_difference"] = compare["posterior_mean_early_stop"] - compare["posterior_mean_baseline"]
    compare["abs_error_change"] = compare["abs_error_early_stop"] - compare["abs_error_baseline"]
    compare["sq_error_change"] = compare["sq_error_early_stop"] - compare["sq_error_baseline"]
    compare["interval_width_change"] = compare["interval_width_early_stop"] - compare["interval_width_baseline"]
    compare["interval_width_ratio"] = compare["interval_width_early_stop"] / compare["interval_width_baseline"]
    compare["coverage_changed"] = compare["in_hdi_early_stop"] != compare["in_hdi_baseline"]
    compare.to_csv(COMPARISON_OUTPUT, index=False)

    def aggregate(frame: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "mean_abs_error_early_stop": frame["abs_error_early_stop"].mean(),
            "mean_abs_error_baseline": frame["abs_error_baseline"].mean(),
            "rmse_early_stop": frame["sq_error_early_stop"].mean() ** 0.5,
            "rmse_baseline": frame["sq_error_baseline"].mean() ** 0.5,
            "mean_interval_width_early_stop": frame["interval_width_early_stop"].mean(),
            "mean_interval_width_baseline": frame["interval_width_baseline"].mean(),
            "coverage_early_stop": frame["in_hdi_early_stop"].mean(),
            "coverage_baseline": frame["in_hdi_baseline"].mean(),
            "max_r_hat_early_stop": frame["r_hat_early_stop"].max(),
            "max_r_hat_baseline": frame["r_hat_baseline"].max(),
            "mean_ess_bulk_early_stop": frame["ess_bulk_early_stop"].mean(),
            "mean_ess_bulk_baseline": frame["ess_bulk_baseline"].mean(),
        })

    by_configuration = compare.groupby(["J", "N", "P"]).apply(
        aggregate, include_groups=False
    ).reset_index()
    overall = aggregate(compare).to_frame().T.assign(J="all", N="all", P="all")
    overview = pd.concat([overall, by_configuration], ignore_index=True)
    overview.to_csv(SUMMARY_OUTPUT, index=False)
    print(f"Wrote {len(early_stop)} rows to {METRICS_OUTPUT}")
    print(f"Wrote {len(compare)} comparison rows to {COMPARISON_OUTPUT}")
    print(f"Wrote {len(overview)} summary rows to {SUMMARY_OUTPUT}")
    print(overview.to_string())


if __name__ == "__main__":
    main()
