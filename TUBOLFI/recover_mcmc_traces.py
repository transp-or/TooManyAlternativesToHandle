#!/usr/bin/env python3
"""Regenerate missing SA TURBOLFI posterior traces from saved GP checkpoints.

This does not retrain BOLFI.  It restores each saved GP checkpoint and reruns
the MCMC stage, saving ``MCMC_trace_early_stop.nc`` beside that checkpoint.
The regenerated NetCDF files are then consumed by the 95%-metrics pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import run_bolfi_canonical_v2 as canonical


CONFIGS = [(j, n) for j in (100, 200, 500) for n in (10_000, 20_000, 50_000)]
ROOT = Path(__file__).resolve().parent / "final_result" / "SA"


def checkpoint_for(j: int, n: int) -> Path:
    folder = f"J{j}_P10_N{n // 1000}k_I0C0M0_D5+6_SIM30"
    return ROOT / folder / "bolfi_training_results_summary(5+6).pkl"


def recover(j: int, n: int, overwrite: bool = False) -> Path:
    checkpoint = checkpoint_for(j, n)
    output = checkpoint.parent / "MCMC_trace_early_stop.nc"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing GP checkpoint: {checkpoint}")
    if output.exists() and not overwrite:
        print(f"Keeping existing trace: {output}")
        return output

    args = SimpleNamespace(
        J=j, N=n, P=10, I=0, C=0, M=0, sim_num=30, dis_type="5+6",
        simulator_type="SA", seed=canonical.SA_DATASET_SEEDS[(j, n)],
        gp_route=str(checkpoint),
    )
    estimator = canonical.gp_retrive(
        args, canonical.DEVICE, tol_quantile=0.85, notes="early_stop"
    )
    if estimator is None:
        raise RuntimeError(f"Could not restore GP checkpoint: {checkpoint}")
    canonical.MCMC_sampling(args, estimator, canonical.DEVICE, notes="early_stop")
    if not output.exists():
        raise RuntimeError(f"MCMC finished without writing expected trace: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--J", type=int, choices=(100, 200, 500))
    parser.add_argument("--N", type=int, choices=(10_000, 20_000, 50_000))
    parser.add_argument("--all", action="store_true", help="recover all nine configurations")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.all == (args.J is not None or args.N is not None):
        parser.error("pass --all, or pass both --J and --N for one configuration")
    configs = CONFIGS if args.all else [(args.J, args.N)]
    for j, n in configs:
        print(f"\nRecovering MCMC trace for J={j}, N={n}...")
        print(recover(j, n, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
