"""
Benchmark Bayesian Inference for Logit Model
=============================================

This script runs Bayesian inference on a multinomial logit model WITHOUT alternative sampling.
It serves as a benchmark to compare against the adaptive sampler with alternative sampling.

The model uses the full choice set for all individuals and only applies Metropolis-Hastings
sampling to the beta parameters.
"""

import pymc as pm
# import pymc.sampling_jax as pmjax
import pytensor.tensor as pt
import pytensor
pytensor.config.mode = "NUMBA"
import numpy as np
import pandas as pd
import arviz as az
import argparse
import os
import logging
import matplotlib.pyplot as plt
import pytensor
from datetime import datetime


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptive_sampler.data import load_data
from adaptive_sampler.utils import (
    TracePrinter, 
    create_X_restaurants, 
    compute_interaction_matrices,
    get_features_for_P
)
from adaptive_sampler.true_parameters import get_true_parameters

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(f"output/run_benchmark_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


def run_benchmark_model(n_steps, burnin, tune, random_seed,
                        J=100, N=10000, P=10, I=0, C=0, M=0, suffix=""):
    """
    Run standard Bayesian inference on MNL model with full choice set.
    
    Parameters:
    -----------
    n_steps : int
        Number of MCMC samples to draw
    burnin : int
        Number of initial samples to discard from trace
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
    J : int
        Number of restaurants/alternatives
    N : int
        Number of individuals
    P : int
        Number of parameters (10, 20, or 50)
    I : int
        Data imbalance factor (0 or 1)
    C : int
        Correlation factor (0 or 1)
    M : int
        Choice randomness factor (0 or 1)
        
    Returns:
    --------
    trace : arviz.InferenceData
        Posterior samples (after burnin)
    model : pymc.Model
        The PyMC model object
    true_params : dict
        True parameter values used to generate the data
    """
    
    np.random.seed(random_seed)
    
    # Load data based on configuration
    logger.info(f"Loading data: J={J}, N={N}, P={P}, I={I}, C={C}, M={M}")
    alternatives, observations, features = load_data(J=J, N=N, P=P, I=I, C=C, M=M)
    
    size = observations.shape[0]
    
    # Build restaurant-only and interaction components
    X_rest, rest_features = create_X_restaurants(alternatives, P)
    inter_mats, interact_features = compute_interaction_matrices(observations, alternatives, P)

    # Use true_parameters ordering to define feature order (ensures consistency)
    feature_names = rest_features.copy()
    feature_names.extend(interact_features)
    logger.info(f"Feature names used (total {len(feature_names)}): {feature_names}")
    logger.info(f"Restaurant features: {rest_features}")
    logger.info(f"Interaction features: {interact_features}")
    logger.info("-" * 60)
    
    # Build X_full from components following feature_names order
    N = len(observations)
    J = len(alternatives)
    Kp1 = len(feature_names)
    mats = []
    for feat in feature_names:
        if feat in rest_features:
            mats.append(np.tile(X_rest[:, rest_features.index(feat)], (N, 1)))
        else:
            mats.append(inter_mats.get(feat, np.zeros((N, J), dtype=np.float32)))
    X_full = np.stack(mats, axis=2).astype(np.float32)
    
    # Basic sanity checks & logging
    logger.info(f"X_full.shape: {X_full.shape}")
    if np.isnan(X_full).any():
        raise ValueError("X_full contains NaNs. Check that all required alternative features exist and are finite.")
    
    # Get chosen alternatives
    chosen_vec = observations["logit_0"].values.astype(int)
    logger.info(f"Number of observations: {size}")
    logger.info(f"Number of alternatives: {J}")
    logger.info(f"Number of features: {Kp1}")
    coords = {"feature": feature_names}
    with pm.Model(coords=coords) as model:
        # 1) Define priors for beta parameters
        logger.info(f"Setting up priors for P={P} parameters")
        
        # Set prior scale as a function of P
        if P == 10:
            prior_sigma = 10.0
        elif P == 20:
            prior_sigma = 10.0
        elif P == 40:
            prior_sigma = 10.0
        else:
            prior_sigma = 10.0
        X_data = pm.Data("X_full", X_full)
        beta = pm.Normal("beta", mu=0.0, sigma=prior_sigma, dims="feature")
        V = pt.tensordot(X_data, beta, axes=[[2], [0]])
        
        # gather chosen utilities (N,)
        arng = pt.arange(V.shape[0])
        chosen_util = V[arng, chosen_vec]

        # loglik: sum over individuals (scalar)
        ll = pt.sum(chosen_util - pt.logsumexp(V, axis=1))
        pm.Potential("mnl_loglike", ll)
        logger.info("Model specification complete")
        
        # 4) Sample using NumPyro/NUTS via the PyMC JAX bridge
        logger.info(f"Starting sampling: {n_steps} draws, {tune} tuning steps")

        # Decide how many chains to run. Use available JAX devices but cap to 4
        # to avoid too much memory pressure by default. You can raise this if
        # your node has more devices and memory.
    
        # Use the vectorized chain method for efficient single-host execution.

        trace = pm.sample(
            draws=n_steps,
            tune=tune,
            chains=4,
            cores=None,
            chain_method="vectorized",  # uses JAX vmap-style parallelism
            progressbar=False, # not available for Custom Compound step, creates error :))
            callback=TracePrinter(every=100),
            compile_kwargs={'mode': 'NUMBA'},
        )
        trace_path = os.path.join(full_outdir, f"trace{suffix}.nc")
        az.to_netcdf(trace, "trace.nc", mode="a")
        logger.info("Sampling complete")
        
        return trace.sel(draw=slice(burnin, None)), model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark Bayesian inference for MNL model (no alternative sampling)"
    )
    
    # Sampling parameters
    parser.add_argument("--n_steps", type=int, default=1000,
                       help="Number of MCMC samples to draw")
    parser.add_argument("--burnin", type=int, default=500,
                       help="Number of initial samples to discard")
    parser.add_argument("--tune", type=int, default=1000,
                       help="Number of tuning steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--outdir", type=str, default="output",
                       help="Output directory for results")
    
    # Data configuration parameters
    parser.add_argument("--J", type=int, choices=[100, 200, 500], default=100,
                       help="Number of restaurants")
    parser.add_argument("--N", type=int, choices=[10000, 20000, 50000], default=10000,
                       help="Number of individuals")
    parser.add_argument("--P", type=int, choices=[10, 20, 50], default=10,
                       help="Number of parameters")
    parser.add_argument("--I", type=int, choices=[0, 1], default=0,
                       help="Data imbalance: 0=balanced, 1=imbalanced")
    parser.add_argument("--C", type=int, choices=[0, 1], default=0,
                       help="Correlation/clones: 0=independent, 1=correlated")
    parser.add_argument("--M", type=int, choices=[0, 1], default=0,
                       help="Choice randomness: 0=selective, 1=random")
    
    args = parser.parse_args()
    
    # Create output directory structure
    base_dir = args.outdir
    data_config = f"J{args.J}_N{args.N//1000}k_P{args.P}_I{args.I}_C{args.C}_M{args.M}"
    full_outdir = '../results/results_pymc/'
    
    os.makedirs(full_outdir, exist_ok=True)
    logger.info(f"Saving results to {full_outdir}")
    suffix = f"full_alternatives_J{args.J}_N{args.N}_P{args.P}_I{args.I}_C{args.C}_M{args.M}_steps{args.n_steps}"
    
    logger.info(f"Running benchmark configuration: J={args.J}, N={args.N}, P={args.P}, "
                f"I={args.I}, C={args.C}, M={args.M}")
    logger.info(f"Sampling parameters: steps={args.n_steps}, burnin={args.burnin}, "
                f"tune={args.tune}, seed={args.seed}")
    
    # Run the benchmark model
    trace, model = run_benchmark_model(
        n_steps=args.n_steps,
        burnin=args.burnin,
        tune=args.tune,
        random_seed=args.seed,
        J=args.J,
        N=args.N,
        P=args.P,
        I=args.I,
        C=args.C,
        M=args.M,
        suffix=suffix
    )
    logger.info("Sampling complete.")
    
    # Save trace
    trace.to_netcdf(trace_path)
    logger.info(f"Trace saved to {trace_path}")
    
    # # Save summary statistics
    # beta_vars = [v for v in trace.posterior.data_vars if v.startswith("beta_")]
    # summary = az.summary(trace, var_names=beta_vars, stat_focus="mean", round_to=None)
    # summary["true_value"] = summary.index.map(lambda name: true_params.get(name.lower(), np.nan))
    
    # # Compute diagnostic statistics
    # p_greater, p_less, credible_includes_true = [], [], []
    # for param in summary.index:
    #     true_val = summary.loc[param, "true_value"]
    #     if not np.isnan(true_val):
    #         samples = trace.posterior[param].values.ravel()
    #         p_greater.append((samples > true_val).mean())
    #         p_less.append((samples < true_val).mean())
    #         hdi_94 = az.hdi(trace, var_names=[param], hdi_prob=0.94).to_dataframe()
    #         lower = hdi_94.loc[param, f"{param}[0]"] if f"{param}[0]" in hdi_94.columns else hdi_94.iloc[0, 0]
    #         upper = hdi_94.loc[param, f"{param}[1]"] if f"{param}[1]" in hdi_94.columns else hdi_94.iloc[0, 1]
    #         credible_includes_true.append(lower <= true_val <= upper)
    #     else:
    #         p_greater.append(np.nan)
    #         p_less.append(np.nan)
    #         credible_includes_true.append(np.nan)
    
    # summary["P(> true)"] = p_greater
    # summary["P(< true)"] = p_less
    # summary["94% HDI includes true"] = credible_includes_true
    # summary = summary.rename(columns={"mean": "posterior_mean", "sd": "std_error"})
    
    # # Save summary in multiple formats
    # summary_path_csv = os.path.join(full_outdir, f"summary{suffix}.csv")
    # summary.to_csv(summary_path_csv)
    # summary.to_latex(os.path.join(full_outdir, f"summary{suffix}.tex"))
    # with open(os.path.join(full_outdir, f"summary{suffix}.txt"), "w") as f:
    #     f.write(summary.to_string())
    # logger.info(f"Summary saved to {summary_path_csv}")
    
    # # Print summary to console
    # logger.info("\n" + "="*80)
    # logger.info("POSTERIOR SUMMARY")
    # logger.info("="*80)
    # logger.info("\n" + summary.to_string())
    
    # # Compute and log overall diagnostics
    # logger.info("\n" + "="*80)
    # logger.info("OVERALL DIAGNOSTICS")
    # logger.info("="*80)
    # coverage = np.nanmean(credible_includes_true)
    # logger.info(f"94% HDI coverage: {coverage:.2%}")
    
    # # Compute RMSE and MAE
    # valid_mask = ~np.isnan(summary["true_value"])
    # if valid_mask.any():
    #     errors = summary.loc[valid_mask, "posterior_mean"] - summary.loc[valid_mask, "true_value"]
    #     rmse = np.sqrt((errors ** 2).mean())
    #     mae = np.abs(errors).mean()
    #     logger.info(f"RMSE: {rmse:.4f}")
    #     logger.info(f"MAE: {mae:.4f}")
    
    # logger.info("\nBenchmark run complete!")
    # logger.info(f"Results saved to: {full_outdir}")
    
    # Optional: Plot trace (commented out by default to avoid display issues)
    # az.plot_trace(trace, var_names=beta_vars[:5])  # Plot first 5 params
    # plt.tight_layout()
    # plot_path = os.path.join(full_outdir, f"trace_plot{suffix}.png")
    # plt.savefig(plot_path)
    # logger.info(f"Trace plot saved to {plot_path}")
