import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
import argparse
import os
import logging
import json
import matplotlib.pyplot as plt
import pytensor
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptive_sampler.alternative_sampler_step import make_alt_step
from adaptive_sampler.data import load_data
from adaptive_sampler.utils import precompute_dist_matrix, TracePrinter, TimeRecorder, create_X_full_matrix, get_features_for_P, create_X_restaurants, compute_interaction_matrices
from adaptive_sampler.true_parameters import get_true_parameters
from adaptive_sampler.reproducibility import (
    set_all_seeds, log_reproducibility_header, get_system_information,
    get_library_versions, validate_data_content, check_x_full_matrix,
    validate_data_files, generate_reproducibility_report
)
from pymc.util import get_value_vars_from_user_vars

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(f"output/run_model_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def rescale_trace(trace, scale_params, feature_names, mode='normalize'):
    """
    Rescale trace parameters back to original scale.
    
    Both modes: beta_original = beta_sampled / std
    (Since both 'normalize' and 'scale' pre-scale X by 1/std)
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Trace with scaled parameters
    scale_params : dict
        Dictionary mapping feature names to {'std': float} or {'mean': float, 'std': float}
    feature_names : list
        List of feature names in order
    mode : str
        'normalize' or 'scale'
        
    Returns:
    --------
    arviz.InferenceData
        Trace with rescaled parameters in original scale
    """
    logger.info("\n" + "=" * 60)
    logger.info("RESCALING TRACE TO ORIGINAL PARAMETER SCALE")
    
    rescaled_trace = trace.copy()
    
    for feat_name in feature_names:
        beta_name = f'beta_{feat_name}'
        
        if beta_name in rescaled_trace.posterior.data_vars:
            std = scale_params[feat_name]['std']
            
            if mode == 'normalize':
                # Feature normalization: divide by std
                rescaled_trace.posterior[beta_name] = rescaled_trace.posterior[beta_name] / std
                logger.info(f"  {beta_name:30s}: beta_orig = beta_norm / {std:.4f}")
            elif mode == 'scale':
                # Beta rescaling: divide by std (same as normalize!)
                rescaled_trace.posterior[beta_name] = rescaled_trace.posterior[beta_name] / std
                logger.info(f"  {beta_name:30s}: beta_orig = beta_scaled / {std:.4f}")
    
    logger.info("=" * 60)
    return rescaled_trace

def binary_logit_potential(V_i_star, V_i_tilde, V_i_s):
    V_i_star_s = V_i_s[:, 0]
    V_i_tilde_s = V_i_s[:, 1]
    log_numerator = V_i_star + V_i_tilde_s
    log_denominator = pm.math.logsumexp(
        pm.math.stack([
            V_i_star + V_i_tilde_s,
            V_i_tilde + V_i_star_s
        ], axis=1),
        axis=1
    )
    mask = pm.math.switch(pm.math.eq(V_i_star_s, V_i_tilde_s), 0.0, 1.0)
    return (log_numerator - log_denominator) * mask

def _make_index_vars_for_mtm(model, N, M, J):
    """Create latent index vars so MTM steps have something to own/update."""
    idx_var = pm.DiscreteUniform("alt_idx", lower=0, upper=J-1, shape=N)
    return idx_var

def run_adaptive_sampler_model(sample_size, n_steps, burnin, tune,
                            random_seed, L=5, m=5, 
                            J=100, N=10000, P=10, I=0, C=0, M=0, scaling_mode='none'):

    # =========================================================================
    # REPRODUCIBILITY: Set all random seeds and log environment
    # =========================================================================
    seed_config = set_all_seeds(random_seed, verbose=False)
    
    logger.info(f"Running: J={J}, N={N}, P={P}, scaling={scaling_mode}, seed={random_seed}")
    
    np.random.seed(random_seed)
    
    # Load data based on configuration
    alternatives, observations, features = load_data(J=J, N=N, P=P, I=I, C=C, M=M)
    
    # =========================================================================
    # REPRODUCIBILITY: Validate loaded data (silent unless error)
    # =========================================================================
    data_validation = validate_data_content(alternatives, observations, verbose=False)
    if not data_validation["all_valid"]:
        logger.error("Data validation failed! Check reproducibility report for details.")
    
    size = observations.shape[0]

    # Build restaurant-only and interaction components (avoid full X_full until necessary)
    X_rest, rest_features = create_X_restaurants(alternatives, P)
    inter_mats, interact_features = compute_interaction_matrices(observations, alternatives, P)

    # Use true_parameters ordering to define feature order (ensures consistency)
    feature_names = rest_features.copy()
    feature_names.extend(interact_features)
    logger.info(f"Feature names used (total {len(feature_names)}): {feature_names}")
    logger.info(f"Restaurant features: {rest_features}")
    logger.info(f"Interaction features: {interact_features}")
    logger.info("-" * 60)
    # Build X_full from components following feature_names order (lazy but explicit)
    N = len(observations)
    J = len(alternatives)
    Kp1 = len(feature_names)
    X_full = np.zeros((N, J, Kp1), dtype=np.float32)
    for idx, feat in enumerate(feature_names):
        if feat in rest_features:
            col = rest_features.index(feat)
            X_full[:, :, idx] = X_rest[:, col][None, :]
        elif feat in inter_mats:
            X_full[:, :, idx] = inter_mats[feat]
        else:
            # missing feature -> zeros
            X_full[:, :, idx] = 0.0

    # Basic sanity checks & logging
    logger.info(f"X_full.shape: {X_full.shape}")
    if np.isnan(X_full).any():
        raise ValueError("X_full contains NaNs. Check that all required alternative features exist and are finite.")
    
    # =========================================================================
    # REPRODUCIBILITY: Validate X_full matrix
    # =========================================================================
    x_full_validation = check_x_full_matrix(X_full, feature_names, verbose=False)
    if not x_full_validation["all_valid"]:
        logger.error("X_full matrix validation failed!")
        raise ValueError("X_full validation failed - see log for details")
    
    chosen_vec = observations["logit_0"].values.astype(int)
    X_star_array = X_full[np.arange(size), chosen_vec, :].astype("float32")

    # =========================================================================
    # SCALING: Apply feature or beta scaling based on mode
    # =========================================================================
    scale_params = {}
    
    if scaling_mode == 'normalize':
        # MODE 1: Feature normalization (changes X)
        logger.info("\n" + "=" * 60)
        logger.info("APPLYING FEATURE NORMALIZATION")
        logger.info("=" * 60)
        
        # Store mean and std for each feature
        for feat_idx, feat_name in enumerate(feature_names):
            feat_values = X_full[:, :, feat_idx].flatten()
            feat_mean = np.mean(feat_values)
            feat_std = np.std(feat_values)
            
            if feat_std < 1e-10:
                feat_std = 1.0
                logger.warning(f"Feature '{feat_name}' has std ~ 0, using std=1.0")
            
            scale_params[feat_name] = {
                'mean': float(feat_mean),
                'std': float(feat_std)
            }
            
            # Normalize: (X - mean) / std
            X_full[:, :, feat_idx] = (X_full[:, :, feat_idx] - feat_mean) / feat_std
            logger.info(f"  {feat_name:20s}: mean={feat_mean:.4f}, std={feat_std:.4f}")
        
        X_star_array = X_full[np.arange(size), chosen_vec, :].astype("float32")
        logger.info("=" * 60)
        logger.info("Feature normalization complete. beta_original = beta_normalized / std")
        logger.info("=" * 60 + "\n")
        
    elif scaling_mode == 'scale':
        # MODE 2: Scale features by 1/std (no mean-centering)
        logger.info(f"Scaling features by 1/std...")
        
        # Scale each feature by 1/std only
        for feat_idx, feat_name in enumerate(feature_names):
            feat_values = X_full[:, :, feat_idx].flatten()
            feat_std = np.std(feat_values)
            
            if feat_std < 1e-10:
                feat_std = 1.0
            
            scale_params[feat_name] = {'std': float(feat_std)}
            X_full[:, :, feat_idx] = X_full[:, :, feat_idx] / feat_std
        
        X_star_array = X_full[np.arange(size), chosen_vec, :].astype("float32")
        
    else:
        # MODE 3: No scaling (normal estimation)
        logger.info("\nNo scaling applied (mode='none')")
        scale_params = None

    # Get true parameters for this configuration
    true_params = get_true_parameters(P)

    with pm.Model() as model:
        # 1) Priors - create priors explicitly for each P configuration
        all_features = feature_names
        prior_sigma = 10.0
        betas = {}
        
        def make_beta(name):
            varname = f'beta_{name}'
            betas[name] = pm.Normal(varname, mu=0.0, sigma=prior_sigma, initval=0.1)

        # Base features (P=10)
        for feat in ['rating', 'cost', 'chinese', 'japanese', 'korean', 'indian', 'french', 'mexican', 'lebanese', 'log_dist']:
            make_beta(feat)

        # Additional features for P >= 20
        if P >= 20:
            for feat in ['cost_x_logincome', 'logdist_x_age', 'rating_x_edu_sec', 'c_16', 'c_19', 'c_20', 'c_21', 'c_22', 'c_24', 'c_25']:
                make_beta(feat)

        # Additional features for P >= 40
        if P >= 40:
            for feat in ['c_26', 'c_27', 'c_29', 'c_30', 'c_32', 'c_33', 'c_34', 'c_35', 
                         'd_36', 'd_37', 'd_38', 'd_39', 'd_41', 'd_42', 'd_43', 'd_44', 'd_45', 'd_46', 'd_49', 'd_50']:
                make_beta(feat)

        # Map features to variables in order
        beta_list = [betas[f] if f in betas else pm.Normal(f"beta_{f}", 0, prior_sigma, initval=0.1) for f in all_features]
        beta_vector = pm.math.stack(beta_list)
        beta_vars = get_value_vars_from_user_vars(beta_list, model)

        # 2) Shared containers (always exist)
        Kp1 = X_full.shape[2]
        X_shared = pm.Data("X", np.zeros((size, sample_size, Kp1), dtype="float32"))
        V_i_s = pm.Data("V_i_s", np.zeros((size, sample_size), dtype="float32"))

        assert sample_size == 2, "Protocol requires sample_size == 2"

        # Compute current-beta utilities from X_shared (X is pre-scaled if needed)
        V = pm.math.dot(X_shared, beta_vector)  # (N, 2)
        V_i_star = V[:, 0]
        V_i_tilde = V[:, 1]
        
        # Original: binary logit with frozen utilities from previous beta
        pm.Potential("loglike_protocol2", binary_logit_potential(V_i_star, V_i_tilde, V_i_s).sum())
        
        same_alt_mask = pm.Data("same_alt_mask", np.zeros((N,), dtype="int32"))
        V_i_tilde_s = pm.Data("V_i_tilde_s", np.zeros((N,), dtype="float32"))
        V_i_star_s  = pm.Data("V_i_star_s",  np.zeros((N,), dtype="float32"))
        idx_var = _make_index_vars_for_mtm(model, N=N, M=None, J=J)
            
        step_alt = make_alt_step(
            model=model,
            X_full=X_full,
            chosen_vec=chosen_vec,
            X_container=X_shared,
            V_i_s=V_i_s,
            beta_vars=beta_vars,
            idx_var=idx_var,
            same_alt_mask=same_alt_mask,
            V_i_tilde_s=V_i_tilde_s,
            V_i_star_s=V_i_star_s,
            sample_size=2,
            m=m,
            random_seed=random_seed,
        )

        # 3) Sampling steps
        step_beta = pm.Metropolis(vars=list(betas.values()))

        trace_printer = TracePrinter(every=100)
        time_recorder = TimeRecorder()

        def _compound_callback(trace, draw):
            trace_printer(trace, draw)
            time_recorder(trace, draw)

        trace = pm.sample(
            draws=n_steps,
            tune=tune,
            step=pm.CompoundStep([step_alt, step_beta]),
            cores=4, #export XLA_FLAGS="--xla_force_host_platform_device_count=<number_of_cores>"
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False, # not available for Custom Compound step, creates error :))
            callback=_compound_callback,
            compile_kwargs={'mode': 'NUMBA'},
            var_names=[v.name for v in beta_list] # ensure we only save beta variables in the trace (exclude auxiliary vars
        )

        # Keep all draws — burnin is recorded as an attribute so downstream
        # analysis can exclude early draws when needed, but the full trace
        # (including warm-up draws) is preserved for the precision-time curve.
        time_recorder.attach_to_trace(trace)   # also strips sample_stats to elapsed_s only
        trace.sample_stats.attrs["burnin"] = burnin
        # Drop observed_data group (X_star, X, V_i_s …) — large arrays that are
        # fully reproducible from the raw data files and not needed for analysis.
        if hasattr(trace, "observed_data"):
            del trace.observed_data
        
        # =========================================================================
        # REPRODUCIBILITY: Attach validation results to trace as metadata
        # =========================================================================
        trace.attrs["reproducibility_seed"] = random_seed
        trace.attrs["reproducibility_seed_config"] = str(seed_config)
        trace.attrs["reproducibility_system_info"] = str(get_system_information())
        trace.attrs["reproducibility_library_versions"] = str(get_library_versions())
        trace.attrs["reproducibility_data_validation"] = str(data_validation)
        trace.attrs["reproducibility_x_full_validation"] = str(x_full_validation)
        
        reproducibility_info = {
            "seed_config": seed_config,
            "data_validation": data_validation,
            "x_full_validation": x_full_validation,
            "system_info": get_system_information(),
            "library_versions": get_library_versions(),
        }
        
        return trace, model, true_params, scale_params, feature_names, scaling_mode, reproducibility_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adaptive sampler for discrete choice model")
    
    # Sampling parameters
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--n_steps",   type=int, default=1000)
    parser.add_argument("--burnin",    type=int, default=500)
    parser.add_argument("--tune",      type=int, default=1000)
    parser.add_argument("--L", type=int, default=5)              # MTM P1
    parser.add_argument("--m", type=int, default=5)              # MTM P2
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="../results/results_pymc",
                       help="Directory where traces and scale parameters are saved")
    
    # Data configuration parameters
    parser.add_argument("--J", type=int, choices=[100, 200, 500], default=100,
                       help="Number of restaurants")
    parser.add_argument("--N", type=int, choices=[10000, 20000, 50000], default=10000,
                       help="Number of individuals")
    parser.add_argument("--P", type=int, choices=[10, 20, 40], default=10,
                       help="Number of parameters")
    parser.add_argument("--I", type=int, choices=[0, 1], default=0,
                       help="Data imbalance: 0=balanced, 1=imbalanced")
    parser.add_argument("--C", type=int, choices=[0, 1], default=0,
                       help="Correlation/clones: 0=independent, 1=correlated")
    parser.add_argument("--M", type=int, choices=[0, 1], default=0,
                       help="Choice randomness: 0=selective, 1=random")
    parser.add_argument("--scaling", type=str, choices=['none', 'normalize', 'scale'], default='none',
                       help="Scaling mode: 'none'=no scaling, 'normalize'=feature normalization, 'scale'=beta rescaling")
    parser.add_argument("--rescale_trace", action="store_true",
                       help="Rescale trace back to original scale before saving")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="Custom suffix to add to output filenames (e.g., '_test' or '_v2')")
    
    args = parser.parse_args()

    full_outdir = os.path.abspath(args.outdir)
    # make the suffix reflect key parameters J, N, P, n_steps
    scale_suffix = f"_{args.scaling.upper()}" if args.scaling != 'none' else ""
    custom_suffix = args.output_suffix if args.output_suffix else ""
    suffix = f"_J{args.J}_N{args.N}_P{args.P}_steps{args.n_steps}{scale_suffix}{custom_suffix}"
    os.makedirs(full_outdir, exist_ok=True)
    logger.info(f"Saving results to {full_outdir}")
    

    logger.info(f"Running configuration: J={args.J}, N={args.N}, P={args.P}, I={args.I}, C={args.C}, M={args.M}")
    logger.info(f"Sampling: (M={args.sample_size}, L={args.L}, m={args.m}, steps={args.n_steps})")
    logger.info(f"Scaling mode: {args.scaling.upper()}")
    if args.scaling != 'none' and args.rescale_trace:
        logger.info(f"Trace rescaling: ENABLED (will save in original scale)")
    
    trace, model, true_params, scale_params, feature_names, scaling_mode, reproducibility_info = run_adaptive_sampler_model(
        sample_size=args.sample_size,
        n_steps=args.n_steps,
        burnin=args.burnin,
        tune=args.tune,
        random_seed=args.seed,
        L=args.L,
        m=args.m,
        J=args.J,
        N=args.N,
        P=args.P,
        I=args.I,
        C=args.C,
        M=args.M,
        scaling_mode=args.scaling,
    )
    logger.info("Sampling complete.")

    # Rescale trace if requested
    if scale_params is not None and args.rescale_trace:
        trace = rescale_trace(trace, scale_params, feature_names, mode=scaling_mode)
        logger.info("Trace has been rescaled to original parameter scale")

    # =========================================================================
    # REPRODUCIBILITY: Save reproducibility metadata
    # =========================================================================
    reproduc_path = os.path.join(full_outdir, f"reproducibility_info{suffix}.json")
    try:
        with open(reproduc_path, 'w') as f:
            json.dump(reproducibility_info, f, indent=2, default=str)
        logger.info(f"Reproducibility metadata saved to {reproduc_path}")
    except Exception as e:
        logger.warning(f"Failed to save reproducibility metadata: {e}")

    # Save trace
    trace_path = os.path.join(full_outdir, f"trace{suffix}.nc")
    trace.to_netcdf(trace_path)
    logger.info(f"Trace saved to {trace_path}")

    logger.info("Wall-clock timestamps embedded in trace.sample_stats['elapsed_s']")
    
    # Save scale parameters if scaling was used
    if scale_params is not None:
        scale_path = os.path.join(full_outdir, f"scale_params{suffix}.json")
        with open(scale_path, 'w') as f:
            json.dump(scale_params, f, indent=2)
        logger.info(f"Scale parameters saved to {scale_path}")
        
        if not args.rescale_trace:
            # Provide mode-specific instructions for rescaling
            logger.info("\n" + "=" * 60)
            logger.info("RESCALING ESTIMATED PARAMETERS TO ORIGINAL SCALE")
            logger.info("=" * 60)
            
            if scaling_mode == 'normalize':
                logger.info("Mode: NORMALIZE (feature normalization)")
                logger.info("To convert normalized betas back to original scale:")
                logger.info("  beta_original = beta_normalized / std")
            elif scaling_mode == 'scale':
                logger.info("Mode: SCALE (beta rescaling)")
                logger.info("To convert rescaled betas back to original scale:")
                logger.info("  beta_original = beta_scaled / std")
            
            logger.info("\nExample Python code:")
            logger.info("  import json")
            logger.info(f"  with open('{scale_path}', 'r') as f:")
            logger.info("      scale_params = json.load(f)")
            logger.info("  ")
            logger.info("  for param_name in trace.posterior.data_vars:")
            logger.info("      if param_name.startswith('beta_'):")
            logger.info("          feat_name = param_name.replace('beta_', '')")
            logger.info("          std = scale_params[feat_name]['std']")
            
            if scaling_mode == 'normalize':
                logger.info("          beta_original = trace.posterior[param_name] / std")
            elif scaling_mode == 'scale':
                logger.info("          beta_original = trace.posterior[param_name] / std")
            
            logger.info("=" * 60)
        else:
            logger.info("\n Trace parameters have been rescaled to original scale")
    else:
        logger.info("\nNo scaling was applied, parameters are in original scale.")
