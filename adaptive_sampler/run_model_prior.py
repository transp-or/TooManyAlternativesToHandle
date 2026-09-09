import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
import argparse
import os
import logging
import matplotlib.pyplot as plt
import pytensor
from datetime import datetime
# pytensor.config.cxx = '/usr/bin/clang++' # Ensure pytensor uses the correct C++ compiler (Apple Silicon problem)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptive_sampler.alternative_sampler_step import make_alt_step
from adaptive_sampler.data import load_data
from adaptive_sampler.utils import precompute_dist_matrix, TracePrinter, create_X_full_matrix, get_features_for_P, create_X_restaurants, compute_interaction_matrices
from adaptive_sampler.true_parameters import get_true_parameters
from pymc.util import get_value_vars_from_user_vars

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(f"output/run_model_prior_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def rescale_trace(trace, normalization_params, feature_names):
    """
    Rescale normalized trace parameters back to original scale.
    
    For normalized features: beta_original = beta_normalized / std
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Trace with normalized parameters
    normalization_params : dict
        Dictionary mapping feature names to {'mean': float, 'std': float}
    feature_names : list
        List of feature names in order
        
    Returns:
    --------
    arviz.InferenceData
        Trace with rescaled parameters
    """
    import xarray as xr
    
    logger.info("\n" + "=" * 60)
    logger.info("RESCALING TRACE TO ORIGINAL PARAMETER SCALE")
    logger.info("=" * 60)
    
    # Create a copy of the trace to modify
    rescaled_trace = trace.copy()
    
    # Rescale each beta parameter
    for feat_name in feature_names:
        beta_name = f'beta_{feat_name}'
        
        if beta_name in rescaled_trace.posterior.data_vars:
            std = normalization_params[feat_name]['std']
            
            # Rescale: beta_original = beta_normalized / std
            rescaled_trace.posterior[beta_name] = rescaled_trace.posterior[beta_name] / std
            
            logger.info(f"  Rescaled {beta_name:30s} by dividing by std={std:.4f}")
    
    logger.info("=" * 60)
    logger.info("Trace rescaling complete. Parameters now in original scale.")
    logger.info("=" * 60 + "\n")
    
    return rescaled_trace

def logp_fn(X_star, X_samp, beta):
    V_star = pt.dot(X_star, beta)                     # (N,)
    V_samp = pt.sum(X_samp * beta, axis=2)            # (N, M)
    log_denom = pt.log(pt.sum(pt.exp(V_samp), axis=1))
    return pt.sum(V_star - log_denom)

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

def _make_index_vars_for_mtm(model, protocol, N, M, J):
    """Create latent index vars so MTM steps have something to own/update."""
    if protocol == 1:
        # Subset indices (duplicates allowed in prior; MTM step overwrites anyway)
        idx_var = pm.DiscreteUniform("subset_idx", lower=0, upper=J-1, shape=(N, M))
    else:
        idx_var = pm.DiscreteUniform("alt_idx", lower=0, upper=J-1, shape=N)
    return idx_var

def run_adaptive_sampler_model(sample_size, num_iters_MH, n_steps, burnin, tune,
                            random_seed, protocol, sampler, L=5, m=5, 
                            J=100, N=10000, P=10, I=0, C=0, M=0, normalize=False):

    np.random.seed(random_seed)
    
    # Load data based on configuration
    alternatives, observations, features = load_data(J=J, N=N, P=P, I=I, C=C, M=M)
    logger.info("\nData loaded:")
    logger.info("-" * 60)
    logger.info(f"Number of features loaded: {len(features)}")
    logger.info(f"Normalization: {'ENABLED' if normalize else 'DISABLED'}")
    
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
    chosen_vec = observations["logit_0"].values.astype(int)
    X_star_array = X_full[np.arange(size), chosen_vec, :].astype("float32")

    # =========================================================================
    # NORMALIZATION: Apply feature scaling if requested
    # =========================================================================
    normalization_params = {}
    if normalize:
        logger.info("\n" + "=" * 60)
        logger.info("APPLYING NORMALIZATION")
        logger.info("=" * 60)
        
        # Store mean and std for each feature to enable rescaling later
        for feat_idx, feat_name in enumerate(feature_names):
            # Extract all values for this feature across all observations and alternatives
            feat_values = X_full[:, :, feat_idx].flatten()
            feat_mean = np.mean(feat_values)
            feat_std = np.std(feat_values)
            
            # Avoid division by zero for constant features
            if feat_std < 1e-10:
                feat_std = 1.0
                logger.warning(f"Feature '{feat_name}' has std ~ 0, using std=1.0")
            
            normalization_params[feat_name] = {
                'mean': float(feat_mean),
                'std': float(feat_std)
            }
            
            # Normalize: (X - mean) / std
            X_full[:, :, feat_idx] = (X_full[:, :, feat_idx] - feat_mean) / feat_std
            
            logger.info(f"  {feat_name:20s}: mean={feat_mean:.4f}, std={feat_std:.4f}")
        
        # Update X_star_array after normalization
        X_star_array = X_full[np.arange(size), chosen_vec, :].astype("float32")
        
        logger.info("=" * 60)
        logger.info("Normalization complete. All features standardized to mean=0, std=1")
        logger.info("=" * 60 + "\n")
    else:
        logger.info("\nNo normalization applied (normalize=False)")
        normalization_params = None

    # Get true parameters for this configuration
    true_params = get_true_parameters(P)

    with pm.Model() as model:
        # 1) Priors with TRUNCATED NORMALS based on known parameter signs
        # We explicitly incorporate our knowledge that certain parameters should be positive or negative
        
        all_features = feature_names

        if Kp1 != len(all_features):
            logger.warning(
                "Feature list length (%d) does not match X_full last dim (%d). Adjusting to X_full shape.",
                len(all_features), Kp1,
            )
            # Adjust feature list length to match X_full columns
            if len(all_features) > Kp1:
                all_features = all_features[:Kp1]
            else:
                extra = [f"_extra_feat_{i}" for i in range(len(all_features), Kp1)]
                all_features = all_features + extra

        prior_sigma = 10.0
        betas = {}
        
        # Helper to create a truncated beta prior
        # We know the sign of parameters from the true data generation process
        def make_beta_positive(name, init=0.5):
            """Create a positive truncated normal prior (lower=0)"""
            varname = f'beta_{name}'
            betas[name] = pm.TruncatedNormal(varname, mu=1.0, sigma=prior_sigma, lower=0.0, initval=init)
            logger.info(f"Created POSITIVE truncated prior for {varname}")
        
        def make_beta_negative(name, init=-0.5):
            """Create a negative truncated normal prior (upper=0)"""
            varname = f'beta_{name}'
            betas[name] = pm.TruncatedNormal(varname, mu=-1.0, sigma=prior_sigma, upper=0.0, initval=init)
            logger.info(f"Created NEGATIVE truncated prior for {varname}")

        # =====================================================================
        # P=10: Base features (restaurant attributes + log_dist)
        # We KNOW these signs from the data generation process
        # =====================================================================
        
        # Restaurant attributes - POSITIVE (people prefer higher ratings, these cuisines)
        make_beta_positive('rating', init=0.5)
        make_beta_positive('chinese', init=0.5)
        make_beta_positive('japanese', init=0.5)
        make_beta_positive('korean', init=0.5)
        make_beta_positive('indian', init=0.5)
        make_beta_positive('french', init=0.5)
        make_beta_positive('mexican', init=0.5)
        make_beta_positive('lebanese', init=0.5)
        
        # Cost and Distance - NEGATIVE (people prefer lower cost, shorter distance)
        make_beta_negative('cost', init=-0.3)
        make_beta_negative('log_dist', init=-0.5)

        # =====================================================================
        # P=20: Add interactions and continuous restaurant features
        # =====================================================================
        if P >= 20:
            # Interactions - based on true parameter signs
            make_beta_positive('cost_x_logincome', init=0.01)      # POSITIVE: wealthier tolerate higher cost
            make_beta_negative('logdist_x_age', init=-0.01)        # NEGATIVE: older people more distance-averse
            # make_beta_positive('rating_x_risk', init=0.01)         # POSITIVE: risk-takers value rating more
            make_beta_positive('rating_x_edu_sec', init=0.1)       # POSITIVE: educated value rating more
            # make_beta_positive('rating_x_edu_tert', init=0.05)     # POSITIVE: highly educated value rating more
            
            # Continuous restaurant features c_16 to c_25 - based on true signs
            make_beta_positive('c_16', init=0.05)    # POSITIVE
            # make_beta_negative('c_17', init=-0.02)   # NEGATIVE
            # make_beta_positive('c_18', init=0.05)    # POSITIVE
            make_beta_negative('c_19', init=-0.02)   # NEGATIVE
            make_beta_positive('c_20', init=0.1)     # POSITIVE
            make_beta_positive('c_21', init=0.1)     # POSITIVE
            make_beta_negative('c_22', init=-0.1)    # NEGATIVE
            # make_beta_positive('c_23', init=0.1)     # POSITIVE
            make_beta_positive('c_24', init=0.1)     # POSITIVE
            make_beta_negative('c_25', init=-0.1)    # NEGATIVE

        # =====================================================================
        # P=40: Add more continuous features c_26 to c_35 and d_36 to d_50
        # =====================================================================
        if P >= 40:
            # Additional continuous restaurant features c_26 to c_35
            make_beta_positive('c_26', init=0.1)     # POSITIVE
            make_beta_positive('c_27', init=0.1)     # POSITIVE
            # make_beta_negative('c_28', init=-0.1)    # NEGATIVE
            make_beta_positive('c_29', init=0.1)     # POSITIVE
            make_beta_positive('c_30', init=0.1)     # POSITIVE
            # make_beta_negative('c_31', init=-0.1)    # NEGATIVE
            make_beta_positive('c_32', init=0.1)     # POSITIVE
            make_beta_positive('c_33', init=0.1)     # POSITIVE
            make_beta_negative('c_34', init=-0.1)    # NEGATIVE
            make_beta_positive('c_35', init=0.1)     # POSITIVE
            
            # Continuous features d_36 to d_50
            make_beta_positive('d_36', init=0.1)     # POSITIVE
            make_beta_positive('d_37', init=0.1)     # POSITIVE
            make_beta_negative('d_38', init=-0.1)    # NEGATIVE
            make_beta_positive('d_39', init=0.1)     # POSITIVE
            # make_beta_positive('d_40', init=0.1)     # POSITIVE
            make_beta_positive('d_41', init=0.1)     # POSITIVE
            make_beta_negative('d_42', init=-0.1)    # NEGATIVE
            make_beta_positive('d_43', init=0.1)     # POSITIVE
            make_beta_positive('d_44', init=0.1)     # POSITIVE
            make_beta_positive('d_45', init=0.1)     # POSITIVE
            make_beta_negative('d_46', init=-0.1)    # NEGATIVE
            # make_beta_positive('d_47', init=0.1)     # POSITIVE
            # make_beta_negative('d_48', init=-0.1)    # NEGATIVE
            make_beta_positive('d_49', init=0.1)     # POSITIVE
            make_beta_positive('d_50', init=0.1)     # POSITIVE

        # Stack betas in the exact order of all_features
        beta_list = []
        for feature in all_features:
            if feature in betas:
                beta_list.append(betas[feature])
            else:
                # If a feature is in all_features but not in betas, create it now with generic positive prior
                logger.warning(f"Feature '{feature}' not explicitly defined; creating generic POSITIVE prior")
                make_beta_positive(feature, init=0.1)
                beta_list.append(betas[feature])

        # Log the priors used
        logger.info("\n" + "=" * 60)
        logger.info("TRUNCATED NORMAL PRIORS BASED ON KNOWN PARAMETER SIGNS")
        logger.info("=" * 60)
        logger.info("We explicitly incorporate knowledge that:")
        logger.info("  - Cost, Distance effects should be NEGATIVE")
        logger.info("  - Rating, Cuisine preferences should be POSITIVE")
        logger.info("  - Each parameter is truncated at 0 according to its known sign")
        logger.info("=" * 60 + "\n")

        # Debug: print created beta variable names
        try:
            logger.info("Created beta vars in order: %s", [v.name for v in beta_list])
        except Exception:
            # Fallback if var naming differs
            logger.info("Created %d beta variables", len(beta_list))
        beta_vector = pm.math.stack(beta_list)
        beta_vars = get_value_vars_from_user_vars(beta_list, model)

        # Debug: verify shapes will match
        logger.info(f"Beta vector length: {len(all_features)}")
        logger.info(f"X_full shape: {X_full.shape} (should be N={N}, J={J}, K+1={X_full.shape[2]})")
        logger.info(f"All features used: {all_features}")
        logger.info("-" * 60)

        # 2) Shared containers (always exist)
        Kp1 = X_full.shape[2]
        X_shared = pm.Data("X", np.zeros((size, sample_size, Kp1), dtype="float32"))
        V_i_s = pm.Data("V_i_s", np.zeros((size, sample_size), dtype="float32"))

        # ~likelihood derivated from Bayes Thm
        if protocol == 1:
            X_star = pm.Data("X_star", X_star_array)
            pm.CustomDist(
                "likelihood",
                X_star,
                X_shared,
                beta_vector,
                logp=lambda value, X_star, X_sampled, beta_vector: logp_fn(X_star, X_sampled, beta_vector),
                observed=np.zeros(size),
                signature="(n,k), (n,m,k), (k)->(n)"
            )

            # If MTM, create the index variable for subsets
            idx_var = None
            if sampler == "mtm":
                idx_var = _make_index_vars_for_mtm(model, protocol=1, N=N, M=sample_size, J=J)

            step_alt = make_alt_step(
                sampler=sampler,
                protocol=1,
                model=model,
                X_full=X_full,
                chosen_vec=chosen_vec,
                X_container=X_shared,
                V_i_s=V_i_s,
                beta_vars=beta_vars,
                idx_var=idx_var,
                sample_size=sample_size,
                L=L,
                random_seed=random_seed,
            )

        elif protocol == 2:
            assert sample_size == 2, "Protocol 2 requires sample_size == 2"

            # Compute current-beta utilities from X_shared; use V_i_s from the step for 'frozen' utilities
            V = pm.math.dot(X_shared, beta_vector)  # (N, 2)
            V_i_star = V[:, 0]
            V_i_tilde = V[:, 1]
            # calls binary logit model
            pm.Potential("loglike_protocol2", binary_logit_potential(V_i_star, V_i_tilde, V_i_s).sum()) 

            # If MTM protocol-2, extra containers expects:
            same_alt_mask = None
            V_i_tilde_s = None # "sampled estimates" from previous MH step
            V_i_star_s  = None
            idx_var = None

            if sampler == "mtm":
                same_alt_mask = pm.Data("same_alt_mask", np.zeros((N,), dtype="int32"))
                V_i_tilde_s = pm.Data("V_i_tilde_s", np.zeros((N,), dtype="float32"))
                V_i_star_s  = pm.Data("V_i_star_s",  np.zeros((N,), dtype="float32"))
                idx_var = _make_index_vars_for_mtm(model, protocol=2, N=N, M=None, J=J)

            step_alt = make_alt_step(
                sampler=sampler,
                protocol=2,
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
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        # 3) Sampling steps
        step_beta = pm.Metropolis(vars=list(betas.values()))
        trace = pm.sample(
            draws=n_steps,
            tune=tune,
            step=pm.CompoundStep([step_alt, step_beta]),
            cores=4, #export XLA_FLAGS="--xla_force_host_platform_device_count=<number_of_cores>"
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False, # not available for Custom Compound step, creates error :))
            callback=TracePrinter(every=100),
            compile_kwargs={'mode': 'NUMBA'},
        )

        return trace.sel(draw=slice(burnin, None)), model, true_params, normalization_params, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adaptive sampler with TRUNCATED NORMAL priors based on known parameter signs")
    
    # Sampling parameters
    parser.add_argument("--protocol", type=int, choices=[1, 2], default=1)
    parser.add_argument("--sampler",  type=str, choices=["epsilon", "mtm"], default="epsilon")
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--n_steps",   type=int, default=1000)
    parser.add_argument("--burnin",    type=int, default=500)
    parser.add_argument("--tune",      type=int, default=1000)
    parser.add_argument("--num_iters_MH", type=int, default=30)  # kept for compatibility
    parser.add_argument("--L", type=int, default=5)              # MTM P1
    parser.add_argument("--m", type=int, default=5)              # MTM P2
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="output")
    
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
    parser.add_argument("--normalize", action="store_true",
                       help="Apply standardization (mean=0, std=1) to all features")
    parser.add_argument("--rescale_trace", action="store_true",
                       help="If normalize=True, rescale trace back to original scale before saving")
    
    args = parser.parse_args()

    full_outdir = '../results/results_pymc_prior/'
    # Suffix reflects that we're using truncated priors
    norm_suffix = "_NORM" if args.normalize else ""
    suffix = f"_J{args.J}_N{args.N}_P{args.P}_prot{args.protocol}_{args.sampler}_steps{args.n_steps}_TRUNCATED{norm_suffix}"
    os.makedirs(full_outdir, exist_ok=True)
    logger.info(f"Saving results to {full_outdir}")
    

    logger.info(f"Running configuration: J={args.J}, N={args.N}, P={args.P}, I={args.I}, C={args.C}, M={args.M}")
    logger.info(f"Sampling: P{args.protocol} / {args.sampler} "
                f"(M={args.sample_size}, L={args.L}, m={args.m}, steps={args.n_steps})")
    logger.info(f"Normalization: {'ENABLED' if args.normalize else 'DISABLED'}")
    if args.normalize and args.rescale_trace:
        logger.info(f"Trace rescaling: ENABLED (will save in original scale)")
    logger.info("=" * 60)
    logger.info("USING TRUNCATED NORMAL PRIORS WITH KNOWN PARAMETER SIGNS")
    logger.info("=" * 60)
    
    trace, model, true_params, normalization_params, feature_names = run_adaptive_sampler_model(
        sample_size=args.sample_size,
        num_iters_MH=args.num_iters_MH,
        n_steps=args.n_steps,
        burnin=args.burnin,
        tune=args.tune,
        random_seed=args.seed,
        protocol=args.protocol,
        sampler=args.sampler,
        L=args.L,
        m=args.m,
        J=args.J,
        N=args.N,
        P=args.P,
        I=args.I,
        C=args.C,
        M=args.M,
        normalize=args.normalize,
    )
    logger.info("Sampling complete.")

    # Rescale trace if requested
    if normalization_params is not None and args.rescale_trace:
        trace = rescale_trace(trace, normalization_params, feature_names)
        logger.info("Trace has been rescaled to original parameter scale")

    # Save trace
    trace_path = os.path.join(full_outdir, f"trace{suffix}.nc")
    trace.to_netcdf(trace_path)
    logger.info(f"Trace saved to {trace_path}")
    
    # Save normalization parameters if normalization was used
    if normalization_params is not None:
        import json
        norm_path = os.path.join(full_outdir, f"normalization{suffix}.json")
        with open(norm_path, 'w') as f:
            json.dump(normalization_params, f, indent=2)
        logger.info(f"Normalization parameters saved to {norm_path}")
        
        if not args.rescale_trace:
            # Provide instructions for rescaling
            logger.info("\n" + "=" * 60)
            logger.info("RESCALING ESTIMATED PARAMETERS TO ORIGINAL SCALE")
            logger.info("=" * 60)
            logger.info("To convert normalized betas back to original scale:")
            logger.info("  beta_original = beta_normalized / std")
            logger.info("\nExample Python code:")
            logger.info("  import json")
            logger.info(f"  with open('{norm_path}', 'r') as f:")
            logger.info("      norm_params = json.load(f)")
            logger.info("  ")
            logger.info("  # For each parameter:")
            logger.info("  for param_name in trace.posterior.data_vars:")
            logger.info("      if param_name.startswith('beta_'):")
            logger.info("          feat_name = param_name.replace('beta_', '')")
            logger.info("          std = norm_params[feat_name]['std']")
            logger.info("          beta_rescaled = trace.posterior[param_name] / std")
            logger.info("=" * 60)
        else:
            logger.info("\n✓ Trace parameters have been rescaled to original scale")
    else:
        logger.info("\nNo normalization was applied, parameters are in original scale.")
