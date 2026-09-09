"""
Comprehensive reproducibility and validation module.

This module provides tools to:
1. Set up all random sources reproducibly
2. Validate data checksums
3. Verify numerical consistency across systems
4. Document all dependencies and versions
5. Check for non-deterministic operations
"""

import numpy as np
import os
import sys
import json
import hashlib
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import platform

logger = logging.getLogger()

# ============================================================================
# VERSION TRACKING
# ============================================================================

def get_system_information() -> Dict[str, str]:
    """Get system and environment information for reproducibility tracing."""
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "hostname": platform.node(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def get_library_versions() -> Dict[str, str]:
    """Get versions of critical libraries."""
    versions = {}
    
    libs = [
        'numpy', 'pandas', 'pymc', 'pytensor', 'arviz',
        'scipy', 'matplotlib', 'numba',
    ]
    
    for lib_name in libs:
        try:
            lib = __import__(lib_name)
            versions[lib_name] = getattr(lib, '__version__', 'unknown')
        except ImportError:
            versions[lib_name] = 'not_installed'
    
    return versions


def log_reproducibility_header(log_file: str = ""):
    """Log system and library information to ensure reproducibility."""
    logger.info("\n" + "=" * 80)
    logger.info("REPRODUCIBILITY INFORMATION")
    logger.info("=" * 80)
    
    sys_info = get_system_information()
    for key, value in sys_info.items():
        logger.info(f"  {key:20s}: {value}")
    
    logger.info("-" * 80)
    logger.info("Library Versions:")
    lib_versions = get_library_versions()
    for lib, version in lib_versions.items():
        logger.info(f"  {lib:20s}: {version}")
    
    logger.info("=" * 80 + "\n")


# ============================================================================
# RANDOM SEED MANAGEMENT
# ============================================================================

def set_all_seeds(seed: int, verbose: bool = True) -> Dict[str, Any]:
    """
    Set ALL random seed sources to ensure reproducibility.
    
    CRITICAL: This must be called BEFORE importing PyMC and creating the model.
    
    Returns a dictionary with seed configuration for logging.
    """
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("SETTING ALL RANDOM SEEDS FOR REPRODUCIBILITY")
        logger.info("=" * 80)
    
    config = {
        "seed": seed,
        "sources_set": []
    }
    
    # 1) NumPy (both old and new API)
    np.random.seed(seed)
    config["sources_set"].append("np.random.seed()")
    
    # 2) NumPy Generator (preferred)
    np.random.default_rng(seed)
    config["sources_set"].append("np.random.default_rng()")
    
    # 3) Python's random module
    import random
    random.seed(seed)
    config["sources_set"].append("random.seed()")
    
    # 4) PyTensor/Theano (if available)
    try:
        import pytensor
        # Note: These config settings can only be changed before PyTensor initialization
        # If PyTensor is already initialized (imported elsewhere), these will fail silently
        try:
            pytensor.config.floatX = 'float32'
            config["sources_set"].append("pytensor.floatX configured to float32")
        except Exception:
            config["sources_set"].append("pytensor.floatX already initialized (skipped)")
        
        try:
            pytensor.config.optimizer = 'fast_compile'
            config["sources_set"].append("pytensor.optimizer configured to fast_compile")
        except Exception:
            config["sources_set"].append("pytensor.optimizer already initialized (skipped)")
    except ImportError:
        pass
    
    # 5) Environment variables for XLA (JAX/NumPy)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    config["sources_set"].append("environment variables set (TF_DETERMINISTIC_OPS, PYTHONHASHSEED)")
    
    if verbose:
        logger.info(f"Random seed: {seed}")
        logger.info("Seed sources set:")
        for source in config["sources_set"]:
            logger.info(f"  ✓ {source}")
        logger.info("=" * 80 + "\n")
    
    return config


# ============================================================================
# DATA VALIDATION & CHECKSUMS
# ============================================================================

def compute_file_checksum(filepath: str, algorithm: str = 'sha256') -> str:
    """Compute cryptographic checksum of a file."""
    hasher = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(65536)  # 64kb chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataframe_checksum(df: pd.DataFrame, algorithm: str = 'sha256') -> str:
    """Compute checksum of a DataFrame's content."""
    hasher = hashlib.new(algorithm)
    # Use the serialized values for consistent hashing
    data_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    hasher.update(data_bytes)
    return hasher.hexdigest()


def validate_data_files(
    alternatives_path: str,
    observations_path: str,
    cached_checksums: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate that data files match expected checksums.
    
    Args:
        alternatives_path: Path to restaurants/alternatives CSV
        observations_path: Path to choices/observations CSV
        cached_checksums: Optional dict with 'alternatives' and 'observations' keys
        verbose: Whether to log validation steps
        
    Returns:
        Dictionary with validation results
    """
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("DATA FILE VALIDATION")
        logger.info("=" * 80)
    
    results = {
        "validation_datetime": datetime.now().isoformat(),
        "files": {},
        "all_valid": True,
        "checksums": {}
    }
    
    for file_type, filepath in [
        ("alternatives", alternatives_path),
        ("observations", observations_path)
    ]:
        if not os.path.exists(filepath):
            results["files"][file_type] = {
                "exists": False,
                "error": f"File not found: {filepath}"
            }
            results["all_valid"] = False
            if verbose:
                logger.error(f"✗ {file_type}: File not found: {filepath}")
            continue
        
        try:
            checksum = compute_file_checksum(filepath)
            results["checksums"][file_type] = checksum
            
            # Verify against cached checksum if provided
            is_valid = True
            if cached_checksums and file_type in cached_checksums:
                is_valid = checksum == cached_checksums[file_type]
                if not is_valid:
                    results["all_valid"] = False
                    if verbose:
                        logger.warning(
                            f"✗ {file_type}: Checksum mismatch!\n"
                            f"  Expected: {cached_checksums[file_type]}\n"
                            f"  Got:      {checksum}"
                        )
            
            results["files"][file_type] = {
                "exists": True,
                "path": filepath,
                "checksum": checksum,
                "size_bytes": os.path.getsize(filepath),
                "checksum_match": is_valid if cached_checksums else None
            }
            
            if verbose:
                status = "✓" if is_valid else "!"
                logger.info(f"{status} {file_type}: {filepath}")
                logger.info(f"    Checksum: {checksum[:16]}...")
                logger.info(f"    Size: {os.path.getsize(filepath):,} bytes")
        
        except Exception as e:
            results["files"][file_type] = {"error": str(e)}
            results["all_valid"] = False
            if verbose:
                logger.error(f"✗ {file_type}: Error computing checksum: {e}")
    
    if verbose:
        logger.info("=" * 80 + "\n")
    
    return results


def validate_data_content(
    alternatives: pd.DataFrame,
    observations: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate data content for consistency and completeness.
    
    Args:
        alternatives: Alternatives DataFrame
        observations: Observations DataFrame
        verbose: Whether to log validation steps
        
    Returns:
        Dictionary with validation results
    """
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("DATA CONTENT VALIDATION")
        logger.info("=" * 80)
    
    results = {
        "validation_datetime": datetime.now().isoformat(),
        "checks": {},
        "all_valid": True,
    }
    
    # Check 1: No NaN values
    alt_nans = alternatives.isna().sum().sum()
    obs_nans = observations.isna().sum().sum()
    results["checks"]["no_nans"] = {
        "alternatives_nans": int(alt_nans),
        "observations_nans": int(obs_nans),
        "valid": alt_nans == 0 and obs_nans == 0
    }
    if not results["checks"]["no_nans"]["valid"]:
        results["all_valid"] = False
        if verbose:
            logger.warning(f"✗ Found NaN values: alt={alt_nans}, obs={obs_nans}")
    else:
        if verbose:
            logger.info("✓ No NaN values found")
    
    # Check 2: No infinite values
    alt_infs = np.isinf(alternatives.select_dtypes(np.number)).sum().sum()
    obs_infs = np.isinf(observations.select_dtypes(np.number)).sum().sum()
    results["checks"]["no_infs"] = {
        "alternatives_infs": int(alt_infs),
        "observations_infs": int(obs_infs),
        "valid": alt_infs == 0 and obs_infs == 0
    }
    if not results["checks"]["no_infs"]["valid"]:
        results["all_valid"] = False
        if verbose:
            logger.warning(f"✗ Found infinite values: alt={alt_infs}, obs={obs_infs}")
    else:
        if verbose:
            logger.info("✓ No infinite values found")
    
    # Check 3: Dimensions
    n_obs = len(observations)
    j_alts = len(alternatives)
    results["checks"]["dimensions"] = {
        "n_observations": n_obs,
        "j_alternatives": j_alts,
        "shape_alternatives": list(alternatives.shape),
        "shape_observations": list(observations.shape),
        "valid": n_obs > 0 and j_alts > 0
    }
    if not results["checks"]["dimensions"]["valid"]:
        results["all_valid"] = False
        if verbose:
            logger.warning("✗ Invalid dimensions")
    else:
        if verbose:
            logger.info(f"✓ Valid dimensions: N={n_obs}, J={j_alts}")
    
    # Check 4: Data types consistency
    alt_dtypes = {col: str(alternatives[col].dtype) for col in alternatives.columns}
    obs_dtypes = {col: str(observations[col].dtype) for col in observations.columns}
    results["checks"]["dtypes"] = {
        "alternatives": alt_dtypes,
        "observations": obs_dtypes
    }
    if verbose:
        logger.info("✓ Data types recorded")
    
    # Check 5: Column existence
    required_alt_cols = {'x', 'y', 'rating', 'cost'}
    required_obs_cols = {'user_x', 'user_y', 'logit_0'}
    
    missing_alt = required_alt_cols - set(alternatives.columns)
    missing_obs = required_obs_cols - set(observations.columns)
    
    results["checks"]["required_columns"] = {
        "alternatives_missing": list(missing_alt),
        "observations_missing": list(missing_obs),
        "valid": len(missing_alt) == 0 and len(missing_obs) == 0
    }
    if not results["checks"]["required_columns"]["valid"]:
        results["all_valid"] = False
        if verbose:
            logger.warning(f"✗ Missing columns: alt={missing_alt}, obs={missing_obs}")
    else:
        if verbose:
            logger.info("✓ All required columns present")
    
    if verbose:
        logger.info("=" * 80 + "\n")
    
    return results


# ============================================================================
# NUMERICAL CONSISTENCY CHECKS
# ============================================================================

def check_x_full_matrix(
    X_full: np.ndarray,
    feature_names: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate X_full feature matrix for numerical consistency.
    
    Args:
        X_full: Feature matrix of shape (N, J, K+1)
        feature_names: List of feature names
        verbose: Whether to log results
        
    Returns:
        Dictionary with validation results
    """
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("X_FULL MATRIX VALIDATION")
        logger.info("=" * 80)
    
    results = {
        "shape": X_full.shape,
        "dtype": str(X_full.dtype),
        "checks": {},
        "all_valid": True,
    }
    
    # Check 1: Shape consistency
    N, J, K = X_full.shape
    if K != len(feature_names):
        logger.warning(f"✗ Feature count mismatch: K={K}, len(feature_names)={len(feature_names)}")
        results["all_valid"] = False
    else:
        if verbose:
            logger.info(f"✓ Shape consistent: N={N}, J={J}, K={K} features")
    
    # Check 2: No NaNs
    n_nans = np.isnan(X_full).sum()
    results["checks"]["nans"] = int(n_nans)
    if n_nans > 0:
        logger.warning(f"✗ Found {n_nans} NaN values in X_full")
        results["all_valid"] = False
    else:
        if verbose:
            logger.info("✓ No NaN values")
    
    # Check 3: No infinite values
    n_infs = np.isinf(X_full).sum()
    results["checks"]["infs"] = int(n_infs)
    if n_infs > 0:
        logger.warning(f"✗ Found {n_infs} infinite values in X_full")
        results["all_valid"] = False
    else:
        if verbose:
            logger.info("✓ No infinite values")
    
    # Check 4: Feature statistics (min, max, mean, std per feature)
    stats = {}
    for k in range(K):
        feature_data = X_full[:, :, k].flatten()
        stats[feature_names[k]] = {
            "min": float(np.min(feature_data)),
            "max": float(np.max(feature_data)),
            "mean": float(np.mean(feature_data)),
            "std": float(np.std(feature_data)),
            "count": int(len(feature_data))
        }
    results["feature_stats"] = stats
    
    if verbose:
        logger.info("\nFeature Statistics:")
        for feat_name, stat in stats.items():
            logger.info(f"  {feat_name:20s}: min={stat['min']:10.4f}, "
                       f"max={stat['max']:10.4f}, mean={stat['mean']:10.4f}, "
                       f"std={stat['std']:10.4f}")
    
    if verbose:
        logger.info("=" * 80 + "\n")
    
    return results


def check_initial_values(
    betas: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate initial beta values.
    
    Args:
        betas: Dictionary of beta variables (PyMC RVs)
        verbose: Whether to log results
        
    Returns:
        Dictionary with validation results
    """
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("INITIAL BETA VALUES VALIDATION")
        logger.info("=" * 80)
    
    results = {
        "num_betas": len(betas),
        "initial_values": {},
        "all_valid": True,
    }
    
    try:
        for name, beta_var in betas.items():
            try:
                # Get initial value (depends on PyMC version)
                if hasattr(beta_var, 'initval'):
                    init_val = float(beta_var.initval)
                elif hasattr(beta_var, 'init'):
                    init_val = float(beta_var.init)
                else:
                    init_val = None
                
                results["initial_values"][name] = {
                    "initval": init_val,
                    "valid": init_val is not None and np.isfinite(init_val)
                }
                
                if not results["initial_values"][name]["valid"]:
                    results["all_valid"] = False
            except Exception as e:
                results["initial_values"][name] = {"error": str(e)}
                results["all_valid"] = False
        
        if verbose:
            for name, info in results["initial_values"].items():
                if "error" not in info:
                    logger.info(f"  {name:20s}: {info['initval']:.6f}")
                else:
                    logger.warning(f"  {name:20s}: ERROR - {info['error']}")
    
    except Exception as e:
        logger.error(f"Error checking initial values: {e}")
        results["all_valid"] = False
    
    if verbose:
        logger.info("=" * 80 + "\n")
    
    return results


# ============================================================================
# REPRODUCIBILITY REPORT
# ============================================================================

def generate_reproducibility_report(
    run_config: Dict[str, Any],
    system_info: Dict[str, str],
    lib_versions: Dict[str, str],
    validation_results: Dict[str, Any],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive reproducibility report.
    
    Args:
        run_config: Run configuration dictionary
        system_info: System information from get_system_information()
        lib_versions: Library versions from get_library_versions()
        validation_results: Combined validation results
        output_path: Optional path to save report as JSON
        
    Returns:
        Dictionary with full reproducibility report
    """
    report = {
        "report_type": "reproducibility_report",
        "generated": datetime.now().isoformat(),
        "system": system_info,
        "libraries": lib_versions,
        "run_configuration": run_config,
        "validation": validation_results,
    }
    
    if output_path:
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Reproducibility report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save reproducibility report: {e}")
    
    return report


# ============================================================================
# QUICK REPRODUCIBILITY CHECK (ONE-LINER)
# ============================================================================

def quick_reproducibility_check(
    random_seed: int,
    alternatives: Optional[pd.DataFrame] = None,
    observations: Optional[pd.DataFrame] = None,
    X_full: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run quick reproducibility checks.
    
    Can be called from scripts with minimal setup.
    """
    logger.info("\n" + "=" * 80)
    logger.info("QUICK REPRODUCIBILITY CHECK")
    logger.info("=" * 80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "seed_config": set_all_seeds(random_seed, verbose=True),
        "system_info": get_system_information(),
        "library_versions": get_library_versions(),
    }
    
    if alternatives is not None and observations is not None:
        results["data_validation"] = validate_data_content(alternatives, observations)
    
    if X_full is not None and feature_names is not None:
        results["x_full_validation"] = check_x_full_matrix(X_full, feature_names)
    
    logger.info("=" * 80 + "\n")
    
    return results
