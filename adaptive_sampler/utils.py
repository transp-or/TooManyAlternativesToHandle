import time
import numpy as np
import pandas as pd
from typing import Any, Callable, List, Dict, Optional

# bc of the ProgressBar problem in CompoundStep, we need to create manual print
class TracePrinter:
    def __init__(self, every=100):
        self.count = 0
        self.every = every

    def __call__(self, trace, draw):
        self.count += 1
        if self.count % self.every == 0:
            print(f"Draw {self.count}")


class TimeRecorder:
    """
    Lightweight sampling callback that records wall-clock elapsed time for
    ALL draws (tuning and sampling alike), keyed by chain and draw index.

    Usage
    -----
    recorder = TimeRecorder()
    pm.sample(..., callback=recorder)
    recorder.attach_to_trace(trace)   # embeds timestamps into trace.sample_stats

    Design notes
    ------------
    - The clock starts at t=0 on the very first callback invocation (first
      tuning draw), so elapsed times include the full tuning phase. This is
      intentional: when comparing to BOLFI (which has a training phase), the
      timeline must be continuous and account for warm-up cost.
    - Tuning and sampling draws are both recorded with an ``is_tuning`` flag.
    - ``draw_idx`` resets to 0 at the start of sampling (PyMC behaviour), so
      tuning and sampling records are distinguished only by ``is_tuning``.
    - ``attach_to_trace`` embeds per-draw ``elapsed_s`` into sample_stats
      (post-tuning draws only, aligned with the posterior ``draw`` dim) and
      stores ``tuning_end_s`` as an attribute so the full timeline is
      recoverable from a single .nc file without a sidecar CSV.
    """

    def __init__(self):
        self._start: Optional[float] = None
        # list of (chain, draw_idx, elapsed_seconds, is_tuning)
        self.records: list[tuple[int, int, float, bool]] = []

    def __call__(self, trace, draw):
        t = time.time()
        if self._start is None:
            self._start = t
        self.records.append((
            int(draw.chain),
            int(draw.draw_idx),
            t - self._start,
            bool(draw.tuning),
        ))

    def tuning_end_s(self) -> float:
        """Wall-clock seconds elapsed at the end of the tuning phase (per chain max)."""
        tuning = [elapsed for _, _, elapsed, is_tuning in self.records if is_tuning]
        return max(tuning) if tuning else 0.0

    def to_xarray(self, n_chains: int, n_draws: int):
        """
        Return a (chain, draw) DataArray of elapsed wall-clock seconds for
        post-tuning draws only.  Times are measured from t=0 (start of tuning),
        so each value already encodes the tuning cost as an offset.
        """
        import xarray as xr
        arr = np.full((n_chains, n_draws), np.nan, dtype=np.float64)
        for chain, draw_idx, elapsed, is_tuning in self.records:
            if not is_tuning and draw_idx < n_draws:
                arr[chain, draw_idx] = elapsed
        return xr.DataArray(arr, dims=["chain", "draw"])

    def attach_to_trace(self, trace) -> None:
        """
        Minimise trace size: drop all sample_stats diagnostic variables
        (accepted, energy, step_size, …) and replace the group with only
        ``elapsed_s`` plus two scalar attributes:

          - elapsed_s     : (chain, draw) wall-clock seconds from t=0
          - tuning_end_s  : attr — seconds at which tuning finished
          - burnin        : attr — set externally after this call

        This keeps the .nc file as small as possible while preserving
        everything needed for the precision-time curve.
        """
        import xarray as xr
        n_chains = trace.posterior.dims["chain"]
        n_draws  = trace.posterior.dims["draw"]
        elapsed_da = self.to_xarray(n_chains, n_draws)
        # Preserve any attrs that were already set (e.g. burnin)
        existing_attrs = dict(trace.sample_stats.attrs)
        # Replace the entire sample_stats group with only elapsed_s
        trace.sample_stats = xr.Dataset(
            {"elapsed_s": elapsed_da},
            attrs={**existing_attrs, "tuning_end_s": self.tuning_end_s()},
        )

def precompute_dist_matrix(observations: pd.DataFrame, alternatives: pd.DataFrame) -> np.ndarray:
    """
    Precompute the distance matrix between alternatives and observations.
    row i corresponds to individual i, and column j corresponds to alternative j.
    """
    obs_xy = observations[["user_x", "user_y"]].values  # shape (n_individuals, 2)
    alt_xy = alternatives[["x", "y"]].values            # shape (n_alternatives, 2)
    # Broadcasting: (n_individuals, 1, 2) - (1, n_alternatives, 2) -> (n_individuals, n_alternatives, 2)
    deltas = obs_xy[:, np.newaxis, :] - alt_xy[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)  # shape (n_individuals, n_alternatives)
    log_distances = np.log(distances + 1e-12)  # Avoid log(0) by adding a small constant
    return log_distances

def get_features_for_P(P: int, interacted_attributes_only: bool = False) -> List[str]:
    """
    Get the list of features for a given number of parameters P.
    
    Args:
        P: Number of parameters (10, 20, or 40)
        interacted_attributes_only: If True and P>=40, return interaction features instead of c_*/d_*
        
    Returns:
        List of feature names
    """
    # Base 9 features for P=10 (excluding log_dist which is computed separately)
    base_features = ['rating', 'cost', 'Chinese', 'Indian', 'French', 'Mexican', 'Lebanese', 'Japanese', 'Korean']
    
    # Socio-economic variables that are used in interaction terms
    socio_features = ['logincome', 'age', 'risk', 'edu_secondary', 'edu_tertiary']

    if P == 10:
        # For P=10 we keep only alternative features; socio-economic vars are per-individual
        return base_features
    
    elif P == 20:
        # Add 5 interaction terms and 5 continuous features
        additional_features = [  
            'c_16',
            # 'c_17',
            # 'c_18',
            'c_19',
            'c_20',
            'c_21',
            'c_22',
            # 'c_23',
            'c_24',
            'c_25'
                               ]
        # Include socio-economic variable names so downstream code can check availability
        return base_features + additional_features + socio_features
    
    elif P == 40:
        if not interacted_attributes_only:
            p20_features = get_features_for_P(20, interacted_attributes_only=False)
            continuous_features = [ 
                'c_26', 
                'c_27', 
                # 'c_28', 
                'c_29', 
                'c_30',
                # 'c_31',
                'c_32', 
                'c_33',
                'c_34',
                'c_35'
            ]
            discrete_features = [
                'd_36',
                'd_37',
                'd_38',
                'd_39',
                # 'd_40',
                'd_41',
                'd_42',
                'd_43',
                'd_44',
                'd_45',
                'd_46',
                # 'd_47',
                # 'd_48',
                'd_49',
                'd_50'
            ]
            return p20_features + continuous_features + discrete_features
        else:
        
            interaction_features = base_features + socio_features  # Base + P=10 socio vars
            
            # Age × 7 cuisines (7 - Ethiopian is base, excluded)
            interaction_features.extend([
                'age_x_chinese', 'age_x_japanese', 'age_x_korean', 'age_x_indian',
                'age_x_french', 'age_x_mexican', 'age_x_lebanese'
            ])
            
            # Gender × 7 cuisines (7)
            interaction_features.extend([
                'gender_x_chinese', 'gender_x_japanese', 'gender_x_korean', 'gender_x_indian',
                'gender_x_french', 'gender_x_mexican', 'gender_x_lebanese'
            ])
            
            # Risk × 7 cuisines (7)
            interaction_features.extend([
                'risk_x_chinese', 'risk_x_japanese', 'risk_x_korean', 'risk_x_indian',
                'risk_x_french', 'risk_x_mexican', 'risk_x_lebanese'
            ])
            
            # Has_children × alternatives (3)
            interaction_features.extend([
                'haschildren_x_rating', 'haschildren_x_cost', 'haschildren_x_logdist'
            ])
            
            # Gender × alternatives (3 NEW - not in P20)
            interaction_features.extend([
                'gender_x_rating', 'gender_x_cost', 'gender_x_logdist'
            ])
            
            # Logincome × alternatives (2 NEW - cost already in P20)
            interaction_features.extend([
                'logincome_x_rating', 'logincome_x_logdist'
            ])
            
            # Edu_secondary × alternatives (2 NEW - rating already in P20)
            interaction_features.extend([
                'edu_sec_x_cost', 'edu_sec_x_logdist'
            ])
            
            # Edu_tertiary × alternatives (2 NEW - rating already in P20)
            interaction_features.extend([
                'edu_tert_x_cost', 'edu_tert_x_logdist'
            ])
            
            # Age × alternatives (2 NEW - log_dist already in P20)
            interaction_features.extend([
                'age_x_rating', 'age_x_cost'
            ])
            
            # Add has_children and gender as individual variables
            interaction_features.extend(['has_children', 'gender'])
            
            return interaction_features
    
    else:
        raise ValueError(f"Unsupported P value: {P}. Must be 10, 20, or 40.")

def get_data_filename(J: int, N: int, P: int, I: int, C: int, M: int, interacted_attributes_only: bool = False, data_type: str = "choices") -> str:
    """
    Generate standardized filename for data files.
    
    Args:
        J: Number of restaurants
        N: Number of individuals  
        P: Number of parameters
        I: Imbalance factor (0/1)
        C: Correlation factor (0/1)
        M: Choice randomness factor (0/1)
        data_type: "choices" or "restaurants"
        
    Returns:
        Filename string
    """
    if data_type == "choices":
        # return f"choices_J{J}_P{P}_N{N//1000}k_I{I}C{C}M{M}_{'True' if interacted_attributes_only else 'False'}.csv"
        return f"choices_J{J}_P{P}_N{N//1000}k_I{I}C{C}M{M}.csv"
    elif data_type == "restaurants":
        # return f"restaurants_J{J}_P{P}_N{N//1000}k_I{I}C{C}M{M}_{'True' if interacted_attributes_only else 'False'}.csv"
        return f"restaurants_J{J}_P{P}_N{N//1000}k_I{I}C{C}M{M}.csv"
    else:
        raise ValueError("data_type must be 'choices' or 'restaurants'")

def create_X_full_matrix(observations: pd.DataFrame, alternatives: pd.DataFrame, P: int) -> np.ndarray:
    """
    Create the full design matrix X_full used in the sampler.
    
    Args:
        observations: Individual data
        alternatives: Restaurant data
        P: Number of parameters
        
    Returns:
        X_full matrix of shape (N, J, K+1) where K+1 includes log_dist
        
    Note:
        For P>=20, interaction terms are not included in this matrix as they require
        individual-level data and cannot be precomputed. Future versions may handle
        this by extending the alternative sampler step.
    """
    N = len(observations)
    J = len(alternatives)
    
    # We'll build features explicitly, one-by-one, so interaction terms can use
    # both alternatives (restaurant-level) and observations (individual-level).
    feature_arrays = []  # list of (N, J) arrays
    feature_names = []

    # Precompute distances early (log distance), used both as a base feature and in interactions
    distances = precompute_dist_matrix(observations, alternatives).astype(np.float32)

    # Case-insensitive column lookup helpers
    def alt_col(ci_name: str):
        for c in alternatives.columns:
            if c.lower() == ci_name.lower():
                return c
        return None

    def obs_col(ci_name: str):
        for c in observations.columns:
            if c.lower() == ci_name.lower():
                return c
        return None

    # 1) Base features in the exact order required by true_parameters
    # Order: rating, cost, log_dist, chinese, japanese, korean, indian, french, mexican, lebanese
    base_order = ['rating', 'cost', 'log_dist', 'chinese', 'japanese', 'korean', 'indian', 'french', 'mexican', 'lebanese']
    for feat in base_order:
        if feat == 'log_dist':
            arr = distances
        else:
            col = alt_col(feat)
            if col is not None:
                arr = alternatives[col].values[None, :].astype(np.float32)
                arr = np.broadcast_to(arr, (N, J))
            else:
                arr = np.zeros((N, J), dtype=np.float32)
                print(f"Warning: alternative feature '{feat}' missing; filling with zeros")
        feature_arrays.append(arr)
        feature_names.append(feat)

    # 2) Interaction terms and socio-economic-derived features for P >= 20
    # Each interaction explicitly uses restaurant attribute(s) and individual attribute(s)
    if P >= 20:
        # cost_x_logincome: cost (restaurant) * logincome (individual)
        cost_col = alt_col('cost')
        logincome_col = obs_col('logincome')
        if cost_col is not None and logincome_col is not None:
            cost_x_logincome = (alternatives[cost_col].values[None, :].astype(np.float32) *
                                observations[logincome_col].values[:, None].astype(np.float32))
        else:
            cost_x_logincome = np.zeros((N, J), dtype=np.float32)
            print("Warning: cost_x_logincome unavailable (missing cost or logincome); filling zeros")
        feature_arrays.append(cost_x_logincome)
        feature_names.append('cost_x_logincome')

        # logdist_x_age: log(distance) * age
        age_col = obs_col('age')
        if age_col is not None:
            logdist_x_age = distances * observations[age_col].values[:, None].astype(np.float32)
        else:
            logdist_x_age = np.zeros((N, J), dtype=np.float32)
            print("Warning: logdist_x_age unavailable (missing age); filling zeros")
        feature_arrays.append(logdist_x_age)
        feature_names.append('logdist_x_age')

        # rating_x_risk: rating (restaurant) * risk (individual)
        rating_col = alt_col('rating')
        risk_col = obs_col('risk')
        if rating_col is not None and risk_col is not None:
            rating_x_risk = (alternatives[rating_col].values[None, :].astype(np.float32) *
                             observations[risk_col].values[:, None].astype(np.float32))
        else:
            rating_x_risk = np.zeros((N, J), dtype=np.float32)
            print("Warning: rating_x_risk unavailable (missing rating or risk); filling zeros")
        feature_arrays.append(rating_x_risk)
        feature_names.append('rating_x_risk')

        # rating_x_edu_sec: rating * edu_secondary
        edu_sec_col = obs_col('edu_secondary')
        if rating_col is not None and edu_sec_col is not None:
            rating_x_edu_sec = (alternatives[rating_col].values[None, :].astype(np.float32) *
                                observations[edu_sec_col].values[:, None].astype(np.float32))
        else:
            rating_x_edu_sec = np.zeros((N, J), dtype=np.float32)
            print("Warning: rating_x_edu_sec unavailable (missing rating or edu_secondary); filling zeros")
        feature_arrays.append(rating_x_edu_sec)
        feature_names.append('rating_x_edu_sec')

        # rating_x_edu_tert: rating * edu_tertiary
        edu_tert_col = obs_col('edu_tertiary')
        if rating_col is not None and edu_tert_col is not None:
            rating_x_edu_tert = (alternatives[rating_col].values[None, :].astype(np.float32) *
                                 observations[edu_tert_col].values[:, None].astype(np.float32))
        else:
            rating_x_edu_tert = np.zeros((N, J), dtype=np.float32)
            print("Warning: rating_x_edu_tert unavailable (missing rating or edu_tertiary); filling zeros")
        feature_arrays.append(rating_x_edu_tert)
        feature_names.append('rating_x_edu_tert')

        # Continuous restaurant features c_16..c_20
        for i in range(16, 21):
            feat = f'c_{i}'
            col = alt_col(feat)
            if col is not None:
                arr = alternatives[col].values[None, :].astype(np.float32)
                arr = np.broadcast_to(arr, (N, J))
            else:
                arr = np.zeros((N, J), dtype=np.float32)
                print(f"Warning: alternative feature '{feat}' missing; filling zeros")
            feature_arrays.append(arr)
            feature_names.append(feat)

    # 3) Additional features for P >= 50
    if P >= 50:
        # Continuous features c_21..c_35 (use case-insensitive lookup)
        for i in range(21, 36):
            feat = f'c_{i}'
            col = alt_col(feat)
            if col is not None:
                arr = alternatives[col].values[None, :].astype(np.float32)
                arr = np.broadcast_to(arr, (N, J))
            else:
                arr = np.zeros((N, J), dtype=np.float32)
                print(f"Warning: alternative feature '{feat}' missing; filling zeros")
            feature_arrays.append(arr)
            feature_names.append(feat)

        # Discrete features d_36..d_50
        for i in range(36, 51):
            feat = f'd_{i}'
            col = alt_col(feat)
            if col is not None:
                arr = alternatives[col].values[None, :].astype(np.float32)
                arr = np.broadcast_to(arr, (N, J))
            else:
                arr = np.zeros((N, J), dtype=np.float32)
                print(f"Warning: alternative feature '{feat}' missing; filling zeros")
            feature_arrays.append(arr)
            feature_names.append(feat)

    # Note: 'log_dist' was already added in the base_order at the correct position

    # Stack along last axis: we need shape (N, J, K)
    X_full = np.stack(feature_arrays, axis=2)

    # Sanity: ensure float32
    X_full = X_full.astype(np.float32, copy=False)

    # Return both the design matrix and the ordered feature names used to construct it
    return X_full, feature_names


def create_X_restaurants(alternatives: pd.DataFrame, P: int):
    """
    Build a restaurant-only feature matrix and ordered feature names.

    Returns:
        X_rest: np.ndarray of shape (J, K_rest)
        rest_feature_names: list of feature names in order
    """
    J = len(alternatives)

    # Case-insensitive lookup helper
    def alt_col(ci_name: str):
        for c in alternatives.columns:
            if c.lower() == ci_name.lower():
                return c
        return None

    rest_features = ['rating', 'cost', 'chinese', 'japanese', 'korean', 'indian', 'french', 'mexican', 'lebanese']

    # P>=20 continuous restaurant features c_16 to c_25 (10 features)
    if P >= 20:
        rest_features += [
            'c_16',
            # 'c_17',
            # 'c_18',
            'c_19',
            'c_20',
            'c_21',
            'c_22',
            # 'c_23',
            'c_24',
            'c_25'
        ]
    # P>=40 extras: c_26 to c_35 and d_36 to d_40
    if P >= 40:
        rest_features += [
            'c_26', 
            'c_27', 
            # 'c_28', 
            'c_29', 
            'c_30',
            # 'c_31',
            'c_32', 
            'c_33',
            'c_34',
            'c_35'
        ] + [
            'd_36',
            'd_37',
            'd_38',
            'd_39',
            # 'd_40',
            'd_41',
            'd_42',
            'd_43',
            'd_44',
            'd_45',
            'd_46',
            # 'd_47',
            # 'd_48',
            'd_49',
            'd_50'
        ]

    arrays = []
    names = []
    for feat in rest_features:
        col = alt_col(feat)
        if col is not None:
            arr = alternatives[col].values.astype(np.float32)  # shape (J,)
        else:
            arr = np.zeros((J,), dtype=np.float32)
        arrays.append(arr)
        names.append(feat)

    if arrays:
        X_rest = np.stack(arrays, axis=1)  # (J, K_rest)
    else:
        X_rest = np.zeros((J, 0), dtype=np.float32)

    return X_rest, names


def compute_interaction_matrices(observations: pd.DataFrame, alternatives: pd.DataFrame, P: int):
    """
    Precompute interaction matrices (N, J) such as log_dist and other interactions.

    Returns:
        dict mapping interaction name -> np.ndarray shape (N, J)
    """
    N = len(observations)
    J = len(alternatives)
    mats = {}

    # log distance
    dists = precompute_dist_matrix(observations, alternatives).astype(np.float32)
    mats['log_dist'] = dists
    names = []

    if P >= 20:
        # cost_x_logincome
        cost_col = None
        for c in alternatives.columns:
            if c.lower() == 'cost':
                cost_col = c
                break
        logincome_col = None
        for c in observations.columns:
            if c.lower() == 'logincome':
                logincome_col = c
                break
        if cost_col is not None and logincome_col is not None:
            mats['cost_x_logincome'] = (alternatives[cost_col].values[None, :].astype(np.float32) *
                                        observations[logincome_col].values[:, None].astype(np.float32))
        else:
            mats['cost_x_logincome'] = np.zeros((N, J), dtype=np.float32)

        # logdist_x_age
        age_col = None
        for c in observations.columns:
            if c.lower() == 'age':
                age_col = c
                break
        if age_col is not None:
            mats['logdist_x_age'] = dists * observations[age_col].values[:, None].astype(np.float32)
        else:
            mats['logdist_x_age'] = np.zeros((N, J), dtype=np.float32)

        rating_col = None
        for c in alternatives.columns:
            if c.lower() == 'rating':
                rating_col = c
                break
        edu_sec = None
        for c in observations.columns:
            if c.lower() == 'edu_secondary':
                edu_sec = c
                break
        if rating_col is not None and edu_sec is not None:
            mats['rating_x_edu_sec'] = (alternatives[rating_col].values[None, :].astype(np.float32) *
                                        observations[edu_sec].values[:, None].astype(np.float32))
        else:
            mats['rating_x_edu_sec'] = np.zeros((N, J), dtype=np.float32)

    names.extend([mats_key for mats_key in mats.keys()])

    return mats, names


def compute_utility_from_components(observations: pd.DataFrame, alternatives: pd.DataFrame,
                                    parameters: Dict[str, float], P: int,
                                    X_rest: Optional[np.ndarray] = None,
                                    rest_names: Optional[list] = None,
                                    inter_mats: Optional[dict] = None) -> np.ndarray:
    """
    Compute utility matrix V (N, J) using restaurant-only matrix and precomputed
    interaction matrices (no broadcasting of a giant X_full).

    Args:
        observations, alternatives, parameters, P: as usual
        X_rest: (J, K_rest) optional precomputed restaurant features
        rest_names: list of names corresponding to X_rest columns
        inter_mats: dict of precomputed (N, J) interaction matrices

    Returns:
        V: np.ndarray shape (N, J)
    """
    N = len(observations)
    J = len(alternatives)
    V = np.zeros((N, J), dtype=np.float32)

    # If not provided, build components
    if inter_mats is None:
        inter_mats = compute_interaction_matrices(observations, alternatives, P)
    if X_rest is None or rest_names is None:
        X_rest, rest_names = create_X_restaurants(alternatives, P)

    # Add contributions from restaurant-only features
    for i, feat in enumerate(rest_names):
        param_name = f'beta_{feat}'
        if param_name in parameters:
            beta = float(parameters[param_name])
            # X_rest column i is length J; broadcast to (N, J)
            V += beta * X_rest[:, i][None, :]

    # Add interactions (including log_dist)
    for name, mat in inter_mats.items():
        param_name = f'beta_{name}'
        if param_name in parameters:
            beta = float(parameters[param_name])
            V += beta * mat.astype(np.float32)

    return V