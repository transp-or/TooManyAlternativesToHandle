# data.py
import pandas as pd
import os
from typing import Tuple
from adaptive_sampler.utils import get_data_filename, get_features_for_P

def load_data(J: int = 100, N: int = 10000, P: int = 10, I: int = 0, C: int = 0, M: int = 0, interacted_attributes_only: bool = False,
              data_dir=None, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Load alternatives and observations data based on configuration parameters.
    
    Args:
        J: Number of restaurants
        N: Number of individuals
        P: Number of parameters (10, 20, or 50)
        I: Data imbalance factor (0 or 1)
        C: Correlation factor (0 or 1)  
        M: Choice randomness factor (0 or 1)
        data_dir: Directory containing data files (default is repo-relative)
        
    Returns:
        Tuple of (alternatives DataFrame, observations DataFrame, features list)
    """
    # Build target filenames according to the standard naming scheme
    restaurants_file = get_data_filename(J, N, P, I, C, M, interacted_attributes_only, "restaurants")
    choices_file = get_data_filename(J, N, P, I, C, M, interacted_attributes_only, "choices")

    # Search candidate directories in order
    # ``adaptive_sampler`` lives directly below the repository root.  Keep this
    # calculation separate from the historical parent-directory fallback below:
    # the generated datasets shipped with this checkout are under
    # ``data/data/final_2310``.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    repo = os.path.abspath(os.path.join(project_root, '..'))
    candidate_dirs = []
    if data_dir:
        candidate_dirs.append(data_dir)
    # Prefer the generated data directory inside the likelihood_free package (if present),
    # then fall back to repo-level synthetic_data_generation and repo root.
    candidate_dirs.extend([
        os.path.join(project_root, 'data', 'data', 'final_2310'),
        os.path.join(repo, 'synthetic_dataset', 'data', 'final_2310'),
        project_root,
        repo,
    ])

    found_rest = None
    found_choices = None
    for d in candidate_dirs:
        if not d:
            continue
        p_rest = os.path.join(d, restaurants_file)
        p_choices = os.path.join(d, choices_file)
        if os.path.exists(p_rest) and os.path.exists(p_choices):
            found_rest = p_rest
            found_choices = p_choices
            print(f"Found data files in directory: {d}")
            print(f"  Restaurants file: {restaurants_file}")
            print(f"  Choices file: {choices_file}")
            break

    if found_rest is None or found_choices is None:
        tried = '\n'.join([os.path.join(d or '', restaurants_file) for d in candidate_dirs])
        raise FileNotFoundError(
            f"Could not find restaurant/choice files using get_data_filename in candidate dirs. Tried:\n{tried}\n\n"
            "Make sure the generated files exist in one of these directories or pass the exact `data_dir` to load_data()."
        )

    # Load data
    if verbose:
        print(f"Loading restaurants file: {found_rest}")
        print(f"Loading choices file: {found_choices}")
    alternatives = pd.read_csv(found_rest)
    observations = pd.read_csv(found_choices)
    
    # Get features for this P configuration
    features = get_features_for_P(P)

    if verbose:
        print(f"Alternatives shape: {alternatives.shape}")
        print(f"Observations shape: {observations.shape}")
        print(f"Features for P={P}: {features}")

    return alternatives, observations, features

# No IO at import time. Call load_data() from scripts/notebooks when needed.
