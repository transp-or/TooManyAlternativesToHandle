#!/usr/bin/env python
"""Entry point: train the GP surrogate, retrieve it, and run the MCMC.

    python run_bolfi_canonical_v2.py [--simulator_type all|SA] [--dis_type 5+6] ...

main() drives, per (J, N): train_gp -> gp_retrive -> MCMC_sampling.  The reusable
machinery is in TuRbolfi_GPU.py / mnl_simulators.py.
"""
import argparse
import os
import pickle
import random
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive: plt.show() is a no-op, never blocks the batch run
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set threading for performance
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"
os.environ["NUMEXPR_NUM_THREADS"] = "24"

# Import BOLFI and Global modules
import GLOBAL_GPU as gv
import TuRbolfi_GPU as bolfi
from TuRbolfi_GPU import BOLFI_Estimator, gv_simulator
from botorch.models.transforms import Normalize

# Suppress warnings
warnings.filterwarnings("ignore")

# Set Default Device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_dtype(torch.float64)
# Run in CPU float64 mode.
DEVICE = "cpu"
torch.set_default_dtype(torch.float64)

# Keep the data outside TUBOLFI
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data" / "final_2310"


def set_determinism(seed):
    """Seed Python, NumPy and PyTorch, configure CUDA, and request deterministic operations."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # CUDA determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ============================================================================
# HELPER FUNCTIONS (From Notebook)
# ============================================================================

def get_base_parameters(P, interacted_attributes_only=False):
    """Return ordered coefficients, extending them at P >= 20 and P >= 40.

    Interaction-only mode omits continuous features and adds 35 interactions at
    P >= 40; the returned coefficient count need not equal P.
    """
    # Base 10 coefficients.
    base_params = {
        'beta_rating': 0.75,
        'beta_cost': -0.4,
        'beta_log_dist': -0.60,
        'beta_chinese': 0.83,
        'beta_japanese': 1.25,
        'beta_korean': 0.75,
        'beta_indian': 1.0,
        'beta_french': 0.95,
        'beta_mexican': 1.30,
        'beta_lebanese': 0.90,
    }
    
    if P >= 20:
        # Add three interaction coefficients.
        base_params.update({
            'beta_cost_x_logincome': 0.02,    
            'beta_logdist_x_age': -0.02,       
            'beta_rating_x_edu_sec': 0.15,    
        })
        if not interacted_attributes_only:
            # Add seven continuous restaurant features.
            base_params.update({
                'beta_c_16': 0.09, 'beta_c_19': -0.04, 'beta_c_20': 0.12,
                'beta_c_21': 0.22, 'beta_c_22': -0.18, 'beta_c_24': 0.28,
                'beta_c_25': -0.15,                
            })

    if P >= 40:
        if not interacted_attributes_only:
            # Add eight continuous features.
            base_params.update({
                'beta_c_26': 0.16, 'beta_c_27': 0.32, 'beta_c_29': 0.24, 'beta_c_30': 0.26,  
                'beta_c_32': 0.20, 'beta_c_33': 0.30, 'beta_c_34': -0.15, 'beta_c_35': 0.34,
                # Add twelve discrete features.
                'beta_d_36': 0.24, 'beta_d_37': 0.28, 'beta_d_38': -0.22, 'beta_d_39': 0.20,  
                'beta_d_41': 0.26, 'beta_d_42': -0.16, 'beta_d_43': 0.22,  
                'beta_d_44': 0.18, 'beta_d_45': 0.30, 'beta_d_46': -0.18, 
                'beta_d_49': 0.26, 'beta_d_50': 0.16,
            })
        else:
            # INTERACTED ATTRIBUTES ONLY: Replace beta_c_16 to beta_d_50 with interactions
            # Add 35 interaction coefficients at P >= 40.
            # Age x all 7 cuisines (7 interactions)
            base_params.update({
                'beta_age_x_chinese': 0.025, 'beta_age_x_japanese': 0.03, 'beta_age_x_korean': -0.02,
                'beta_age_x_indian': 0.02, 'beta_age_x_french': 0.04, 'beta_age_x_mexican': -0.025,
                'beta_age_x_lebanese': 0.023,
            })
            # Gender x all 7 cuisines (7 interactions)
            base_params.update({
                'beta_gender_x_chinese': 0.06, 'beta_gender_x_japanese': 0.08, 'beta_gender_x_korean': 0.05,
                'beta_gender_x_indian': -0.03, 'beta_gender_x_french': 0.07, 'beta_gender_x_mexican': -0.04,
                'beta_gender_x_lebanese': 0.04,
            })
            # Risk x all 7 cuisines (7 interactions)
            base_params.update({
                'beta_risk_x_chinese': 0.03, 'beta_risk_x_japanese': 0.04, 'beta_risk_x_korean': 0.06,
                'beta_risk_x_indian': 0.07, 'beta_risk_x_french': 0.02, 'beta_risk_x_mexican': 0.03,
                'beta_risk_x_lebanese': 0.05,
            })
            # Has_children x alternatives (3 interactions)
            base_params.update({
                'beta_haschildren_x_rating': 0.05, 'beta_haschildren_x_cost': -0.04, 'beta_haschildren_x_logdist': -0.03,
            })
            # Gender x alternatives (3 interactions - NEW, not in P20)
            base_params.update({
                'beta_gender_x_rating': 0.03, 'beta_gender_x_cost': -0.02, 'beta_gender_x_logdist': -0.025,
            })
            # Logincome x alternatives (2 NEW - cost already in P20)
            base_params.update({
                'beta_logincome_x_rating': 0.04, 'beta_logincome_x_logdist': 0.03,
            })
            # Edu_secondary x alternatives (2 NEW - rating already in P20)
            base_params.update({
                'beta_edu_sec_x_cost': 0.025, 'beta_edu_sec_x_logdist': 0.02,
            })
            # Tertiary education x cost and log distance (2 interactions).
            base_params.update({
                'beta_edu_tert_x_cost': 0.03, 'beta_edu_tert_x_logdist': 0.025,
            })
            # Age x alternatives (2 NEW - log_dist already in P20)
            base_params.update({
                'beta_age_x_rating': 0.025, 'beta_age_x_cost': -0.02,
            })
    return base_params

def get_scale(M, scale_trial=0.4):
    """Return scale_trial (default 0.4) for M == 1, otherwise 1.0; log the choice."""
    if M == 1:
        scale = scale_trial  # Random behavior = small scale = flat probabilities
        print(f"Applied rescaling (M=1): mu={scale}")
    else:
        scale = 1.0  # Selective behavior = large scale = clear preferences
        print("No rescaling applied (M=0)")
    return scale

def compute_utility(user_x, user_y, user_data, df_restaurants, parameters, P, interacted_attributes_only=False):
    """Return restaurant utilities for one individual using the implemented features.

    At P >= 40, interaction-only mode adds cuisine interactions and has_children
    terms; other interaction coefficients from get_base_parameters are unused.
    """
    dists = np.sqrt((df_restaurants['x'] - user_x) ** 2 + (df_restaurants['y'] - user_y) ** 2)
    V = (
        parameters['beta_rating'] * df_restaurants["rating"].values +
        parameters['beta_cost'] * df_restaurants["cost"].values +
        parameters['beta_log_dist'] * np.log(dists) +
        sum(parameters[f'beta_{c.lower()}'] * df_restaurants[c].values 
            for c in ["Chinese", "Japanese", "Korean", "Indian", "French", "Mexican", "Lebanese"] 
            if c in df_restaurants.columns and f'beta_{c.lower()}' in parameters)
    )
    if P >= 20:
        V += parameters['beta_cost_x_logincome'] * (df_restaurants["cost"].values * user_data['logincome'])
        V += parameters['beta_logdist_x_age'] * (np.log(dists) * user_data['age'])
        V += parameters['beta_rating_x_edu_sec'] * (df_restaurants["rating"].values * user_data['edu_secondary'])
        if not interacted_attributes_only:
            for c_num in [16, 19, 20, 21, 22, 24, 25]:
                c_col = f'c_{c_num}'
                if c_col in df_restaurants.columns and f'beta_{c_col}' in parameters:
                    V += parameters[f'beta_{c_col}'] * df_restaurants[c_col].values
    if P >= 40:
        if not interacted_attributes_only:
            for c_num in [26, 27, 29, 30, 32, 33, 34, 35]:
                c_col = f'c_{c_num}'
                if c_col in df_restaurants.columns and f'beta_{c_col}' in parameters:
                    V += parameters[f'beta_{c_col}'] * df_restaurants[c_col].values
            for d_num in [36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 49, 50]:
                d_col = f'd_{d_num}'
                if d_col in df_restaurants.columns and f'beta_{d_col}' in parameters:
                    V += parameters[f'beta_{d_col}'] * df_restaurants[d_col].values
        else:
            # Interaction-based utility logic
            for cuisine in ["chinese", "japanese", "korean", "indian", "french", "mexican", "lebanese"]:
                cuisine_col = cuisine.capitalize()
                for prefix in ["age", "gender", "risk"]:
                    param_key = f'beta_{prefix}_x_{cuisine}'
                    if param_key in parameters and cuisine_col in df_restaurants.columns:
                        V += parameters[param_key] * (df_restaurants[cuisine_col].values * user_data[prefix])
            # Add has_children interactions.
            for alt in ["rating", "cost"]:
                if f'beta_haschildren_x_{alt}' in parameters:
                    V += parameters[f'beta_haschildren_x_{alt}'] * (df_restaurants[alt].values * user_data['has_children'])
            if 'beta_haschildren_x_logdist' in parameters:
                V += parameters['beta_haschildren_x_logdist'] * (np.log(dists) * user_data['has_children'])
            # Remaining interaction coefficients are not evaluated here.
    return V

def generate_attribute_matrix(user_x, user_y, user_data, df_restaurants, parameters, P, J, interacted_attributes_only=False):
    """Return a (J, P) NumPy feature array for one individual.

    Populate features at the P thresholds. Interaction-only mode is partial:
    at P >= 40 it adds only age-by-cuisine terms starting at column 20.
    Unfilled columns remain zero; parameters is unused.
    """
    attributes = np.zeros((J, P))
    # Base attributes (P=10)
    attributes[:, 0] = df_restaurants["rating"].values
    attributes[:, 1] = df_restaurants["cost"].values
    dists = np.sqrt((df_restaurants['x'] - user_x) ** 2 + (df_restaurants['y'] - user_y) ** 2)
    attributes[:, 2] = np.log(dists)  # log_dist
    
    index_c = 3
    for c in ["Chinese", "Japanese", "Korean", "Indian", "French", "Mexican", "Lebanese"]:
        attributes[:, index_c] = df_restaurants[c].values
        index_c += 1
 
    # Add interactions for P>=20
    if P >= 20:
        attributes[:, 10] = df_restaurants["cost"].values * user_data['logincome']
        attributes[:, 11] = np.log(dists) * user_data['age']
        attributes[:, 12] = df_restaurants["rating"].values * user_data['edu_secondary']
       
        index_c = 13
        if not interacted_attributes_only:
            # Additional continuous features (c_16 to c_25)
            for c_num in [16, 19, 20]:
                c_col = f'c_{c_num}'
                if c_col in df_restaurants.columns:
                    attributes[:, index_c] = df_restaurants[c_col].values
                    index_c += 1
            for c_num in [21, 22, 24, 25]:
                c_col = f'c_{c_num}'
                if c_col in df_restaurants.columns:
                    attributes[:, index_c] = df_restaurants[c_col].values
                    index_c += 1
    
    if P >= 40:
        index_c = 20
        if not interacted_attributes_only:
            # More continuous features (c_26 to c_35)
            for c_num in [26, 27, 29, 30, 32, 33, 34, 35]:
                c_col = f'c_{c_num}'
                if c_col in df_restaurants.columns:
                    attributes[:, index_c] = df_restaurants[c_col].values
                    index_c += 1
            
            # Discrete features (d_36 to d_50)
            for d_num in [36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 49, 50]:
                d_col = f'd_{d_num}'
                if d_col in df_restaurants.columns:
                    attributes[:, index_c] = df_restaurants[d_col].values
                    index_c += 1         
        else:
            # INTERACTED ATTRIBUTES ONLY: Replace beta_c_16 to beta_d_50 with interactions
            # Age x all 7 cuisines (7 interactions)
            for cuisine in ["chinese", "japanese", "korean", "indian", "french", "mexican", "lebanese"]:
                cuisine_col = cuisine.capitalize()
                if cuisine_col in df_restaurants.columns:
                    attributes[:, index_c] = df_restaurants[cuisine_col].values * user_data['age']
                    index_c += 1
            # Other interaction-only columns remain zero.
    return attributes

def merge_attributes_matrix(df_individuals, df_restaurants, real_parameter, P, J, interacted_attributes_only=False):
    """Stack individual feature arrays into a NumPy array of shape (N, J, P)."""
    N = len(df_individuals)
    all_attributes_matrices = np.zeros((N, J, P))
    for i in range(N):
        all_attributes_matrices[i, :, :] = generate_attribute_matrix(
            user_x=df_individuals.loc[i, 'user_x'],
            user_y=df_individuals.loc[i, 'user_y'],
            user_data=df_individuals.loc[i],
            df_restaurants=df_restaurants, 
            parameters=real_parameter,
            P=P, J=J,
            interacted_attributes_only=interacted_attributes_only
        )
    return all_attributes_matrices
def betai_function(iter:int, param_dim: int,**params):                
    """Return the UCB exploration weight from the training-point counter and dimension."""
   
    betai = 2 * torch.log((iter+1)**(param_dim/2+2)*(np.pi ** 2)/(3*1))  # (Srinivas et al., 2010)


    return torch.tensor(betai,dtype=gv.DATA_TYPE_torch,device=gv.DEVICE)  

def train_gp(args, DEVICE):
    """Load data, configure the simulator, train BOLFI and save diagnostics/checkpoints.

    Return the fitted estimator, or None if data files are missing.
    Use 5 * P initial samples and a training-point budget of 150 * P.
    """
    base_dir = Path(__file__).parent
    data_dir = DATA_DIR
    
    # Dataset identifier (Match filename pattern)
    N_str = f"{args.N // 1000}k"
    data_id = f"J{args.J}_P{args.P}_N{N_str}_I{args.I}C{args.C}M{args.M}"
    restaurants_file = data_dir / f"restaurants_{data_id}.csv"
    choices_file = data_dir / f"choices_{data_id}.csv"
    
    print(f"Loading data for {data_id}...")
    if not (restaurants_file.exists() and choices_file.exists()):
        print(f"Error: Data files not found!")
        print(f"Expected: {restaurants_file}")
        return None

    df_restaurants = pd.read_csv(restaurants_file)
    df_individuals = pd.read_csv(choices_file)

    ####### Setting global variables  #########
    gv.DEVICE = DEVICE
    gv.DATA_TYPE_torch = torch.float64
    gv.DATA_TYPE_np = np.float64
    gv.MASTER_SEED = args.seed
    gv.EPSILION = torch.tensor(1e-6, device=DEVICE)
    
    # Determinism
    seed = int(gv.MASTER_SEED)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # for CUDA determinism
    # AV: TODO remove or not?
    # torch.set_num_threads(1)   # Prevent thread explosion on high-CPU machines
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize data tensors in GV
    gv.OBSERVATION_MATRIX = torch.tensor(df_individuals['logit_0'].values, dtype=torch.int64, device=DEVICE)
    gv.REAL_BETA_dict = get_base_parameters(args.P)
    gv.REAL_BETA_U = torch.tensor(list(gv.REAL_BETA_dict.values()), dtype=gv.DATA_TYPE_torch, device=DEVICE)
    gv.REAL_BETA = gv.REAL_BETA_U.clone().to(dtype=gv.DATA_TYPE_torch,device = gv.DEVICE)
    gv.size = torch.tensor(args.N, dtype=torch.int64, device=DEVICE)
    gv.J = torch.tensor(args.J, dtype=torch.int64, device=DEVICE)
    gv.P = torch.tensor(args.P, dtype=torch.int64, device=DEVICE)

    torch.set_default_dtype(gv.DATA_TYPE_torch)

    print("Generating Attribute Matrix...")
    attr_matrix = merge_attributes_matrix(df_individuals, df_restaurants, gv.REAL_BETA_dict, args.P, args.J)
    gv.ATTRIBUTE_MATRIX = torch.tensor(attr_matrix, dtype=gv.DATA_TYPE_torch, device=DEVICE)

    # Attribute discrepancy uses an unscaled copy.
    gv.SCALED_ATTRIBUTE_MATRIX = gv.ATTRIBUTE_MATRIX.clone().detach()

    # --- choose the simulator ("all" = utility+epsilon argmax; "SA" = Multiple-Try MH on the choice) ---
    gv.SIMULATOR_TYPE = getattr(args, "simulator_type", "all")
    if gv.SIMULATOR_TYPE == "SA":
        mhw = getattr(args, "mh_steps_warm", "") or ""
        mhw = [int(x) for x in str(mhw).replace(",", " ").split()] or None   # e.g. "10,5" -> [10,5]
        bolfi.reset_sa_simulator(num_alt=args.J, obs_choices=gv.OBSERVATION_MATRIX,
                                 mh_steps=getattr(args, "mh_steps", 1),
                                 mh_steps_warm=mhw,
                                 mtm_m=getattr(args, "sa_mtm_m", 1),
                                 carry=True,            # persistent per-slot choice chain
                                 cold_bo_iters=1,       # first BO iter follows the init rule
                                 device=DEVICE)
        print(f"Simulator: SA/MTM  (mtm_m={bolfi._SA.mtm_m}, "
              f"mh_steps init={bolfi._SA.mh_steps_init}, "
              f"warm/stage={bolfi._SA.mh_steps_stage}, carry={bolfi._SA.carry}, "
              f"cold_bo_iters={bolfi._SA.cold_bo_iters})")
    else:
        print("Simulator: all  (utility + epsilon, argmax over alternatives)")

    # Results folder
    folder_path = base_dir / "final_result" / f"{args.simulator_type}" / f"{data_id}_D{args.dis_type}_SIM{args.sim_num}"
    folder_path.mkdir(parents=True, exist_ok=True)
    bolfi_estimator = None
    # Start BOLFI
    print(f"\nStarting BOLFI Estimation on {DEVICE}...")
    tracemalloc.start()
    start_time = time.time()
    
    SIM_NUM = torch.tensor(args.sim_num, dtype=torch.int64, device=DEVICE) # number of simulations to run per discrepancy evaluation;
    
    bolfi_estimator = bolfi.BOLFI_Estimator(
        dim=gv.P,
        simulator_fun=bolfi.simulate,        # dispatcher -> gv_simulator / sa_simulator per gv.SIMULATOR_TYPE
        sim_num=SIM_NUM,
        dis_type=args.dis_type,
        epsilon=gv.EPSILION,
        file_route=str(folder_path) + "/"
    )

    # ========================================================================
    # ATTR-BASED SETUP (Fix for "global_sd" Error)
    # ========================================================================
    if bolfi_estimator.dis_type in ["6", "6+5"]:
        print("Calibrating attribute-based discrepancy weights (global_sd) via pilot simulation...")
        pilot_num = 50
        attribute_s = torch.zeros((pilot_num, bolfi_estimator.dim), dtype=gv.DATA_TYPE_torch, device=DEVICE)
        
        for i in range(pilot_num):
            beta_prior = torch.empty((bolfi_estimator.dim,), dtype=gv.DATA_TYPE_torch, device=DEVICE).uniform_(-3, 3, generator=bolfi_estimator.init_train_X_rng)
            res_chosen_matrix, _ = bolfi.gv_simulator(
                estimable_parameters_vector=beta_prior,
                attributes_matrix=gv.ATTRIBUTE_MATRIX,
                noise_type="gumbel",
                num_sim=1,
                device=DEVICE,
                rgn_seed=gv.MASTER_SEED + i  # Ensure different seed for each pilot simulation
            )
            idx = res_chosen_matrix.T
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, bolfi_estimator.dim)
            attribute_s[i, :] = torch.gather(gv.SCALED_ATTRIBUTE_MATRIX, 1, idx_exp).sum(dim=(0, 1))

        diff_pilot_var = torch.cov(attribute_s.T, correction=1)
        # eye = torch.eye(bolfi_estimator.dim, device=DEVICE, dtype=gv.DATA_TYPE_torch)
        # pilot_var = torch.linalg.solve(A=diff_pilot_var + 1e-6 * eye, B=eye)
        pilot_var = torch.linalg.solve(A = diff_pilot_var,B = torch.eye(bolfi_estimator.dim, device=gv.DEVICE, dtype=gv.DATA_TYPE_torch))  # (param_num,param_num)
        bolfi_estimator.global_sd = torch.linalg.cholesky(pilot_var).to(device=DEVICE, dtype=gv.DATA_TYPE_torch)

    elif bolfi_estimator.dis_type in ["5+6"]:
        print("Using Identity matrix for global_sd (Stage-1 focus discrepancy as per notebook)...")
        bolfi_estimator.global_sd = torch.eye(bolfi_estimator.dim, device=DEVICE, dtype=gv.DATA_TYPE_torch)

    # Initialize GP
    print("Initializing GP...")
    # warning scale !!!! TODO
    # normal corresponds to v1 and v3 results !!
    prior_param = {
    "loc": torch.zeros_like(gv.REAL_BETA_U).to(
        dtype=gv.DATA_TYPE_torch,
        device=gv.DEVICE
    ),
    "scale": torch.ones_like(gv.REAL_BETA_U).to(
        dtype=gv.DATA_TYPE_torch,
        device=gv.DEVICE
    ),
}
    if gv.SIMULATOR_TYPE == "SA":
        bolfi._SA.warm = False        # init phase: reseed choices by random sampling w/ replacement
    bolfi_estimator.initialize_gp(
    prior_type="normal",
    prior_param=prior_param,
    init_num=int(5 * args.P),
    proportion_of_sobol=torch.tensor(
        1,
        device=gv.DEVICE,
        dtype=gv.DATA_TYPE_torch
    ),
)
    ## New version
    bolfi_estimator.system = args.system

    bolfi_estimator.save_train_data(train_X=bolfi_estimator.train_X_int, train_Y=bolfi_estimator.train_Y_int, note="INITIAL")

    # Train
    print("Training BOLFI...")
    # (SA: train_bolfi flips _SA.warm per BO iteration - the first `cold_bo_iters`
    #  iterations still follow the init rule, then the persistent chain takes over)
    max_iters = int(150 * args.P)

    batch_size = args.batch_size
    convergence_type = args.convergence_type if args.convergence_type else "auto"
    bolfi_estimator.train_bolfi(
        max_iters=max_iters,
        convergence_check_length= 10, 
        batch_size=batch_size,
        betai_function=betai_function,
        convergence_type=convergence_type,
        tol_gap = 0.05,tol_step=0.05,tol_sigma=0.05, 
        callback_interval=500, global_percentage=0.8
        )

    # Post-processing
    bolfi_estimator.gp_training_time = time.time() - start_time
    print(f"\nTraining completed in {bolfi_estimator.gp_training_time:.1f} seconds.")
    print(pd.DataFrame(bolfi_estimator.train_X_neat.detach().cpu().numpy()).describe())
    print(pd.DataFrame(bolfi_estimator.train_Y_neat.detach().cpu().numpy()).describe())
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak memory usage: {peak/1024/1024:.2f} MB")
    tracemalloc.stop()

    bolfi_estimator.plot_training_trajectory()
    bolfi_estimator.est_sigma2()
    bolfi_estimator.est_tol()
    bolfi_estimator.save_train_data() # Save FINAL CSVs
    bolfi_estimator.result_output()
    bolfi_estimator.plt_time_cost()

    # Extra Fitness Checks (from Notebook)
    gp = bolfi_estimator.gp
    
    best_lfi_beta = bolfi_estimator.train_X_neat[bolfi_estimator.train_Y_neat[:,-1].argmax()]
    if gv.SIMULATOR_TYPE == "SA":
        print(f"SA running best-match to observed choices: {bolfi._SA.best_match:.4f}")
        print(f"SA last-call MH acceptance rate: {bolfi._SA.accept_rate:.4f}")
    print(f"The true parameter is: {gv.REAL_BETA_U}")
    print(f"The best train_X is: {best_lfi_beta}")

    # Return the estimator after saving diagnostics.
    return bolfi_estimator


def gp_retrive(args, DEVICE, tol_quantile=0.95, gp_iters=None,notes = "max_iters"):
    """Restore saved arrays/settings and return an estimator with a refitted GP.

    Use args.gp_route when present, otherwise the canonical result path.
    gp_iters limits stored rows; otherwise notes="early_stop" selects the recorded
    stopping count, and other notes use all rows. Counts are capped at stored size.
    Two-stage runs always fit the final target column. Recompute tolerance from
    all stored targets. Return None if data or checkpoint files are missing.
    """
    base_dir = Path(__file__).parent
    data_dir = DATA_DIR

    # Dataset identifier (Match filename pattern used by train_gp)
    N_str = f"{args.N // 1000}k"
    data_id = f"J{args.J}_P{args.P}_N{N_str}_I{args.I}C{args.C}M{args.M}"
    print(f"Retrieving GP for {data_id} (D{args.dis_type}), interacted_attributes_only=False...")

    restaurants_file = data_dir / f"restaurants_{data_id}.csv"
    choices_file = data_dir / f"choices_{data_id}.csv"
    if not (restaurants_file.exists() and choices_file.exists()):
        print(f"Error: Data files not found! Expected: {restaurants_file}")
        return None
    df_restaurants = pd.read_csv(restaurants_file)
    df_individuals = pd.read_csv(choices_file)

    # Saved training-results pickle: explicit override, else canonical train_gp path
    if getattr(args, "gp_route", None):
        gp_route = Path(args.gp_route)
    else:
        gp_route = (base_dir / "final_result" / f"{args.simulator_type}"
                    / f"{data_id}_D{args.dis_type}_SIM{args.sim_num}"
                    / f"bolfi_training_results_summary({args.dis_type}).pkl")
    if not gp_route.exists():
        print(f"Error: GP pickle not found: {gp_route}")
        return None
    with open(gp_route, "rb") as f:
        bolfi_info = pickle.load(f)

    ####### Setting global variables (mirror train_gp) #########
    gv.DEVICE = DEVICE
    gv.DATA_TYPE_torch = torch.float64
    gv.DATA_TYPE_np = np.float64
    gv.MASTER_SEED = args.seed
    torch.set_default_dtype(gv.DATA_TYPE_torch)
    set_determinism(args.seed)   # Seed before rebuilding state.

    gv.OBSERVATION_MATRIX = torch.tensor(df_individuals['logit_0'].values, dtype=torch.int64, device=DEVICE)
    gv.REAL_BETA_dict = get_base_parameters(args.P, interacted_attributes_only=False)
    gv.REAL_BETA_U = torch.tensor(list(gv.REAL_BETA_dict.values()), dtype=gv.DATA_TYPE_torch, device=DEVICE)
    gv.REAL_BETA = gv.REAL_BETA_U.clone().to(dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)
    gv.size = torch.tensor(args.N, dtype=torch.int64, device=DEVICE)
    gv.J = torch.tensor(args.J, dtype=torch.int64, device=DEVICE)
    gv.P = torch.tensor(args.P, dtype=torch.int64, device=DEVICE)

    # Values from the saved run take precedence over CLI args
    gv.EPSILION = bolfi_info['epsilon']
    gv.MASTER_SEED = bolfi_info['master seed']
    sim_num = bolfi_info['sim_num']  # number of simulations per discrepancy evaluation

    print("Generating Attribute Matrix...")
    attr_matrix = merge_attributes_matrix(
        df_individuals, df_restaurants, gv.REAL_BETA_dict, args.P, args.J,
        interacted_attributes_only=False,
    )
    gv.ATTRIBUTE_MATRIX = torch.tensor(attr_matrix, dtype=gv.DATA_TYPE_torch, device=DEVICE)
    gv.SCALED_ATTRIBUTE_MATRIX = gv.ATTRIBUTE_MATRIX.clone().detach()

    folder_path = (base_dir / "final_result" / f"{args.simulator_type}"
                   / f"{data_id}_D{args.dis_type}_SIM{args.sim_num}")   # Use the canonical output folder, even with a custom input checkpoint.
    folder_path.mkdir(parents=True, exist_ok=True)

    bolfi_estimator = bolfi.BOLFI_Estimator(
        dim=bolfi_info['dim'],
        simulator_fun=bolfi.gv_simulator,
        sim_num=sim_num,
        dis_type=args.dis_type,
        epsilon=gv.EPSILION,
        file_route=str(folder_path) + "/",
    )

    # Restore estimator state from the saved run
    bolfi_estimator.CHILD_SEED = bolfi_info['child seed']
    bolfi_estimator.bounds = bolfi_info['bounds']
    bolfi_estimator.prior_type = bolfi_info['prior_type']
    bolfi_estimator.prior_param = bolfi_info['prior_param']
    bolfi_estimator.init_num = bolfi_info['init_num']
    bolfi_estimator.batch_size = bolfi_info['batch_size']
    bolfi_estimator.tol_gap = bolfi_info['tol_gap']
    bolfi_estimator.tol_ucb = bolfi_info['tol_ucb']
    bolfi_estimator.tol_step = bolfi_info['tol_step']
    bolfi_estimator.tol_sigma = bolfi_info['tol_sigma']
    bolfi_estimator.cov = bolfi_info['gp_cov']
    bolfi_estimator.training_results = bolfi_info['training_results']
    bolfi_estimator.tol_quantile = bolfi_info['gp_tol_quantile']
    bolfi_estimator.train_X_normed = bolfi_info['train_X_normed']
    bolfi_estimator.train_X_neat = bolfi_info['train_X_neat']
    bolfi_estimator.train_Y_neat = bolfi_info['train_Y_neat']
    bolfi_estimator.global_sd = bolfi_info['global_sd']
    bolfi_estimator.time_log = bolfi_info['time_log']
    bolfi_estimator.iter_min = bolfi_info['iter_min']
    bolfi_estimator.iter_early_stop = bolfi_info['iter_early_stop']
    bolfi_estimator.gp_training_time = bolfi_info['gp_training_time']
    bolfi_estimator.time_storage = bolfi_info['time_storage']

    bolfi_estimator.normalize = Normalize(d=bolfi_estimator.dim, bounds=bolfi_estimator.bounds)
    bolfi_estimator.tol = bolfi_estimator.est_tol(tol_quantile=tol_quantile)

    # Interpret the requested point count as a capped slice of stored neat rows.
    n_stored = bolfi_estimator.train_X_normed.size(0)
    early_stop_iters = bolfi_estimator.iter_early_stop if bolfi_estimator.iter_early_stop is not None else n_stored
    two_stage = bolfi_estimator.dis_type in ["5+6", "6+5"]
    stage = 1 if two_stage else 0

    if gp_iters is not None:
        train_index = min(int(gp_iters), n_stored)
    elif notes == "early_stop":
        train_index = min(early_stop_iters, n_stored)
    else:                       # notes in {"iter_max", "max_iters", None, ...}: use all data
        train_index = n_stored

    if two_stage and train_index < bolfi_estimator.iter_min:
        print(f"Warning: train_index ({train_index}) < iter_min ({bolfi_estimator.iter_min}); "
              f"GP is refit on stage-1 data only and may be under-trained.")

    # re-seed with the run's master seed right before the GP hyper-parameter fit
    set_determinism(gv.MASTER_SEED)
    
    bolfi_estimator.gp_fit(
        stage=stage,
        train_X=bolfi_estimator.train_X_normed[:train_index, :],
        train_Y=bolfi_estimator.train_Y_neat[:train_index],
    )
    
    # ------------------------------------------------------------------
    # Approximate-likelihood profiles: sweep one beta at a time around the
    # best acquired point, holding the rest fixed, and plot the GP-approx
    # likelihood. Uses the same GP / best-point / slice conventions as the
    # rest of gp_retrive and MCMC_sampling (train_index, train_Y_neat argmax).
    # ------------------------------------------------------------------
    # gp = bolfi_estimator.gp
    # beta_name = list(gv.REAL_BETA_dict.keys())
    # best_lfi_beta = bolfi_estimator.train_X_neat[
    #     bolfi_estimator.train_Y_neat[:train_index, stage].argmax()
    # ].detach().view(-1)

    # steps = 30
    # real_beta = gv.REAL_BETA_U.detach().view(-1)
    # lows = torch.minimum(real_beta, best_lfi_beta) - 1.0
    # highs = torch.maximum(real_beta, best_lfi_beta) + 1.0

    # with torch.no_grad():
    #     for j, name in enumerate(beta_name):
    #         shift_thetas = torch.linspace(lows[j].item(), highs[j].item(), steps,
    #                                       dtype=gv.DATA_TYPE_torch, device=DEVICE)
    #         shift_llk = np.zeros(steps)
    #         for i in range(steps):
    #             tr_theta = best_lfi_beta.clone()
    #             tr_theta[j] = shift_thetas[i]
    #             shift_llk[i] = bolfi_estimator.calculate_likelihood(
    #                 gp=gp, beta=tr_theta.view(1, -1)
    #             ).item()

    #         plt.figure()
    #         plt.plot(shift_thetas.cpu().numpy(), shift_llk, color="black")
    #         plt.axvline(real_beta[j].item(), linestyle="--", c="blue", label="true")
    #         plt.axvline(best_lfi_beta[j].item(), linestyle="--", c="brown", label="lfi best point")
    #         plt.legend(fontsize=13, loc="upper right")
    #         plt.grid()
    #         plt.tick_params(axis="both", labelsize=15)
    #         plt.xlabel(name, fontsize=20)
    #         plt.ylabel("approximate likelihood")
    #         plt.savefig(
    #             fname=f"{bolfi_estimator.file_route}MLBA_joint_{name}_{notes}.pdf",
    #             dpi=1080, bbox_inches="tight",
    #         )
    #         plt.close()

    return bolfi_estimator



def MCMC_sampling(args, bolfi_estimator, DEVICE,notes=None):
    """Return four DEMetropolisZ chains and save trace/summary files.

    Use Normal(0, 10) priors and a GP likelihood with a variance penalty.
    Each chain uses 2,500 tuning steps and 5,000 draws, with seed 42.
    notes selects the initialization/threshold slice and output suffix.
    Return None if MCMC dependencies are unavailable.
    """
    print("\nStarting MCMC Sampling (PyMC)...")
    os.environ["PYTENSOR_FLAGS"] = "openmp=False"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    try:
        import pymc as pm
        import pytensor
        import pytensor.tensor as pt
        pytensor.config.openmp = False
        pytensor.config.floatX = "float64"   # match the float64 torch GP (no float32 rounding)
        from pytensor.graph.op import Op
        from pytensor.graph.basic import Apply
        import arviz as az
    except ImportError:
        print("MCMC dependencies (pymc, pytensor, arviz) not found. Skipping.")
        return

    set_determinism(42)   # reproducible chain seeds / init
    gp = bolfi_estimator.gp

    # Penalize high GP variance with a bounded log-sigmoid gate.
    bolfi_estimator.llk_mode = "trust"
    bolfi_estimator.sigma_trust_c = 3.0
    bolfi_estimator.gate_sharpness = 10.0
    print(f"MCMC likelihood: llk_mode={bolfi_estimator.llk_mode}"
          + (f", sigma_trust_c={bolfi_estimator.sigma_trust_c}, gate_sharpness={bolfi_estimator.gate_sharpness}"
             if bolfi_estimator.llk_mode == "trust" else ""))

    if notes == "early_stop":
        n_stored = bolfi_estimator.iter_early_stop if bolfi_estimator.iter_early_stop is not None else bolfi_estimator.train_X_normed.size(0)
    else:
        n_stored = bolfi_estimator.train_X_normed.size(0)

    best_lfi_beta = bolfi_estimator.train_X_neat[bolfi_estimator.train_Y_neat[:n_stored,-1].argmax()]
    beta_name = list(gv.REAL_BETA_dict.keys())

    if bolfi_estimator.llk_mode == "trust":
        bolfi_estimator._sigma_trust_n = n_stored
        bolfi_estimator._sigma_trust_max_val = None          # force recompute for this GP slice
        sig_max = float(bolfi_estimator._sigma_trust_max())
        with torch.no_grad():
            sig_best = float(gp.posterior(bolfi_estimator.normalize(best_lfi_beta.view(1, -1))).variance.view(-1))
        cov0 = float(torch.as_tensor(bolfi_estimator.cov))
        print(f"  trust gate: sigma_trust_max={sig_max:.4g} (c={bolfi_estimator.sigma_trust_c}); "
              f"Sigma(best_beta)={sig_best:.4g}; cov0={cov0:.4g}")
        if sig_best >= sig_max:
            print("  WARNING: Sigma(best_beta) >= sigma_trust_max - the gate would suppress the "
                  "MCMC init region; raise sigma_trust_c.")

    # Approximate log-likelihood wrapper
    def approximate_llk_np(beta):
        return float(bolfi_estimator.calculate_likelihood(gp=gp, beta=torch.Tensor(beta).to(DEVICE).view(1, -1)))

    class LogLike(Op):
        def make_node(self, beta) -> Apply:
            beta = pt.as_tensor_variable(beta).astype(pytensor.config.floatX)
            outputs = pt.scalar(dtype=pytensor.config.floatX)
            return Apply(self, [beta], [outputs])

        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            beta = inputs[0]
            loglike_eval = approximate_llk_np(beta)
            outputs[0][0] = np.asarray(loglike_eval, dtype=node.outputs[0].dtype)

    loglike_op = LogLike()

    with pm.Model() as model:
        betas = [pm.Normal(name, mu=0.0, sigma=10.0) for name in beta_name]
        beta_vector = pm.math.stack([pt.cast(betas[i], pytensor.config.floatX) for i in range(len(beta_name))])
        pm.Potential("ll", loglike_op(beta_vector))

        # Initial values from BOLFI best point
        init_val = best_lfi_beta.cpu().numpy()
        start_dict = {name: init_val[i] for i, name in enumerate(beta_name)}
        step = pm.DEMetropolisZ(tune_drop_fraction =0.9,
                                scaling = 5*1e-4,
                                tune="lambda") # Adapt the proposal scale during tuning.

        print("Running MCMC Chains...")
        trace = pm.sample(
            draws=5000,
            tune=2500,
            chains=4,
            cores=4,
            step = step,
            initvals=start_dict,
            random_seed=42,
            progressbar=True
        )
    
    # Save trace summary
    summary = az.summary(trace)

    # add true value + whether the HDI (credible interval) covers it
    true_map = {name: float(v) for name, v in gv.REAL_BETA_dict.items()}
    hdi_cols = sorted([c for c in summary.columns if c.startswith("hdi_")],
                      key=lambda c: float(c.replace("hdi_", "").rstrip("%")))
    lo_col, hi_col = hdi_cols[0], hdi_cols[-1]
    summary["true"] = [true_map.get(ix, np.nan) for ix in summary.index]
    summary["covered"] = (summary["true"] >= summary[lo_col]) & (summary["true"] <= summary[hi_col])

    n_cov = int(summary["covered"].sum())
    print("\nMCMC Summary:")
    print(summary)
    print(f"HDI ({lo_col}..{hi_col}) covers true value for {n_cov}/{len(summary)} parameters")
    summary.to_csv(bolfi_estimator.file_route + f"MCMC_summary_{notes}.csv")
    #  save trace for further analysis
    trace.to_netcdf(bolfi_estimator.file_route + f"MCMC_trace_{notes}.nc")
    return trace

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Canonical SA training seeds for (J, N), with P=10 and I=C=M=0.
SA_DATASET_SEEDS = {
    (100, 10000): 42,
    (100, 20000): 42,
    (100, 50000): 42,
    (200, 10000): 42,
    (200, 20000): 41,
    (200, 50000): 42,
    (500, 10000): 41,
    (500, 20000): 42,
    (500, 50000): 42,
}


def main(P,J,N):
    """Parse CLI settings, train BOLFI, then run final and early-stop GP refits and MCMC."""
    parser = argparse.ArgumentParser(description="Run BOLFI Estimation")
    parser.add_argument("--J", type=int, default=J)
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--P", type=int, default=P)
    parser.add_argument("--I", type=int, default=0)
    parser.add_argument("--C", type=int, default=0)
    parser.add_argument("--M", type=int, default=0)
    
    parser.add_argument("--system", type=str, default="windows")
    # parser.add_argument("--system", type=str, default="os")
    
    parser.add_argument("--sim_num", type=int, default=30)
    parser.add_argument("--dis_type", type=str, default="5+6")
    parser.add_argument("--seed", type=int, default=None,
                        help="Training seed; defaults to the canonical SA dataset seed, otherwise 42")
    # version suffix and max_iters are fixed: folder = final_result/{sim}/{data_id}_D{dis}_SIM{sim_num}
    # (no suffix), max_iters = 150 * P.
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--convergence_type", type=str, default="fixed")
    parser.add_argument("--simulator_type", type=str, default="SA", choices=["all", "SA"],
                        help="'all' = utility+epsilon argmax over all alternatives; "
                             "'SA' = sequential-adjustment, Multiple-Try Metropolis step(s) ")
    parser.add_argument("--mh_steps", type=int, default=30,
                        help="SA only: MTM sweeps per call during the init phase")
    parser.add_argument("--mh_steps_warm", type=str, default="20,10",
                        help="SA only: MTM sweeps per call during BO - int, or per-stage ")
    parser.add_argument("--sa_mtm_m", type=int, default=5,
                        help="SA only: Multiple-Try Metropolis tries per sweep. 1 = plain "
                             "independence MH (uniform proposal); larger values mix much "
                             "faster when J is large, at O(m) cost per sweep")
    args = parser.parse_args()
    if args.seed is None:
        canonical_sa = (
            args.simulator_type == "SA" and args.P == 10
            and (args.I, args.C, args.M) == (0, 0, 0)
        )
        args.seed = SA_DATASET_SEEDS.get((args.J, args.N), 42) if canonical_sa else 42
    print(f"Training seed: {args.seed}")

    bolfi_estimator = train_gp(args, DEVICE)
    if bolfi_estimator:
    # estimation with early stop
        early_stop = bolfi_estimator.iter_early_stop if bolfi_estimator else None
        
        bolfi_estimator = gp_retrive(args, DEVICE, 
                                            tol_quantile=0.85, 
                                            notes="early_stop")
        MCMC_sampling(args, bolfi_estimator, DEVICE,notes="early_stop")
                
        
        # estimation when iter_max
        if args.convergence_type == "fixed":
            bolfi_estimator = gp_retrive(args, DEVICE, tol_quantile=0.85, notes="iter_max")
            MCMC_sampling(args, bolfi_estimator, DEVICE,notes="iter_max")
        


if __name__ == "__main__":
    P = 10
    J = 100
    N = 10000
    print(f"\nRunning BOLFI for P={P}, J={J}, N={N}...")
    main(P, J, N)
