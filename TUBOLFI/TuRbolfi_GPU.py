"""TuRbolfi_GPU - TuRBO-BOLFI likelihood-free inference for discrete-choice models.

Layout
------
    1. MNL utility + choice simulators  (mnl_choice_utility, gv_simulator, the SA
       simulator, the `simulate` dispatcher)                  -> mnl_simulators.py
    2. compute_discrepancy  - negative discrepancy between simulated and observed choices
    3. TuRBO trust-region state + candidate generation  (TurboState, turbo_update_state,
       generate_batch)
    4. BOLFI_Estimator  - the end-to-end estimator: init GP -> BO training loop ->
       sigma^2 / tolerance / approximate-likelihood -> IO & plots

`run_bolfi_canonical_v2.py` is the entry point that drives all of this.
"""
import gc
import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
from joblib import Parallel, delayed
from torch.quasirandom import SobolEngine

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.generation import MaxPosteriorSampling
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood

import GLOBAL_GPU as gv
# simulators live in their own module; re-exported here so `import TuRbolfi_GPU as bolfi`
# keeps giving `bolfi.gv_simulator`, `bolfi.sa_simulator`, `bolfi.simulate`, `bolfi._SA`, ...
from mnl_simulators import (
    mnl_choice_utility, gv_simulator,
    _SAState, _SA, reset_sa_simulator, sa_simulator, simulate, _obs_sim_util,
)


def set_seed(seed):
    """Seed Python, NumPy and PyTorch; request deterministic operations when available."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# (1) MNL utility + choice simulators -> mnl_simulators.py  (re-imported above)


# ============================================================================
# (2) DISCREPANCY   -  negative discrepancy between simulated and observed choices
# ============================================================================
def compute_discrepancy(beta: torch.Tensor,  sim_num: int,dis_type: str, global_sd:torch.Tensor, rgn_seed: Optional[int] = None, sequential: Optional[int]=0,simulator_fun: callable = simulate, *args, **kwargs):
        """Return negative discrepancy using GLOBAL_GPU data/settings; larger means a better data match.

        Types: 1/nllk = frequency NLL; 2/mse = frequency squared error;
        3/cp2 = log population-frequency L1 distance; 4/utility+frequency = NLL plus
        utility discrepancy; 5/utility-based = utility discrepancy; 6/attribute =
        attribute discrepancy. 5+6 and 6+5 return both components in that order.

        The simulator returns choices (sim_num, N) and utilities (sim_num, N, 2).
        Attribute discrepancy uses log(norm(diff @ global_sd)); one simulation with
        no weights uses the unlogged Euclidean norm. Types 1 and 2 require sim_num >= 2.
        Offset the seed by sequential. Extra args/kwargs are unused.
        """
        
        if rgn_seed is None:
            rgn_seed = int(gv.MASTER_SEED+sequential) % (2**32)
            
        else:
            rgn_seed = int(rgn_seed+sequential) % (2**32)
            
        res_chosen_matrix,res_observed_matrix = simulator_fun(estimable_parameters_vector = beta,attributes_matrix = gv.ATTRIBUTE_MATRIX,num_sim = sim_num, device=gv.DEVICE,rgn_seed=rgn_seed)
        size = gv.size
        J = gv.J
        ar_size = torch.arange(size, device=gv.DEVICE, dtype=torch.long)
        if dis_type in ("1", "nllk", "2", "mse") and (sim_num<2):  #, "3", "cp2"
            raise ValueError("This is frequency-based discrepancy, please set sim_num >= 2.")
        elif dis_type in ("4","utility+frequency","6","utility+attribute") and (gv.SCALED_ATTRIBUTE_MATRIX is None):
            raise ValueError("Please provide scaled_attributes_matrix for discrepancy types 4 and 6 for attribute-related discrepancy.")
        
        elif dis_type in ("1", "nllk"):
            offset = ar_size * J                     # (size,)
            flat = (res_chosen_matrix + offset.unsqueeze(0)).reshape(-1)    # (sim_num*size,)
            counts = torch.bincount(flat, minlength=size * J)  # (size*J,)
            choice_freq = counts.reshape(size, J)/ float(sim_num)                # (size, J) float
            picked = choice_freq[ar_size,gv.OBSERVATION_MATRIX].clamp_min(gv.EPSILION)
            discrepancy = -torch.log(picked).mean()
            # Average NLL of the observed choices using clamped frequencies.
            
        elif dis_type in ("2", "mse"):
            # Offset bins per column, then one global bincount
            offset = ar_size * J                     # (size,)
            flat = (res_chosen_matrix + offset.unsqueeze(0)).reshape(-1)    # (sim_num*size,)
            counts = torch.bincount(flat, minlength=size * J)  # (size*J,)
            choice_freq = counts.reshape(size, J)/ float(sim_num)               # (size, J) float
            
            chosen = choice_freq[ar_size, gv.OBSERVATION_MATRIX]
            chosen_se = torch.sum((1.0 - chosen) ** 2)
            unchosen_mask = (torch.arange(J, device=gv.DEVICE).unsqueeze(0) != gv.OBSERVATION_MATRIX.unsqueeze(0))  # (size, J)
            unchosen_se = torch.sum((choice_freq[unchosen_mask]) ** 2)
            discrepancy = (chosen_se + unchosen_se) / float(self.size)
            
            
        elif dis_type in ("3", "cp2"):
            sim_freq = torch.bincount(res_chosen_matrix.reshape(-1), minlength=J).to(dtype=torch.int64) / float(sim_num * size)
            real_freq = torch.bincount(gv.OBSERVATION_MATRIX.reshape(-1), minlength=J).to(dtype=torch.int64) / float(size)
            discrepancy = torch.log((sim_freq - real_freq).abs().sum().clamp_min(gv.EPSILION))
            
        elif dis_type in ("4","utility+frequency"):
            offset = ar_size * J                     # (size,)
            flat = (res_chosen_matrix + offset.unsqueeze(0)).reshape(-1)    # (sim_num*size,)
            counts = torch.bincount(flat, minlength=size * J)  # (size*J,)
            choice_freq = counts.reshape(size, J)/ float(sim_num)               # (size, J) float
    
            picked = choice_freq[ar_size, gv.OBSERVATION_MATRIX].clamp_min(gv.EPSILION)
            discrepancy1 = -torch.log(picked).mean()

            true_obs_utility, sim_obs_utility = _obs_sim_util(res_observed_matrix)          # (size,1), (size,sim_num)
            utility_diff = true_obs_utility - sim_obs_utility                               # (size, sim_num)
            discrepancy_vec2 = torch.logsumexp(utility_diff, dim=1) - torch.log(
                torch.as_tensor(float(sim_num * J), dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)
            )
            discrepancy2 = -discrepancy_vec2.mean()
            discrepancy = discrepancy1 + discrepancy2
                
        elif dis_type in ("5","utility-based"):
            true_obs_utility, sim_obs_utility = _obs_sim_util(res_observed_matrix)          # (size,1), (size,sim_num)
            utility_diff = true_obs_utility - sim_obs_utility                               # (size, sim_num)
            discrepancy_vec = torch.logsumexp(utility_diff, dim=1) - torch.log(
                torch.as_tensor(float(sim_num * J), dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)
            )
            discrepancy = -discrepancy_vec.mean()
            
        elif dis_type in ("6","attribute"):
            attribute_o = gv.SCALED_ATTRIBUTE_MATRIX[ar_size, gv.OBSERVATION_MATRIX, :].sum(dim=0)   # (param_num)
            idx = res_chosen_matrix.T  

            idx_exp = idx.unsqueeze(-1).expand(-1, -1,gv.SCALED_ATTRIBUTE_MATRIX.size(-1))  
            attribute_s = torch.gather(gv.SCALED_ATTRIBUTE_MATRIX, 1, idx_exp).sum(dim=0)     # (sim_num, K)

            # Weight attribute differences using global_sd.
            if global_sd is None and sim_num>1:
                raise ValueError("Please provide the global covariance matrix of attribute-based statistics when using attribute-based discrepancy when sim_num >1.")
            elif global_sd is None and sim_num==1:
                warnings.warn("When sim_num=1, the variance of attribute-based statistics is zero. Hence, unweighted Euclidean distance is used for attribute-based discrepancy.",stacklevel = 2)
               
                discrepancy = torch.linalg.vector_norm(attribute_o - attribute_s[0, :])
            elif global_sd is not None:
                diff = (attribute_o - attribute_s.mean(dim=0)) # (sim_num, K)
                # discrepancy = torch.linalg.norm(diff@global_sd, ord=2, dim=1).log()
                discrepancy = torch.linalg.norm(diff@global_sd, ord=2).log()
                # discrepancy = (1/torch.linalg.norm(diff@global_sd, ord=2, dim=1).mean()-1)/(-1/2.0)  # Box-Cox transformation with lambda = -1/2
        elif dis_type in ["5+6","6+5"]:
            true_obs_utility, sim_obs_utility = _obs_sim_util(res_observed_matrix)          # (size,1), (size,sim_num)
            utility_diff = true_obs_utility - sim_obs_utility                               # (size, sim_num)
            discrepancy_vec = torch.logsumexp(utility_diff, dim=1) - torch.log(
                torch.as_tensor(float(sim_num * J), dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)
            )
            discrepancy1 = -discrepancy_vec.mean()
            
            attribute_o = gv.SCALED_ATTRIBUTE_MATRIX[ar_size, gv.OBSERVATION_MATRIX, :].sum(dim=0)   # (param_num)
            idx = res_chosen_matrix.T  

            idx_exp = idx.unsqueeze(-1).expand(-1, -1,gv.SCALED_ATTRIBUTE_MATRIX.size(-1))  
            attribute_s = torch.gather(gv.SCALED_ATTRIBUTE_MATRIX, 1, idx_exp).sum(dim=0)     # (sim_num, K)
            if global_sd is None and sim_num>1:
                raise ValueError("Please provide the global covariance matrix of attribute-based statistics when using attribute-based discrepancy when sim_num >1.")
            elif global_sd is None and sim_num==1:
                warnings.warn("When sim_num=1, the variance of attribute-based statistics is zero. Hence, unweighted Euclidean distance is used for attribute-based discrepancy.",stacklevel = 2)
               
                discrepancy2 = torch.linalg.vector_norm(attribute_o - attribute_s[0, :])
            elif global_sd is not None:
                diff = (attribute_o - attribute_s.mean(dim=0)) # (param_num,)
                discrepancy2 = torch.linalg.norm(diff@global_sd, ord=2).log()
                # discrepancy2 = (1/torch.linalg.norm(diff@global_sd, ord=2, dim=1).mean()-1)/(-1/2.0)  # Box-Cox transformation with lambda = -1/2
            if dis_type== "5+6":    
                discrepancy = torch.stack((discrepancy1,discrepancy2),dim = 0)    
            else:
                discrepancy = torch.stack((discrepancy2,discrepancy1),dim = 0)

        else:
            raise ValueError("please indicate the discrepancy type.")
        # the larger the better, to maximize the negative discrepancy
        return -discrepancy.to(dtype=gv.DATA_TYPE_torch)


# ============================================================================
# (3) TuRBO   -  trust-region state, its update rule, and candidate generation
# ============================================================================
@dataclass
class TurboState:
    """Trust-region bookkeeping for TuRBO (length grows on success, shrinks on failure)."""
    dim: int
    batch_size: int
    length: float = 0.5              # trust-region side length (relative)
    length_min: float = 0.5**7
    length_max: float =  1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 4 # 5~8    # can tune
    failure_tolerance: int = 0       # set in __post_init__
    global_best_value: torch.Tensor = torch.tensor([1.0]) * float('-inf')
    global_best_point: Optional[torch.Tensor] =None
    global_best_index: int = 0
    center_value: torch.Tensor = torch.tensor([1.0]) * float('-inf')
    x_current_center: Optional[torch.Tensor] =None
    restart_triggered: bool = False
    flag:int=0 
    # message:int=0
    def __post_init__(self):
        # same heuristic as TuRBO-1 paper/tutorial
        self.failure_tolerance = math.ceil(
            min(max(4.0 / float(self.batch_size), float(self.dim) / float(self.batch_size)),3)
        )
        # self.failure_tolerance = 2

def turbo_update_state(state: TurboState, Y_next: torch.Tensor, X_next:torch.Tensor) -> TurboState:
    """Update best points, counters and trust-region length in place; return state.

    X_next has shape (q, dim); Y_next contains q negative-discrepancy values.
    Flag 0 compares with the global best; other flags compare with the local center.
    """
    
    # treat improvement relative to current best
    if state.global_best_value == torch.tensor([1.0]) * float('-inf'):
        improved = True
    else:
        y_new_idx = Y_next.argmax()
        y_new_max = Y_next[y_new_idx]
        if state.flag==0:
            improved = (y_new_max > state.global_best_value + 1e-3 * abs(state.global_best_value)) 
        
        else:  # when state.flag==1
            if state.center_value == torch.tensor([1.0]) * float('-inf'):
                improved = True
            else:    
                # warnings.warn(f"current center value {state.center_value.item()}")
                improved = (y_new_max>state.center_value+1e-3 * abs(state.center_value))
            
            
     
    if improved:
        state.success_counter += 1
        state.failure_counter = 0
        
        if state.flag ==0: # explore around global best point
            state.global_best_point = X_next[y_new_idx].clone().detach()
            state.global_best_value =  y_new_max.clone().detach()
        else: # continue explore around proposed local point
            state.x_current_center = X_next[y_new_idx].clone().detach()
            state.center_value =y_new_max.clone().detach()
            state.flag = 1
            if state.center_value>state.global_best_value:
                state.global_best_point = state.x_current_center.clone().detach()
                state.global_best_value = state.center_value.clone().detach()
                
    else:
        state.failure_counter += 1
        state.success_counter = 0
        
        

             

    # expand / shrink TR
    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance and state.failure_tolerance > 0:
        state.length = max(state.length/2.0,state.length_min)
        state.failure_counter = 0



    if state.length <= state.length_min:
        state.restart_triggered = True
        state.flag = 0
        state.failure_counter = 0
        # state.x_current_center = state.global_best_point.clone().detach()
        # state.center_value=state.global_best_point.clone().detach()
    return state



def generate_batch(
    state: TurboState,
    model: SingleTaskGP,  # GP model
    X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
    Y: torch.Tensor,  # Function values
    batch_size: int,
    n_candidates: Optional[int] = None,  # Number of candidates for Thompson sampling
    num_restarts: int = 10,
    raw_samples: int = 512,
    generator1 = None,
    generator2 = None,
    iter:int=0,
    betai:int = 9, 
    dis_type: str = "6",
    sobol_seed_base: int = 0,
    thompson_seed_base: int = 0,
) -> torch.Tensor:
    """Return (candidates, lower_bound, upper_bound) in normalized coordinates.

    Use local Thompson sampling unless (iter + 1) is divisible by 20, then optimize
    global qUCB. Apply GP lengthscale weights when iter is divisible by 4.
    Returned bounds describe the local trust region, including on UCB calls.
    """
    flag = 1
    assert X.min() >= 0.0
    assert X.max() <= 1.0
    assert torch.all(torch.isfinite(Y))
    assert state.length>=state.length_min
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
        # 

    # Use lengthscale weights every fourth point count.
   
    dim = X.shape[-1]
    if iter%4!=0:
        weights = torch.ones(size = (1,dim),dtype=gv.DATA_TYPE_torch,device=gv.DEVICE)*1.0
    
    else:
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        weights = torch.clamp(weights,min= 0.25,max = 8.0)
    # Flags: 0 = global best, 1 = local center, 2 = propose a new center.
    alpha = torch.rand(1, dtype=gv.DATA_TYPE_torch, device=gv.DEVICE, generator=generator2)
    if alpha<0.02 and state.flag ==2:   # Start at the normalized midpoint.
        x_center =  torch.ones(size = (1,dim),dtype=gv.DATA_TYPE_torch,device=gv.DEVICE)*0.5
       
    elif alpha<0.05 and state.flag ==2:  # Randomize selected coordinates of the midpoint.
        x_center =  torch.ones(size = (1,dim),dtype=gv.DATA_TYPE_torch,device=gv.DEVICE)*0.5
        local_idx = torch.randint(low=0, high=2, size=(dim,), dtype=torch.bool,device=gv.DEVICE, generator=generator1) #,generator=generator
        local_sample_sobol = torch.rand(size=(dim,), dtype=gv.DATA_TYPE_torch,device=gv.DEVICE, generator=generator2) #,generator=generator
        x_center = torch.clamp(x_center*local_idx+local_sample_sobol*(~local_idx), min=0.0+1e-4, max=1.0-1e-4)
       
    elif alpha<0.35 and state.flag ==2: # Randomize selected coordinates of the global best.
        x_center = state.global_best_point.clone().detach()
        local_idx = torch.randint(low=0, high=2, size=(dim,), dtype=torch.bool,device=gv.DEVICE, generator=generator1) #,generator=generator
        local_sample_sobol = torch.rand(size=(dim,), dtype=gv.DATA_TYPE_torch,device=gv.DEVICE, generator=generator2) #,generator=generator
        x_center = torch.clamp(x_center*local_idx+local_sample_sobol*(~local_idx), min=0.0+1e-4, max=1.0-1e-4)
        
    elif  state.flag ==1:   #and state.message== 2
           # continue local variation
        x_center = state.x_current_center.clone().detach()
        
    else:
        # continue explore around best global point
        x_center = state.global_best_point.clone().detach()
    
    
    box_half_width = torch.clamp(weights*state.length/2.0,min = 0.5**7, max = 1.0)
    tr_lb = torch.clamp(x_center - box_half_width, min = 0.0, max = 1.0)
    tr_ub = torch.clamp(x_center + box_half_width, min = 0.0, max = 1.0)
    
    local_chance = 0
    if (iter+1)%20!= 0: #(iter+1)%100!=0 and # thompson sampling
        
        sobol = SobolEngine(dim, scramble=True, seed=int((sobol_seed_base + iter) % (2**31 - 1)))
        pert = sobol.draw(n_candidates).to(dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=gv.DATA_TYPE_torch, device=gv.DEVICE, generator=generator2) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=gv.DEVICE, generator=generator1)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        ts_seed = int((thompson_seed_base + iter) % (2**31 - 1))
        # replicability setting for "cuda" device
        device_str = str(gv.DEVICE)
        if torch.cuda.is_available() and "cuda" in device_str:
            if isinstance(gv.DEVICE, torch.device) and gv.DEVICE.index is not None:
                cuda_device = gv.DEVICE.index
            elif isinstance(gv.DEVICE, str) and ":" in gv.DEVICE:
                cuda_device = int(gv.DEVICE.split(":")[-1])
            else:
                cuda_device = 0
            cuda_devices = [cuda_device]
        else:
            cuda_devices = []


        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(ts_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(ts_seed)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)
        
    else:  # Global qUCB acquisition.
        sobol = SobolEngine(dim, scramble=True, seed=int((sobol_seed_base + iter) % (2**31 - 1)))
        pert = sobol.draw(n_candidates * batch_size).to(dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)
        pert = pert.view(n_candidates, batch_size, dim)
        
        alpha = torch.rand(1, dtype=gv.DATA_TYPE_torch, device=gv.DEVICE, generator=generator2) #prob for local search
        if alpha<local_chance:
            pert = tr_lb + (tr_ub - tr_lb) * pert   # (n_candidates, q, d)
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(
            (n_candidates, batch_size, dim),
            dtype=gv.DATA_TYPE_torch,
            device=gv.DEVICE,
            generator=generator2
        ) <= prob_perturb
        ind = torch.where(mask.sum(dim=2) == 0)  # tuple of (idx_n, idx_q)
        mask[ind[0], ind[1], torch.randint(
            low=0, high=dim, size=(len(ind[0]),),
            device=gv.DEVICE, generator=generator1
        )] = 1

        X_cand = x_center.expand(n_candidates, batch_size, dim).clone()
        X_cand[mask] = pert[mask]
        X_flat = X_cand.view(-1, dim)            # (n_candidates * batch_size, dim)

        with torch.no_grad():
            posterior = model.posterior(X_flat)
            raw_mean = posterior.mean                # (n_candidates * batch_size, 1) 

        raw_mean = raw_mean.view(-1)
        
        
        topk_idx = torch.topk(raw_mean, k=30*batch_size, largest=True, dim=0)[1]
        X_flat_top = X_flat[topk_idx].view(30, batch_size, dim)
    
        
    
        UCB = qUpperConfidenceBound(model = model, beta=betai)
        if alpha<local_chance:
            # local search
                X_next = optimize_acqf(
                    UCB,
                    bounds=torch.vstack([tr_lb, tr_ub]),
                    q=batch_size,
                    num_restarts=30,
                    # raw_samples=raw_samples,
                    batch_initial_conditions=X_flat_top,
                    options={"maxiter": 100}
                )[0]
        else:
            # global search
            X_next = optimize_acqf(
                UCB,
                bounds = torch.vstack([
                    torch.zeros(dim, device=gv.DEVICE),
                    torch.ones(dim, device=gv.DEVICE)]),
                q=batch_size,
                num_restarts=30,
                # raw_samples=raw_samples,
                batch_initial_conditions=X_flat_top,
                options={"maxiter": 100}
            )[0]

    # message, 
    return X_next,  tr_lb, tr_ub


# ============================================================================
# (4) BOLFI_ESTIMATOR
# ----------------------------------------------------------------------------
# End-to-end estimator.  Typical use (see run_bolfi_canonical_v2.train_gp):
#   est = BOLFI_Estimator(dim, simulator_fun, sim_num, dis_type, file_route)
#   est.initialize_gp(...)          # Latin-hypercube/prior samples and targets
#   est.train_bolfi(...)            # TuRBO BO loop over the discrepancy surface
#   est.est_sigma2(); est.est_tol() # discrepancy noise + tolerance for the likelihood
#   est.result_output()            # pickle the trained state
# then est.calculate_likelihood(gp, beta) is the approximate log-likelihood for MCMC.
# ============================================================================
class BOLFI_Estimator:


    def __init__(self, dim: int, simulator_fun: Callable, sim_num: int, dis_type: str,  file_route:str,  epsilon:float = 1e-8, max_iters:int= 3000,   *args, **kwargs):
        """Store estimator settings and initialize RNGs from gv.MASTER_SEED.

        See compute_discrepancy for simulator outputs and discrepancy types.
        Training arrays, bounds and GP are initialized later. max_iters limits
        the training-point counter, including initial samples.
        """
        
        #  attributes_matrix: np.ndarray, scaled_attributes_matrix: Optional[np.ndarray], Obs_response: np.ndarray, 
        
        self.dim = dim
        self.bounds: Optional[npt.NDArray] =  None
    

        self.simulator_fun = simulator_fun
        
        self.sim_num = sim_num
        self.dis_type = dis_type
        self.epsilon = epsilon
        self.global_sd = None
        self.train_X_int = None
        self.train_Y_int =  None
        self.max_iters = max_iters
        self.file_route = file_route
        self.train_X = None
        self.train_Y = None
        self.train_X_neat = self.train_X_int
        self.train_Y_neat = self.train_Y_int
        self.train_X_normed = None
        self.max_cholesky_size = float("inf")
        self.betai_function =None
        self.init_num = None 
        self.tol_ucb = 0.05
        self.tol_gap = 0.05
        self.tol_step = 0.01
        self.tol_sigma = 0.05 
        self.raw_samples = int(1024)
        self.num_restarts = int(24)        
        self.training_results = None
        self.cov = None
        self.tol = None
        self.likelihood = None
        self.gp_training_time = None
        self.batch_size = 5
        self.J = gv.J
        self.size = gv.size
        self.ATTRIBUTE_MATRIX  =gv.ATTRIBUTE_MATRIX   
        self.SCALED_ATTRIBUTE_MATRIX =  gv.SCALED_ATTRIBUTE_MATRIX
        self.OBSERVATION_MATRIX  =  gv.OBSERVATION_MATRIX 
        self.normalize =   None 
        self.system = "windows"
         # random seed configuration
        self.MASTER_SEED = gv.MASTER_SEED
        self.CHILD_SEED= torch.randint(low=0, 
                                       high=2**31, 
                                       size=(3+self.batch_size,), 
                                       dtype=torch.int64, 
                                       generator= torch.Generator(device=gv.DEVICE).manual_seed(self.MASTER_SEED))
      
        self.discrepancy_rng =  [torch.Generator(gv.DEVICE).manual_seed(int(self.CHILD_SEED[m])) for m in range(self.batch_size)]  # First batch_size seeds serve discrepancies; the last three serve initialization.
    
        
        # torch.Generator(gv.DEVICE).manual_seed(int(self.CHILD_SEED[0]))
        self.init_train_X_rng = torch.Generator(gv.DEVICE).manual_seed(int(self.CHILD_SEED[self.batch_size]))
        self.X_init_local_1 = torch.Generator(gv.DEVICE).manual_seed(int(self.CHILD_SEED[self.batch_size+1]))
        self.X_init_local_2 =torch.Generator(gv.DEVICE).manual_seed(int(self.CHILD_SEED[self.batch_size+2]))
        
       
        # if self.bounds is None:
        #     raise  ValueError("Please offer finite bounds for estimable parameters. For unbounded prior, please temporarily use [-6*sd,6*sd] to be its bounds and adjust if the optimal points are around boundaries.")







    # ==================================================================
    # discrepancy-sensitivity diagnostics (optional; not on the main path)
    # ==================================================================



# Plot the discrepancy over parameter space     
    @torch.no_grad()
    def discrepancy_FixAllButOne(self,tr_theta, idx, param_index, shift_thetas):
        """Set one coordinate of tr_theta in place and evaluate its negative discrepancy."""
        tr_theta[param_index] = shift_thetas[idx]
        
        res = compute_discrepancy(beta =tr_theta, simulator_fun = self.simulator_fun, sim_num = self.sim_num,
                                  dis_type = self.dis_type, global_sd = self.global_sd, rgn_seed = gv.MASTER_SEED, sequential=idx)
        return  res
    @torch.no_grad()
    def plot_discrepancy_sensitivity(self,file_route:Optional[str] =None, bounds:Optional[torch.tensor]=None, split_num:int=30, beta_name:Optional[list]=None,
                                     beta_name_plt:Optional[str]=None, reference_beta:Optional[torch.tensor]=None,*args, **kwargs):
        """Save negative-discrepancy curves while varying one parameter at a time.

        Use split_num points per bound interval; fix other coordinates at reference_beta
        (default gv.REAL_BETA_U). Bounds and output prefix default to estimator settings.
        Names default to beta0, beta1, ...; beta_name_plt overrides axis labels.
        """
        if (file_route is None) and (self.file_route is not None):  
            file_route = self.file_route
        elif (self.file_route is None) and (file_route is not None):
            self.file_route = file_route
        elif (self.file_route is None) and (file_route is None):
            raise warnings('No available file_route for this class.') # type: ignore
        
        if (bounds is None) and (self.bounds is not None):
            bounds = self.bounds
        elif self.bounds is None:
            raise warnings('Both bounds of the class attribute and this function arguments have not been defined.') # type: ignore
        
        if beta_name is None:
            beta_name = [ 'beta'+str(item)  for item in range(self.dim)]
        if beta_name_plt is None:
            beta_name_plt = beta_name # type: ignore
        
        if (reference_beta is None) and (gv.REAL_BETA_U is not None):
            reference_beta = gv.REAL_BETA_U
        elif (reference_beta is None) and (gv.REAL_BETA_U is None):
            raise warnings('Both gv.REAL_BETA_U and the reference_beta of this function arguments have not been defined.') # type: ignore
       
        lows_u_prior = bounds[0,:]
        highs_u_prior =bounds[1,:]
        print("Bounds",bounds)
          
        
        for param_index in range(len(beta_name)):
            shift_thetas =torch.linspace(lows_u_prior[param_index],highs_u_prior[param_index],steps=split_num, dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)             # optional
            
            shift_marlls =Parallel(n_jobs = -1,backend="threading")(delayed(self.discrepancy_FixAllButOne)(tr_theta = reference_beta.clone(),idx = idx,param_index = param_index,
                                                                                                           shift_thetas = shift_thetas) for idx in range(split_num))

            # Parallel returns a Python list; stack into a tensor so the two-stage branch
            # can slice columns (shift_marlls[:,0] / [:,1]). Move everything to CPU for matplotlib.
            shift_marlls = torch.stack([torch.as_tensor(m, dtype=gv.DATA_TYPE_torch, device=gv.DEVICE).reshape(-1)
                                        for m in shift_marlls]).detach().cpu()
            if shift_marlls.size(1) == 1:
                shift_marlls = shift_marlls.squeeze(1)
            shift_thetas = shift_thetas.detach().cpu()

            if self.dis_type not in ["5+6","6+5"]:
                plt.figure()
                plt.plot(shift_thetas,shift_marlls, color="black")
                plt.axvline(reference_beta[param_index],linestyle="--",c='blue',label='reference point')
                plt.legend()
                plt.legend(fontsize=13,loc='upper right')
                plt.grid()
                plt.tick_params(axis='both', labelsize=15)
                plt.xlabel(beta_name_plt[param_index],fontsize=20)
                plt.ylabel("negative discrepancy",fontsize=20)
                
            else:
                fig, ax1 = plt.subplots()
                ax1.plot(shift_thetas,shift_marlls[:,0], color="C0",label = "1st stage discrepancy")
                ax1.set_ylabel('1st Stage Dis', color="C0")
                ax1.axvline(reference_beta[param_index],linestyle="--",c='black',label='reference point')
                ax1.tick_params(axis='y', labelcolor='C0')
                ax2 = ax1.twinx()

                ax2.plot(shift_thetas,shift_marlls[:,1], color="C1",label = "2nd stage discrepancy")
                ax2.set_ylabel('2nd Stage Dis', color="C1")
                ax2.tick_params(axis='y', labelcolor='C1')
                fig.tight_layout()

            plt.savefig(fname = f'{file_route}MLBA_discrepancy_{beta_name[param_index]}_{self.dis_type}({self.sim_num}).pdf',dpi = 1080,bbox_inches='tight') 
            plt.show()
            
        return  print("Please check in the file folder.")





###########################################################
################## Gaussian Process Model #################
###########################################################
      
# generate initial training set for Gaussian Process model
    @torch.no_grad()
    def initialize_gp(self, prior_type:str,prior_param:dict,bounds:Optional[torch.Tensor] = None, init_num: int = 0, proportion_of_sobol:torch.Tensor = torch.tensor(0.7, device=gv.DEVICE,dtype=gv.DATA_TYPE_torch),*args, **kwargs):
        """Create initial samples and store negative discrepancies; return None.

        init_num=0 selects 5 * dim samples. Despite its name, proportion_of_sobol sets
        the Latin-hypercube fraction; remaining samples come from the prior.
        Normal priors use loc/scale and default bounds loc +/- 3 * scale; uniform
        priors use low/high. Explicit bounds override defaults. Store train_X_int
        as (init_num, dim) and train_Y_int as (init_num, 1 or 2).
        """
    
        if init_num == 0:
            self.init_num = 5*self.dim # initial samples for bolfi training,suggested would be 2-5 times n_dim
            warnings.warn(f"the dimension of initial training data isn't determined, hence choose {self.init_num} automatically.",stacklevel = 2)
        else:
            self.init_num = init_num
        
        if bounds is None:
            if prior_type=="normal":
                self.prior_type = prior_type
                self.prior_param = prior_param
                self.bounds =torch.stack((prior_param['loc']-prior_param['scale']*3.0,prior_param['loc']+prior_param['scale']*3.0)).to(dtype=gv.DATA_TYPE_torch)
                warnings.warn(f"The parameter space of the bounds for parameter space doesn't given, hence we automatically use [mean-6*sd,mean+6*sd] of the priors to be its bounds. Please manually adjust if the optimal points are around boundaries.",stacklevel = 2)
            elif prior_type=="uniform":
                self.prior_type = prior_type
                self.prior_param = prior_param
                self.bounds = torch.stack((prior_param["low"],prior_param["high"])).to(dtype=gv.DATA_TYPE_torch)
                warnings.warn(f"The parameter space of the bounds for parameter space doesn't given, hence we automatically use [low,high] of the priors to be its bounds.",stacklevel = 2)
            else:
                raise  ValueError("the `prior_type` input doesn't belong to either `normal` or `uniform`. Please generate the initial value of parameters and associated discrepancy value manually.")
        
        else:
            self.bounds = torch.tensor(bounds,dtype= gv.DATA_TYPE_torch) 
        self.bounds_init = self.bounds
        m = torch.round(self.init_num *proportion_of_sobol).int().item()  # number of initial points from spacing strategy
        
        # initial value from prior
        if prior_type=="normal":
           
            X_prior = torch.normal(mean=prior_param["loc"].expand(self.init_num-m, -1), std=prior_param["scale"].expand(self.init_num-m, -1),generator=self.init_train_X_rng)
        elif prior_type=="uniform":
            X_prior = torch.empty((self.init_num-m, self.dim), dtype=gv.DATA_TYPE_torch,device=gv.DEVICE).uniform_(low=prior_param["low"], high=prior_param["high"],generator=self.init_train_X_rng)
        else:
            raise  ValueError("the `prior_type` input doesn't belong to either `normal` or `uniform`. Please generate the initial value of parameters and associated discrepancy value manually.")
        # initial value from spacing strategy
        
        # X_space = np.array(draw_sobol_samples(bounds=bounds, n=m, q=1).squeeze(1),dtype=gv.DATA_TYPE_np)
        
        # Latin hypercube: jitter within strata and permute each column.
        u_center = (torch.arange(m, device=gv.DEVICE, dtype=gv.DATA_TYPE_torch) +  torch.rand(m, device=gv.DEVICE, dtype=gv.DATA_TYPE_torch, generator=self.init_train_X_rng)) / m  # (m,)
        X_unit = torch.empty((m, self.dim), device=gv.DEVICE, dtype=gv.DATA_TYPE_torch)
        for j in range(self.dim):
            perm = torch.randperm(m,  device=gv.DEVICE, generator=self.init_train_X_rng)
            X_unit[:, j] = u_center[perm]
        # scale to bounds
        X_space = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X_unit  # (m, d)

        self.train_X_int = torch.vstack((X_space, X_prior))

        # calculate discrepancy for train_X_int
        if self.dis_type in ["5+6","6+5"]:
            self.train_Y_int = torch.zeros((self.init_num,2),dtype=gv.DATA_TYPE_torch) 
        else:
            self.train_Y_int = torch.zeros((self.init_num,1),dtype=gv.DATA_TYPE_torch) 
        
        for row in range(self.init_num):
            self.train_Y_int[row,:] = compute_discrepancy(beta =self.train_X_int[row].view(-1), simulator_fun = self.simulator_fun, sim_num = self.sim_num,
                                  dis_type = self.dis_type, global_sd = self.global_sd, rgn_seed = gv.MASTER_SEED*2, sequential=row)
            
       
        return print("The initial value for Gaussian Process training has been done successfully. Check the initial value by attribute train_X_int and its discrepancy value by train_Y_int.")

# Fit a Gaussian Process Model      
    def gp_fit(self, train_X: Optional[torch.Tensor] = None,
               train_Y: Optional[torch.Tensor] = None, stage: int = 0):
        """Fit and store a Matern-2.5 ARD GP with standardized negative discrepancies.

        Supply both normalized train_X and train_Y, or use stored arrays.
        stage selects column 0 or 1. Reseed before fitting; return None.
        """
        set_seed(self.MASTER_SEED)     # narrow (not eliminate) fit_gpytorch_mll drift
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-2))
        covar_module = ScaleKernel(  # Constrain ARD lengthscales below.
                MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 3.0))
            )
        if (train_X is None) and (train_Y is None):
            # for unnormed X
            if stage==0: # first stage fitness
                self.gp = SingleTaskGP(self.train_X_normed, 
                                        self.train_Y_neat[:,0].view(-1,1),
                                    covar_module=covar_module, likelihood=likelihood,
                                    outcome_transform=Standardize(1)
                                    )
            else: #second stage fitness
                self.gp = SingleTaskGP(self.train_X_normed, 
                                        self.train_Y_neat[:,1].view(-1,1),
                                    covar_module=covar_module, likelihood=likelihood,
                                    outcome_transform=Standardize(1)
                                    )
        else:
            if stage==0:
                self.gp = SingleTaskGP(train_X, train_Y[:,0].view(-1,1),
                                    covar_module=covar_module, likelihood=likelihood,
                                        # input_transform=Normalize(self.dim, bounds=self.bounds),   # handles the unit-cube mapping internally
                                        outcome_transform=Standardize(1))
            else:
                self.gp = SingleTaskGP(train_X, train_Y[:,1].view(-1,1),
                                    covar_module=covar_module, likelihood=likelihood,
                                        # input_transform=Normalize(self.dim, bounds=self.bounds),   # handles the unit-cube mapping internally
                                        outcome_transform=Standardize(1))
        #
        with gpytorch.settings.cholesky_jitter(1e-7), gpytorch.settings.cholesky_max_tries(8):
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_mll(mll) 






    def average_duplicates(self,decimals:int=3):

        """Average targets at rounded input rows in first-occurrence order; store neat arrays."""
        X_rounded = self.train_X.round(decimals=decimals)
        unique_vals, inverse = torch.unique(X_rounded, dim=0, return_inverse=True)
        inverse = inverse.to(device=gv.DEVICE)
        n_unique = unique_vals.size(0)

        # Restore first-occurrence order after grouping rounded inputs.
        rev = torch.arange(inverse.size(0) - 1, -1, -1, device=gv.DEVICE)
        first_pos = torch.empty(n_unique, dtype=torch.long, device=gv.DEVICE)
        first_pos[inverse[rev]] = rev                       # last write wins -> smallest index
        order = torch.argsort(first_pos)                    # unique-row idx -> chronological slot
        new_slot = torch.empty_like(order)
        new_slot[order] = torch.arange(n_unique, device=gv.DEVICE)
        inverse = new_slot[inverse]

        self.train_X_neat = unique_vals[order].to(dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)

        width = 2 if self.dis_type in ["5+6","6+5"] else 1
        self.train_Y_neat = torch.zeros((n_unique, width), dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)
        counts = torch.zeros((n_unique, width), dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)

        for i in range(len(inverse)):
            self.train_Y_neat[inverse[i],:] += self.train_Y[i,:]
            counts[inverse[i],:] += 1
        self.train_Y_neat /= counts

    # ==================================================================
    # BO training loop
    # ==================================================================
    def train_bolfi(self,train_X:Optional[torch.Tensor]=None, train_Y:Optional[torch.Tensor]=None, global_percentage:float=0.9,
                    convergence_check_length: int=10,convergence_type: str = "auto", batch_size:int = 5, 
                    max_iters: Optional[int]= None, betai_function: Optional[Callable] = None, 
                    tol_gap: float = 0.05, tol_ucb: float = 0.05,tol_step: float = 0.05, tol_sigma: float = 0.05, file_route: Optional[str]=None,callback_interval: int=50,*args, **kwargs): 
        """Train and store the GP through batched TuRBO acquisition; return None.

        Use train_X_int/train_Y_int even when train_X/train_Y are supplied. Targets
        have one column, or two for 5+6/6+5. The counter includes initial points
        and advances by batch_size; max_iters is a point budget, not a loop count.
        Auto mode stops after repeated stalls; fixed mode continues to the budget.
        For 5+6, recalibrate attribute weights when switching stages.

        global_percentage and convergence_check_length are unused. The tol_* values
        are stored but do not control stopping. betai_function supplies the UCB weight;
        callback_interval controls logging by point count.
        """
        # sanity checks and update self attributes
        # if gv.DATA_TYPE_torch==torch.float32:
        #     warnings.filterwarnings("ignore")
        torch.set_default_dtype(gv.DATA_TYPE_torch) #set tensor float precision
        
        start_time_inside = time.time() # initialize the start time for training BOLFI, used for callback report
        # ###############################################
        if (train_X is None) and (train_Y is None):
            self.train_X = self.train_X_int.detach().to(device = gv.DEVICE)
            self.train_Y = self.train_Y_int.detach().to(device = gv.DEVICE)
            
        else:
            self.train_X = train_X.detach().to(device = gv.DEVICE)
            self.train_Y = train_Y.detach().to(device = gv.DEVICE)
        
            
        assert torch.isfinite(self.train_X).all(), "NaN/Inf in train_X"
        assert torch.isfinite(self.train_Y).all(), "NaN/Inf in train_Y"
        
        self.normalize =  Normalize(d=self.dim, bounds=self.bounds)
        
        
        self.train_X_normed = self.normalize(X = self.train_X_int)
        
        

        self.train_X = self.train_X_int
        self.train_Y = self.train_Y_int
        
        self.train_X_neat = self.train_X
        self.train_Y_neat = self.train_Y

        if max_iters is not None:
            self.max_iters = max_iters
        if betai_function is not None:
            self.betai_function = betai_function
        if file_route is not None:
            self.file_route = file_route
        if batch_size is not None:
            self.batch_size = batch_size
        self.tol_gap = tol_gap
        self.tol_ucb = tol_ucb
        self.tol_step= tol_step
        self.tol_sigma = tol_sigma
       
       
        
        n_samples = self.train_X.size(0)
        i = n_samples
        box_width = (self.bounds[1] - self.bounds[0]) # for unnormed X
        stage = 0  # the stage of the discrepancy used for GP fitness

        # TuRBO state initialization
        state = TurboState(self.dim, batch_size=batch_size)  #length_min=2*self.tol_step/min(box_width).item()
        state.flag = 0
        N_CANDIDATES = min(5000, max(2000, 200 * self.dim)) #
        last_best = None
        count_best = 0
        count_stop = 0
        
        self.average_duplicates()
        self.train_X_normed = self.normalize(X = self.train_X_neat)
        state.global_best_index = int(self.train_Y_neat[:,stage].argmax()) # Current stage.
        state.global_best_point = self.train_X_normed[state.global_best_index].clone().detach()  
        state.global_best_value = self.train_Y_neat[state.global_best_index,stage].clone().detach()
        self.time_storage = []
        self.time_log = []
        self.iter_min = None
        self.iter_early_stop = None
        while(stage<=1 and i<=self.max_iters):
            # let the SA simulator pick a per-stage mh_steps, and keep the first
            # `cold_bo_iters` BO iterations on the init rule (reseed, no carry)
            _SA.stage = stage
            _SA.warm = (_SA._bo_iter >= _SA.cold_bo_iters)
            _SA._bo_iter += 1

            # Fit a GP model
            self.gp_fit(stage = stage)
            
          
            
            # if  ((count_stop>( self.dim//10-1) and stage==0) or ((count_stop>(self.dim//10) or (i>=self.max_iters)) and stage==1)) and self.dis_type in ["5+6","6+5"]:
            if  ((count_stop>1) or ((i>=self.max_iters) and stage==1)) and self.dis_type in ["5+6","6+5"]:
                    
                
                if not (convergence_type=="fixed" and stage==1 and i<self.max_iters):
                    state.global_best_index = int(self.train_Y_neat[:,stage].argmax()) # Current stage.
                    beta_prior = self.train_X_neat[state.global_best_index].clone().detach() 
                    print(f"[EARLY-STOP @ iter {i}] Best value for stage {stage+1}:")
                    print(f"| [Predicted] {self.gp.posterior(state.global_best_point.view(1,-1)).mean}")
                    print("Best acquired point:{}\t".format(beta_prior.tolist()))
                
                if self.dis_type=="5+6" and stage==0:
                    self.iter_min = i # Point count at the stage switch.
                    pilot_num = 30
                    # attribute_s = torch.zeros((pilot_num,self.dim),dtype=gv.DATA_TYPE_torch,device = gv.DEVICE)# (sim_num,param_num)
                
                    # for sim_i in range(pilot_num):
                    res_chosen_matrix, _ = simulate(estimable_parameters_vector = beta_prior,
                                                                            attributes_matrix=gv.ATTRIBUTE_MATRIX,
                                                                            noise_type="gumbel",
                                                                            num_sim = pilot_num ,#pilot_num,  #1,
                                                                            device=gv.DEVICE,
                                                                            rgn_seed=self.MASTER_SEED)
                    ar_size = torch.arange(self.size, device=gv.DEVICE, dtype=torch.long)
                    offset = ar_size * self.J                     # (size,)
                    flat = (res_chosen_matrix + offset.unsqueeze(0)).reshape(-1)    # (sim_num*size,)
                    counts = torch.bincount(flat, minlength=self.size * self.J)  # (size*J,)
                    choice_freq = counts.reshape(self.size, self.J)/ self.sim_num    # (size, J) float
                    
                    idx = res_chosen_matrix.T  # (N, pilot_num)
                    
                    # gather along J dimension -> need index shape (N, pilot_num, D)
                    idx_exp = idx.unsqueeze(-1).expand(self.size, pilot_num, self.dim)                 # (N, SIM, D)
                    X_rep =gv.SCALED_ATTRIBUTE_MATRIX.unsqueeze(1).expand(self.size, pilot_num, self.J, self.dim)                  # (N, SIM, J, D)
                    x_chosen = torch.gather(X_rep, 2, idx_exp.unsqueeze(2)).squeeze(2) # (N, S, D)
        
                    mu_hat = x_chosen.mean(dim=1,keepdim=True)                 # (N,1,dim)
                    g_i = gv.SCALED_ATTRIBUTE_MATRIX - mu_hat       # (N,J,dim)

                    # # Omega = Var(g_i) = E[g_i g_i^T] ≈ (1/N) sum_i (1/(S-1)) sum_s g_i g_i^T
                    Omega_hat = torch.einsum('ij,ijk,ijl->kl', choice_freq, g_i, g_i)
                    # Omega_hat = torch.cov(x_chosen.sum(dim=0).T, correction=1)
                    # # Omega_hat = Omega_hat 
                    W_hat = torch.linalg.inv(Omega_hat+ self.epsilon * torch.eye(self.dim, device=gv.DEVICE, dtype=gv.DATA_TYPE_torch))
                    self.global_sd = torch.linalg.cholesky(W_hat).to(device = gv.DEVICE,dtype = gv.DATA_TYPE_torch)
                   
                    self.dis_type="6" # switch to attribute-based discrepancy for second stage (temporary)
                   
                    vals_Y = self.train_Y_neat[:,1]
                    for idx in range(self.train_X_neat.shape[0]):
                        self.train_Y_neat[idx,1] = compute_discrepancy(beta = self.train_X_neat[idx].flatten(), simulator_fun = simulate, sim_num = self.sim_num, 
                                                                       dis_type = self.dis_type, global_sd = self.global_sd, rgn_seed = gv.MASTER_SEED*5, sequential=idx)
                    
                    self.dis_type="5+6" # switch back
                
                stage +=1
                
                # Reset the best point and stall counters for the next stage.
                if stage==1:
                    state.global_best_index = int(self.train_Y_neat[:,stage].argmax()) # Current stage.
                    state.global_best_point = self.train_X_normed[state.global_best_index].clone().detach()  
                    state.global_best_value = self.train_Y_neat[state.global_best_index,stage].clone().detach()
                    last_best =None
                    count_best = 0
                    count_stop = 0

                elif stage==2 and self.iter_early_stop is None: # Record the final-stage stopping count.
                    self.iter_early_stop = i
                    if convergence_type=="auto":
                        break
                    else:
                        stage = 1 # if convergence_type = "fixed"
                elif stage==2 and self.iter_early_stop is not None and convergence_type=="fixed":
                    if i<self.max_iters:
                        stage = 1
                    else:
                        break   
                else:
                    break
                
            elif   (count_stop> (self.dim//10) or i>=self.max_iters) and (self.dis_type not in ["5+6","6+5"]):

                if self.iter_early_stop is None:
                    self.iter_early_stop = i        # record where it would have stopped

                if convergence_type == "fixed" and i < self.max_iters:
                    count_stop = 0                  # keep training to max_iters
                else:
                    print(f"[EARLY-STOP @ iter {i}] Best value for stage {stage+1}:")
                    print(f"| [Predicted] {self.gp.posterior(state.global_best_point.view(1,-1)).mean}")
                    print("Best acquired point:{}\t".format(self.normalize.untransform(state.global_best_point).tolist()))
                    break
                
                # Create a batch  message,
            candidate, lb,ub = generate_batch(state=state,model=self.gp,X=self.train_X_normed, Y= self.train_Y_neat[:,stage], # Y_turbo
                                                                                    batch_size=self.batch_size,n_candidates=N_CANDIDATES,num_restarts=self.num_restarts,raw_samples=self.raw_samples,
                                                                                    betai = self.betai_function(iter = i, param_dim = self.dim),
                                                                                    generator1 = self.X_init_local_1,generator2 = self.X_init_local_2,iter = i,
                                                                                    dis_type=self.dis_type,
                                                                                    sobol_seed_base=self.MASTER_SEED,
                                                                                    thompson_seed_base=self.MASTER_SEED)
        
            
            candidate_unnorm = self.normalize.untransform(X = candidate)
            
            
            if self.system == "windows":
                vals_Y = Parallel(n_jobs = -3,backend="threading")(delayed(compute_discrepancy)(beta = candidate_unnorm[batch_idx].flatten(), sim_num = self.sim_num, dis_type = self.dis_type, global_sd = self.global_sd, rgn_seed = gv.MASTER_SEED*6, sequential=batch_idx) for batch_idx in range(self.batch_size))
            
            else: #os or linux
                vals_Y = Parallel(n_jobs = -3,backend="loky")(delayed(compute_discrepancy)(beta = candidate_unnorm[batch_idx].detach().cpu().numpy().flatten(),
                                                                                           sim_num = self.sim_num, dis_type = self.dis_type, global_sd = self.global_sd.detach().cpu(), rgn_seed = gv.MASTER_SEED*6, sequential=batch_idx) for batch_idx in range(self.batch_size))
            
            candidate_Y =  torch.stack(vals_Y)
            if candidate_Y.ndim == 1:          # single-stage dis_type: (batch,) -> (batch,1)
                candidate_Y = candidate_Y.unsqueeze(-1)

            self.train_X = torch.cat((self.train_X, candidate_unnorm)).to(gv.DATA_TYPE_torch)
            self.train_Y = torch.cat((self.train_Y, candidate_Y)).to(gv.DATA_TYPE_torch)
            
            self.average_duplicates()
            self.train_X_normed = self.normalize(X = self.train_X_neat)
             
            # >>> TuRBO: update TR state with new batch outcome ---------------
            # self.computate_discrepancy_normalized(beta = x_current_state)
            state = turbo_update_state(state = state, Y_next = candidate_Y[:,stage] ,X_next = candidate) 
            
            if state.restart_triggered:
                if last_best  is None:
                    count_best = 0  # counter for the restart at the center
                    count_stop = 0  # counter for the restart given the same global best
                    last_best =state.global_best_value.clone().detach()
                elif last_best==state.global_best_value:
                    count_best +=1
                else:
                    count_best = 0
                    count_stop = 0 
                    last_best =state.global_best_value.clone().detach()
                    state.flag = 0
                    state.x_current_center = state.global_best_point.clone().detach()
                    state.center_value = state.global_best_value.clone().detach()
                # simple "global restart": reset length & best_value
                # print(f"[TuRBO] Trust region collapsed at iter {i}, restarting TR.")
                state.length = state.length_max
                
                state.restart_triggered = False
                
            if count_best >1:
                state.flag = 2
                count_best  = 0
                count_stop +=1

                # print("the state information is",state)
            if i%callback_interval in range(self.batch_size+1):
                
                print(f"Iteration:{i} Stage{stage+1},  Best value: {state.global_best_value.item():.2e}, TR length: {state.length:.2e}, count stop: {count_stop}")
                print("best point ever:",self.normalize.untransform(state.global_best_point).view(-1))
                if state.flag==1 and state.center_value is not None:
                    print(f"Iteration:{i},  Best center value: {state.center_value.item():.2e}")
                    print("current center point:", self.normalize.untransform(state.x_current_center))
                
                # print("Trust Region width:",(ub-lb)*box_width)
            
                print("New parameter point:{}\t".format(self.train_X[-1].tolist()))
                print(f'neg-discrepancy:[Real]{self.train_Y[-1,stage].item():.6g}| [Predicted] {self.gp.posterior(candidate).mean[-1].item():.6g}')
            
            self.time_log.append(time.time()-start_time_inside)    
            start_time_inside =  time.time()    
            
            # save mid-point results in case the system crashes;
            # if i % 500 in range(self.batch_size):
            #     time_start = time.time()
            #     df_X = pd.DataFrame(self.train_X_neat.clone().detach().cpu().numpy(),dtype=gv.DATA_TYPE_np)
            #     filename_X = f'{self.file_route}bolfi_{i}iter_train_X({self.dis_type}).csv'
            #     df_X.to_csv(filename_X,header=False,index=False)
            #     df_X = None
            #     df_Y = pd.DataFrame(self.train_Y_neat.clone().detach().cpu().numpy(),dtype=gv.DATA_TYPE_np)
            #     filename_Y = f'{self.file_route}bolfi_{i}iter_train_Y({self.dis_type}).csv'
            #     df_Y.to_csv(filename_Y,header=False,index=False)
            #     df_Y = None
            #     self.time_storage.append(time.time()-time_start)
            
            gc.collect()
            torch.cuda.empty_cache()
            i+= candidate_Y.size(0)

        if self.dis_type in ["5+6","6+5"]:
            self.gp_fit(stage=1)
        else:
            self.gp_fit(stage=0)
        print("The BO process has been finished.:)")
        

# function to generate bolfi training results and plots
    def plot_training_trajectory(self,gp=None, file_route:Optional[str] =None, train_X:Optional[torch.Tensor]=None,train_Y:Optional[torch.Tensor]=None,*args, **kwargs):
        """Save observed and GP-predicted negative discrepancies to self.file_route.

        Defaults to stored normalized inputs and final-stage targets, recording
        [index, observed, predicted, log-density score] in self.training_results.
        With explicit inputs, targets still come from self.train_Y; train_Y and
        file_route do not override stored values. Return None.
        """
        training_results = np.zeros((self.train_Y_neat.size(0), 4))
        if gp is None:
            gp = self.gp
        if (train_X is None) and (train_Y is None): 
            train_X_plt = self.train_X_normed
            if self.dis_type in ["5+6","6+5"]:
                train_Y_plt = self.train_Y_neat[:,1].view(-1,1)
            else:
                train_Y_plt = self.train_Y_neat
            
            # train_X_plt = self.train_X
            # train_Y_plt = self.train_Y
            
            for i in range(train_Y_plt.size(0)):
                # real_obj = self.train_Y_neat[i,:]
                real_obj = train_Y_plt[i,:]
                pred_obj = gp.posterior(train_X_plt[i,:].view(1,-1)).mean
                pred_var = gp.posterior(train_X_plt[i,:].view(1,-1)).variance

                botorchlikelihood = torch.distributions.Normal(loc=pred_obj[0][0], scale=pred_var[0][0]).log_prob(real_obj[0])
                training_results[i,:] = [i,real_obj.item(),pred_obj[0][0].item(),botorchlikelihood.item()]
                # df = pd.DataFrame(training_results,columns=["iter","real_discrepancy","predicted_discrepancy","log_likelihood"])
            self.training_results =  training_results   
            
        else:
            train_X_plt = train_X
            if self.dis_type in ["5+6","6+5"]:
                train_Y_plt = self.train_Y[:,1].view(-1,1)
            else:
                train_Y_plt = self.train_Y
            for i in range(train_Y_plt.size(0)):
                real_obj = train_Y_plt[i,:]
                pred_obj = gp.posterior(train_X_plt[i,:].view(1,-1)).mean
                pred_var = gp.posterior(train_X_plt[i,:].view(1,-1)).variance
                botorchlikelihood = torch.distributions.Normal(loc=pred_obj[0][0], scale=pred_var[0][0]).log_prob(real_obj[0])
                training_results[i,:] = [i,real_obj.item(),pred_obj[0][0].item(),botorchlikelihood.item()]
        plt.figure()  
        fig, axs = plt.subplots(1, 1, figsize=(9,5))
        if (self.init_num is not None) and (self.init_num!=0):
            axs.plot(training_results[self.init_num:,0]-self.init_num, training_results[self.init_num:,1], color='red', label='real')
            axs.plot(training_results[self.init_num:,0]-self.init_num, training_results[self.init_num:,2],'green',label='pred',)
        else:
            axs.plot(training_results[:,0], training_results[:,1], color='red', label='real')
            axs.plot(training_results[:,0], training_results[:,2],'green',label='pred',)
  
        axs.legend()
        axs.grid()
        axs.tick_params(axis='both')
        axs.set_xlabel(r'$Iter$')
        axs.set_ylabel(r'Discrepancy value',rotation='vertical',labelpad=30)
        plt.savefig(fname = f'{self.file_route}GP_fitness_trajectory_{self.dis_type}({self.sim_num}).pdf',dpi = 1080,bbox_inches='tight') 
        fig.show()
        # return fig.show()
        return print("GP fitness trajectory has been generated.")


# a function to save training data of BOLFI of different period
    def save_train_data(self,train_X:Optional[torch.Tensor]=None, train_Y:Optional[torch.Tensor]=None,note:Optional[str]="",file_route:Optional[str]=None):
        
        """Save headerless training CSVs; return None.

        With neither array supplied, save neat arrays with the FINAL prefix. Otherwise
        supply both arrays; note is inserted verbatim into filenames. A supplied
        file_route also updates the estimator output prefix.
        """
        if file_route is not None:
            self.file_route = file_route
        if self.file_route is None:
            raise ValueError("Please specify the file route")
        
        if (train_X is None) and (train_Y is None):
            df_X = pd.DataFrame(self.train_X_neat.detach().cpu().numpy())
            filename_X = f'{self.file_route}bolfi_FINAL_train_X({self.dis_type}).csv'
            df_X.to_csv(filename_X,header=False,index=False)
            df_Y = pd.DataFrame(self.train_Y_neat.detach().cpu().numpy())
            filename_Y = f'{self.file_route}bolfi_FINAL_train_Y({self.dis_type}).csv'
            df_Y.to_csv(filename_Y,header=False,index=False)
            return print(f'All train data has been saved with defalt name:,{filename_X} and {filename_Y}.')
        else:
            df_X = pd.DataFrame(train_X.detach().cpu().numpy())
            filename_X = f'{self.file_route}bolfi_{note}_train_X({self.dis_type}).csv'
            df_X.to_csv(filename_X, header=False, index=False)
            df_Y = pd.DataFrame(train_Y.detach().cpu().numpy())
            filename_Y = f'{self.file_route}bolfi_{note}_train_Y({self.dis_type}).csv'
            df_Y.to_csv(filename_Y, header=False, index=False)
            return print(f'All train data has been saved with defalt name:,{filename_X} and {filename_Y}.')
        

        
  

###########################################################
################# Parameter Inference #####################
###########################################################

# function to calculate additional gaussian noise of the trained GP
    def est_sigma2(self,repeat_num:int = 50,training_results:Optional[torch.Tensor]=None, train_X:Optional[torch.Tensor]=None): 
        """Estimate discrepancy variance at the row with the highest last-column score.

        Use stored training_results/train_X_neat or supply both arrays. Two-stage runs
        use the final discrepancy component. The loop always evaluates 50 replicates;
        repeat_num only sets buffer size, so use 50. Store and return self.cov.
        """
        # Select the row with the highest stored log-density score.
        self.repeate_num = repeat_num
       
        
        Y_cov = torch.zeros((repeat_num,1))
        if (training_results is None) and (train_X is None):
            training_results = torch.from_numpy(self.training_results).to(dtype = gv.DATA_TYPE_torch).to(device = gv.DEVICE)
            index = torch.argmax(training_results[:,-1])
            beta_bolfi_best = self.train_X_neat[index,:].flatten()
        elif training_results.size(0)==train_X.size(0):
            index = torch.argmax(training_results[:,-1])
            beta_bolfi_best = train_X[index,:].flatten()
        else:
            raise warnings("the size of training_results input does not match the size of train_X input.")
        
        rgn_sigma2 = torch.Generator(device=gv.DEVICE).manual_seed(self.MASTER_SEED)
        if self.dis_type in ["5+6","6+5"]:
            for i in range(50):
               
                Y_cov[i,:] = compute_discrepancy(beta = beta_bolfi_best, simulator_fun = simulate, sim_num = self.sim_num, 
                                                 dis_type = self.dis_type, global_sd = self.global_sd, rgn_seed = gv.MASTER_SEED*7, sequential=i)[-1]
        else:
            for i in range(50):
                
                Y_cov[i,:] = compute_discrepancy(beta = beta_bolfi_best, simulator_fun = simulate, sim_num = self.sim_num,
                                                 dis_type = self.dis_type, global_sd = self.global_sd, rgn_seed = gv.MASTER_SEED*7, sequential=i)
                                                 
                      
        self.cov = torch.cov(input = Y_cov.detach().cpu().clone().T).cpu().numpy()
        
        return self.cov

# function to determine the satisfactory discrepancy (closer to the true value) of the trained GP
    def est_tol(self,tol_quantile:float= 0.99, train_Y:Optional[torch.Tensor]=None):
        """Return a quantile of negative discrepancies, using the final stage when present.

        With no train_Y, use stored neat targets and update self.tol.
        Higher quantiles impose a stricter likelihood threshold.
        """
        self.tol_quantile = tol_quantile
        if train_Y is None:
            if self.dis_type in ["5+6","6+5"]:
                self.tol = torch.quantile(self.train_Y_neat[:,1],q=tol_quantile,dim=0)#.tolist()
            else:
                self.tol = torch.quantile(self.train_Y_neat,q=tol_quantile,dim=0)#.tolist()
            return self.tol
        else:
            if self.dis_type in ["5+6","6+5"]:
                tol = torch.quantile(train_Y[:,1],q=tol_quantile,dim=0)#.tolist()
            else:
                tol = torch.quantile(train_Y,q=tol_quantile,dim=0)#.tolist()    
        return tol

# approximate log-likelihood function of the given parameter vector
    def calculate_likelihood(self,beta:torch.Tensor, cov:Optional[float]=None,tol:Optional[float]=None, *args, **kwargs):
        """Return a one-element approximate log-likelihood tensor for an unscaled beta.

        Use self.gp, self.cov and self.tol; cov/tol arguments and kwargs are ignored.
        Marginal mode includes GP variance; trust mode adds a bounded log-sigmoid
        penalty when GP variance exceeds the cached threshold.
        """
        beta = self.normalize(beta.view(1,-1))
        with gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.cholesky_max_tries(8):
            preds = self.gp.posterior(beta)
            means, variances = preds.mean, preds.variance
        means = means.view(-1)
        Sigma = variances[0]
        cov0 = torch.tensor(self.cov, dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)

        # Marginal likelihood; trust mode additionally penalizes high GP variance.
        mode = getattr(self, "llk_mode", "marginal")
        z = (means - self.tol) / torch.sqrt(cov0 + Sigma)          # Include simulator noise and GP variance.
        log_likelihood = torch.special.log_ndtr(z)                 # stable log-CDF of N(0,1)
        if mode == "trust":
            # Clamp the log-sigmoid penalty to a finite floor.
            sig_max = self._sigma_trust_max()
            s = float(getattr(self, "gate_sharpness", 6.0))
            floor = float(getattr(self, "llk_gate_floor", -50.0))
            gate = torch.nn.functional.logsigmoid(-s * (Sigma / sig_max - 1.0)).clamp_min(floor)
            log_likelihood = log_likelihood + gate
        log_likelihood = log_likelihood.to(dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)

        return log_likelihood

    def _sigma_trust_max(self):
        """Return the cached GP-variance threshold, computing it if absent.

        Use sigma_trust_c (default 3) times the 95th percentile over the first
        _sigma_trust_n normalized training rows, or all rows when unset.
        """
        v = getattr(self, "_sigma_trust_max_val", None)
        if v is None:
            c = float(getattr(self, "sigma_trust_c", 3.0))
            n = int(getattr(self, "_sigma_trust_n", 0)) or self.train_X_normed.size(0)
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4):
                st = self.gp.posterior(self.train_X_normed[:n]).variance.view(-1)
            v = (c * torch.quantile(st, 0.95)).to(dtype=gv.DATA_TYPE_torch, device=gv.DEVICE)
            self._sigma_trust_max_val = v
        return v
    
    def plt_time_cost(self):
        """Save cumulative BO-loop time in minutes with stage/stopping markers; return None."""
        # Infer batch indices from the stored point count.
        iter_list = np.arange((self.train_Y_neat.size(0) - self.init_num) / self.batch_size)

        accumulated_time = np.cumsum(self.time_log)/60
        plt.figure()
        plt.plot(iter_list, accumulated_time, color="black")
        if self.dis_type in ["5+6","6+5"] and self.iter_min is not None:
            plt.axvline((self.iter_min-self.init_num)/self.batch_size,linestyle="--",c='blue',label='stage switch point')

        if self.iter_early_stop is not None:
            plt.axvline((self.iter_early_stop-self.init_num)/self.batch_size,linestyle="--",c='red',label='early stop point')
        plt.legend()
        plt.legend(fontsize=13,loc='upper right')
        plt.grid()
        plt.tick_params(axis='both', labelsize=15)
        plt.xlabel("iterations for GP training",fontsize=20)
        plt.ylabel("accumulated time (min)",fontsize=20)
        
        

        plt.savefig(fname = f'{self.file_route}MLBA_timecost_{self.dis_type}({self.sim_num}).pdf',dpi = 1080,bbox_inches='tight') 
        plt.show()
        return print(f"The time cost of each iteration and the time cost of storing data have been plotted and saved at {self.file_route}MLBA_timecost_{self.dis_type}({self.sim_num}).pdf")
    

            

    def result_output(self):
        """Pickle training arrays, settings and diagnostics for later GP refitting; return None."""
  
        res = {"sim_num":self.sim_num,
                "dis_type":self.dis_type,
                "data_type":[gv.DATA_TYPE_np,gv.DATA_TYPE_torch],
                "master seed":gv.MASTER_SEED,
                "child seed":self.CHILD_SEED,
                "epsilon":gv.EPSILION,
                "betai_function":self.betai_function.__name__,
                "dim":self.dim,
                "bounds":self.bounds,
                "prior_type":self.prior_type,
                "prior_param":self.prior_param,
                "init_num":self.init_num,
                "train_X_init": f'{self.file_route}bolfi_INITIAL_train_X({self.dis_type}).csv',
                "train_Y_init": f'{self.file_route}bolfi_INITIAL_train_Y({self.dis_type}).csv',
                "train_X":self.train_X,
                "train_Y":self.train_Y,
                "train_X_neat":self.train_X_neat,
                "train_Y_neat":self.train_Y_neat,
                "train_X_normed":self.train_X_normed,
                "train_X_neat_route":f'{self.file_route}bolfi_FINAL_train_X({self.dis_type}).csv',
                "train_Y_neat_route":f'{self.file_route}bolfi_FINAL_train_Y({self.dis_type}).csv',
                "train_iters":self.train_X.size(0),
                "tol_gap":self.tol_gap,
                "tol_ucb":self.tol_ucb,
                "tol_step":self.tol_step,
                "tol_sigma":self.tol_sigma,
                "max_iters":self.max_iters,
                "gp_training_time":self.gp_training_time,
                "gp_cov":self.cov,
                "gp_cov_simple_num":self.repeate_num,
                "gp_tol":self.tol,
                "gp_tol_quantile":self.tol_quantile,
                "gp_raw_samples":self.raw_samples,
                "gp_num_restarts":self.num_restarts,
                "training_results":self.training_results,
                "global_sd": self.global_sd ,
                "time_log":self.time_log,
                "time_storage":self.time_storage,
                "iter_min":self.iter_min,
                "iter_early_stop":self.iter_early_stop,
                "batch_size":self.batch_size
                } 
        
        # save res as a pickle file
        with open(f'{self.file_route}bolfi_training_results_summary({self.dis_type}).pkl', 'wb') as f:
            pickle.dump(res, f) 
        print(f'The training results summary has been saved at {self.file_route}bolfi_training_results_summary({self.dis_type}).pkl')
        