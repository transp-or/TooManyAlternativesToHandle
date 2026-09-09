"""MNL choice simulators for TuRBO-BOLFI.

Both return (choices (num_sim, N), utilities (num_sim, N, 2)); utilities
contain V at the observed choice and at each simulated choice, respectively.
gv_simulator enumerates alternatives with additive noise. sa_simulator uses
finite Multiple-Try Metropolis sweeps, with optional per-seed chain reuse.
Call reset_sa_simulator before SA use; simulate dispatches on gv.SIMULATOR_TYPE.
"""
import threading as _threading
from typing import Optional

import torch

import GLOBAL_GPU as gv


# ============================================================================
# 1. systematic utility
# ============================================================================
def mnl_choice_utility(estimable_parameters_vector: torch.Tensor,
                       attributes_matrix: torch.Tensor, device="cpu") -> torch.Tensor:
    """Return X @ beta with shape (N, J); device placement follows the input tensors."""
    return torch.matmul(attributes_matrix, estimable_parameters_vector)


# ============================================================================
# 2. random-utility MNL simulator (enumerates all J alternatives)
# ============================================================================
def gv_simulator(estimable_parameters_vector: torch.Tensor, attributes_matrix: torch.Tensor,
                 noise_type="gumbel", noise_parameters=(0, 1), num_sim=1, device="cpu",
                 rgn_seed: Optional[int] = None):
    """Choose argmax(V + noise) over all alternatives; return the module output tuple.

    Support Gumbel and normal noise with location/scale parameters; other noise_type
    values add no noise. Gumbel uniforms are clamped to [1e-6, 1 - 1e-6].
    """
    if rgn_seed is None:
        rgn_seed = int(getattr(gv, "MASTER_SEED", 42))
    rgn = torch.Generator(device=device).manual_seed(int(rgn_seed))

    utilities0 = mnl_choice_utility(estimable_parameters_vector, attributes_matrix, device)   # (N, J)
    dtype = utilities0.dtype
    N, J = utilities0.shape
    shape = (num_sim, N, J)
    if noise_type == "normal":
        noise = torch.normal(noise_parameters[0], noise_parameters[1], size=shape,
                             device=device, dtype=dtype, generator=rgn)
    elif noise_type == "gumbel":
        U = torch.rand(shape, device=device, dtype=dtype, generator=rgn).clamp_(1e-6, 1 - 1e-6)
        noise = noise_parameters[0] - noise_parameters[1] * torch.log(-torch.log(U))
    else:
        noise = torch.zeros(shape, device=device, dtype=dtype)

    chosen_matrix = torch.argmax(noise + utilities0, dim=2)                                   # (num_sim, N)

    rows = torch.arange(N, device=device)
    v_obs = utilities0[rows, gv.OBSERVATION_MATRIX].unsqueeze(0).expand(num_sim, -1)          # (num_sim, N)
    v_sim = utilities0[rows.unsqueeze(0), chosen_matrix]                                      # (num_sim, N)
    obs_util = torch.stack((v_obs, v_sim), dim=-1)                                            # (num_sim, N, 2)
    return (chosen_matrix, obs_util)


# ============================================================================
# 3. Sequential adjustment (Multiple-Try Metropolis)
# ============================================================================
def _mix64(*vals):
    """Hash an ordered sequence of integers into a stable non-negative PyTorch seed."""
    h = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for v in vals:
        h = (h ^ (int(v) & mask)) & mask
        h = (h * 0xBF58476D1CE4E5B9) & mask
        h ^= h >> 31
    return h % (2 ** 63)
class _SAState:
    def __init__(self):
        self.J = None
        self.obs = None                 # (N,) observed choice indices
        self.mh_steps_init = 1          # MTM sweeps per call during the init phase (warm=False)
        self.mh_steps_stage = (1,)     # MTM sweeps per call during BO, indexed by discrepancy stage
        self.mtm_m = 1                  # Multiple-Try Metropolis tries per sweep (1 = plain independence MH)
        self.carry = True
        self.warm = False               # Reuse chains only when warm and carry are both true.
        self.stage = 0                  # current BO discrepancy stage (set by train_bolfi)
        self.cold_bo_iters = 0          # leading BO iterations that still use the init rule
        self._bo_iter = 0               # BO loop counter (set by train_bolfi)
        self.anchor = {}                # {rgn_seed -> (num_sim, N) running choice state}, one chain per slot
        self.best_match = 0.0           # running best fraction-matching-observed (diagnostic)
        self.accept_rate = 0.0          # mean MH acceptance rate over the last call (diagnostic)
        self.attr_getter = None         # Optional broadcast-aware attribute lookup.
        self._calls = 0
        self._lock = _threading.Lock()


_SA = _SAState()


def reset_sa_simulator(num_alt, obs_choices, mh_steps=1, mh_steps_warm=None,
                       mtm_m=1, carry=True, cold_bo_iters=0, attr_getter=None, device="cpu"):
    """Reset SA chains, diagnostics and stage counters for a new run.

    mh_steps sets cold sweeps; mh_steps_warm sets warm sweeps per stage (int or
    sequence, default mh_steps). mtm_m sets proposal tries, clamped to at least 1.
    Reuse chains only when warm and carry are true. train_bolfi controls warm
    using cold_bo_iters. attr_getter must broadcast row/alternative indices for
    (S, N) choices and (S, N, m) proposals, returning a final feature axis.
    """
    _SA.J = int(num_alt)
    _SA.obs = obs_choices.to(device).long().view(-1)
    _SA.mh_steps_init = int(mh_steps)
    w = mh_steps if mh_steps_warm is None else mh_steps_warm
    _SA.mh_steps_stage = (int(w),) if isinstance(w, (int, float)) else tuple(int(x) for x in w)
    _SA.mtm_m = max(1, int(mtm_m))
    _SA.carry = bool(carry)
    _SA.cold_bo_iters = int(cold_bo_iters)
    _SA.attr_getter = attr_getter
    _SA.warm = False
    _SA.stage = 0
    _SA._bo_iter = 0
    _SA.anchor = {}
    _SA.best_match = 0.0
    _SA.accept_rate = 0.0
    _SA._calls = 0


def _sa_mtm_sweep(get_attr, beta, rows, cur, v_cur, J, m, gen, device):
    """Advance (S, N) choices by one Multiple-Try Metropolis sweep.

    Draw m uniform proposals, select by softmax utility, and form a backward set
    from the current choice plus m - 1 uniform draws. Accept using the forward /
    backward weight-sum ratio. Return choices, utilities and mean acceptance.
    Utility evaluation costs O(S * N * m * P); m == 1 gives independence MH.
    """
    S, N = cur.shape
    rows_t = rows.reshape(1, N, 1)          # broadcast against (S, N, tries) alt-index tensors

    # ---- forward: m uniform tries, select one proportional to exp(V) ----
    cand = torch.randint(0, J, (S, N, m), device=device, generator=gen)          # (S,N,m)
    v_cand = get_attr(rows_t, cand) @ beta                                       # (S,N,m)
    lse_f = torch.logsumexp(v_cand, dim=-1)                                      # (S,N)  log sum_i w(y_i)
    pick = torch.multinomial(torch.softmax(v_cand, dim=-1).reshape(S * N, m),
                             1, generator=gen).reshape(S, N, 1)                  # (S,N,1)
    prop = cand.gather(-1, pick).squeeze(-1)                                     # (S,N)
    v_prop = v_cand.gather(-1, pick).squeeze(-1)                                 # (S,N)

    # ---- backward reference set: {current} u (m-1) fresh uniform tries ----
    if m > 1:
        back = torch.randint(0, J, (S, N, m - 1), device=device, generator=gen)  # (S,N,m-1)
        v_back = get_attr(rows_t, back) @ beta                                   # (S,N,m-1)
        lse_b = torch.logsumexp(torch.cat((v_cur.unsqueeze(-1), v_back), dim=-1), dim=-1)  # (S,N)
    else:
        lse_b = v_cur

    # ---- MTM acceptance ----
    u = torch.rand((S, N), device=device, generator=gen)
    accept = torch.log(u) < torch.clamp(lse_f - lse_b, max=0.0)                  # (S,N)
    cur = torch.where(accept, prop, cur)
    v_cur = torch.where(accept, v_prop, v_cur)
    return cur, v_cur, accept.to(v_cur.dtype).mean()


def sa_simulator(estimable_parameters_vector, attributes_matrix, noise_type="gumbel",
                 noise_parameters=(0, 1), num_sim=1, device="cpu", rgn_seed=None, **kwargs):
    """Advance per-seed SA chains and return choices/utilities (see module docstring).

    Requires reset_sa_simulator. Reuse matching chain shapes only when warm and carry
    are true. Seed each call from (rgn_seed, stage, _bo_iter); distinct slots avoid
    shared choice state while stage settings remain fixed. Noise arguments are ignored.
    """
    if _SA.J is None:
        raise RuntimeError("SA simulator not initialised - call reset_sa_simulator(...) first.")
    S = int(num_sim)
    beta = estimable_parameters_vector.to(device)
    N = int(attributes_matrix.shape[0])

    if _SA.attr_getter is not None:
        get_attr = _SA.attr_getter
    else:
        _am = attributes_matrix.to(device)
        def get_attr(rows, alts):
            return _am[rows, alts]                          # Broadcast indices; retain the final feature axis.
    rows = torch.arange(N, device=device)

    # Seed by slot and BO state; keep stage settings fixed during a batch.
    slot = int(rgn_seed if rgn_seed is not None else getattr(gv, "MASTER_SEED", 42))
    with _SA._lock:
        _SA._calls += 1                       # diagnostic count only; not used for seeding
        stage, bo_iter = _SA.stage, _SA._bo_iter
    gen = torch.Generator(device=device).manual_seed(_mix64(slot, stage, bo_iter))

    cur = None
    if _SA.warm and _SA.carry:
        with _SA._lock:
            prev = _SA.anchor.get(slot)
        if prev is not None and tuple(prev.shape) == (S, N):
            cur = prev.clone()
    if cur is None:
        cur = torch.randint(0, _SA.J, (S, N), device=device, generator=gen)     # sample w/ replacement

    k = _SA.mh_steps_stage[min(_SA.stage, len(_SA.mh_steps_stage) - 1)] if _SA.warm else _SA.mh_steps_init
    m_tries = max(1, int(_SA.mtm_m))
    v_cur = get_attr(rows, cur) @ beta                                          # (S,N) seed, carried across sweeps
    acc = 0.0
    for _ in range(k):
        cur, v_cur, a = _sa_mtm_sweep(get_attr, beta, rows, cur, v_cur, _SA.J, m_tries, gen, device)
        acc += float(a)

    obs = _SA.obs.to(device)
    match = (cur == obs.unsqueeze(0)).float().mean().item()
    with _SA._lock:
        _SA.anchor[slot] = cur.detach().clone()   # per-slot chain: only this rgn_seed touches this key
        _SA.best_match = max(_SA.best_match, match)
        _SA.accept_rate = acc / max(k, 1)

    v_obs = get_attr(rows, obs.unsqueeze(0).expand(S, N)) @ beta                 # (S, N)
    v_sim = v_cur                                                               # (S, N)  carried, no re-gather
    obs_util = torch.stack((v_obs, v_sim), dim=-1)                              # (S, N, 2)
    return (cur, obs_util)


# ============================================================================
# 4. dispatcher + discrepancy-side helper
# ============================================================================
def simulate(*args, **kwargs):
    """Dispatch case-insensitive sa/seq/mh to SA; all other values use gv_simulator."""
    stype = str(getattr(gv, "SIMULATOR_TYPE", "all")).lower()
    if stype in ("sa", "seq", "mh"):
        return sa_simulator(*args, **kwargs)
    return gv_simulator(*args, **kwargs)


def _obs_sim_util(res_observed_matrix):
    """Unpack a simulator's (sim_num, N, 2) utility output into
    (true_obs_utility (N, 1), sim_obs_utility (N, sim_num))."""
    true_obs = res_observed_matrix[0, :, 0].unsqueeze(1)            # (N, 1)  V at observed choice
    sim_obs = res_observed_matrix[:, :, 1].transpose(0, 1)         # (N, sim_num)  V at simulated choice
    return true_obs, sim_obs
