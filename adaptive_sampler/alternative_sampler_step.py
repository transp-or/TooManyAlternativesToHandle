# alternative_samplers_step.py
import numpy as np
from typing import Tuple, Optional
from pymc.step_methods.arraystep import ArrayStep
from pymc.blocking import RaveledVars
from pymc.util import get_value_vars_from_user_vars

# ---------------------------------------------
# MTM for Protocol 2 (single index, writes V_i_s)
# ---------------------------------------------
class IndexRWMTMStepP2(ArrayStep):
    """
    Protocol-2 MTM over single indices:
      - Draw m forward candidates uniformly, pick 1 ∝ exp(V_frozen).
      - Backward set = {old} ∪ (m-1) fresh uniform draws.
      - Accept with α = min(1, sum_w_fwd / sum_w_bwd).
      - Refresh X_shared, same_alt_mask, V_i_*_s and V_i_s (stack of [star, tilde]).
    """
    def __init__(self, *, idx_var, beta_vars, model, X_full, chosen_vec,
                 X_shared, same_alt_mask, V_i_tilde_s, V_i_star_s, V_i_s,
                 step_size=10, random_seed=42, m=5):

        self.idx_value_var   = get_value_vars_from_user_vars([idx_var], model)[0]
        self.beta_value_vars = get_value_vars_from_user_vars(beta_vars, model)
        super().__init__(vars=[self.idx_value_var], fs=[])

        self.X_full = X_full.astype("float32")
        self.N, self.J, self.Kp1 = self.X_full.shape
        self.chosen_vec = np.asarray(chosen_vec, dtype=np.int64)
        self.X_shared = X_shared
        self.same_alt_mask = same_alt_mask
        self.V_i_tilde_s = V_i_tilde_s
        self.V_i_star_s  = V_i_star_s
        self.V_i_s = V_i_s 
        self.m = int(m)
        self.step_size = int(step_size)
        self.rng = np.random.default_rng(random_seed)

        # I've added stats I wanted to check afterwards
        self.stats_dtypes = [{
            # "idx_accept_rate": np.float64,
            # "idx_n_accept": np.int64,
            "idx_n_same_as_chosen_prop": np.int64,
            # "idx_mean_delta": np.float64,
            # "idx_prop_idx_mean": np.float64,
            # "idx_V_star_s_mean": np.float64,
            # "idx_V_tilde_s_mean": np.float64,
            # "idx_beta_norm": np.float64,
            "tune": bool,
        }]

    def _utilities_for_indices(self, beta: np.ndarray, idx: np.ndarray) -> np.ndarray:
        X_sel = self.X_full[np.arange(self.N), idx, :]
        return (X_sel * beta).sum(axis=1)

    def _refresh_containers(self, idx_new: np.ndarray, beta: np.ndarray) -> None:
        X_star  = self.X_full[np.arange(self.N), self.chosen_vec, :]
        X_tilde = self.X_full[np.arange(self.N), idx_new, :]

        self.X_shared.set_value(np.stack([X_star, X_tilde], axis=1).astype(np.float32))
        same_mask = (self.chosen_vec == idx_new).astype(np.int16)
        self.same_alt_mask.set_value(same_mask)

        V_star_s  = (X_star  * beta).sum(axis=1).astype(np.float32)
        V_tilde_s = (X_tilde * beta).sum(axis=1).astype(np.float32)
        self.V_i_star_s.set_value(V_star_s)
        self.V_i_tilde_s.set_value(V_tilde_s)

        self.V_i_s.set_value(np.stack([V_star_s, V_tilde_s], axis=1))

    def step(self, point):
        q = point[self.idx_value_var.name]
        beta = np.array([point[v.name] for v in self.beta_value_vars],
                        dtype=self.X_full.dtype).ravel()
        new_q, stats = self.astep(q, beta)
        new_point = point.copy()
        new_point[self.idx_value_var.name] = new_q
        return new_point, [stats]

    def astep(self, q: np.ndarray, beta: np.ndarray):
        N, J = self.N, self.J
        beta_frozen = beta.astype(self.X_full.dtype, copy=False)

        def util_for_idx_matrix(beta_vec, idx_mat):
            flat = idx_mat.reshape(-1)
            X_sel = self.X_full[np.repeat(np.arange(N), idx_mat.shape[1]), flat, :]
            U = (X_sel * beta_vec).sum(axis=1)
            return U.reshape(N, idx_mat.shape[1])

        K_fwd = self.rng.integers(low=0, high=J, size=(N, self.m), endpoint=False, dtype=np.int64)
        U_fwd = util_for_idx_matrix(beta_frozen, K_fwd)
        W_fwd = np.exp(np.clip(U_fwd, -50.0, 50.0))
        S_fwd = W_fwd.sum(axis=1, keepdims=True)

        u = self.rng.random(N)[:, None] * S_fwd
        cdf = np.cumsum(W_fwd, axis=1)
        choice_idx = (cdf >= u).argmax(axis=1)
        idx_prop = K_fwd[np.arange(N), choice_idx]
        sum_w_fwd = S_fwd[:, 0]

        K_bwd_rest = self.rng.integers(low=0, high=J, size=(N, self.m - 1),
                                       endpoint=False, dtype=np.int64)
        K_bwd = np.concatenate([q.reshape(N, 1), K_bwd_rest], axis=1)
        U_bwd = util_for_idx_matrix(beta_frozen, K_bwd)
        W_bwd = np.exp(np.clip(U_bwd, -50.0, 50.0))
        sum_w_bwd = W_bwd.sum(axis=1)

        log_alpha = np.log(sum_w_fwd) - np.log(sum_w_bwd)
        accept = np.log(self.rng.random(N)) < np.minimum(0.0, log_alpha)
        idx_new = np.where(accept, idx_prop, q).astype(np.int64)

        self._refresh_containers(idx_new, beta_frozen)

        stats = {
            # "idx_accept_rate": float(accept.mean()),
            # "idx_n_accept": int(accept.sum()),
            "idx_n_same_as_chosen_prop": int((self.chosen_vec == idx_prop).sum()),
            # "idx_mean_delta": float((np.log(sum_w_fwd + 1e-12) - np.log(sum_w_bwd + 1e-12)).mean()),
            # "idx_prop_idx_mean": float(idx_prop.mean()),
            # "idx_V_star_s_mean": float(((self.X_full[np.arange(N), self.chosen_vec, :] * beta_frozen).sum(axis=1)).mean()),
            # "idx_V_tilde_s_mean": float(((self.X_full[np.arange(N), idx_new,       :] * beta_frozen).sum(axis=1)).mean()),
            # "idx_beta_norm": float(np.linalg.norm(beta_frozen)),
            "tune": False,
        }
        return idx_new.astype(q.dtype, copy=False), stats


# ---------------------------
# Convenience factory
# ---------------------------
def make_alt_step(
    *,
    model,
    X_full,
    chosen_vec,
    X_container,
    V_i_s,
    beta_vars,
    idx_var=None,            # required for MTM
    same_alt_mask=None,      # protocol 2
    V_i_tilde_s=None,        # protocol 2
    V_i_star_s=None,         # protocol 2
    sample_size: int = 2,
    m: int = 5,
    random_seed: int = 42,
):
    
    return IndexRWMTMStepP2(
        idx_var=idx_var,
        beta_vars=beta_vars,
        model=model,
        X_full=X_full,
        chosen_vec=chosen_vec,
        X_shared=X_container,
        same_alt_mask=same_alt_mask,
        V_i_tilde_s=V_i_tilde_s,
        V_i_star_s=V_i_star_s,
        V_i_s=V_i_s,
        step_size=10,
        random_seed=random_seed,
        m=m,
    )

