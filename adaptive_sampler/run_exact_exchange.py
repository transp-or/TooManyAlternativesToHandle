"""Exact exchange sampler for the full multinomial-logit posterior.

This module implements the exchange transition described in
``bayesian_choice_set_v2.tex``.  It deliberately does not use the binary
pairwise potential from ``run_model.py`` and it does not use the MTM
alternative transition from ``alternative_sampler_step.py``.

At every outer iteration the custom PyMC step:

1. proposes beta with a symmetric Gaussian random walk;
2. draws one auxiliary alternative independently for every observation from
   the full J-alternative softmax at the proposed beta; and
3. accepts the proposal with the exchange ratio

       prior/proposal ratio * product_n
       exp(V_i(beta') + V_j(beta) - V_i(beta) - V_j(beta')).

The choice-set normalizers are never evaluated.  Because this implementation
uses an explicitly materialized finite choice set, the auxiliary categorical
draw is exact up to floating-point arithmetic.  For a truly implicit choice
set, an exact sampler (or an exactness-preserving alternative) would still be
needed; a finite inner MH run would turn this into approximate exchange.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Iterable

import numpy as np
import pymc as pm
from pymc.step_methods.arraystep import ArrayStep
from pymc.util import get_value_vars_from_user_vars


# Allow execution as ``python run_exact_exchange.py`` from this directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_sampler.data import load_data
from adaptive_sampler.reproducibility import (
    check_x_full_matrix,
    set_all_seeds,
    validate_data_content,
)
from adaptive_sampler.utils import (
    compute_interaction_matrices,
    create_X_restaurants,
)


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("output", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.FileHandler(f"output/run_exact_exchange_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )


def _build_x_full(alternatives, observations, p: int):
    """Build the same ``(N, J, K)`` design matrix as ``run_model.py``."""
    x_rest, rest_features = create_X_restaurants(alternatives, p)
    interaction_matrices, interaction_features = compute_interaction_matrices(
        observations, alternatives, p
    )

    feature_names = list(rest_features) + list(interaction_features)
    n = len(observations)
    j = len(alternatives)
    x_full = np.zeros((n, j, len(feature_names)), dtype=np.float32)

    for k, feature in enumerate(feature_names):
        if feature in rest_features:
            x_full[:, :, k] = x_rest[:, rest_features.index(feature)][None, :]
        else:
            x_full[:, :, k] = interaction_matrices[feature]

    return x_full, feature_names


def _scale_x_full(x_full: np.ndarray, feature_names: Iterable[str], mode: str):
    """Apply the existing pipeline's feature scaling and return its metadata."""
    if mode == "none":
        return x_full, None

    scaled = x_full.copy()
    scale_params = {}

    for k, feature in enumerate(feature_names):
        values = scaled[:, :, k]
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-10:
            std = 1.0

        if mode == "normalize":
            scaled[:, :, k] = (values - mean) / std
            scale_params[feature] = {"mean": mean, "std": std}
        elif mode == "scale":
            scaled[:, :, k] = values / std
            scale_params[feature] = {"std": std}
        else:
            raise ValueError("scaling_mode must be one of: none, normalize, scale")

    return scaled.astype(np.float32, copy=False), scale_params


class ExactExchangeStep(ArrayStep):
    """One exact exchange-MH transition for the beta vector.

    The proposal is a fixed symmetric Gaussian random walk.  The auxiliary
    alternatives are sampled independently from the full finite choice set at
    the proposed beta, so the exchange ratio is exact for the model represented
    by ``x_full`` and the specified prior.
    """

    name = "exact_exchange"

    def __init__(
        self,
        *,
        model,
        beta_vars,
        x_full: np.ndarray,
        chosen_vec: np.ndarray,
        proposal_scale: float = 0.05,
        prior_sigma: float = 10.0,
        random_seed: int = 42,
        batch_size: int = 2048,
    ):
        self.beta_value_vars = get_value_vars_from_user_vars(beta_vars, model)
        super().__init__(vars=self.beta_value_vars, fs=[])

        self.x_full = np.asarray(x_full, dtype=np.float32)
        self.chosen_vec = np.asarray(chosen_vec, dtype=np.int64)
        self.x_star = self.x_full[
            np.arange(self.x_full.shape[0]), self.chosen_vec, :
        ]
        self.proposal_scale = float(proposal_scale)
        self.prior_sigma = float(prior_sigma)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(random_seed)

        if self.proposal_scale <= 0:
            raise ValueError("proposal_scale must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.x_full.ndim != 3:
            raise ValueError("x_full must have shape (N, J, K)")
        if self.x_full.shape[0] != self.chosen_vec.shape[0]:
            raise ValueError("chosen_vec and x_full have incompatible N")
        if np.any(self.chosen_vec < 0) or np.any(
            self.chosen_vec >= self.x_full.shape[1]
        ):
            raise ValueError("chosen_vec contains an invalid alternative index")

        self.stats_dtypes = [
            {
                "accepted": bool,
                "acceptance_prob": np.float64,
                "log_exchange_ratio": np.float64,
                "tune": bool,
            }
        ]

    def _beta_from_point(self, point) -> np.ndarray:
        return np.asarray(
            [point[var.name] for var in self.beta_value_vars], dtype=np.float64
        ).reshape(-1)

    def _log_prior(self, beta: np.ndarray) -> float:
        # Independent Normal(0, prior_sigma) priors, matching the PyMC model.
        sigma = self.prior_sigma
        return float(
            -0.5 * np.sum((beta / sigma) ** 2)
            - beta.size * np.log(sigma * np.sqrt(2.0 * np.pi))
        )

    def _draw_auxiliary(self, beta: np.ndarray) -> np.ndarray:
        """Draw independent j_n ~ softmax(x_nj beta) without computing Z_n."""
        n, j, _ = self.x_full.shape
        auxiliary = np.empty(n, dtype=np.int64)

        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            x_batch = self.x_full[start:stop]

            # Subtracting the row maximum is numerically stable and does not
            # change the categorical probabilities.
            utilities = np.einsum("bjk,k->bj", x_batch, beta)
            utilities -= utilities.max(axis=1, keepdims=True)
            weights = np.exp(utilities)
            cumulative = np.cumsum(weights, axis=1)

            uniforms = self.rng.random(stop - start) * cumulative[:, -1]
            auxiliary[start:stop] = np.argmax(
                cumulative >= uniforms[:, None], axis=1
            )

        return auxiliary

    def _log_exchange_ratio(self, beta: np.ndarray, proposal: np.ndarray) -> float:
        """Draw auxiliaries at proposal and return log R for beta -> proposal."""
        auxiliary = self._draw_auxiliary(proposal)
        x_auxiliary = self.x_full[
            np.arange(self.x_full.shape[0]), auxiliary, :
        ]

        v_star_current = self.x_star @ beta
        v_star_proposal = self.x_star @ proposal
        v_aux_current = x_auxiliary @ beta
        v_aux_proposal = x_auxiliary @ proposal

        log_r = self._log_prior(proposal) - self._log_prior(beta)
        log_r += float(
            np.sum(
                v_star_proposal
                + v_aux_current
                - v_star_current
                - v_aux_proposal
            )
        )
        return float(log_r)

    def step(self, point):
        beta = self._beta_from_point(point)
        proposal = beta + self.rng.normal(
            loc=0.0, scale=self.proposal_scale, size=beta.shape
        )

        log_r = self._log_exchange_ratio(beta, proposal)
        log_acceptance = min(0.0, log_r)
        accepted = bool(np.log(self.rng.random()) < log_acceptance)
        acceptance_prob = float(np.exp(log_acceptance))

        new_point = point.copy()
        if accepted:
            for var, value in zip(self.beta_value_vars, proposal):
                new_point[var.name] = np.asarray(value, dtype=np.float64)

        stats = {
            "accepted": accepted,
            "acceptance_prob": acceptance_prob,
            "log_exchange_ratio": log_r,
            "tune": False,
        }
        return new_point, [stats]


def run_exact_exchange(
    *,
    n_steps: int,
    tune: int,
    random_seed: int,
    proposal_scale: float = 0.05,
    batch_size: int = 2048,
    J: int = 100,
    N: int = 10000,
    P: int = 10,
    I: int = 0,
    C: int = 0,
    M: int = 0,
    scaling_mode: str = "none",
    chains: int = 1,
    cores: int = 1,
):
    """Run the exact exchange sampler and return the ArviZ trace."""
    set_all_seeds(random_seed, verbose=False)
    np.random.seed(random_seed)

    alternatives, observations, _ = load_data(
        J=J, N=N, P=P, I=I, C=C, M=M
    )
    validation = validate_data_content(alternatives, observations, verbose=False)
    if not validation["all_valid"]:
        LOGGER.warning("Data validation reported invalid content: %s", validation)

    x_full, feature_names = _build_x_full(alternatives, observations, P)
    x_full, scale_params = _scale_x_full(x_full, feature_names, scaling_mode)
    x_validation = check_x_full_matrix(x_full, feature_names, verbose=False)
    if not x_validation["all_valid"]:
        raise ValueError(f"X_full validation failed: {x_validation}")

    chosen_vec = observations["logit_0"].to_numpy(dtype=np.int64)
    n, j, k = x_full.shape
    LOGGER.info(
        "Exact exchange model: N=%d, J=%d, K=%d, scaling=%s",
        n,
        j,
        k,
        scaling_mode,
    )

    with pm.Model() as model:
        beta_vars = [
            pm.Normal(
                f"beta_{feature}",
                mu=0.0,
                sigma=10.0,
                initval=0.0,
            )
            for feature in feature_names
        ]

        exchange_step = ExactExchangeStep(
            model=model,
            beta_vars=beta_vars,
            x_full=x_full,
            chosen_vec=chosen_vec,
            proposal_scale=proposal_scale,
            prior_sigma=10.0,
            random_seed=random_seed,
            batch_size=batch_size,
        )

        trace = pm.sample(
            draws=n_steps,
            tune=tune,
            step=exchange_step,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
            var_names=[var.name for var in beta_vars],
        )

    trace.attrs["sampler"] = "exact_exchange"
    trace.attrs["proposal_scale"] = proposal_scale
    trace.attrs["prior_sigma"] = 10.0
    trace.attrs["scaling_mode"] = scaling_mode
    trace.attrs["feature_names"] = json.dumps(feature_names)
    trace.attrs["scale_params"] = json.dumps(scale_params)
    trace.attrs["x_shape"] = json.dumps([n, j, k])
    trace.attrs["reproducibility_seed"] = random_seed
    trace.attrs["x_full_validation"] = str(x_validation)

    return trace, scale_params, feature_names


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Run exact exchange sampling for the full logit posterior"
    )
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proposal_scale", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--outdir", type=str, default="../results/results_exact_exchange")
    parser.add_argument("--J", type=int, choices=[100, 200, 500], default=100)
    parser.add_argument("--N", type=int, choices=[10000, 20000, 50000], default=10000)
    parser.add_argument("--P", type=int, choices=[10, 20, 40], default=10)
    parser.add_argument("--I", type=int, choices=[0, 1], default=0)
    parser.add_argument("--C", type=int, choices=[0, 1], default=0)
    parser.add_argument("--M", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--scaling",
        choices=["none", "normalize", "scale"],
        default="none",
    )
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()

    trace, scale_params, feature_names = run_exact_exchange(
        n_steps=args.n_steps,
        tune=args.tune,
        random_seed=args.seed,
        proposal_scale=args.proposal_scale,
        batch_size=args.batch_size,
        J=args.J,
        N=args.N,
        P=args.P,
        I=args.I,
        C=args.C,
        M=args.M,
        scaling_mode=args.scaling,
        chains=args.chains,
        cores=args.cores,
    )

    os.makedirs(args.outdir, exist_ok=True)
    suffix = (
        f"_J{args.J}_N{args.N}_P{args.P}_steps{args.n_steps}"
        f"_{args.scaling.upper()}"
    )
    trace_path = os.path.join(args.outdir, f"trace_exact_exchange{suffix}.nc")
    trace.to_netcdf(trace_path)
    LOGGER.info("Trace saved to %s", trace_path)

    if scale_params is not None:
        scale_path = os.path.join(args.outdir, f"scale_params{suffix}.json")
        with open(scale_path, "w", encoding="utf-8") as handle:
            json.dump(scale_params, handle, indent=2)
        LOGGER.info("Scale parameters saved to %s", scale_path)

    LOGGER.info("Features: %s", feature_names)


if __name__ == "__main__":
    main()
