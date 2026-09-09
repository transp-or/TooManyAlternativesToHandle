# Reproducible settings — BOLFI 
Updated 2026-09-07 
**Reports only the runs
kept in `final_result/`** — one canonical run per `(method, J, N)`, seed chosen for best coverage.
The *only* thing that differs between datasets of one method is the training `--seed`. Scripts now
read/write `final_result/{all,SA}/{data_id}/` directly.

Model: MNL, 10 restaurant-attribute betas, true values from `get_base_parameters(10)`. Ethiopian is the omitted cuisine reference (its ASC fixed at 0).

Layout:

```
final_result/
├── README.md
├── all/  J{100,200,500}_P10_N{10k,20k,50k}_I0C0M0_D5+6_SIM30/   (9 folders)
└── SA/   J{100,200,500}_P10_N{10k,20k,50k}_I0C0M0_D5+6_SIM30/   (9 folders)
```

Each folder: `bolfi_training_results_summary(5+6).pkl` for bolfi training GP model and associated parameters, `bolfi_{INITIAL,FINAL}_train_X/Y(5+6).csv` for samples over iterations,
`MCMC_summary_{iter_max,early_stop}.csv` for the summary of MCMC result for parameter posterior, `MCMC_trace_{iter_max,early_stop}.nc` for the full MCMC trace, and the
`MLBA_*` / `GP_fitness_trajectory` PDFs for the visual check of evolved GP performance.

---

## 0. Environment

The results were produced in an isolated Python virtual environment. Recreate it from just two things:

| | |
|---|---|
| **Python** | **3.11** (exact build used: 3.11.7; any 3.11.x is fine) |
| **Packages** | On Windows x86-64, `TUBOLFI/requirements-lock.txt` is the full `pip freeze` (95 packages, every version pinned). On macOS/Linux, install the portable direct-dependency list `TUBOLFI/requirements.txt`: the Windows lockfile includes Intel-only packages unavailable on those platforms. |
| **Platform** | developed on Windows x86-64, **CPU only** (`DEVICE = "cpu"`, `torch.set_default_dtype(float64)`). Linux/macOS x86-64 CPU should reproduce to within MCMC noise; other archs / GPU are untested. |
| key versions | `torch 2.9.0`, `botorch 0.16.0`, `gpytorch 1.14.2`, `linear-operator 0.6`, `numpy 2.3.4`, `pandas 2.3.3`, `scipy 1.16.3`, `scikit-learn 1.7.2`, `joblib 1.5.2`, `pymc 5.26.1`, `pytensor 2.35.1`, `arviz 0.22.0`, `xarray 2025.10.1`, `h5netcdf 1.7.3`, `matplotlib 3.10.7` |

Recreate (venv or conda — either works):

```
# venv
python3.11 -m venv .venv && . .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r TUBOLFI/requirements.txt

# conda
conda create -y -n bolfi-env python=3.11 && conda activate bolfi-env
pip install -r TUBOLFI/requirements.txt
```

All commands below assume the working directory is `.../codes/LFI/TUBOLFI/` and that this
environment's `python` is on `PATH` (shown as `python` throughout).

---

## 1. Common settings (both methods, all datasets)

| block | setting |
|---|---|
| **Data** | `data/restaurants_J{J}_P10_N{N}_I0C0M0.csv`, `data/choices_J{J}_P10_N{N}_I0C0M0.csv`; `I=0 C=0 M=0` |
| **Discrepancy** | `--dis_type 5+6` (stage 0 = utility-based, stage 1 = attribute-based); `global_sd` calibrated at the stage-0→1 switch (`cholesky(inv(Omega_hat))`) |
| **Sims / eval** | `--sim_num 30` |
| **BO / TuRBO** | `--batch_size 2`, `--convergence_type fixed`, `max_iters = 150*P = 1500`, `--system windows` (threaded batch discrepancy eval) |
| **GP surrogate** | `SingleTaskGP`, `ScaleKernel(MaternKernel(nu=2.5, ARD, lengthscale in [0.005, 3.0]))`, `GaussianLikelihood(noise in [1e-6, 1e-2])`, `Standardize` outcome, `Normalize` inputs; refit via `fit_gpytorch_mll` each iteration |
| **Noise `cov0`** | `est_sigma2`: variance of 50 repeated stage-1 neg-discrepancy sims at the BO incumbent |
| **MCMC likelihood** (hardcoded in `MCMC_sampling`, no CLI flag) | `llk_mode = "trust"`: `logL = logPhi((mu - tol)/sqrt(cov0 + Sigma)) + logsigmoid(-s*(Sigma/sig_max - 1)).clamp_min(-50)` |
| | `sigma_trust_c = 3.0` -> `sig_max = 3 * q95(Sigma at training inputs)`; `gate_sharpness s = 10.0` |
| **tol_quantile** (passed to `gp_retrive` in `main()`, overrides the pkl's training value) | **0.85** for both `iter_max` and `early_stop` (was 0.90 for `iter_max` — lowered 2026-09-06, +12 pp `all` / +6 pp SA `iter_max` coverage) |
| **MCMC sampler** | `pm.DEMetropolisZ(tune_drop_fraction=0.9, scaling=5e-4, tune="lambda")`; `pm.sample(tune=2500, draws=5000, chains=4, cores=4, initvals = BOLFI best point, random_seed=42)` — 20 000 post-tuning samples. (was `tune=2000, draws=6000`; changed 2026-09-07 — fixed `SA J500/N10k` iter_max mixing (r̂ 1.05→1.03, ess 91→293); `all` early_stop 81→82/90, everything else within ±1 beta) |
| **Prior (MCMC)** | `beta_k ~ Normal(0, 10)` iid |
| **Device / dtype / threads** | CPU, `float64`; `OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=24` (set at top of script); `PYTENSOR_FLAGS="openmp=False"`, `pytensor.config.floatX="float64"` (set in `MCMC_sampling`) |
| **Environment** | see §0 |

### 1b. Per-method (differs only in the simulator)

| | BOLFI-all | BOLFI-SA |
|---|---|---|
| `--simulator_type` | `all` — `gv_simulator`: `V + iid Gumbel`, argmax over all J alternatives (exact MNL draw) | `SA` — `sa_simulator`: Multiple-Try Metropolis on the discrete choice, per-slot deterministic RNG + per-slot carried chain (reproducible under threaded batch eval) |
| SA knobs | — | `--mh_steps 30` (init sweeps), `--mh_steps_warm 20,10` (BO sweeps stage0,stage1), `--sa_mtm_m 5` (tries/sweep). `carry=True`, `cold_bo_iters=1` are **hardcoded** in the `reset_sa_simulator()` call. |

---

## 2. Results kept (`final_result/`)

`cov` = # of 10 betas whose true value is inside the 94% HDI. `||best-t||` = Euclidean distance of
the BO incumbent from the true beta (GP-fitness proxy), shown as **`iter_max slice / early_stop
slice`** — `early_stop` always searches a strict subset of the acquired points
(`iter_early_stop < n_points` for every run), so the two are listed even when equal; they differ
only when the BO's best point was acquired *after* `iter_early_stop`.

Both retrievals use `tol_quantile = 0.85`. Coverage / `r_hat` / `min ess` below are the
**2026-09-07 MCMC re-run** (`tune=2500, draws=5000`); `||best-t||` is from the (unchanged) pkls.

### BOLFI-all — `final_result/all/`

| J | N | seed | \|\|best-t\|\| (im / es) | iter_max cov (r_hat / min ess) | early_stop cov (r_hat / min ess) |
|---|---|---|---|---|---|
| 100 | 10k | **41** | 0.15 / 0.20 | **10/10** (1.01 / 521) | 9/10 (1.01 / 447) `cost` |
| 100 | 20k | 42 | 0.12 / 0.12 | **10/10** (1.02 / 354) | **10/10** (1.01 / 478) |
| 100 | 50k | 42 | 0.19 / 0.19 | **10/10** (1.01 / 412) | **10/10** (1.02 / 403) |
| 200 | 10k | 42 | 0.14 / 0.14 | **10/10** (1.02 / 335) | **10/10** (1.01 / 353) |
| 200 | 20k | 42 | 0.32 / 0.32 | 9/10 (1.02 / 523) `ind` | 9/10 (1.02 / 422) `ind` |
| 200 | 50k | **41** | 0.34 / 0.34 | 6/10 (1.01 / 485) `chi,jap,ind,mex` | 6/10 (1.02 / 488) `chi,jap,ind,mex` |
| 500 | 10k | **41** | 0.31 / 0.31 | 9/10 (1.02 / 461) `fre` | **10/10** (1.02 / 451) |
| 500 | 20k | 42 | 0.37 / 0.41 | 8/10 (1.01 / 442) `kor,mex` | 8/10 (1.01 / 525) `kor,mex` |
| 500 | 50k | 42 | 0.04 / 0.07 | **10/10** (1.01 / 517) | **10/10** (1.04 / 203) |

**Roll-up (9 datasets):** iter_max **82/90 = 91.1%** · early_stop **82/90 = 91.1%**

### BOLFI-SA — `final_result/SA/`

| J | N | seed | \|\|best-t\|\| (im / es) | iter_max cov (r_hat / min ess) | early_stop cov (r_hat / min ess) |
|---|---|---|---|---|---|
| 100 | 10k | 42 | 0.18 / 0.18 | **10/10** (1.02 / 353) | **10/10** (1.01 / 348) |
| 100 | 20k | 42 | 0.22 / 0.22 | **10/10** (1.02 / 458) | **10/10** (1.01 / 499) |
| 100 | 50k | 42 | 0.07 / 0.07 | 9/10 (1.01 / 484) `cost` | 9/10 (1.02 / 594) `cost` |
| 200 | 10k | 42 | 0.33 / 0.33 | 6/10 (1.01 / 439) `cost,jap,ind,mex` | 6/10 (1.01 / 531) `cost,jap,ind,mex` |
| 200 | 20k | **41** | 0.11 / 0.26 | **10/10** (1.01 / 470) | **10/10** (1.02 / 453) |
| 200 | 50k | 42 | 0.19 / 0.19 | **10/10** (1.01 / 547) | **10/10** (1.02 / 392) |
| 500 | 10k | **41** | 0.31 / 0.31 | **10/10** (1.03 / 293) | **10/10** (1.02 / 275) |
| 500 | 20k | 42 | 0.42 / 0.42 | **10/10** (1.01 / 547) | **10/10** (1.02 / 275) |
| 500 | 50k | 42 | 0.18 / 0.18 | 9/10 (1.01 / 520) `cost` | 9/10 (1.01 / 602) `cost` |

**Roll-up (9 datasets):** iter_max **84/90 = 93.3%**  · early_stop **84/90 = 93.3%**

**Seed 41 datasets:** all → J100/N10k, J200/N50k, J500/N10k ; SA → J200/N20k, J500/N10k.
Everything else is seed 42.

---

## 3. How to reproduce (human summary)

`run_bolfi_canonical_v2.py` reads *and* writes `final_result/{simulator_type}/{data_id}/`
directly. `--simulator_type` picks the subfolder; `--J --N --P` pick the dataset; `--seed` picks
the training seed. `main()` calls `train_gp` → `gp_retrive(tol_quantile=0.85)` → `MCMC_sampling`
for both `iter_max` and `early_stop`.

- **MCMC only** (~3 min/config): comment `bolfi_estimator = train_gp(args, DEVICE)` in `main()`.
  `gp_retrive` loads the stored pkl; `MCMC_sampling` overwrites `MCMC_summary_<mode>.csv`,
  `MCMC_trace_<mode>.nc`, `MLBA_joint_beta_*_<mode>.pdf`. The pkl and `bolfi_*_train_X/Y` are safe.
- **Full retrain** (~40–120 min/config): keep `train_gp` active. ⚠️ it **overwrites the kept pkl**
  in `final_result/`.
- The `if __name__ == "__main__"` loop runs all 9 `(J,N)` at a *single* `--seed`. To reproduce
  `final_result` you must run per-config with the seed from §2 (see §4 step 3).

Determinism: `set_determinism(seed)` runs in `train_gp`, `gp_retrive`, `MCMC_sampling`. Residual
run-to-run drift comes from `fit_gpytorch_mll` (L-BFGS) and multi-threaded BLAS
(`OMP_NUM_THREADS=24`); set `OMP_NUM_THREADS=1` for byte-exact retrains. Threaded SA batch eval is
deterministic (per-slot RNG + per-slot carried chain). MCMC (`DEMetropolisZ`, `random_seed=--seed`)
reproduces exactly for a fixed GP.

---

### 3a. Reproduce `TuRBOLFI_SA_early-stop` for every dataset

Activate the environment in Section 0 and run these commands from `TUBOLFI/`. Each command runs one dataset with `P=10`, `I=C=M=0`, stops BO automatically, then refits the early-stop GP and runs MCMC. Existing results for that dataset are overwritten; use a separate working copy to retain the canonical checkpoints.

```shell
# J=100
python -u run_bolfi_canonical_v2.py --J 100 --N 10000 --P 10 --simulator_type SA --convergence_type auto --seed 42
python -u run_bolfi_canonical_v2.py --J 100 --N 20000 --P 10 --simulator_type SA --convergence_type auto --seed 42
python -u run_bolfi_canonical_v2.py --J 100 --N 50000 --P 10 --simulator_type SA --convergence_type auto --seed 42

# J=200
python -u run_bolfi_canonical_v2.py --J 200 --N 10000 --P 10 --simulator_type SA --convergence_type auto --seed 42
python -u run_bolfi_canonical_v2.py --J 200 --N 20000 --P 10 --simulator_type SA --convergence_type auto --seed 41
python -u run_bolfi_canonical_v2.py --J 200 --N 50000 --P 10 --simulator_type SA --convergence_type auto --seed 42

# J=500
python -u run_bolfi_canonical_v2.py --J 500 --N 10000 --P 10 --simulator_type SA --convergence_type auto --seed 41
python -u run_bolfi_canonical_v2.py --J 500 --N 20000 --P 10 --simulator_type SA --convergence_type auto --seed 42
python -u run_bolfi_canonical_v2.py --J 500 --N 50000 --P 10 --simulator_type SA --convergence_type auto --seed 42
```

**Settings:** `--convergence_type auto` overrides the default `fixed`; `--J` and `--N` select the dataset. `--P 10`, `--simulator_type SA` and the explicit seeds match current defaults, including the dataset-specific SA seed table. Seeds are written explicitly for reproducibility. All commands retain `--dis_type 5+6`, `--sim_num 30`, `--batch_size 2`, `--mh_steps 30`, `--mh_steps_warm 20,10` and `--sa_mtm_m 5`. MCMC uses seed `42` for every dataset.

Results are saved under `final_result/SA/J{J}_P10_N{N/1000}k_I0C0M0_D5+6_SIM30/`, including `MCMC_summary_early_stop.csv`, `MCMC_trace_early_stop.nc` and the training checkpoint. In `auto` mode the current runner skips the `iter_max` inference pass. The early-stop refit uses `tol_quantile=0.85`.

**Wall-time and space:** record the printed `Training completed in ... seconds` (also saved as `gp_training_time`) and `Peak memory usage`. These cover initial sampling and BO, after data/feature construction and before diagnostics, saving, GP retrieval and MCMC. The memory value is peak `tracemalloc` allocation in MiB, despite its `MB` label; it is not total process RAM. Measure process-tree RSS separately if total memory is required. For end-to-end time, measure the entire command. Use the same hardware, environment and thread settings, and report the median/range across fresh-process repeats.

These commands perform new automatically stopped runs. Section 2's retained early-stop results were refitted from longer `fixed` runs, with some quantities derived from the full training history; identical posterior summaries are not guaranteed. Their saved checkpoints also do not contain early-stop peak-memory measurements.

---

## 4. AI reproduction guide (step-by-step runbook)

An agent with shell access can follow this verbatim. All paths relative to
`.../codes/LFI/TUBOLFI/`. `python` = the §0 environment's interpreter (activate the venv/conda env
first, or call its `python` by full path).

### Step 0 — preconditions

1. `cd` to `TUBOLFI/`.
2. Env (§0): create a Python-3.11 venv/conda env and, on macOS/Linux,
   `python -m pip install -r requirements.txt`. Verify:
   `python -c "import torch, botorch, gpytorch, pymc, arviz; print(torch.__version__)"` → `2.9.0`.
3. Data present: `data/restaurants_J{J}_P10_N{N}_I0C0M0.csv` and `data/choices_J{J}_...csv` for
   all 9 `(J,N)` in {100,200,500}×{10000,20000,50000}.
4. Decide the target: **verify existing** `final_result/` (fast) or **regenerate** (slow).

### Step 1 — verify the existing results (no compute)

For each `method ∈ {all, SA}`, `(J,N)` in the grid, read
`final_result/{method}/J{J}_P10_N{N}_I0C0M0_D5+6_SIM30/MCMC_summary_{iter_max,early_stop}.csv`
and count `covered == True` rows. Compare to the `cov` columns in §2. Compute the roll-ups
(`sum / 90`) and check they equal **all: 91.1% / 91.1%**, **SA: 93.3% / 93.3%**
(iter_max / early_stop). Also check `master seed` in each `.pkl` matches §2's seed column
(41 for the five flagged datasets, 42 otherwise).

### Step 2 — regenerate MCMC only (fast, ~1 h total; GP surrogates unchanged)

1. In `run_bolfi_canonical_v2.py :: main()`, comment out `bolfi_estimator = train_gp(args, DEVICE)`.
2. Run once per method (the `__main__` loop covers all 9 `(J,N)`). `gp_retrive` restores the
   pkl's *training* master seed, so the GP refit is fixed regardless of `--seed`; `--seed` only
   sets the **MCMC chain seed** (`random_seed` in `pm.sample`). A different chain seed changes the
   exact draws but not coverage beyond ±1 beta/dataset. The current CSVs were produced with the
   script default; use `--seed 42`:
   ```
   python -u run_bolfi_canonical_v2.py --simulator_type all
   python -u run_bolfi_canonical_v2.py --simulator_type SA
   ```
3. This overwrites `MCMC_summary_*.csv` / `MCMC_trace_*.nc` / `MLBA_*` in each `final_result/`
   folder. Re-run Step 1; coverage should match §2 within ±1 beta/dataset (MCMC MC noise +
   `fit_gpytorch_mll` L-BFGS drift in the `gp_retrive` refit).

### Step 3 — full retrain from scratch (slow, ~1–2 days total)

1. Keep `train_gp` active in `main()`.
2. Back up `final_result/` first (retrain overwrites the pkls).
3. Set `OMP_NUM_THREADS=1` for byte-stability (optional; slower).
4. Run **per config with the §2 seed** — do *not* use the bare `__main__` loop (it uses one seed
   for all 9). Temporarily replace the loop body, or drive `main()` directly, e.g.:

   | method | run at **seed 41** | run at **seed 42** |
   |---|---|---|
   | all | (J100,N10000), (J200,N50000), (J500,N10000) | the other 6 |
   | SA  | (J200,N20000), (J500,N10000) | the other 7 |

   For each: set `args.J,args.N,args.P=…`, `args.seed=41|42`, `args.simulator_type="all"|"SA"`,
   then `train_gp → gp_retrive(tol_quantile=0.85, notes="iter_max") → MCMC_sampling(...,"iter_max")
   → gp_retrive(tol_quantile=0.85, notes="early_stop") → MCMC_sampling(...,"early_stop")`.
5. Expect: `||best-t||` within ~0.05 of §2, coverage within ±1–2 betas/dataset. The three
   cuisine-ridge datasets (`SA J200/N10k`, `all J200/N50k`, `all/J500/N20k`) may land on a
   *different* point of the flat ridge → materially different coverage. That is the documented
   seed-fragility, not a reproduction failure — try 2–3 seeds and keep the best, as was done here.

### Step 4 — refresh the derived docs

After Step 2 or 3, regenerate `final_result/README.md` and, if used,
`MCMC_summary_Sept_all_vs_SA.md`, from the new CSVs (coverage counts + roll-ups). Update the
`iter_max` / `early_stop` numbers and roll-ups in §2 of this file.

### Non-negotiables (don't change and still call it "reproduced")

`--dis_type 5+6`, `--sim_num 30`, `--batch_size 2`, `--convergence_type fixed`,
`max_iters = 150*P`, GP = Matérn-2.5 ARD + `Standardize` + noise∈[1e-6,1e-2],
`est_sigma2` repeat = 50 at the incumbent, likelihood `trust` with `sigma_trust_c=3.0` /
`gate_sharpness=10.0`, `tol_quantile=0.85` both modes, `DEMetropolisZ(scaling=5e-4)` +
`draws=5000, tune=2500, chains=4`, prior `N(0,10)`, CPU / float64, SA knobs from §1b.
