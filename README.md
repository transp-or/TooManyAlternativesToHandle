# Too Many Alternatives to Handle

This repository contains two approaches for Bayesian estimation of large
discrete-choice models: an Adaptive Sampler based on sampled choice sets, and
TurboLFI/BOLFI, a likelihood-free inference approach. Each method has its own
environment and run instructions below.

## Authors and supervision

- Anne-Valérie Preto — [anne-valerie.preto@epfl.ch](mailto:anne-valerie.preto@epfl.ch)
- Xinwei Li — [xinwei.li@u.nus.edu](mailto:xinwei.li@u.nus.edu)

Supervised by Prof. Michel Bierlaire and Prof. Prateek Bansal.

# Adaptive Sampler

This repository contains a Adaptive Sampler implementation for discrete choice models using PyMC and a custom adaptive alternative sampling step.

## 1. Environment Setup

To ensure reproducibility (matching the Scitas computation environment), create a Conda environment using the provided requirements:

```bash
# 1. Create the environment
conda create -n adasampler python=3.11 -y
conda activate adasampler

# 2. From the repository root, install dependencies
pip install -r requirements_repro.txt
```

## 2. Data Configuration

The sampler expects CSV files following the naming scheme:
`choices_J{J}_P{P}_N{N/1000}k_I{I}C{C}M{M}.csv` (for example,
`choices_J100_P10_N10k_I0C0M0.csv`).

- **Bundled path**: `data/data/final_2310/`
- **Change path**: Pass `data_dir` when calling `load_data()` from Python, or add a
  directory to `candidate_dirs` in `adaptive_sampler/data.py`.

## 3. Running the Model

Run the sampler from the `adaptive_sampler/` directory (this also keeps the log
and default `output/` directory together):

```bash
python run_model.py \
  --sample_size 2 \
  --L 5 \
  --m 5 \
  --n_steps 10000 \
  --burnin 0 \
  --tune 1000 \
  --seed 42 \
  --J 100 \
  --N 10000 \
  --P 10 \
  --I 0 \
  --C 0 \
  --M 0 \
  --outdir output \
  --scaling scale \
  --rescale_trace \
  --output_suffix "_260603"
```

### Parameter Reference

| Parameter | Meaning |
| :--- | :--- |
| `sample_size` | Number of alternatives in the sampled set (must be 2 for this protocol). |
| `L` | MTM Parameter: Number of proposals for the first stage. |
| `m` | MTM Parameter: Number of proposals for the second stage. |
| `n_steps` | Number of MCMC draws to keep. |
| `burnin` | Number of initial draws to discard (metadata only). |
| `tune` | Number of tuning/warmup steps for PyMC. |
| `seed` | Random seed for reproducibility. |
| `J`, `N`, `P` | Data config: Alternatives (100-500), Observations (10k-50k), Parameters (10-40). |
| `I`, `C`, `M` | Data config: Imbalance, Correlation, Randomness (0 or 1). |
| `outdir` | Directory where `.nc` traces and `.json` reports are saved. |
| `scaling` | Scaling mode (`scale` or `normalize`) to improve convergence. |
| `rescale_trace` | Automatically convert parameters back to original units before saving. |
| `output_suffix` | String appended to output filenames for versioning. |

## 4. Outputs

Results are saved in the defined `--outdir`:
- `trace_...nc`: The posterior samples.
- `reproducibility_info_...json`: System info and validation logs.
- `scale_params_...json`: Parameters used for feature scaling.


# TurboLFI / BOLFI

This directory contains the CPU implementation of TurboLFI/BOLFI for the
synthetic multinomial-logit experiments.  The entry point is
`run_bolfi_canonical_v2.py`; there is no `run_bolfi_canonical.py` in this
repository.

## Environment

TurboLFI has its own Python 3.11 environment.  Create it from the repository
root so that it stays separate from the adaptive-sampler environment:

```bash
conda create -y -n turbolfi python=3.11
conda activate turbolfi
python -m pip install --upgrade pip
python -m pip install -r TUBOLFI/requirements.txt
```

Use `requirements.txt` on macOS and Linux.  `requirements-lock.txt` is a full
package snapshot from the Windows/Intel machine used for the recorded results;
it includes platform-specific packages such as `intel-cmplr-lib-ur`, which are
not published for macOS and therefore cannot be installed there.  The shorter
`requirements.txt` contains the portable direct dependencies required by the
runner.

Verify the installation before submitting a long run:

```bash
cd TUBOLFI
python -c "import torch, botorch, gpytorch, pymc, arviz; print(torch.__version__)"
python run_bolfi_canonical_v2.py --help
```

## Data and outputs

For `--J 100 --N 10000 --P 10`, the runner reads these bundled files:

```text
data/restaurants_J100_P10_N10k_I0C0M0.csv
data/choices_J100_P10_N10k_I0C0M0.csv
```

It writes results directly below:

```text
final_result/SA/J100_P10_N10k_I0C0M0_D5+6_SIM30/
```

The default simulator is `SA`.  A full run trains the GP and then runs two
four-chain PyMC MCMC passes (`early_stop` and `iter_max` when convergence is
`fixed`).  It overwrites checkpoints and MCMC artifacts for the selected
configuration.  Use a separate copy of `final_result/` if its checked-in
results must be retained.

## Run on Slurm

After activating the `turbolfi` environment and changing to `TUBOLFI/`, the
requested command is:

```bash
srun /usr/bin/time -v python run_bolfi_canonical_v2.py --J 100 --N 10000 --P 10
```

For an explicitly reproducible training seed, use:

```bash
srun /usr/bin/time -v python run_bolfi_canonical_v2.py --J 100 --N 10000 --P 10 --seed 42
```

`script_Bolfi.run` is the corresponding batch-script template.  Before
`sbatch`, set its `#SBATCH --chdir`, virtual-environment activation path, and
`#SBATCH --output` path to locations on the target cluster.  The file now calls
the existing `_v2` runner and only uses supported command-line flags.

Useful options are `--simulator_type {SA,all}`, `--seed`, and
`--convergence_type {fixed,auto}`.  The full fixed run is computationally
substantial: its BO budget is `150 * P` iterations and it subsequently performs
two MCMC jobs with 4 chains, 2,500 tuning draws, and 5,000 retained draws each.
See [reproducible_setting.md](reproducible_setting.md) for the canonical
settings and retained-result summary.
