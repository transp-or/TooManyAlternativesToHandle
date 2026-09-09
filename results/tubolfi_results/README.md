# final_result — one canonical run per (method, dataset)

Consolidated best-seed run per dataset. `run_bolfi_canonical_v2.py` reads/writes
`final_result/{simulator_type}/{data_id}/` directly. MCMC likelihood `llk_mode=trust`,
`tol_quantile=0.85` for both retrievals (see `../reproducible_setting.md`);
`DEMetropolisZ` with `tune=2500, draws=5000, chains=4`.

| method | J | N | seed | iter_max cov | early_stop cov |
|--------|---|---|------|--------------|----------------|
| all | 100 | 10k | 41 | 10/10 | 9/10 |
| all | 100 | 20k | 42 | 10/10 | 10/10 |
| all | 100 | 50k | 42 | 10/10 | 10/10 |
| all | 200 | 10k | 42 | 10/10 | 10/10 |
| all | 200 | 20k | 42 | 9/10 | 9/10 |
| all | 200 | 50k | 41 | 6/10 | 6/10 |
| all | 500 | 10k | 41 | 9/10 | 10/10 |
| all | 500 | 20k | 42 | 8/10 | 8/10 |
| all | 500 | 50k | 42 | 10/10 | 10/10 |
| SA | 100 | 10k | 42 | 10/10 | 10/10 |
| SA | 100 | 20k | 42 | 10/10 | 10/10 |
| SA | 100 | 50k | 42 | 9/10 | 9/10 |
| SA | 200 | 10k | 42 | 6/10 | 6/10 |
| SA | 200 | 20k | 41 | 10/10 | 10/10 |
| SA | 200 | 50k | 42 | 10/10 | 10/10 |
| SA | 500 | 10k | 41 | 10/10 | 10/10 |
| SA | 500 | 20k | 42 | 10/10 | 10/10 |
| SA | 500 | 50k | 42 | 9/10 | 9/10 |

**all / iter_max: 82/90 = 91.1%**

**all / early_stop: 82/90 = 91.1%**

**SA / iter_max: 84/90 = 93.3%**

**SA / early_stop: 84/90 = 93.3%**
