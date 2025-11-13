# AOD_case_study# STGP: State-Space Filtering for Convolution-Generated Spatio-Temporal GP

This repo provides two frequency-domain state-space filters for spatio-temporal fields on a regular 2-D grid:
- **Proposed model** (Galerkin + Euler, derivative-aware): `src/train_proposed.py`
- **Conventional baseline** (augmented α/β with spectral transition): `src/train_conventional.py`

Both ingest a time–space matrix CSV (rows = time, columns = flattened grid), and output filtered fields, modal coefficients, and logs in a `runs/` directory.

## Environment

Tested with:
- Python ≥ 3.9
- PyTorch ≥ 2.1 (CPU or CUDA)
- NumPy ≥ 1.23
- pandas ≥ 2.0
- Matplotlib ≥ 3.7

Install:
```bash
pip install -r requirements.txt
```
*The code uses double precision (float64). A CUDA GPU is recommended but not required.*

## Data

Place your CSV at:
```
data/csv/
```
- Shape: **[T, 2500]** (50×50 grid flattened row-wise).
- Time step (`--dt`): typically **0.1** (adjust as needed).


## Proposed Model

```bash
python -m src.train_proposed \
  --csv data/csv/la_aod_50x50_patches_2.csv \
  --run_dir runs/proposed_aod \
  --dt 0.1 \
  --grid_n 50 \
  --J1 20 --J2 20 \
  --R 0.0025 \
  --iters 3000 --lr 1e-2
```

**Artifacts (under `--run_dir`):**
- `theta_raw.pt` — learned raw parameter vector (PyTorch tensor)
- `meta.json` — run config & CLI args used for this run
- `proposed_outputs.pt` — a dict of torch tensors:
  - `x_recon`: reconstructed field (T × 2500)
  - `xdot_recon`: derivative field (T × 2500)
  - `alpha_* / beta_*`: modal components & magnitudes
  - `Omega1_list`, `Omega2_list`: exact mode lists used by the run

**Notes**
- Uses **chunked measurement updates** (default chunk size 625 for 50×50) to control memory; this is equivalent to a full update under white measurement noise.
- Uses **robust Cholesky with jitter** for numerical stability.

## Conventional Baseline

```bash
python -m src.train_conventional \
  --csv data/csv/la_aod_50x50_patches_2.csv \
  --run_dir runs/conv_aod \
  --dt 0.1 \
  --grid_n 50 \
  --J1 30 --J2 30 \
  --iters 600 --lr 1e-2 \
  --q_alpha 1e-4 --q_beta 1e-6 --r_meas 1.0 \
  --zeta 0.1 --v1 0.0 --v2 0.0 --diff1 0.01 --diff2 0.01
```

**Typical outputs:**
- `alpha_estimates.csv`, `beta_estimates.csv`
- `alpha_field_estimated.csv`, `beta_field_estimated.csv`
- `K_list.json` — the mode list used during training (consumed by the plotting script)


## Plotting

### Conventional CSVs
```bash
python -m src.stgp.plotting.plot_alpha_beta_all \
  --alpha_csv alpha_estimates.csv \
  --beta_csv  beta_estimates.csv \
  --run_dir   runs/conv_aod \
  --out_dir   plots_alpha_beta \
  --vline_idx 9
```
- Creates per‑mode time series under `plots_alpha_beta/`.
- Writes `plots_alpha_beta/modes_index.csv` to map mode index → `(k1,k2)`.

