# TFIM QSL Modular Pipeline

This repo modularizes the `qsl_check.ipynb` workflow and extends it to the transverse-field Ising model (TFIM) using a hardware-efficient ansatz. It saves a deterministic Hamiltonian, runs multiple optimizers, verifies the QSL relation, and stores all results in JSON with metadata. It also produces plots for QSL diagnostics and iteration count vs parameter count.

## Setup

Activate the provided virtual environment:

```bash
source ../qiskit2.x/bin/activate
```

## Run

```bash
python run_tfim_qsl.py
```

This will:
- Build a deterministic TFIM Hamiltonian (no random couplings by default)
- Run multiple optimizers with a shared initial parameter vector
- Compute QSL metrics (S0, S_N, N_min, prefix checks)
- Save metadata + arrays to JSON inside `result/seed_<seed>/`
- Generate plots

## Outputs

- Hamiltonian JSON: `hamiltonian/tfim_n4_j1.0_h0.8_open.json`
- Results JSON: `result/seed_7/tfim_qsl_results.json`
- Per-optimizer diagnostics plots: `result/seed_7/qsl_diagnostics_*.png`
- Sweep plot (iterations vs parameters): `result/seed_7/iter_vs_params_COBYLA.png`

## Project Layout

- `module/` core modules
- `hamiltonian/` saved Hamiltonians
- `result/seed_<seed>/` plots and JSON outputs grouped by seed
- `test/` minimal smoke test

## Configuration

Edit `run_tfim_qsl.py` to change:
- TFIM parameters: `n_qubits`, `j`, `h`, `boundary`
- Ansatz depth: `layers`, `entanglement`
- Randomization: `seed`, `random_couplings`, `disorder_scale`, `init_mode`
- Optimizers: `optimizers`, `maxiter`
- Parameter sweep: `sweep_optimizer`, `sweep_layers`

## Quick Smoke Test

```bash
python test/run_smoke.py
```
This uses a tiny TFIM instance to validate the pipeline quickly.
