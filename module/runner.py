from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .ansatz import AnsatzSpec, hardware_efficient_ansatz
from .experiment import ExperimentConfig, run_optimizer, build_hamiltonian, generate_initial_params
from .hamiltonian import save_hamiltonian_json
from .plotting import plot_param_sweep, plot_qsl_diagnostics
from .utils import ensure_dir, save_json


DEFAULT_SWEEP_LAYERS = (1, 2, 3, 4)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_parameter_sweep_for_optimizer(
    config: ExperimentConfig,
    hamiltonian,
    seed_output_dir: Path,
    *,
    sweep_method: str,
    sweep_layers: Tuple[int, ...],
) -> Dict:
    sweep_results: List[Dict] = []
    for layer in sweep_layers:
        cfg_layer = replace(config, layers=int(layer))
        rng_layer = np.random.default_rng(config.seed + 2000 + int(layer))
        ansatz_spec_layer = AnsatzSpec(
            n_qubits=cfg_layer.n_qubits,
            layers=cfg_layer.layers,
            entanglement=cfg_layer.entanglement,
        )
        _, params_layer = hardware_efficient_ansatz(ansatz_spec_layer)
        init_params = generate_initial_params(
            cfg_layer.init_mode, rng_layer, len(params_layer)
        )

        result = run_optimizer(
            cfg_layer,
            hamiltonian,
            rng_layer,
            method=sweep_method,
            init_params=init_params,
        )

        actual_iterations = len(result.trajectory.params) - 1
        sweep_results.append(
            {
                "layers": int(layer),
                "num_params": result.num_params,
                "actual_iterations": actual_iterations,
                "qsl_iterations": result.metrics.n_min,
                "final_relation_holds": result.metrics.final_relation_holds,
                "prefix_relation_holds": result.metrics.prefix_relation_holds,
            }
        )

    sweep_plot_path = seed_output_dir / f"iter_vs_params_{sweep_method.replace(' ', '_')}.png"
    plot_param_sweep(sweep_results, sweep_plot_path, optimizer_name=sweep_method)

    return {
        "optimizer": sweep_method,
        "layers": list(sweep_layers),
        "results": sweep_results,
        "plot": str(sweep_plot_path),
    }


def run_tfim_experiment(
    config: ExperimentConfig,
    *,
    output_dir: str | Path = "result",
    hamiltonian_dir: str | Path = "hamiltonian",
    sweep_optimizer: str | None = None,
    sweep_optimizers: Tuple[str, ...] | None = None,
    sweep_layers: Tuple[int, ...] = DEFAULT_SWEEP_LAYERS,
) -> Dict:
    output_root = ensure_dir(output_dir)
    seed_output_dir = ensure_dir(Path(output_root) / f"seed_{config.seed}")
    hamiltonian_dir = ensure_dir(hamiltonian_dir)

    hamiltonian, h_metadata, _ = build_hamiltonian(config)

    ham_name = (
        f"tfim_n{config.n_qubits}_j{config.j}_h{config.h}_"
        f"{config.boundary}.json"
    )
    ham_path = Path(hamiltonian_dir) / ham_name
    save_hamiltonian_json(ham_path, hamiltonian, h_metadata)

    summary: Dict = {
        "timestamp_utc": _timestamp(),
        "config": asdict(config),
        "result_dir": str(seed_output_dir),
        "hamiltonian_file": str(ham_path),
        "hamiltonian_metadata": asdict(h_metadata),
        "optimizer_runs": {},
        "sweeps": [],
        "sweep": None,
    }

    # Use a separate RNG stream for initial parameters, to keep results reproducible.
    base_rng = np.random.default_rng(config.seed + 1234)

    # Build the ansatz once to determine parameter count.
    ansatz_spec = AnsatzSpec(
        n_qubits=config.n_qubits,
        layers=config.layers,
        entanglement=config.entanglement,
    )
    _, params = hardware_efficient_ansatz(ansatz_spec)
    num_params_main = len(params)
    init_params_main = generate_initial_params(config.init_mode, base_rng, num_params_main)

    # Main optimizer runs with fixed ansatz depth.
    for method in config.optimizers:
        result = run_optimizer(
            config,
            hamiltonian,
            base_rng,
            method=method,
            init_params=init_params_main,
        )

        plot_qsl_diagnostics(
            result.trajectory,
            result.metrics,
            seed_output_dir / f"qsl_diagnostics_{method.replace(' ', '_')}.png",
            title_suffix=f"TFIM | {method} | params={result.num_params}",
        )

        summary["optimizer_runs"][method] = {
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "nit": result.nit,
            "final_energy": result.final_energy,
            "final_params": result.final_params,
            "num_params": result.num_params,
            "ansatz_spec": asdict(result.ansatz_spec),
            "init_params": init_params_main,
            "qsl": {
                "s0": result.metrics.s0,
                "s_n_final": float(result.metrics.s_n[-1]),
                "final_relation_holds": result.metrics.final_relation_holds,
                "prefix_relation_holds": result.metrics.prefix_relation_holds,
                "n_min": result.metrics.n_min,
                "n_min_raw": result.metrics.n_min_raw,
            },
            "arrays": {
                "params": result.trajectory.params,
                "energies": result.trajectory.energies,
                "delta_e": result.trajectory.delta_e,
                "raw_dt": result.metrics.raw_dt,
                "dt_star": result.metrics.dt_star,
                "s_n": result.metrics.s_n,
                "s_raw": result.metrics.s_raw,
                "geo_prefix": result.metrics.geo_prefix,
            },
        }

    # Parameter sweeps.
    if sweep_optimizers is not None:
        sweep_methods = tuple(dict.fromkeys(sweep_optimizers))
    elif sweep_optimizer is not None:
        sweep_methods = (sweep_optimizer,)
    else:
        sweep_methods = (config.optimizers[0],) if config.optimizers else ()

    if sweep_methods:
        sweeps = [
            run_parameter_sweep_for_optimizer(
                config,
                hamiltonian,
                seed_output_dir,
                sweep_method=method,
                sweep_layers=sweep_layers,
            )
            for method in sweep_methods
        ]
        summary["sweeps"] = sweeps
        if len(sweeps) == 1:
            summary["sweep"] = sweeps[0]

    results_path = seed_output_dir / "tfim_qsl_results.json"
    save_json(results_path, summary)

    return summary
