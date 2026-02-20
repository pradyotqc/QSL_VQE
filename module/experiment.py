from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .ansatz import AnsatzSpec, hardware_efficient_ansatz
from .hamiltonian import TFIMMetadata, build_tfim_hamiltonian, set_global_seed
from .optimize import run_minimize
from .qsl import Trajectory, TrajectoryRecorder, QSLMetrics, compute_qsl_metrics


@dataclass
class OptimizerResult:
    method: str
    success: bool
    message: str
    nfev: int
    nit: int | None
    final_energy: float
    final_params: np.ndarray
    num_params: int
    ansatz_spec: AnsatzSpec
    trajectory: Trajectory
    metrics: QSLMetrics


@dataclass
class ExperimentConfig:
    n_qubits: int = 4
    j: float = 1.0
    h: float = 0.8
    boundary: str = "open"
    layers: int = 2
    entanglement: str = "linear"
    seed: int = 7
    random_couplings: bool = True
    disorder_scale: float = 0.1
    init_mode: str = "random"
    optimizers: Tuple[str, ...] = ("COBYLA", "Nelder-Mead", "L-BFGS-B")
    maxiter: int = 160


def generate_initial_params(mode: str, rng: np.random.Generator, num_params: int) -> np.ndarray:
    if mode == "zeros":
        return np.zeros(num_params, dtype=float)
    if mode == "random":
        return rng.uniform(0.0, 2 * np.pi, size=num_params)
    raise ValueError("init_mode must be 'zeros' or 'random'")


def build_hamiltonian(config: ExperimentConfig):
    rng = set_global_seed(config.seed)
    hamiltonian, metadata = build_tfim_hamiltonian(
        config.n_qubits,
        config.j,
        config.h,
        boundary=config.boundary,
        rng=rng,
        random_couplings=config.random_couplings,
        disorder_scale=config.disorder_scale,
        seed=config.seed,
    )
    return hamiltonian, metadata, rng


def run_optimizer(
    config: ExperimentConfig,
    hamiltonian,
    rng: np.random.Generator,
    *,
    method: str,
    init_params: np.ndarray | None = None,
) -> OptimizerResult:
    ansatz_spec = AnsatzSpec(
        n_qubits=config.n_qubits,
        layers=config.layers,
        entanglement=config.entanglement,
    )
    ansatz, params = hardware_efficient_ansatz(ansatz_spec)

    if init_params is None:
        init_params = generate_initial_params(config.init_mode, rng, len(params))
    else:
        init_params = np.asarray(init_params, dtype=float)

    recorder = TrajectoryRecorder(ansatz, params, hamiltonian)
    recorder.record(init_params)

    def objective(theta: np.ndarray) -> float:
        energy, _, _ = recorder._energy_delta_state(theta)
        return energy

    def callback(theta: np.ndarray) -> None:
        recorder.record(theta)

    result = run_minimize(
        objective,
        init_params,
        method=method,
        callback=callback,
        options={"maxiter": config.maxiter},
    )

    recorder.record(result.x)
    trajectory = recorder.to_trajectory()
    metrics = compute_qsl_metrics(trajectory)

    return OptimizerResult(
        method=method,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(getattr(result, "nfev", -1)),
        nit=getattr(result, "nit", None),
        final_energy=float(trajectory.energies[-1]),
        final_params=np.asarray(result.x, dtype=float),
        num_params=len(params),
        ansatz_spec=ansatz_spec,
        trajectory=trajectory,
        metrics=metrics,
    )
