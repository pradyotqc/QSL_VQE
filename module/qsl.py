from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector


@dataclass
class Trajectory:
    params: np.ndarray
    energies: np.ndarray
    delta_e: np.ndarray
    states: List[Statevector]


@dataclass
class QSLMetrics:
    raw_dt: np.ndarray
    dt_star: np.ndarray
    terms: np.ndarray
    s_n: np.ndarray
    s_raw: np.ndarray
    s0: float
    geo_prefix: np.ndarray
    final_relation_holds: bool
    prefix_relation_holds: bool
    n_min: int | None
    n_min_raw: int | None


class TrajectoryRecorder:
    def __init__(
        self,
        circuit,
        param_list: Sequence,
        hamiltonian: SparsePauliOp,
        *,
        dedupe_tol: float = 1e-12,
    ) -> None:
        self.circuit = circuit
        self.param_list = list(param_list)
        self.hamiltonian = hamiltonian
        self.hamiltonian_sq = (hamiltonian @ hamiltonian).simplify()
        self.dedupe_tol = dedupe_tol

        self.params: List[np.ndarray] = []
        self.energies: List[float] = []
        self.delta_e: List[float] = []
        self.states: List[Statevector] = []

    def _state_from_params(self, params: Sequence[float]) -> Statevector:
        binding = {p: float(v) for p, v in zip(self.param_list, params)}
        bound = self.circuit.assign_parameters(binding, inplace=False)
        return Statevector.from_instruction(bound)

    def _energy_delta_state(self, params: Sequence[float]):
        state = self._state_from_params(params)
        energy = float(np.real(state.expectation_value(self.hamiltonian)))
        energy_sq = float(np.real(state.expectation_value(self.hamiltonian_sq)))
        var = max(0.0, energy_sq - energy ** 2)
        delta_e = float(np.sqrt(var))
        return energy, delta_e, state

    def record(self, params: Sequence[float]) -> bool:
        params_arr = np.asarray(params, dtype=float)
        if self.params and np.linalg.norm(params_arr - self.params[-1]) < self.dedupe_tol:
            return False
        energy, delta_e, state = self._energy_delta_state(params_arr)
        self.params.append(params_arr.copy())
        self.energies.append(energy)
        self.delta_e.append(delta_e)
        self.states.append(state)
        return True

    def to_trajectory(self) -> Trajectory:
        return Trajectory(
            params=np.asarray(self.params),
            energies=np.asarray(self.energies),
            delta_e=np.asarray(self.delta_e),
            states=self.states,
        )


def compute_qsl_metrics(traj: Trajectory, *, hbar: float = 1.0) -> QSLMetrics:
    states = traj.states
    if len(states) < 2:
        raise ValueError("Need at least one optimization step to evaluate the discrete QSL sum.")

    params = traj.params
    delta_e = traj.delta_e

    n_steps = len(states) - 1

    raw_dt = np.linalg.norm(params[1:] - params[:-1], axis=1)
    speed = 2.0 * delta_e[1:] / hbar

    fs_step = np.zeros(n_steps)
    dt_star = np.zeros(n_steps)
    terms = np.zeros(n_steps)

    for k in range(n_steps):
        overlap = np.clip(abs(np.vdot(states[k].data, states[k + 1].data)), 0.0, 1.0)
        ds_k = float(np.arccos(overlap))
        fs_step[k] = ds_k

        if speed[k] > 1e-12:
            dt_star[k] = ds_k / speed[k]
            terms[k] = speed[k] * dt_star[k]
        else:
            dt_star[k] = 0.0 if ds_k < 1e-12 else np.inf
            terms[k] = np.nan

    s_n = np.cumsum(np.nan_to_num(terms, nan=0.0))
    terms_raw = speed * raw_dt
    s_raw = np.cumsum(terms_raw)

    ov_0f = np.clip(abs(np.vdot(states[0].data, states[-1].data)), 0.0, 1.0)
    s0 = float(np.arccos(ov_0f))

    geo_prefix = np.array(
        [
            np.arccos(np.clip(abs(np.vdot(states[0].data, states[k].data)), 0.0, 1.0))
            for k in range(1, len(states))
        ]
    )

    final_relation_holds = bool(s_n[-1] + 1e-10 >= s0)
    prefix_relation_holds = bool(np.all(s_n + 1e-10 >= geo_prefix))

    cross = np.where(s_n >= s0 - 1e-10)[0]
    n_min = int(cross[0] + 1) if cross.size else None

    cross_raw = np.where(s_raw >= s0 - 1e-10)[0]
    n_min_raw = int(cross_raw[0] + 1) if cross_raw.size else None

    return QSLMetrics(
        raw_dt=raw_dt,
        dt_star=dt_star,
        terms=terms,
        s_n=s_n,
        s_raw=s_raw,
        s0=s0,
        geo_prefix=geo_prefix,
        final_relation_holds=final_relation_holds,
        prefix_relation_holds=prefix_relation_holds,
        n_min=n_min,
        n_min_raw=n_min_raw,
    )
