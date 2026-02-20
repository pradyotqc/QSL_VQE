from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .utils import save_json


@dataclass
class TFIMMetadata:
    model: str
    n_qubits: int
    j: float
    h: float
    boundary: str
    random_couplings: bool
    disorder_scale: float
    seed: int | None
    sign_convention: str
    j_list: List[float]
    h_list: List[float]


def set_global_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _pauli_label(n_qubits: int, positions: Dict[int, str]) -> str:
    label = ["I"] * n_qubits
    for idx, pauli in positions.items():
        label[idx] = pauli
    return "".join(label)


def build_tfim_hamiltonian(
    n_qubits: int,
    j: float,
    h: float,
    *,
    boundary: str = "open",
    rng: np.random.Generator | None = None,
    random_couplings: bool = False,
    disorder_scale: float = 0.1,
    seed: int | None = None,
) -> Tuple[SparsePauliOp, TFIMMetadata]:
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")

    if rng is None:
        rng = np.random.default_rng(seed)

    j_list = []
    h_list = []
    if random_couplings:
        for _ in range(n_qubits - (0 if boundary == "periodic" else 1)):
            j_list.append(float(j * (1.0 + disorder_scale * rng.standard_normal())))
        for _ in range(n_qubits):
            h_list.append(float(h * (1.0 + disorder_scale * rng.standard_normal())))
    else:
        for _ in range(n_qubits - (0 if boundary == "periodic" else 1)):
            j_list.append(float(j))
        for _ in range(n_qubits):
            h_list.append(float(h))

    terms: List[Tuple[str, float]] = []

    # Coupling: -J sum Z_i Z_{i+1}
    for i in range(n_qubits - 1):
        label = _pauli_label(n_qubits, {i: "Z", i + 1: "Z"})
        terms.append((label, -j_list[i]))

    if boundary == "periodic" and n_qubits > 2:
        label = _pauli_label(n_qubits, {n_qubits - 1: "Z", 0: "Z"})
        terms.append((label, -j_list[-1]))

    # Transverse field: -h sum X_i
    for i in range(n_qubits):
        label = _pauli_label(n_qubits, {i: "X"})
        terms.append((label, -h_list[i]))

    hamiltonian = SparsePauliOp.from_list(terms)

    metadata = TFIMMetadata(
        model="TFIM",
        n_qubits=n_qubits,
        j=float(j),
        h=float(h),
        boundary=boundary,
        random_couplings=random_couplings,
        disorder_scale=float(disorder_scale),
        seed=seed,
        sign_convention="H = -sum J_i Z_i Z_{i+1} - sum h_i X_i",
        j_list=j_list,
        h_list=h_list,
    )

    return hamiltonian, metadata


def save_hamiltonian_json(
    path: str | Path, hamiltonian: SparsePauliOp, metadata: TFIMMetadata
) -> None:
    data = {
        "paulis": [
            {"label": label, "coeff": float(np.real(coeff))}
            for label, coeff in hamiltonian.to_list()
        ],
        "metadata": metadata,
    }
    save_json(path, data)
