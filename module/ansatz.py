from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


@dataclass
class AnsatzSpec:
    n_qubits: int
    layers: int
    entanglement: str = "linear"
    rotation_gates: Tuple[str, ...] = ("ry", "rz")


def hardware_efficient_ansatz(spec: AnsatzSpec) -> Tuple[QuantumCircuit, List[Parameter]]:
    if spec.layers < 1:
        raise ValueError("layers must be >= 1")
    if spec.entanglement not in {"linear", "circular", "full"}:
        raise ValueError("entanglement must be 'linear', 'circular', or 'full'")

    qc = QuantumCircuit(spec.n_qubits)
    params: List[Parameter] = []

    def add_entanglement() -> None:
        if spec.entanglement == "linear":
            for i in range(spec.n_qubits - 1):
                qc.cx(i, i + 1)
        elif spec.entanglement == "circular":
            for i in range(spec.n_qubits - 1):
                qc.cx(i, i + 1)
            if spec.n_qubits > 2:
                qc.cx(spec.n_qubits - 1, 0)
        else:
            for i in range(spec.n_qubits):
                for j in range(i + 1, spec.n_qubits):
                    qc.cx(i, j)

    for layer in range(spec.layers):
        for q in range(spec.n_qubits):
            for gate in spec.rotation_gates:
                param = Parameter(f"theta_{layer}_{q}_{gate}")
                params.append(param)
                getattr(qc, gate)(param, q)
        add_entanglement()

    return qc, params
