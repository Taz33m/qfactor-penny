"""Qiskit circuit utilities for the frozen QFactor-Penny QNN."""

from __future__ import annotations

from typing import Any

import numpy as np


def require_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("Qiskit is required for hardware audit utilities. Install `qfactor-penny[hardware]`.") from exc


def build_qiskit_qnn_circuit(features: list[float] | np.ndarray, weights: list[Any] | np.ndarray):
    """Build the Qiskit circuit matching the PennyLane MVP QNN ansatz."""

    require_qiskit()
    import pennylane as qml
    from pennylane.wires import Wires
    from qiskit import QuantumCircuit

    feature_array = np.asarray(features, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if feature_array.shape != (4,):
        raise ValueError("The hardware QNN circuit expects exactly 4 feature values.")
    if weight_array.shape != (1, 4, 3):
        raise ValueError("The hardware QNN circuit expects weights with shape (1, 4, 3).")

    circuit = QuantumCircuit(4)
    angle_ops = qml.AngleEmbedding.compute_decomposition(feature_array, wires=Wires(range(4)), rotation=qml.RY)
    ranges = [1]
    entangling_ops = qml.StronglyEntanglingLayers.compute_decomposition(
        weight_array,
        wires=Wires(range(4)),
        ranges=ranges,
        imprimitive=qml.CNOT,
    )
    for op in [*angle_ops, *entangling_ops]:
        _append_pennylane_op(circuit, op)
    return circuit


def qiskit_pauli_z0_observable(num_qubits: int = 4):
    require_qiskit()
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)


def qiskit_statevector_score(
    features: list[float] | np.ndarray,
    weights: list[Any] | np.ndarray,
    bias: float = 0.0,
) -> float:
    """Return the local Qiskit Statevector expectation score plus bias."""

    require_qiskit()
    from qiskit.quantum_info import Statevector

    circuit = build_qiskit_qnn_circuit(features, weights)
    observable = qiskit_pauli_z0_observable(circuit.num_qubits)
    expectation = Statevector.from_instruction(circuit).expectation_value(observable)
    return float(np.real(expectation)) + float(bias)


def _append_pennylane_op(circuit, op) -> None:
    name = op.name
    wires = [int(wire) for wire in op.wires]
    params = [float(param) for param in op.parameters]
    if name == "RY":
        circuit.ry(params[0], wires[0])
        return
    if name == "RZ":
        circuit.rz(params[0], wires[0])
        return
    if name == "RX":
        circuit.rx(params[0], wires[0])
        return
    if name == "Rot":
        phi, theta, omega = params
        circuit.rz(phi, wires[0])
        circuit.ry(theta, wires[0])
        circuit.rz(omega, wires[0])
        return
    if name in {"CNOT", "CX"}:
        circuit.cx(wires[0], wires[1])
        return
    raise ValueError(f"Unsupported PennyLane operation for Qiskit export: {name}")
