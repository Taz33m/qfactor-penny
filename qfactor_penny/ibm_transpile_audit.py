"""Transpile frozen QNN circuits against an IBM Quantum backend."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from .hardware_utils import (
    flattened_samples,
    ibm_credentials_available,
    load_frozen_qnn_subset,
    load_env_file,
    qiskit_available,
    qiskit_runtime_available,
    scrub_error_message,
    skipped_rows,
    write_rows,
)
from .qiskit_qnn import build_qiskit_qnn_circuit


def run_transpile_audit(
    *,
    input_path: str | Path,
    output: str | Path,
    backend_name: str | None = None,
    optimization_level: int = 1,
) -> list[dict[str, Any]]:
    load_env_file()
    artifact = load_frozen_qnn_subset(input_path)
    if not qiskit_available():
        rows = skipped_rows(
            artifact,
            status="skipped",
            error_type="QiskitUnavailable",
            error_message="Qiskit is not installed; install `qfactor-penny[hardware]`.",
            extra={"backend_name": backend_name or "", "optimization_level": optimization_level},
        )
        write_rows(rows, output)
        return rows
    if not qiskit_runtime_available():
        rows = skipped_rows(
            artifact,
            status="skipped",
            error_type="QiskitIBMRuntimeUnavailable",
            error_message="qiskit-ibm-runtime is not installed; install `qfactor-penny[hardware]`.",
            extra={"backend_name": backend_name or "", "optimization_level": optimization_level},
        )
        write_rows(rows, output)
        return rows
    if not ibm_credentials_available():
        rows = skipped_rows(
            artifact,
            status="skipped",
            error_type="MissingCredentials",
            error_message="QISKIT_IBM_TOKEN is not set; rotate the exposed key and export the new token locally.",
            extra={"backend_name": backend_name or "", "optimization_level": optimization_level},
        )
        write_rows(rows, output)
        return rows

    try:
        backend = _load_backend(backend_name)
        rows = _transpile_rows(artifact, backend, optimization_level=optimization_level)
    except Exception as exc:  # pragma: no cover - depends on live IBM service
        rows = skipped_rows(
            artifact,
            status="failed",
            error_type=type(exc).__name__,
            error_message=scrub_error_message(str(exc)),
            extra={"backend_name": backend_name or "", "optimization_level": optimization_level},
        )
    write_rows(rows, output)
    return rows


def _load_backend(backend_name: str | None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs = {"channel": "ibm_quantum_platform", "token": os.environ["QISKIT_IBM_TOKEN"]}
    instance = os.environ.get("QISKIT_IBM_INSTANCE")
    if instance:
        kwargs["instance"] = instance
    service = QiskitRuntimeService(**kwargs)
    if backend_name:
        return service.backend(backend_name)
    try:
        return service.least_busy(operational=True, simulator=False, min_num_qubits=4)
    except TypeError:  # pragma: no cover - SDK compatibility
        backends = service.backends(simulator=False, operational=True)
        usable = [backend for backend in backends if getattr(backend, "num_qubits", 0) >= 4]
        if not usable:
            raise RuntimeError("No operational IBM backend with at least 4 qubits was available.")
        return usable[0]


def _transpile_rows(artifact: dict[str, Any], backend, *, optimization_level: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples = flattened_samples(artifact)
    backend_name = getattr(backend, "name", None)
    if callable(backend_name):
        backend_name = backend_name()
    for sample in samples:
        start = time.perf_counter()
        try:
            circuit = build_qiskit_qnn_circuit(sample["feature_values"], sample["weights"])
            transpiled = _transpile(circuit, backend, optimization_level=optimization_level)
            rows.append(
                {
                    "status": "success",
                    "error_type": "",
                    "error_message": "",
                    "backend_name": backend_name,
                    "backend_num_qubits": getattr(backend, "num_qubits", ""),
                    "backend_basis_gates": ",".join(_basis_gates(backend)),
                    "backend_calibration_timestamp": _calibration_timestamp(backend),
                    "optimization_level": int(optimization_level),
                    "model_id": sample["model_id"],
                    "split_id": sample["split_id"],
                    "seed": sample["seed"],
                    "date": sample["date"],
                    "ticker": sample["ticker"],
                    "sample_index": sample["sample_index"],
                    "pre_depth": int(circuit.depth()),
                    "post_depth": int(transpiled.depth()),
                    "pre_two_qubit_gate_count": _two_qubit_gate_count(circuit),
                    "post_two_qubit_gate_count": _two_qubit_gate_count(transpiled),
                    "post_swap_count": int(transpiled.count_ops().get("swap", 0)),
                    "post_cx_count": int(transpiled.count_ops().get("cx", 0)),
                    "physical_layout": _layout_summary(transpiled),
                    "runtime_seconds": time.perf_counter() - start,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": scrub_error_message(str(exc)),
                    "backend_name": backend_name,
                    "optimization_level": int(optimization_level),
                    "model_id": sample["model_id"],
                    "split_id": sample["split_id"],
                    "seed": sample["seed"],
                    "date": sample["date"],
                    "ticker": sample["ticker"],
                    "sample_index": sample["sample_index"],
                    "runtime_seconds": time.perf_counter() - start,
                }
            )
    return rows


def _transpile(circuit, backend, *, optimization_level: int):
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=int(optimization_level))
        return pass_manager.run(circuit)
    except Exception:
        from qiskit import transpile

        return transpile(circuit, backend=backend, optimization_level=int(optimization_level))


def _two_qubit_gate_count(circuit) -> int:
    return int(sum(1 for instruction in circuit.data if getattr(instruction.operation, "num_qubits", 0) == 2))


def _basis_gates(backend) -> list[str]:
    try:
        target = getattr(backend, "target", None)
        if target is not None:
            names = getattr(target, "operation_names", None)
            if names:
                return sorted(str(name) for name in names)
    except Exception:
        pass
    try:
        config = backend.configuration()
        return sorted(str(name) for name in getattr(config, "basis_gates", []) or [])
    except Exception:
        return []


def _calibration_timestamp(backend) -> str:
    try:
        props = backend.properties()
        value = getattr(props, "last_update_date", "")
        return str(value) if value else ""
    except Exception:
        return ""


def _layout_summary(circuit) -> str:
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return ""
    return scrub_error_message(str(layout))[:500]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Frozen QNN subset JSON.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--backend", default=None, help="Optional IBM backend name. Defaults to least busy 4+ qubit backend.")
    parser.add_argument("--optimization-level", type=int, default=1, choices=[0, 1, 2, 3])
    args = parser.parse_args(argv)
    run_transpile_audit(
        input_path=args.input,
        output=args.output,
        backend_name=args.backend,
        optimization_level=args.optimization_level,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
