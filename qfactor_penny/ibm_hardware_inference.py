"""Run inference-only frozen QNN circuits through IBM Runtime Estimator."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

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
from .ibm_transpile_audit import _basis_gates, _calibration_timestamp, _load_backend, _transpile
from .qiskit_qnn import build_qiskit_qnn_circuit, qiskit_pauli_z0_observable


def run_hardware_inference(
    *,
    input_path: str | Path,
    output: str | Path,
    backend_name: str | None = None,
    optimization_level: int = 1,
    resilience_levels: list[int] | None = None,
    precision: float | None = None,
) -> list[dict[str, Any]]:
    load_env_file()
    artifact = load_frozen_qnn_subset(input_path)
    resilience_levels = [0] if resilience_levels is None else [int(level) for level in resilience_levels]
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
        rows = []
        for level in resilience_levels:
            rows.extend(
                _estimator_rows(
                    artifact,
                    backend,
                    optimization_level=optimization_level,
                    resilience_level=level,
                    precision=precision,
                )
            )
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


def _estimator_rows(
    artifact: dict[str, Any],
    backend,
    *,
    optimization_level: int,
    resilience_level: int,
    precision: float | None,
) -> list[dict[str, Any]]:
    Estimator = _estimator_class()
    backend_name = getattr(backend, "name", None)
    if callable(backend_name):
        backend_name = backend_name()
    samples = flattened_samples(artifact)
    circuits = []
    observables = []
    for sample in samples:
        circuit = build_qiskit_qnn_circuit(sample["feature_values"], sample["weights"])
        isa_circuit = _transpile(circuit, backend, optimization_level=optimization_level)
        observable = qiskit_pauli_z0_observable(isa_circuit.num_qubits)
        try:
            observable = observable.apply_layout(isa_circuit.layout)
        except Exception:
            pass
        circuits.append(isa_circuit)
        observables.append(observable)

    estimator = _make_estimator(Estimator, backend, resilience_level=resilience_level, precision=precision)
    pubs = list(zip(circuits, observables))
    start = time.perf_counter()
    job = _run_estimator(estimator, pubs, precision=precision)
    result = job.result()
    runtime_seconds = time.perf_counter() - start
    job_id = _job_id(job)
    pub_results = list(result)
    rows = []
    for sample, pub_result in zip(samples, pub_results):
        expectation, std, shots = _extract_pub_result(pub_result)
        hardware_score = expectation + float(sample["bias"])
        rows.append(
            {
                "status": "success",
                "error_type": "",
                "error_message": "",
                "backend_name": backend_name,
                "backend_num_qubits": getattr(backend, "num_qubits", ""),
                "backend_basis_gates": ",".join(_basis_gates(backend)),
                "backend_calibration_timestamp": _calibration_timestamp(backend),
                "job_id": job_id,
                "optimization_level": int(optimization_level),
                "resilience_level": int(resilience_level),
                "requested_precision": precision if precision is not None else "",
                "shots": shots,
                "model_id": sample["model_id"],
                "split_id": sample["split_id"],
                "seed": sample["seed"],
                "date": sample["date"],
                "ticker": sample["ticker"],
                "sample_index": sample["sample_index"],
                "analytic_score": float(sample["analytic_score"]),
                "hardware_expectation": expectation,
                "hardware_expectation_std": std,
                "bias": float(sample["bias"]),
                "hardware_score": hardware_score,
                "score_diff": hardware_score - float(sample["analytic_score"]),
                "runtime_seconds": runtime_seconds,
            }
        )
    return rows


def _estimator_class():
    import qiskit_ibm_runtime as runtime

    estimator = getattr(runtime, "EstimatorV2", None)
    if estimator is not None:
        return estimator
    estimator = getattr(runtime, "Estimator", None)
    if estimator is None:
        raise RuntimeError("qiskit-ibm-runtime does not expose an Estimator primitive.")
    return estimator


def _make_estimator(Estimator, backend, *, resilience_level: int, precision: float | None):
    try:
        estimator = Estimator(mode=backend)
    except TypeError:  # pragma: no cover - SDK compatibility
        estimator = Estimator(backend=backend)
    for path, value in [
        (("resilience_level",), resilience_level),
        (("default_precision",), precision),
    ]:
        if value is None:
            continue
        try:
            target = estimator.options
            for attribute in path[:-1]:
                target = getattr(target, attribute)
            setattr(target, path[-1], value)
        except Exception:
            pass
    return estimator


def _run_estimator(estimator, pubs, *, precision: float | None):
    if precision is not None:
        try:
            return estimator.run(pubs, precision=float(precision))
        except TypeError:
            pass
    return estimator.run(pubs)


def _extract_pub_result(pub_result) -> tuple[float, float, int | str]:
    data = getattr(pub_result, "data", None)
    evs = getattr(data, "evs", np.nan)
    stds = getattr(data, "stds", np.nan)
    expectation = float(np.asarray(evs, dtype=float).reshape(-1)[0])
    std = float(np.asarray(stds, dtype=float).reshape(-1)[0]) if np.asarray(stds).size else float("nan")
    metadata = getattr(pub_result, "metadata", {}) or {}
    shots = metadata.get("shots", metadata.get("target_precision", ""))
    return expectation, std, shots


def _job_id(job) -> str:
    value = getattr(job, "job_id", "")
    if callable(value):
        try:
            return str(value())
        except Exception:
            return ""
    return str(value) if value else ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Frozen QNN subset JSON.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--backend", default=None, help="Optional IBM backend name. Defaults to least busy 4+ qubit backend.")
    parser.add_argument("--optimization-level", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--resilience-levels", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--precision", type=float, default=None, help="Optional Estimator precision target.")
    args = parser.parse_args(argv)
    run_hardware_inference(
        input_path=args.input,
        output=args.output,
        backend_name=args.backend,
        optimization_level=args.optimization_level,
        resilience_levels=args.resilience_levels,
        precision=args.precision,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
