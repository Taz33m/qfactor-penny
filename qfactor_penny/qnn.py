"""Small PennyLane QNN scorer."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class QNNResult:
    score: np.ndarray
    train_seconds: float
    inference_seconds: float
    parameter_count: int
    qubits: int
    layers: int
    selected_features: list[str]
    epochs_ran: int
    best_validation_loss: float
    shot_sensitivity_samples: int
    shot_sensitivity_shots: int
    shot_score_correlation: float
    shot_mean_abs_score_diff: float
    shot_ranking_flip_rate: float


def _sigmoid(values):
    import pennylane.numpy as pnp

    return 1.0 / (1.0 + pnp.exp(-values))


def fit_predict_qnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    *,
    selected_features: list[str],
    epochs: int,
    learning_rate: float,
    patience: int,
    random_state: int,
    shot_sensitivity_samples: int = 8,
    shot_sensitivity_shots: int = 1024,
) -> QNNResult:
    try:
        import pennylane as qml
        import pennylane.numpy as pnp
    except Exception as exc:  # pragma: no cover - depends on installed dependency
        raise RuntimeError("PennyLane is required for the QNN MVP. Install `pennylane`.") from exc

    qubits = 4
    layers = 1
    if x_train.shape[1] != qubits or x_test.shape[1] != qubits:
        raise ValueError("The MVP QNN requires exactly 4 selected/scaled features mapped to 4 qubits.")

    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(features, weights):
        qml.AngleEmbedding(features, wires=range(qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(qubits))
        return qml.expval(qml.PauliZ(0))

    def logits(weights, bias, x_values):
        outputs = [circuit(row, weights) for row in x_values]
        return pnp.stack(outputs) + bias

    def loss(weights, bias, x_values, y_values):
        raw = logits(weights, bias, x_values)
        probs = pnp.clip(_sigmoid(raw), 1e-6, 1.0 - 1e-6)
        labels = pnp.asarray(y_values, dtype=float)
        return -pnp.mean(labels * pnp.log(probs) + (1.0 - labels) * pnp.log(1.0 - probs))

    rng = np.random.default_rng(random_state)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=qubits)
    weights = pnp.array(0.05 * rng.normal(size=shape), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    x_train_p = pnp.asarray(x_train, dtype=float)
    y_train_p = pnp.asarray(y_train, dtype=float)
    x_val_p = pnp.asarray(x_validation, dtype=float)
    y_val_p = pnp.asarray(y_validation, dtype=float)
    optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    best_weights = weights.copy()
    best_bias = bias.copy()
    best_val = float("inf")
    stale = 0
    epochs_ran = 0
    start = time.perf_counter()
    for epoch in range(max(1, epochs)):
        weights, bias = optimizer.step(lambda w, b: loss(w, b, x_train_p, y_train_p), weights, bias)
        epochs_ran = epoch + 1
        validation_loss = float(loss(weights, bias, x_val_p, y_val_p)) if len(y_validation) else float(loss(weights, bias, x_train_p, y_train_p))
        if validation_loss + 1e-7 < best_val:
            best_val = validation_loss
            best_weights = weights.copy()
            best_bias = bias.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    train_seconds = time.perf_counter() - start
    infer_start = time.perf_counter()
    score = np.asarray(logits(best_weights, best_bias, pnp.asarray(x_test, dtype=float)), dtype=float)
    inference_seconds = time.perf_counter() - infer_start
    shot_samples = min(max(0, int(shot_sensitivity_samples)), len(x_test))
    shot_corr = np.nan
    shot_diff = np.nan
    flip_rate = np.nan
    if shot_samples >= 2 and shot_sensitivity_shots > 0:
        shot_dev = qml.device("default.qubit", wires=qubits, shots=int(shot_sensitivity_shots))

        @qml.qnode(shot_dev)
        def shot_circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(qubits))
            return qml.expval(qml.PauliZ(0))

        subset = np.asarray(x_test[:shot_samples], dtype=float)
        analytic_subset = score[:shot_samples]
        shot_scores = np.asarray([shot_circuit(row, best_weights) + float(best_bias) for row in subset], dtype=float)
        shot_diff = float(np.mean(np.abs(analytic_subset - shot_scores)))
        if len(np.unique(analytic_subset)) > 1 and len(np.unique(shot_scores)) > 1:
            shot_corr = float(np.corrcoef(analytic_subset, shot_scores)[0, 1])
        total_pairs = 0
        flipped_pairs = 0
        for left in range(shot_samples):
            for right in range(left + 1, shot_samples):
                analytic_order = np.sign(analytic_subset[left] - analytic_subset[right])
                shot_order = np.sign(shot_scores[left] - shot_scores[right])
                if analytic_order == 0 or shot_order == 0:
                    continue
                total_pairs += 1
                flipped_pairs += int(analytic_order != shot_order)
        if total_pairs:
            flip_rate = float(flipped_pairs / total_pairs)
    return QNNResult(
        score=score,
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        parameter_count=int(np.prod(shape) + 1),
        qubits=qubits,
        layers=layers,
        selected_features=selected_features,
        epochs_ran=epochs_ran,
        best_validation_loss=best_val,
        shot_sensitivity_samples=shot_samples,
        shot_sensitivity_shots=int(shot_sensitivity_shots),
        shot_score_correlation=shot_corr,
        shot_mean_abs_score_diff=shot_diff,
        shot_ranking_flip_rate=flip_rate,
    )
