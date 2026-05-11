<p align="center">
  <img src="assets/qfactor-penny-logo.png" alt="QFactor-Penny logo" width="260" />
</p>

<h1 align="center">QFactor-Penny</h1>

<p align="center">
  <strong>A leakage-aware PennyLane benchmark for trainable quantum circuits in cross-sectional sector ETF return ranking.</strong>
</p>

<p align="center">
  <a href="https://youtu.be/jAKgiCHIuVY">Demo Video</a>
  ·
  <a href="paper/qfactor_penny_paper.pdf">Paper PDF</a>
</p>

<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-7fc7a6" />
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-536b66" />
  <img alt="PennyLane" src="https://img.shields.io/badge/PennyLane-QNN-7fc7a6" />
  <img alt="tests" src="https://img.shields.io/badge/tests-pytest-7fc7a6" />
  <img alt="claim" src="https://img.shields.io/badge/claim-no%20quantum%20advantage-d77768" />
</p>

<p align="center">
  <a href="https://youtu.be/jAKgiCHIuVY">
    <img src="https://img.youtube.com/vi/jAKgiCHIuVY/maxresdefault.jpg" alt="QFactor-Penny demo video thumbnail" width="860" />
  </a>
</p>

<p align="center">
  <a href="paper/qfactor_penny_paper.pdf">
    <img src="assets/paper-thumbnail.png" alt="QFactor-Penny paper thumbnail" width="520" />
  </a>
</p>

> **TL;DR:** Cross-sectional-aware feature selection removed QNN constant-score collapse, but the QNN still showed negative rank IC, near-random precision@3, and no post-cost alpha versus SPY.

## Best Way to Review

1. Watch the [demo video](https://youtu.be/jAKgiCHIuVY).
2. Read the [paper abstract](paper/qfactor_penny_paper.pdf) and the standard vs cross-sectional-aware QNN table.
3. Inspect [`results/experimental_variant_comparison.csv`](results/experimental_variant_comparison.csv).
4. Run [`configs/mvp.yaml`](configs/mvp.yaml) for the fast benchmark path.

## Overview

QFactor-Penny evaluates whether a small trainable PennyLane quantum neural network can produce useful cross-sectional ranking scores for sector ETFs under realistic financial validation constraints. The benchmark ranks 11 SPDR sector ETFs over non-overlapping five-trading-day rebalance windows, trains on SPY-relative top/bottom labels, scores all sectors at inference, and evaluates long top-3 portfolios using absolute ETF returns after transaction costs.

The result is intentionally not framed as a performance win. The project found a measurable QNN failure mode: standard feature selection sometimes chose calendar-heavy inputs with weak within-date cross-sectional dispersion, causing constant or near-constant QNN scores. Cross-sectional-aware feature selection removed that collapse, but it did not create a robust ranking signal or post-cost alpha.

> Cross-sectional-aware feature selection removed the observed QNN constant-score collapse, but QNN rank IC remained negative, precision@3 stayed near random, and no net alpha versus SPY survived transaction costs.

This is a reproducible failure-mode benchmark, not a quantum-advantage claim and not a trading system.

## Why This Matters

Finance QML benchmarks can look impressive if leakage, overlapping labels, weak baselines, or post-hoc tuning are ignored. QFactor-Penny is designed to expose those failure modes directly. The negative result is the point: a cleaner QNN diagnostic did not create a reliable ranking signal.

## Current Result Snapshot

| QNN diagnostic | Standard MVP | Cross-sectional-aware MVP |
| --- | ---: | ---: |
| Constant-score groups | 4 | 0 |
| Calendar-heavy selected feature share | 40% | 0% |
| Rank IC | -0.0735 | -0.0494 |
| Precision@3 | 0.2292 | 0.2708 |
| Random precision@3 baseline | 0.2727 | 0.2727 |
| Alpha vs SPY after costs | -0.0004 | -0.0006 |
| Turnover | 0.5521 | 0.6562 |

The diagnostic improved the mechanics of the QNN scoring behavior, but the investment-style signal remained weak. Precision@3 stayed near the random baseline of `3 / 11 = 0.2727`, rank IC remained negative, and no net alpha versus SPY survived transaction costs.

## Architecture

```mermaid
flowchart LR
    A["Frozen sector ETF data"] --> B["prepare_data"]
    B --> C["Non-overlapping 5-day rebalance panel"]
    C --> D["Expanding walk-forward split builder"]
    D --> E["Train-only feature selection and scaling"]
    E --> F1["Classical baselines"]
    E --> F2["4-feature / 4-qubit PennyLane QNN"]
    F1 --> G["Scores for all 11 sector ETFs"]
    F2 --> G
    G --> H["Model rank positions"]
    H --> I["Long equal-weight top 3 sectors"]
    I --> J["Portfolio accounting with costs"]
    G --> K["Prediction, split, status, feature, QNN audits"]
    J --> L["Results summary, figures, paper"]
    K --> L

    M["SPY benchmark/reference"] --> B
    M --> J
```

## Methodology

| Component | Design |
| --- | --- |
| Benchmark ticker | `SPY` |
| Tradable universe | `XLB`, `XLC`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLRE`, `XLU`, `XLV`, `XLY` |
| Horizon | Next 5 trading days |
| Rebalance dates | Non-overlapping 5-trading-day windows |
| Label | Top 3 SPY-relative sectors = `1`; bottom 3 = `0` |
| Middle sectors | Dropped from classification training, included in inference/ranking |
| Validation | Expanding walk-forward splits with train-only preprocessing |
| Leakage control | Configurable `purge_trading_days` around target horizons |
| Portfolio | Equal-weight long top 3 sectors, held for the next 5 trading days |
| Return accounting | Absolute ETF forward returns, not SPY-relative target returns |
| Transaction cost | `turnover * 0.0005` |
| Alpha definition | `portfolio_net_return - spy_forward_return` |

SPY is a benchmark/reference ticker only. It is never selected in sector portfolios.

## Models

The QNN is deliberately small:

- 4 selected/scaled features
- 4 qubits
- angle encoding
- 1 `StronglyEntanglingLayers` layer
- analytic PennyLane simulator for training
- scalar expectation output used as the ranking score
- inference-only 1024-shot sensitivity check after analytic training

Classical baselines:

- naive momentum rank
- logistic regression
- ridge linear classifier
- random forest
- RBF SVM
- small one-hidden-layer MLP
- XGBoost when installed; skipped cleanly otherwise

The goal is not to exhaust every classical model family. The goal is to compare the QNN against nontrivial lightweight baselines while keeping the experiment auditable.

## Repository Map

```text
qfactor_penny/                  Benchmark package and CLI modules
configs/                        MVP, full, and feature-selection configs
tests/                          Unit and smoke tests
paper/                          Markdown, LaTeX, PDF, and bibliography
results/                        Current standard MVP artifact outputs
results_cross_sectional_mvp/    Current cross-sectional-aware MVP artifact outputs
results_hardware/               Optional frozen-QNN hardware audit outputs
assets/                         README logo and paper thumbnail
requirements.txt                Runtime/test requirements
requirements-hardware.txt       Optional Qiskit/IBM Runtime requirements
requirements.lock.txt           Local environment lock snapshot
```

Generated cache directories, local data extracts, stale full-run outputs, and rendered video files are ignored by git. The demo video is hosted externally on YouTube.

## Quickstart

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Prepare data from the frozen MarketMind-Q sector ETF dataset when available:

```bash
.venv/bin/python -m qfactor_penny.prepare_data \
  --input mktmind-qtm/data/marketmind_qml_dataset.csv \
  --output data/qfactor_dataset.csv
```

The project does not silently download market data. If the input path is unavailable, `prepare_data` creates deterministic synthetic demo data and marks the output as synthetic for smoke testing.

Run the fast MVP benchmark:

```bash
.venv/bin/python -m qfactor_penny.run_benchmark --config configs/mvp.yaml
.venv/bin/python -m qfactor_penny.make_report --config configs/mvp.yaml
```

Run the cross-sectional-aware diagnostic MVP:

```bash
.venv/bin/python -m qfactor_penny.run_benchmark --config configs/cross_sectional_mvp.yaml
.venv/bin/python -m qfactor_penny.make_report --config configs/cross_sectional_mvp.yaml
```

Run tests:

```bash
.venv/bin/python -m pytest
```

## Optional IBM Quantum Hardware Audit

QFactor-Penny includes an opt-in IBM Quantum audit path for studying simulator-to-hardware degradation. This path is inference-only: it freezes a small trained QNN subset, transpiles the matching Qiskit circuits for an IBM backend, and compares analytic scores against hardware Estimator outputs. It is a systems diagnostic, not evidence of quantum advantage, tradable alpha, or real-hardware validation of a finance signal.

The current v1 hardware audit used RPI's IBM Quantum System One backend through the `General-dedicated` instance. It did not retrain the QNN and did not alter the benchmark conclusions.

| Hardware diagnostic | Result |
| --- | ---: |
| Backend | `ibm_rensselaer` |
| Backend size | 127 qubits |
| Instance | `General-dedicated` |
| Samples | 11 frozen QNN inference samples |
| Shots | 100 |
| Optimization / resilience | level 0 / level 0 |
| Mean circuit depth | 8.0000 -> 26.6364 |
| Mean two-qubit gates | 4.0000 -> 8.9091 |
| Mean SWAP count | 0.0000 |
| Analytic vs hardware score correlation | -0.0901 |
| Mean absolute score difference | 0.3857 |
| Pairwise ranking flip rate | 0.4630 |
| Top-3 overlap | 0.3333 |

Analytic top 3 was `XLP, XLV, XLC`; hardware top 3 was `XLK, XLU, XLP`. A first batched optimization-level-1 Estimator attempt failed with IBM HAL error `9604`; the conservative optimization-level-0 run succeeded. This is evidence of hardware execution fragility, not evidence of a useful finance signal.

Install optional hardware dependencies:

```bash
.venv/bin/python -m pip install -r requirements-hardware.txt
```

Rotate any exposed IBM Quantum API key before use, then export the fresh token locally:

```bash
read -rsp "IBM Quantum token: " QISKIT_IBM_TOKEN
export QISKIT_IBM_TOKEN
export QISKIT_IBM_INSTANCE="General-dedicated"
```

Export a frozen cross-sectional-aware QNN subset and run the audit:

```bash
.venv/bin/python -m qfactor_penny.export_qnn_hardware_subset \
  --config configs/cross_sectional_mvp.yaml \
  --output results_hardware/frozen_qnn_subset.json

.venv/bin/python -m qfactor_penny.ibm_transpile_audit \
  --input results_hardware/frozen_qnn_subset.json \
  --output results_hardware/ibm_transpilation_audit.csv \
  --backend ibm_rensselaer

.venv/bin/python -m qfactor_penny.ibm_hardware_inference \
  --input results_hardware/frozen_qnn_subset.json \
  --output results_hardware/ibm_hardware_scores.csv \
  --backend ibm_rensselaer \
  --optimization-level 0 \
  --resilience-levels 0 \
  --precision 0.1

.venv/bin/python -m qfactor_penny.make_hardware_report \
  --input-dir results_hardware
```

The hardware report records backend/transpilation details, Estimator settings, score drift, ranking flips, top-3 overlap, diagnostic figures, and `hardware_run_manifest.json`. Missing Qiskit packages or credentials are reported as skipped status rows instead of crashing the workflow.

## Configs

| Config | Purpose | Output directory |
| --- | --- | --- |
| `configs/mvp.yaml` | Fast standard smoke/research MVP; `max_splits: 4`, `seeds: [42]` | `results/` |
| `configs/cross_sectional_mvp.yaml` | Matched diagnostic MVP with cross-sectional-aware feature selection | `results_cross_sectional_mvp/` |
| `configs/cross_sectional_features.yaml` | Longer cross-sectional-aware diagnostic variant | configured in file |
| `configs/full.yaml` | Slower research config with more splits/seeds | `results_full/` |

The paper conclusions use the current standard MVP and cross-sectional-aware MVP artifacts. `results_full/` is treated as stale/local unless regenerated under the current manifest/status/feature-stability schema.

## Reproducibility Artifacts

Each benchmark output directory includes audit and status files:

| Artifact | Purpose |
| --- | --- |
| `run_manifest.json` | Config hash, dataset hash, package versions, model list, seeds, runtime, git SHA |
| `split_audit.csv` | Train/validation/test dates, purge gaps, and split coverage |
| `prediction_audit.csv` | Confirms all 11 sectors are scored per model/date/seed |
| `model_run_status.csv` | Success/skipped/failed status, prediction counts, constant-score counts |
| `feature_stability_summary.csv` | Train-only feature dispersion and calendar-heavy flags |
| `qnn_failure_audit.csv` | Constant-score QNN date groups and related diagnostics |
| `quantum_diagnostics.csv` | Qubit count, layer count, parameters, timing, shot-sensitivity metrics |
| `portfolio_selection_audit.csv` | Top-3 selection, deterministic rank handling, no SPY portfolio inclusion |
| `undefined_metric_audit.csv` | Metrics that became `NaN` because they were undefined |

The report files `results/results_summary.md` and `results_cross_sectional_mvp/results_summary.md` summarize these artifacts and preserve the no-overclaiming stance.

## Paper and Video

- Paper title: **QFactor-Penny: A Reproducible Benchmark of Trainable Quantum Circuits for Cross-Sectional Sector Return Ranking**
- Subtitle: **When Small QNNs Fail to Rank Assets Under Walk-Forward Validation**
- PDF: [`paper/qfactor_penny_paper.pdf`](paper/qfactor_penny_paper.pdf)
- Demo video: [youtu.be/jAKgiCHIuVY](https://youtu.be/jAKgiCHIuVY)

The video is a short companion explainer. It walks through the task, the QNN constant-score collapse, the cross-sectional-aware diagnostic, and the negative research conclusion.

## What This Project Does Not Claim

QFactor-Penny does not claim:

- quantum advantage
- statistically significant QNN outperformance
- tradable alpha
- real-hardware validation
- investment advice
- a production trading system

The benchmark is designed to expose fragility before making performance claims.

## Limitations

- The current paper-backed MVP comparison uses only 4 walk-forward splits and one seed.
- The tradable universe is small: 11 sector ETFs.
- Financial labels are noisy and short-horizon returns are difficult to rank.
- The QNN is intentionally small and not tuned for best possible performance.
- Shot sensitivity is simulator-based inference after analytic training, not real-hardware execution.
- No statistical-significance claim is made.

## License

MIT License. See [`LICENSE`](LICENSE).
