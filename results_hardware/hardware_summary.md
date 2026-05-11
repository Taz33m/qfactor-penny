# IBM Quantum Hardware Audit

This optional report treats IBM Quantum execution as an inference-time systems diagnostic. It does not claim quantum advantage, tradable alpha, or real-hardware validation of a finance signal.

## Executive Interpretation

The frozen QNN circuits reached IBM hardware, but the real-device scores were unstable relative to analytic simulation: score correlation `-0.0901`, mean absolute score difference `0.3857`, pairwise ranking flip rate `0.4630`, and mean top-3 overlap `0.3333`. The transpiled circuits had mean SWAP count `0.0000`. This supports a hardware robustness reading, not a performance claim.

## Artifacts

| Artifact | Status |
| --- | --- |
| `frozen_qnn_subset.json` | present |
| `ibm_transpilation_audit.csv` | present |
| `ibm_hardware_scores.csv` | present |
| `hardware_run_manifest.json` | present |

## Transpilation Audit

Status counts: `{'success': 11}`.

| backend_name | circuits | pre_depth_mean | post_depth_mean | post_two_qubit_gate_count_mean | post_swap_count_mean |
| --- | --- | --- | --- | --- | --- |
| ibm_rensselaer | 11 | 8.0000 | 26.6364 | 8.9091 | 0.0000 |

## Hardware Score Diagnostics

Status counts: `{'success': 11}`.

| backend_name | resilience_level | samples | score_correlation | mean_abs_score_diff | pairwise_ranking_flip_rate | mean_top3_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| ibm_rensselaer | 0 | 11 | -0.0901 | 0.3857 | 0.4630 | 0.3333 |

Top-3 ranking comparison:

| model_id | date | analytic_top3 | hardware_top3 | top3_overlap |
| --- | --- | --- | --- | --- |
| split_00_2018-12__seed_42 | 2018-12-07 | XLP, XLV, XLC | XLK, XLU, XLP | 0.3333 |

## HAL/Runtime Execution Note

| event | status | backend_name | optimization_level | error_type | error_code | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| initial_batched_estimator_attempt | failed | ibm_rensselaer | 1 | RuntimeJobFailureError | 9604 | Batched optimization-level-1 Estimator execution failed with IBM HAL configuration error; conservative optimization-level-0 execution succeeded. |

## Diagnostic Figure

- `figures/hardware_score_scatter.png`

## Interpretation Guardrail

A successful hardware run only shows how the frozen QNN circuits behave under a selected backend, transpilation pass, finite-shot estimator execution, and optional mitigation settings. It should be read alongside the existing negative simulator benchmark, not as a new performance result.
