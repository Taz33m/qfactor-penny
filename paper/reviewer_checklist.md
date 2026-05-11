# QFactor-Penny Reviewer Checklist

This checklist maps the paper's main claims to the generated artifacts that support them. It is intended for a skeptical reviewer who wants to verify that the paper is a benchmark and failure-mode analysis, not a quantum-advantage or trading-edge claim.

## Claim/Evidence Map

| Paper Claim | Evidence Artifact | Expected Evidence |
| --- | --- | --- |
| The task is cross-sectional sector ETF return ranking, not point return forecasting. | `paper/qfactor_penny_paper.md`, `results/prediction_audit.csv` | Each model/date/seed scores all 11 sector ETFs; middle sectors are included in inference. |
| SPY is a benchmark/reference, not a tradable sector portfolio member. | `results/rebalance_predictions.csv`, `results/portfolio_selection_audit.csv` | Selected sector portfolios contain sector ETFs only. |
| Walk-forward validation is leakage-audited. | `results/split_audit.csv` | Train, validation, and test windows are ordered and forward-label windows are purged before validation/test boundaries. |
| Feature selection and scaling are train-only. | `qfactor_penny/preprocessing.py`, tests | Preprocessor is fit on training rows within each split. |
| Tie handling is deterministic. | `results/prediction_audit.csv`, `results/portfolio_selection_audit.csv` | Rankings sort by model score and use ticker as the deterministic tie-breaker. |
| Shot sensitivity is inference-only simulation. | `results/quantum_diagnostics.csv`, `paper/qfactor_penny_paper.md` | 1024-shot sensitivity is simulator sampling after analytic training, not real-hardware validation. |
| The standard QNN exhibited constant-score collapse. | `results/qnn_failure_audit.csv` | QNN constant-score groups equal `4` in the standard MVP run. |
| Standard feature selection chose calendar-heavy features. | `results/feature_stability_summary.csv` | Calendar-heavy selected feature share is `40%` in the standard MVP comparison. |
| Cross-sectional-aware selection removed the collapse. | `results_cross_sectional_mvp/qnn_failure_audit.csv` | QNN constant-score groups equal `0` in the cross-sectional-aware MVP run. |
| Cross-sectional-aware selection did not create a robust QNN ranking signal. | `results/experimental_variant_comparison.csv` | QNN rank IC remains negative (`-0.0494`) and precision@3 remains near random (`0.2708` vs random `3 / 11 = 0.2727`). |
| No QNN net alpha versus SPY survived costs. | `results/experimental_variant_comparison.csv` | QNN alpha vs SPY after costs is negative in both variants (`-0.0004 -> -0.0006`). |
| IBM hardware execution is a robustness audit, not a performance claim. | `results_hardware/hardware_summary.md`, `results_hardware/hardware_run_manifest.json`, `results_hardware/ibm_hardware_scores.csv` | Frozen-QNN hardware inference used `ibm_rensselaer`, 11 samples, 100 shots, optimization/resilience level 0, and reports score drift plus ranking instability. |
| The first IBM Runtime attempt exposed an execution failure mode. | `results_hardware/hardware_run_manifest.json`, `results_hardware/hardware_summary.md` | The failed batched optimization-level-1 attempt is documented with IBM HAL error `9604`; the conservative optimization-level-0 run succeeded. |
| Stale full-run outputs are excluded from current paper conclusions. | `results/experimental_variant_comparison.csv` | Comparison rows come only from current-schema result directories and have known feature-selection modes plus nonzero selected-feature rows. |

## Reviewer Questions To Check

- Does every model/date/seed produce scores for all 11 sector ETFs?
- Are top-3 and bottom-3 labels used only for classification training, while middle sectors remain in inference?
- Are split boundaries purged enough to avoid overlapping five-trading-day target windows?
- Are undefined metrics reported as `NaN` instead of being hidden?
- Is the QNN failure mode visible in diagnostics rather than patched away?
- Does the paper avoid interpreting collapse removal as quantum advantage?
- Does the IBM hardware section stay inference-only and avoid treating hardware noise as finance-signal validation?
- Are all figure references relative repo paths?
- Do limitations explicitly state 4 MVP splits, one seed, only a small inference-only hardware audit, no statistical-significance claim, no quantum advantage, and no trading-edge claim?

## Current Supported Conclusion

Cross-sectional-aware feature selection removed the observed QNN constant-score collapse, but QNN rank IC remained negative, precision@3 stayed near random, and no net alpha versus SPY survived transaction costs. The current artifact supports a leakage-audited QML robustness benchmark and failure-mode analysis. It does not support a quantum-advantage or trading-edge claim.
