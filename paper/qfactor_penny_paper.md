# QFactor-Penny: A Reproducible Benchmark of Trainable Quantum Circuits for Cross-Sectional Sector Return Ranking

## When Small QNNs Fail to Rank Assets Under Walk-Forward Validation

### Abstract

Quantum machine learning is often proposed for financial prediction, but finance benchmarks are vulnerable to noisy labels, small samples, leakage, and post-hoc overclaiming. We introduce QFactor-Penny, a reproducible PennyLane benchmark for trainable quantum neural networks in cross-sectional sector ETF ranking. The task ranks 11 sector ETFs using non-overlapping five-trading-day rebalance windows, SPY-relative top/bottom labels, expanding walk-forward validation, train-only preprocessing, and transaction-cost-aware portfolio accounting. A 4-qubit PennyLane QNN with angle encoding and one entangling layer is compared against naive momentum, linear models, random forest, RBF-SVM, XGBoost, and a small MLP. The standard QNN did not produce a stable positive ranking signal and exhibited a measurable failure mode: calendar-heavy selected features caused weak cross-sectional sensitivity and constant-score collapse. Cross-sectional-aware feature selection removed the observed QNN constant-score collapse, but QNN rank IC remained negative, precision@3 stayed near random, and no net alpha versus SPY survived transaction costs. The contribution is a leakage-audited QML robustness benchmark and failure-mode analysis, not a quantum-advantage claim.

## 1. Introduction

Quantum machine learning is an appealing direction for financial modeling because trainable quantum circuits can represent nonlinear transformations with compact parameterizations. Financial ranking tasks, however, are especially vulnerable to small-sample overfitting, target leakage, noisy labels, overlapping forward-return horizons, and post-hoc performance narratives. For quantum finance research to be credible, benchmarks need to report fragility and failure modes with the same care as positive results.

QFactor-Penny studies a constrained but realistic task: cross-sectional sector ETF return ranking over non-overlapping five-trading-day rebalance windows. The project does not attempt to forecast exact returns. Instead, each model produces ranking scores for the sector universe, and the benchmark evaluates whether those scores sort the sectors usefully under expanding walk-forward validation.

Small PennyLane QNNs can be implemented cleanly for cross-sectional sector ETF ranking, but under leakage-audited walk-forward validation they did not produce a stable positive ranking signal or post-cost alpha. The main contribution is a reproducible benchmark and failure-mode analysis showing that calendar-heavy feature selection can induce QNN constant-score collapse, and that cross-sectional-aware feature selection can remove the collapse without creating a tradable signal.

The empirical result is negative but informative. The standard QNN run produced constant-score collapse in some rebalance groups and did not generate stable positive rank IC or post-cost alpha. A cross-sectional-aware feature-selection variant removed the constant-score collapse, but the QNN still remained weak as a ranking model. This makes the project a failure-mode benchmark rather than a QNN performance claim.

Cross-sectional-aware feature selection removed the observed QNN constant-score collapse, but QNN rank IC remained negative, precision@3 stayed near random, and no net alpha versus SPY survived transaction costs.

The paper contributes:

- A reproducible QML benchmark for finance using non-overlapping five-trading-day sector ETF ranking windows and SPY-relative labels.
- A leakage-aware walk-forward evaluation framework with train-only preprocessing, target-window purging, split audits, prediction-shape audits, and deterministic rank handling.
- A PennyLane QNN comparison against classical baselines, including linear models, tree models, kernel methods, XGBoost, naive momentum, and a small MLP.
- A feature-stability audit showing that calendar-heavy features can break cross-sectional ranking by providing time variation without within-date sector dispersion.
- A measured QNN failure mode: constant or near-constant sector scores under standard feature selection.
- A diagnostic variant showing that cross-sectional-aware feature selection removes the collapse but does not create stable positive rank signal or post-cost alpha.

## 2. Data and Prediction Task

The benchmark uses SPY as the benchmark ticker and excludes SPY from tradable sector portfolios. The tradable universe is:

```text
XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY
```

For each non-overlapping rebalance date, the benchmark computes each sector ETF's next-five-trading-day return and its SPY-relative excess return over the same horizon. The top 3 sectors by forward SPY-relative return receive label `1`; the bottom 3 receive label `0`; the middle 5 are dropped from classification training. During inference, every model scores all 11 sector ETFs, including the middle sectors.

| Design Item | Choice |
| --- | --- |
| Benchmark ticker | SPY |
| Tradable universe | 11 SPDR sector ETFs |
| Horizon | Next 5 trading days |
| Rebalance schedule | Non-overlapping 5-trading-day rebalance dates |
| Training label | SPY-relative top 3 = 1, bottom 3 = 0 |
| Middle sectors | Dropped from classification training, included in inference |
| Portfolio | Long equal-weight top 3 sectors |
| Portfolio return | Absolute realized ETF forward return |
| Transaction cost | `turnover * 0.0005` |
| Alpha definition | `portfolio_net_return - spy_forward_return` |
| Tie handling | Model score descending, then ticker as deterministic tie-breaker |

## 3. Leakage Controls and Walk-Forward Design

The benchmark uses expanding walk-forward splits. Each split trains on historical rebalance dates, reserves the final training-history slice for validation, and tests on the next month of non-overlapping rebalance dates. Feature selection, imputation, scaling, and model fitting are performed only on training data inside each split.

The split builder applies `purge_trading_days` before validation and test periods so that five-trading-day forward target windows do not overlap across boundaries. Each run emits `split_audit.csv`, which records train, validation, test, purge, and forward-window boundary dates. This audit is central to the paper because it makes leakage controls visible rather than implicit.

## 4. Models

The QNN uses PennyLane's analytic simulator for training. The required MVP architecture maps 4 selected and scaled features to 4 qubits using angle encoding, applies one `StronglyEntanglingLayers` layer, and returns a scalar expectation value used as the ranking score. The QNN has 13 trainable parameters in the current implementation.

Classical baselines are intentionally lightweight but nontrivial:

- naive momentum rank
- logistic regression
- ridge linear classifier
- random forest
- RBF SVM
- small one-hidden-layer MLP
- optional XGBoost when installed

The goal is not to exhaust every classical model family. The goal is to prevent the QNN from being compared only to weak straw baselines.

## 5. Metrics and Portfolio Accounting

The benchmark reports both machine-learning and portfolio-style metrics. Ranking quality is summarized using rank IC and precision@3. Classification behavior is summarized using ROC-AUC, balanced accuracy, and F1 where defined. Metrics that are undefined due to constant scores, one-class labels, tiny samples, or zero volatility are reported as `NaN` with warnings instead of crashing or silently disappearing.

Portfolio evaluation ranks sectors by model score at each rebalance date and selects the top 3 sectors as an equal-weight long portfolio. Gross and net returns use absolute realized ETF returns, not SPY-relative target returns.

```text
turnover = 0.5 * sum(abs(new_weights - old_weights))
transaction_cost = turnover * 0.0005
net_return = gross_return - transaction_cost
alpha_vs_spy = net_return - spy_forward_return
```

## 6. Results

### 6.1 Standard Feature Selection

In the standard MVP run, train-only mutual-information feature selection selected several calendar-heavy features, including `month_cos`, `weekday_cos`, and `month_sin`. These features varied over time but had zero within-date cross-sectional dispersion across ETFs. That makes them weak inputs for a task that requires ranking sectors against one another on the same rebalance date.

The QNN standard run produced 4 constant-score date groups and 1 row with undefined QNN metrics. Its mean rank IC was `-0.0735`, precision@3 was `0.2292`, and alpha versus SPY after transaction costs was `-0.0004`.

### 6.2 Cross-Sectional-Aware Feature Selection

The cross-sectional-aware variant filters low within-date dispersion features using only the training slice. This variant is a diagnostic experiment, not post-hoc performance tuning. It tests whether the measured collapse is associated with feature geometry rather than QNN optimization alone.

The variant removed calendar-heavy selected features from the learned-model selected feature set and reduced QNN constant-score groups from `4` to `0`. However, the ranking result remained weak: QNN rank IC stayed negative at `-0.0494`, precision@3 was `0.2708`, and alpha versus SPY after costs worsened slightly to `-0.0006`.

Because the task selects 3 sectors out of 11 and there are 3 true positive sectors, random precision@3 is approximately `3 / 11 = 0.2727`. The cross-sectional-aware QNN precision@3 of `0.2708` is therefore near random.

| QNN Metric | Standard | Cross-Sectional-Aware |
| --- | ---: | ---: |
| Constant-score groups | 4 | 0 |
| Calendar-heavy selected feature share | 40% | 0% |
| Rank IC | -0.0735 | -0.0494 |
| Precision@3 | 0.2292 | 0.2708 |
| Alpha vs SPY after costs | -0.0004 | -0.0006 |
| Turnover | 0.5521 | 0.6562 |

Cross-sectional-aware feature selection removed the observed QNN constant-score collapse, but QNN rank IC remained negative, precision@3 stayed near random, and no net alpha versus SPY survived transaction costs.

### 6.3 Classical Baseline Context

The classical baselines also showed unstable behavior in the short MVP comparison. This matters: the negative QNN result is not merely "classical models win." The broader result is that this small, noisy financial ranking task is difficult, and robust reporting should emphasize instability, turnover, and path dependence.

| Model | Feature Selection | Rank IC | Precision@3 | Alpha vs SPY | Turnover |
| --- | --- | ---: | ---: | ---: | ---: |
| logistic regression | standard | -0.0369 | 0.2292 | -0.0010 | 0.7188 |
| naive momentum | standard | -0.0540 | 0.2917 | -0.0023 | 0.3021 |
| PennyLane QNN | standard | -0.0735 | 0.2292 | -0.0004 | 0.5521 |
| random forest | standard | -0.1350 | 0.2292 | -0.0038 | 0.6771 |
| RBF SVM | standard | 0.0551 | 0.3542 | 0.0004 | 0.6354 |
| ridge linear | standard | -0.0403 | 0.2292 | -0.0010 | 0.7188 |
| small MLP | standard | -0.0614 | 0.2083 | -0.0023 | 0.7604 |
| XGBoost | standard | -0.1564 | 0.2083 | -0.0044 | 0.6979 |
| logistic regression | cross-sectional-aware | 0.0682 | 0.2292 | -0.0004 | 0.7188 |
| PennyLane QNN | cross-sectional-aware | -0.0494 | 0.2708 | -0.0006 | 0.6562 |
| RBF SVM | cross-sectional-aware | 0.0653 | 0.2917 | -0.0001 | 0.6771 |
| ridge linear | cross-sectional-aware | 0.0710 | 0.2500 | 0.0003 | 0.6771 |
| XGBoost | cross-sectional-aware | -0.0175 | 0.3333 | 0.0004 | 0.7812 |

### 6.4 Shot Sensitivity

The QNN is trained analytically and then evaluated on a small fixed subset with 1024-shot simulator sampling. This is an inference-only simulation diagnostic, not real-hardware validation. It does not retrain the QNN with shots.

Under standard feature selection, later splits showed weak or negative analytic-vs-shot score correlation and high ranking flip rates, especially around the constant-score failure. Under cross-sectional-aware feature selection, shot-score correlations were high in the MVP run, but this did not translate into a stable positive ranking signal.

| Variant | Split | Shot Score Correlation | Mean Absolute Score Difference | Ranking Flip Rate |
| --- | --- | ---: | ---: | ---: |
| standard | split_00_2018-12 | 0.9764 | 0.0197 | 0.0000 |
| standard | split_01_2019-01 | 0.8979 | 0.0268 | 0.1481 |
| standard | split_02_2019-02 | -0.4807 | 0.0259 | 0.7143 |
| standard | split_03_2019-03 | -0.6488 | 0.0202 | 0.7692 |
| cross-sectional-aware | split_00_2018-12 | 0.9950 | 0.0219 | 0.0385 |
| cross-sectional-aware | split_01_2019-01 | 0.9969 | 0.0232 | 0.0357 |
| cross-sectional-aware | split_02_2019-02 | 0.9941 | 0.0307 | 0.0000 |
| cross-sectional-aware | split_03_2019-03 | 0.9932 | 0.0203 | 0.1071 |

## 7. Failure-Mode Analysis

The paper's central failure mode is QNN constant-score collapse: the trained circuit emits identical or nearly identical scores for all sector ETFs on a rebalance date. This is damaging for cross-sectional ranking because it destroys sector ordering and can make ROC-AUC, rank IC, balanced accuracy, or F1 undefined.

Feature-stability analysis suggests one mechanism. Standard feature selection sometimes selected calendar-heavy features with time-series variation but no within-date cross-sectional dispersion. A feature such as `month_cos` can describe the date, but all sector ETFs on that date receive the same value. If a model leans on these inputs, it may learn date-level context without learning sector-level differences.

The cross-sectional-aware diagnostic supports this interpretation: removing low-dispersion features eliminated the observed QNN collapse. But the diagnostic also shows the limits of the fix. Removing collapse is not the same as finding a robust ranking signal. The QNN still had negative rank IC, near-random precision@3, and negative post-cost alpha versus SPY.

## 8. Reproducibility and Artifacts

QFactor-Penny is designed to make experiment state auditable. Each benchmark run writes raw results, aggregate summaries, and diagnostic files.

| Artifact | Purpose |
| --- | --- |
| `run_manifest.json` | Records config hash, dataset hash, dependency versions, seeds, runtime, and output directory |
| `split_audit.csv` | Records train/validation/test dates, purge gaps, and target-window boundaries |
| `prediction_audit.csv` | Confirms every model/date/seed scores all 11 sector ETFs and includes middle-sector inference rows |
| `model_run_status.csv` | Records success, skip, failure, constant-score dates, NaN scores, and diagnostics availability |
| `feature_stability_summary.csv` | Reports selected-feature dispersion and calendar-heavy flags |
| `qnn_failure_audit.csv` | Summarizes QNN constant-score groups, undefined metrics, and selected features |
| `quantum_diagnostics.csv` | Records qubits, layers, parameter count, train/inference time, and shot-sensitivity diagnostics |
| `experimental_variant_comparison.csv` | Compares standard and cross-sectional-aware feature-selection variants |

Generated figures are available in:

- `results/figures/model_rank_ic.png`
- `results/figures/roc_auc_by_model.png`
- `results/figures/split_rank_ic_by_model.png`
- `results/figures/portfolio_equity.png`
- `results/figures/alpha_vs_spy_by_model.png`
- `results/figures/turnover_vs_return.png`
- `results/figures/qnn_shot_sensitivity.png`
- `results_cross_sectional_mvp/figures/model_rank_ic.png`
- `results_cross_sectional_mvp/figures/qnn_shot_sensitivity.png`

## 9. Limitations

This benchmark is not a trading system and is not investment advice. The current MVP summaries use only 4 walk-forward splits and one seed (`42`) for fast iteration. The tradable universe is small: 11 sector ETFs. Financial labels are noisy, sector returns are path dependent, and five-trading-day horizons can be unstable. The QNN is trained on an analytic simulator rather than real hardware, and the 1024-shot experiment is inference-only simulator sampling, not real-hardware validation. The reported cross-sectional-aware variant is diagnostic, not an optimized production feature pipeline.

The paper makes no statistical-significance claim, no quantum-advantage claim, and no trading-edge claim. Most importantly, a cleaner diagnostic result should not be overread. The cross-sectional-aware QNN removed constant-score collapse, but it did not create stable positive rank IC or post-cost alpha.

## 10. Future Work

Future work should run the full benchmark across more splits and seeds, expand robustness checks, and report confidence intervals for ranking and portfolio metrics. A larger universe could test whether QNN fragility changes with a broader cross-section, though this would also increase the need for careful multiple-testing controls. Additional QNN variants should be introduced only as pre-registered experimental variants, not as post-hoc tuning to make results look better. Real-hardware inference could be added later as a separate experiment, but no real-hardware results are reported in this paper.

## 11. Conclusion

QFactor-Penny demonstrates that small PennyLane QNNs can be evaluated rigorously in a financial ranking setting, but the tested QNN did not produce stable positive ranking behavior. The most informative result was a failure mode: standard feature selection sometimes chose calendar-heavy inputs with weak cross-sectional dispersion, leading to QNN constant-score collapse. Cross-sectional-aware feature selection removed the observed QNN constant-score collapse, but QNN rank IC remained negative, precision@3 stayed near random, and no net alpha versus SPY survived transaction costs. These results do not support a quantum-advantage or trading-edge claim. Instead, they support the value of leakage-aware QML benchmarks that expose model fragility before making performance claims.
