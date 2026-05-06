# QFactor-Penny Results Summary

This project evaluates whether small trainable PennyLane QNNs provide stable or useful ranking behavior under walk-forward financial validation. It compares QNNs against strong classical baselines and reports limitations, failure cases, and sensitivity to circuit design and shots. It does not claim quantum advantage.

## Executive Summary

This run evaluated `8` model families across `4` walk-forward splits and `1` seed(s). The report is descriptive: it compares ranking stability, portfolio accounting, and QNN shot sensitivity without asserting a tradable edge or quantum advantage.

## Methodology

The benchmark uses non-overlapping five-trading-day rebalance dates. Labels are SPY-relative: top 3 sectors are labeled 1, bottom 3 sectors are labeled 0, and middle sectors are dropped from classification training. Portfolio returns use absolute realized ETF returns.

This report was generated from results directory `results_cross_sectional_mvp` with config split cap `4` and seeds `[42]`. `configs/mvp.yaml` is the fast smoke path; `configs/full.yaml` is the slower research path.

## Dataset And Coverage

- Metric rows: `32`
- Portfolio rows: `160`
- QNN diagnostic rows: `4`
- Split date range: `2018-01-31` to `2019-03-28`
- Config hash: `72d81de7f0255f7bed98dd57f294d907c622ad38c2692b99afc723700be9d94e`
- Dataset hash: `1e2dd6ecabac15151c6861d9dca6dac40aa5a449cf7a6f4603156d93a66dc07d`
- Runtime seconds: `92.3441`

## Leakage Audit Interpretation

The split audit records train, validation, and test date boundaries plus the final realized target-window end before validation and test. Valid rows should have `train_last_forward_end < validation_start` and `validation_last_forward_end < test_start`; this report preserves those columns for manual review.

## Split Audit

| split_id | train_start | train_end | validation_start | validation_end | test_start | test_end | purge_gap_days | purge_trading_days_required | train_purge_trading_days | validation_purge_trading_days | train_last_forward_end | validation_last_forward_end | num_train_dates | num_validation_dates | num_test_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00_2018-12 | 2018-01-31 | 2018-10-10 | 2018-10-24 | 2018-11-21 | 2018-12-07 | 2018-12-31 | 7 | 5 | 5 | 6 | 2018-10-17 | 2018-11-29 | 36 | 5 | 4 |
| split_01_2019-01 | 2018-01-31 | 2018-11-07 | 2018-11-21 | 2018-12-21 | 2019-01-08 | 2019-01-30 | 7 | 5 | 5 | 6 | 2018-11-14 | 2018-12-31 | 40 | 5 | 4 |
| split_02_2019-02 | 2018-01-31 | 2018-12-07 | 2018-12-21 | 2019-01-23 | 2019-02-06 | 2019-02-28 | 7 | 5 | 5 | 5 | 2018-12-14 | 2019-01-30 | 44 | 5 | 4 |
| split_03_2019-03 | 2018-01-31 | 2019-01-08 | 2019-01-23 | 2019-02-21 | 2019-03-07 | 2019-03-28 | 7 | 5 | 6 | 5 | 2019-01-15 | 2019-02-28 | 48 | 5 | 4 |

## Prediction Shape Audit

Every model/date/seed group should score all 11 sector ETFs while retaining exactly 6 labeled training-comparable rows and 5 middle-sector inference rows. `model_rank_position` should match the model's own score ordering; realized future rank is reported separately as `realized_rank_position` in the prediction CSV.

| split_id | date | model | seed | num_scores | num_sector_etfs | num_labeled | num_middle_included | all_sector_scores | model_rank_matches_score | passes_shape_check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00_2018-12 | 2018-12-07 | logistic_regression | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-07 | naive_momentum | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-07 | pennylane_qnn | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-07 | random_forest | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-07 | rbf_svm | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-07 | ridge_linear | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-07 | small_mlp | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-07 | xgboost | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-14 | logistic_regression | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-14 | naive_momentum | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-14 | pennylane_qnn | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-14 | random_forest | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-14 | rbf_svm | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-14 | ridge_linear | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-14 | small_mlp | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-14 | xgboost | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-21 | logistic_regression | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-21 | naive_momentum | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-21 | pennylane_qnn | 42 | 11 | 11 | 6 | 5 | True | True | True |
| split_00_2018-12 | 2018-12-21 | random_forest | 42 | 11 | 11 | 6 | 5 | True | True | True |

## Model Run Status

Failed, skipped, constant-score, or diagnostics-missing runs should be visible rather than silently absorbed into aggregates.

| model | status | rows | total_constant_score_dates | total_nan_scores | diagnostics_available_rows |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | success | 4 | 0 | 0 | 0 |
| naive_momentum | success | 4 | 0 | 0 | 0 |
| pennylane_qnn | success | 4 | 0 | 0 | 4 |
| random_forest | success | 4 | 0 | 0 | 0 |
| rbf_svm | success | 4 | 0 | 0 | 0 |
| ridge_linear | success | 4 | 0 | 0 | 0 |
| small_mlp | success | 4 | 0 | 0 | 0 |
| xgboost | success | 4 | 0 | 0 | 0 |

## Portfolio Selection Audit

Portfolio top-3 selections should match the score-ranked prediction table with deterministic ticker tie-breaking.

| selection_matches_score_rank | size |
| --- | --- |
| True | 128 |

## Undefined Metric Audit

No undefined model-level metrics were produced in this run.

## Seed Stability Audit

| model | num_seeds | rank_ic_mean | rank_ic_std | rank_ic_nan_count | portfolio_net_return_seed_min | portfolio_net_return_seed_max | portfolio_net_return_seed_range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_linear | 1 | 0.0710 | 0.2144 | 0 | 0.0066 | 0.0066 | 0.0000 |
| logistic_regression | 1 | 0.0682 | 0.2043 | 0 | 0.0059 | 0.0059 | 0.0000 |
| rbf_svm | 1 | 0.0653 | 0.0887 | 0 | 0.0062 | 0.0062 | 0.0000 |
| xgboost | 1 | -0.0175 | 0.1790 | 0 | 0.0068 | 0.0068 | 0.0000 |
| random_forest | 1 | -0.0386 | 0.1888 | 0 | 0.0044 | 0.0044 | 0.0000 |
| pennylane_qnn | 1 | -0.0494 | 0.0541 | 0 | 0.0058 | 0.0058 | 0.0000 |
| naive_momentum | 1 | -0.0540 | 0.1389 | 0 | 0.0040 | 0.0040 | 0.0000 |
| small_mlp | 1 | -0.0869 | 0.0843 | 0 | 0.0047 | 0.0047 | 0.0000 |

## Feature Stability Audit

Calendar-heavy selected features have low within-date cross-sectional dispersion relative to their time-series dispersion; these can produce weak cross-sectional ranking behavior.

| selected_feature | selected_for_model | selected_count | calendar_heavy_rate | mean_cross_sectional_std_by_date | mean_time_series_std_by_ticker | mean_variance_ratio | mean_missing_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| relative_strength_20d | learned_models | 4 | 0.0000 | 0.0281 | 0.0270 | 1.0919 | 0.0000 |
| relative_strength_20d | naive_momentum | 4 | 0.0000 | 0.0281 | 0.0270 | 1.0919 | 0.0000 |
| ret_1d | learned_models | 4 | 0.0000 | 0.0070 | 0.0118 | 0.3518 | 0.0000 |
| volume_ratio_20d | learned_models | 4 | 0.0000 | 0.2675 | 0.4597 | 0.3386 | 0.0000 |
| volume_ratio_5d | learned_models | 3 | 0.0000 | 0.1926 | 0.3046 | 0.3999 | 0.0000 |
| ret_20d | learned_models | 1 | 0.0000 | 0.0263 | 0.0335 | 0.6178 | 0.0000 |

## Path Dependence Audit

| model | mean_net_return | mean_alpha_vs_spy | mean_turnover | total_net_return | best_split | best_split_net_return | best_split_share_of_total | mean_net_return_ex_best_split | worst_split | worst_split_net_return | best_date | best_date_net_return | best_date_share_of_total | mean_net_return_ex_best_date | worst_date | worst_date_net_return | positive_alpha_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | 0.0068 | 0.0004 | 0.7812 | 0.1080 | split_03_2019-03 | 0.0570 | 0.5280 | 0.0042 | split_00_2018-12 | -0.0027 | 2018-12-21 | 0.0417 | 0.3864 | 0.0044 | 2018-12-14 | -0.0669 | 0.5625 |
| ridge_linear | 0.0066 | 0.0003 | 0.6771 | 0.1059 | split_01_2019-01 | 0.0791 | 0.7470 | 0.0022 | split_00_2018-12 | -0.0300 | 2018-12-31 | 0.0295 | 0.2782 | 0.0051 | 2018-12-14 | -0.0780 | 0.6875 |
| spy_benchmark | 0.0063 | 0.0000 | 0.0000 | 0.1012 | split_01_2019-01 | 0.0608 | 0.6011 | 0.0034 | split_00_2018-12 | -0.0166 | 2018-12-21 | 0.0383 | 0.3786 | 0.0042 | 2018-12-14 | -0.0705 | 0.0000 |
| rbf_svm | 0.0062 | -0.0001 | 0.6771 | 0.0989 | split_01_2019-01 | 0.0742 | 0.7502 | 0.0021 | split_00_2018-12 | -0.0150 | 2018-12-31 | 0.0407 | 0.4115 | 0.0039 | 2018-12-14 | -0.0827 | 0.5000 |
| logistic_regression | 0.0059 | -0.0004 | 0.7188 | 0.0950 | split_01_2019-01 | 0.0684 | 0.7197 | 0.0022 | split_00_2018-12 | -0.0300 | 2018-12-31 | 0.0295 | 0.3101 | 0.0044 | 2018-12-14 | -0.0780 | 0.6250 |
| pennylane_qnn | 0.0058 | -0.0006 | 0.6562 | 0.0922 | split_01_2019-01 | 0.0767 | 0.8319 | 0.0013 | split_00_2018-12 | -0.0223 | 2018-12-21 | 0.0350 | 0.3794 | 0.0038 | 2018-12-14 | -0.0723 | 0.4375 |
| equal_weight_sector | 0.0057 | -0.0006 | 0.0313 | 0.0910 | split_01_2019-01 | 0.0565 | 0.6211 | 0.0029 | split_00_2018-12 | -0.0204 | 2018-12-31 | 0.0296 | 0.3247 | 0.0041 | 2018-12-14 | -0.0676 | 0.4375 |
| small_mlp | 0.0047 | -0.0016 | 0.5521 | 0.0753 | split_01_2019-01 | 0.0608 | 0.8069 | 0.0012 | split_00_2018-12 | -0.0319 | 2018-12-31 | 0.0295 | 0.3910 | 0.0031 | 2018-12-14 | -0.0780 | 0.4375 |
| random_forest | 0.0044 | -0.0020 | 0.7188 | 0.0698 | split_03_2019-03 | 0.0615 | 0.8809 | 0.0007 | split_00_2018-12 | -0.0217 | 2018-12-21 | 0.0451 | 0.6468 | 0.0016 | 2018-12-14 | -0.0783 | 0.2500 |
| naive_momentum | 0.0040 | -0.0023 | 0.3021 | 0.0637 | split_01_2019-01 | 0.0512 | 0.8039 | 0.0010 | split_00_2018-12 | -0.0407 | 2019-03-07 | 0.0237 | 0.3726 | 0.0027 | 2018-12-14 | -0.0581 | 0.6250 |

## Model Aggregate Summary

| model | roc_auc_mean | roc_auc_std | balanced_accuracy_mean | balanced_accuracy_std | f1_mean | f1_std | precision_at_3_mean | precision_at_3_std | rank_ic_mean | rank_ic_std | portfolio_net_return_mean_mean | portfolio_net_return_mean_std | portfolio_alpha_mean_mean | portfolio_alpha_mean_std | portfolio_sharpe_mean | portfolio_sharpe_std | portfolio_max_drawdown_mean | portfolio_max_drawdown_std | portfolio_turnover_mean_mean | portfolio_turnover_mean_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_linear | 0.4948 | 0.1010 | 0.5000 | 0.1179 | 0.5000 | 0.1179 | 0.2500 | 0.0680 | 0.0710 | 0.2144 | 0.0066 | 0.0000 | 0.0003 | 0.0000 | 1.7472 | 0.0000 | -0.0780 | 0.0000 | 0.6771 | 0.0000 |
| logistic_regression | 0.4896 | 0.0991 | 0.5000 | 0.1179 | 0.5000 | 0.1179 | 0.2292 | 0.0417 | 0.0682 | 0.2043 | 0.0059 | 0.0000 | -0.0004 | 0.0000 | 1.5892 | 0.0000 | -0.0780 | 0.0000 | 0.7188 | 0.0000 |
| rbf_svm | 0.5122 | 0.0630 | 0.5417 | 0.0481 | 0.5417 | 0.0481 | 0.2917 | 0.1076 | 0.0653 | 0.0887 | 0.0062 | 0.0000 | -0.0001 | 0.0000 | 1.5431 | 0.0000 | -0.0827 | 0.0000 | 0.6771 | 0.0000 |
| xgboost | 0.5000 | 0.1174 | 0.5208 | 0.1049 | 0.5208 | 0.1049 | 0.3333 | 0.1179 | -0.0175 | 0.1790 | 0.0068 | 0.0000 | 0.0004 | 0.0000 | 1.9681 | 0.0000 | -0.0669 | 0.0000 | 0.7812 | 0.0000 |
| random_forest | 0.4948 | 0.1343 | 0.4583 | 0.1443 | 0.4583 | 0.1443 | 0.2917 | 0.0833 | -0.0386 | 0.1888 | 0.0044 | 0.0000 | -0.0020 | 0.0000 | 1.1201 | 0.0000 | -0.0783 | 0.0000 | 0.7188 | 0.0000 |
| pennylane_qnn | 0.4549 | 0.0430 | 0.5000 | 0.0680 | 0.5000 | 0.0680 | 0.2708 | 0.1250 | -0.0494 | 0.0541 | 0.0058 | 0.0000 | -0.0006 | 0.0000 | 1.6848 | 0.0000 | -0.0723 | 0.0000 | 0.6562 | 0.0000 |
| naive_momentum | 0.4427 | 0.1447 | 0.4375 | 0.1049 | 0.4375 | 0.1049 | 0.2917 | 0.0833 | -0.0540 | 0.1389 | 0.0040 | 0.0000 | -0.0023 | 0.0000 | 1.4070 | 0.0000 | -0.0581 | 0.0000 | 0.3021 | 0.0000 |
| small_mlp | 0.4705 | 0.1064 | 0.5000 | 0.0680 | 0.5000 | 0.0680 | 0.2708 | 0.1423 | -0.0869 | 0.0843 | 0.0047 | 0.0000 | -0.0016 | 0.0000 | 1.2895 | 0.0000 | -0.0780 | 0.0000 | 0.5521 | 0.0000 |

## Model Metrics

| split_id | seed | model | rows | labeled_rows | roc_auc | balanced_accuracy | f1 | precision_at_3 | rank_ic | portfolio_net_return_mean | portfolio_alpha_mean | portfolio_sharpe | portfolio_max_drawdown | portfolio_turnover_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_01_2019-01 | 42 | ridge_linear | 44 | 24 | 0.6319 | 0.6667 | 0.6667 | 0.3333 | 0.3182 | 0.0066 | 0.0003 | 1.7472 | -0.0780 | 0.6771 |
| split_01_2019-01 | 42 | logistic_regression | 44 | 24 | 0.6250 | 0.6667 | 0.6667 | 0.2500 | 0.2977 | 0.0059 | -0.0004 | 1.5892 | -0.0780 | 0.7188 |
| split_03_2019-03 | 42 | random_forest | 44 | 24 | 0.6944 | 0.6667 | 0.6667 | 0.4167 | 0.2432 | 0.0044 | -0.0020 | 1.1201 | -0.0783 | 0.7188 |
| split_03_2019-03 | 42 | xgboost | 44 | 24 | 0.6701 | 0.6667 | 0.6667 | 0.5000 | 0.2319 | 0.0068 | 0.0004 | 1.9681 | -0.0669 | 0.7812 |
| split_01_2019-01 | 42 | rbf_svm | 44 | 24 | 0.5903 | 0.5833 | 0.5833 | 0.4167 | 0.1886 | 0.0062 | -0.0001 | 1.5431 | -0.0827 | 0.6771 |
| split_03_2019-03 | 42 | ridge_linear | 44 | 24 | 0.4722 | 0.4167 | 0.4167 | 0.2500 | 0.1818 | 0.0066 | 0.0003 | 1.7472 | -0.0780 | 0.6771 |
| split_03_2019-03 | 42 | logistic_regression | 44 | 24 | 0.4583 | 0.4167 | 0.4167 | 0.2500 | 0.1818 | 0.0059 | -0.0004 | 1.5892 | -0.0780 | 0.7188 |
| split_02_2019-02 | 42 | naive_momentum | 44 | 24 | 0.6319 | 0.5833 | 0.5833 | 0.4167 | 0.1227 | 0.0040 | -0.0023 | 1.4070 | -0.0581 | 0.3021 |
| split_00_2018-12 | 42 | rbf_svm | 44 | 24 | 0.4722 | 0.5000 | 0.5000 | 0.2500 | 0.0591 | 0.0062 | -0.0001 | 1.5431 | -0.0827 | 0.6771 |
| split_03_2019-03 | 42 | rbf_svm | 44 | 24 | 0.5347 | 0.5833 | 0.5833 | 0.3333 | 0.0341 | 0.0062 | -0.0001 | 1.5431 | -0.0827 | 0.6771 |
| split_01_2019-01 | 42 | pennylane_qnn | 44 | 24 | 0.5139 | 0.5833 | 0.5833 | 0.4167 | 0.0136 | 0.0058 | -0.0006 | 1.6848 | -0.0723 | 0.6562 |
| split_02_2019-02 | 42 | small_mlp | 44 | 24 | 0.4861 | 0.5000 | 0.5000 | 0.4167 | 0.0045 | 0.0047 | -0.0016 | 1.2895 | -0.0780 | 0.5521 |
| split_02_2019-02 | 42 | rbf_svm | 44 | 24 | 0.4514 | 0.5000 | 0.5000 | 0.1667 | -0.0205 | 0.0062 | -0.0001 | 1.5431 | -0.0827 | 0.6771 |
| split_02_2019-02 | 42 | pennylane_qnn | 44 | 24 | 0.4167 | 0.4167 | 0.4167 | 0.3333 | -0.0227 | 0.0058 | -0.0006 | 1.6848 | -0.0723 | 0.6562 |
| split_01_2019-01 | 42 | naive_momentum | 44 | 24 | 0.4653 | 0.4167 | 0.4167 | 0.2500 | -0.0273 | 0.0040 | -0.0023 | 1.4070 | -0.0581 | 0.3021 |
| split_00_2018-12 | 42 | xgboost | 44 | 24 | 0.4514 | 0.5000 | 0.5000 | 0.2500 | -0.0282 | 0.0068 | 0.0004 | 1.9681 | -0.0669 | 0.7812 |
| split_01_2019-01 | 42 | small_mlp | 44 | 24 | 0.5833 | 0.5833 | 0.5833 | 0.3333 | -0.0364 | 0.0047 | -0.0016 | 1.2895 | -0.0780 | 0.5521 |
| split_02_2019-02 | 42 | logistic_regression | 44 | 24 | 0.3889 | 0.4167 | 0.4167 | 0.1667 | -0.0841 | 0.0059 | -0.0004 | 1.5892 | -0.0780 | 0.7188 |
| split_02_2019-02 | 42 | xgboost | 44 | 24 | 0.4757 | 0.5000 | 0.5000 | 0.3333 | -0.0852 | 0.0068 | 0.0004 | 1.9681 | -0.0669 | 0.7812 |
| split_00_2018-12 | 42 | pennylane_qnn | 44 | 24 | 0.4306 | 0.5000 | 0.5000 | 0.1667 | -0.0886 | 0.0058 | -0.0006 | 1.6848 | -0.0723 | 0.6562 |

## Portfolio Aggregate Summary

| model | gross_return_mean | gross_return_std | net_return_mean | net_return_std | alpha_vs_spy_mean | alpha_vs_spy_std | turnover_mean | turnover_std | transaction_cost_mean | transaction_cost_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | 0.0071 | 0.0244 | 0.0068 | 0.0244 | 0.0004 | 0.0077 | 0.7812 | 0.2488 | 0.0004 | 0.0001 |
| ridge_linear | 0.0070 | 0.0269 | 0.0066 | 0.0269 | 0.0003 | 0.0057 | 0.6771 | 0.2315 | 0.0003 | 0.0001 |
| spy_benchmark | 0.0063 | 0.0252 | 0.0063 | 0.0252 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rbf_svm | 0.0065 | 0.0284 | 0.0062 | 0.0284 | -0.0001 | 0.0105 | 0.6771 | 0.2885 | 0.0003 | 0.0001 |
| logistic_regression | 0.0063 | 0.0266 | 0.0059 | 0.0265 | -0.0004 | 0.0056 | 0.7188 | 0.2562 | 0.0004 | 0.0001 |
| pennylane_qnn | 0.0061 | 0.0244 | 0.0058 | 0.0243 | -0.0006 | 0.0081 | 0.6562 | 0.3247 | 0.0003 | 0.0002 |
| equal_weight_sector | 0.0057 | 0.0233 | 0.0057 | 0.0233 | -0.0006 | 0.0036 | 0.0313 | 0.1250 | 0.0000 | 0.0001 |
| small_mlp | 0.0050 | 0.0260 | 0.0047 | 0.0259 | -0.0016 | 0.0057 | 0.5521 | 0.3146 | 0.0003 | 0.0002 |
| random_forest | 0.0047 | 0.0277 | 0.0044 | 0.0276 | -0.0020 | 0.0085 | 0.7188 | 0.2254 | 0.0004 | 0.0001 |
| naive_momentum | 0.0041 | 0.0201 | 0.0040 | 0.0201 | -0.0023 | 0.0097 | 0.3021 | 0.2127 | 0.0002 | 0.0001 |

## Portfolio Summary

| model | net_return_mean | alpha_vs_spy_mean | turnover_mean |
| --- | --- | --- | --- |
| equal_weight_sector | 0.0057 | -0.0006 | 0.0313 |
| logistic_regression | 0.0059 | -0.0004 | 0.7188 |
| naive_momentum | 0.0040 | -0.0023 | 0.3021 |
| pennylane_qnn | 0.0058 | -0.0006 | 0.6562 |
| random_forest | 0.0044 | -0.0020 | 0.7188 |
| rbf_svm | 0.0062 | -0.0001 | 0.6771 |
| ridge_linear | 0.0066 | 0.0003 | 0.6771 |
| small_mlp | 0.0047 | -0.0016 | 0.5521 |
| spy_benchmark | 0.0063 | 0.0000 | 0.0000 |
| xgboost | 0.0068 | 0.0004 | 0.7812 |

## Quantum Diagnostics

| split_id | seed | model | n_qubits | n_layers | trainable_parameter_count | train_seconds | inference_seconds | simulation_mode | shots | shot_sensitivity_samples | shot_sensitivity_shots | shot_score_correlation | shot_mean_abs_score_diff | shot_ranking_flip_rate | epochs_ran | best_validation_loss | selected_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00_2018-12 | 42 | pennylane_qnn | 4 | 1 | 13 | 21.1756 | 0.1289 | analytic_default_qubit | analytic | 8 | 1024 | 0.9950 | 0.0219 | 0.0385 | 10 | 0.7180 | ret_1d,ret_20d,volume_ratio_20d,relative_strength_20d |
| split_01_2019-01 | 42 | pennylane_qnn | 4 | 1 | 13 | 23.1143 | 0.1278 | analytic_default_qubit | analytic | 8 | 1024 | 0.9969 | 0.0232 | 0.0357 | 10 | 0.7425 | ret_1d,volume_ratio_5d,volume_ratio_20d,relative_strength_20d |
| split_02_2019-02 | 42 | pennylane_qnn | 4 | 1 | 13 | 14.9009 | 0.1279 | analytic_default_qubit | analytic | 8 | 1024 | 0.9941 | 0.0307 | 0.0000 | 6 | 0.6884 | ret_1d,volume_ratio_5d,volume_ratio_20d,relative_strength_20d |
| split_03_2019-03 | 42 | pennylane_qnn | 4 | 1 | 13 | 30.0744 | 0.1281 | analytic_default_qubit | analytic | 8 | 1024 | 0.9932 | 0.0203 | 0.1071 | 10 | 0.7143 | ret_1d,volume_ratio_5d,volume_ratio_20d,relative_strength_20d |

## QNN Shot-Sensitivity Interpretation

The QNN is trained analytically, then a fixed subset of inference samples is re-evaluated with 1024-shot sampling. Large score differences, weak score correlation, or high pairwise ranking flip rates indicate sensitivity to finite-shot execution and should be treated as a limitation rather than a positive result.

## QNN Failure Audit

| split_id | seed | qnn_date_groups | constant_score_groups | mean_score_std | min_unique_scores | max_unique_scores | roc_auc | balanced_accuracy | f1 | precision_at_3 | rank_ic | train_seconds | inference_seconds | shot_score_correlation | shot_mean_abs_score_diff | shot_ranking_flip_rate | epochs_ran | best_validation_loss | selected_features | any_undefined_metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00_2018-12 | 42 | 4 | 0 | 0.2254 | 11 | 11 | 0.4306 | 0.5000 | 0.5000 | 0.1667 | -0.0886 | 21.1756 | 0.1289 | 0.9950 | 0.0219 | 0.0385 | 10 | 0.7180 | ret_1d,ret_20d,volume_ratio_20d,relative_strength_20d | False |
| split_01_2019-01 | 42 | 4 | 0 | 0.2470 | 11 | 11 | 0.5139 | 0.5833 | 0.5833 | 0.4167 | 0.0136 | 23.1143 | 0.1278 | 0.9969 | 0.0232 | 0.0357 | 10 | 0.7425 | ret_1d,volume_ratio_5d,volume_ratio_20d,relative_strength_20d | False |
| split_02_2019-02 | 42 | 4 | 0 | 0.2939 | 11 | 11 | 0.4167 | 0.4167 | 0.4167 | 0.3333 | -0.0227 | 14.9009 | 0.1279 | 0.9941 | 0.0307 | 0.0000 | 6 | 0.6884 | ret_1d,volume_ratio_5d,volume_ratio_20d,relative_strength_20d | False |
| split_03_2019-03 | 42 | 4 | 0 | 0.2427 | 11 | 11 | 0.4583 | 0.5000 | 0.5000 | 0.1667 | -0.1000 | 30.0744 | 0.1281 | 0.9932 | 0.0203 | 0.1071 | 10 | 0.7143 | ret_1d,volume_ratio_5d,volume_ratio_20d,relative_strength_20d | False |

## Experimental Variant Comparison

If both standard and cross-sectional-aware feature-selection runs are available, this table compares whether the variant reduced QNN constant-score collapse, improved rank IC or precision@3, helped models beyond QNN, and survived transaction costs.

Cross-sectional-aware selection is treated as a diagnostic variant, not post-hoc performance tuning. For the QNN, constant-score groups reduced (4 -> 0), rank IC improved (-0.0735 -> -0.0494), precision@3 improved (0.2292 -> 0.2708), and alpha vs SPY after costs worsened (-0.0004 -> -0.0006). A positive diagnostic movement should still not be read as quantum advantage or a trading result.

| results_dir | feature_selection_mode | max_splits | seeds | model | rank_ic_mean | precision_at_3_mean | portfolio_net_return_mean | alpha_vs_spy_mean | turnover_mean | qnn_constant_groups | qnn_failure_rows | qnn_undefined_metric_rows | calendar_heavy_selected_features | selected_feature_rows | calendar_heavy_selected_feature_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results | standard | 4 | 42 | logistic_regression | -0.0369 | 0.2292 | 0.0054 | -0.0010 | 0.7188 | 4 | 1 | 1 | 8 | 20 | 0.4000 |
| results | standard | 4 | 42 | naive_momentum | -0.0540 | 0.2917 | 0.0040 | -0.0023 | 0.3021 | 4 | 1 | 1 | 8 | 20 | 0.4000 |
| results | standard | 4 | 42 | pennylane_qnn | -0.0735 | 0.2292 | 0.0059 | -0.0004 | 0.5521 | 4 | 1 | 1 | 8 | 20 | 0.4000 |
| results | standard | 4 | 42 | random_forest | -0.1350 | 0.2292 | 0.0025 | -0.0038 | 0.6771 | 4 | 1 | 1 | 8 | 20 | 0.4000 |
| results | standard | 4 | 42 | rbf_svm | 0.0551 | 0.3542 | 0.0067 | 0.0004 | 0.6354 | 4 | 1 | 1 | 8 | 20 | 0.4000 |
| results | standard | 4 | 42 | ridge_linear | -0.0403 | 0.2292 | 0.0054 | -0.0010 | 0.7188 | 4 | 1 | 1 | 8 | 20 | 0.4000 |
| results | standard | 4 | 42 | small_mlp | -0.0614 | 0.2083 | 0.0040 | -0.0023 | 0.7604 | 4 | 1 | 1 | 8 | 20 | 0.4000 |
| results | standard | 4 | 42 | xgboost | -0.1564 | 0.2083 | 0.0020 | -0.0044 | 0.6979 | 4 | 1 | 1 | 8 | 20 | 0.4000 |
| results_cross_sectional_mvp | cross_sectional_aware | 4 | 42 | logistic_regression | 0.0682 | 0.2292 | 0.0059 | -0.0004 | 0.7188 | 0 | 0 | 0 | 0 | 20 | 0.0000 |
| results_cross_sectional_mvp | cross_sectional_aware | 4 | 42 | naive_momentum | -0.0540 | 0.2917 | 0.0040 | -0.0023 | 0.3021 | 0 | 0 | 0 | 0 | 20 | 0.0000 |
| results_cross_sectional_mvp | cross_sectional_aware | 4 | 42 | pennylane_qnn | -0.0494 | 0.2708 | 0.0058 | -0.0006 | 0.6562 | 0 | 0 | 0 | 0 | 20 | 0.0000 |
| results_cross_sectional_mvp | cross_sectional_aware | 4 | 42 | random_forest | -0.0386 | 0.2917 | 0.0044 | -0.0020 | 0.7188 | 0 | 0 | 0 | 0 | 20 | 0.0000 |
| results_cross_sectional_mvp | cross_sectional_aware | 4 | 42 | rbf_svm | 0.0653 | 0.2917 | 0.0062 | -0.0001 | 0.6771 | 0 | 0 | 0 | 0 | 20 | 0.0000 |
| results_cross_sectional_mvp | cross_sectional_aware | 4 | 42 | ridge_linear | 0.0710 | 0.2500 | 0.0066 | 0.0003 | 0.6771 | 0 | 0 | 0 | 0 | 20 | 0.0000 |
| results_cross_sectional_mvp | cross_sectional_aware | 4 | 42 | small_mlp | -0.0869 | 0.2708 | 0.0047 | -0.0016 | 0.5521 | 0 | 0 | 0 | 0 | 20 | 0.0000 |
| results_cross_sectional_mvp | cross_sectional_aware | 4 | 42 | xgboost | -0.0175 | 0.3333 | 0.0068 | 0.0004 | 0.7812 | 0 | 0 | 0 | 0 | 20 | 0.0000 |

## Figures

- `results_cross_sectional_mvp/figures/model_rank_ic.png`
- `results_cross_sectional_mvp/figures/roc_auc_by_model.png`
- `results_cross_sectional_mvp/figures/split_rank_ic_by_model.png`
- `results_cross_sectional_mvp/figures/portfolio_equity.png`
- `results_cross_sectional_mvp/figures/alpha_vs_spy_by_model.png`
- `results_cross_sectional_mvp/figures/turnover_vs_return.png`
- `results_cross_sectional_mvp/figures/qnn_shot_sensitivity.png`

## Limitations

This benchmark is a sensitivity study, not a trading product. Undefined metrics are reported as NaN when samples are too small, scores are constant, labels are one-class, or volatility is zero. Classical baselines may dominate, multi-seed variation may be large, and any QNN behavior should be interpreted as experimental.

Failure cases matter here: negative rank IC, high turnover, weak shot-score correlation, or classical dominance are all valid benchmark outcomes and should not be hidden.
