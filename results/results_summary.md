# QFactor-Penny Results Summary

This project evaluates whether small trainable PennyLane QNNs provide stable or useful ranking behavior under walk-forward financial validation. It compares QNNs against strong classical baselines and reports limitations, failure cases, and sensitivity to circuit design and shots. It does not claim quantum advantage.

## Executive Summary

This run evaluated `8` model families across `4` walk-forward splits and `1` seed(s). The report is descriptive: it compares ranking stability, portfolio accounting, and QNN shot sensitivity without asserting a tradable edge or quantum advantage.

## Methodology

The benchmark uses non-overlapping five-trading-day rebalance dates. Labels are SPY-relative: top 3 sectors are labeled 1, bottom 3 sectors are labeled 0, and middle sectors are dropped from classification training. Portfolio returns use absolute realized ETF returns.

This report was generated from results directory `results` with config split cap `4` and seeds `[42]`. `configs/mvp.yaml` is the fast smoke path; `configs/full.yaml` is the slower research path.

## Dataset And Coverage

- Metric rows: `32`
- Portfolio rows: `160`
- QNN diagnostic rows: `4`
- Split date range: `2018-01-31` to `2019-03-28`
- Config hash: `7ba93f5047f7c458379d2f0d8ec46bf0080b08395ee1535e97c00d69b6cf6000`
- Dataset hash: `1e2dd6ecabac15151c6861d9dca6dac40aa5a449cf7a6f4603156d93a66dc07d`
- Runtime seconds: `43.9120`

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
| pennylane_qnn | success | 4 | 4 | 0 | 4 |
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

| metric | nan_count |
| --- | --- |
| roc_auc | 1 |
| balanced_accuracy | 1 |
| f1 | 1 |
| rank_ic | 1 |

## Seed Stability Audit

| model | num_seeds | rank_ic_mean | rank_ic_std | rank_ic_nan_count | portfolio_net_return_seed_min | portfolio_net_return_seed_max | portfolio_net_return_seed_range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_svm | 1 | 0.0551 | 0.1436 | 0 | 0.0067 | 0.0067 | 0.0000 |
| logistic_regression | 1 | -0.0369 | 0.1232 | 0 | 0.0054 | 0.0054 | 0.0000 |
| ridge_linear | 1 | -0.0403 | 0.1132 | 0 | 0.0054 | 0.0054 | 0.0000 |
| naive_momentum | 1 | -0.0540 | 0.1389 | 0 | 0.0040 | 0.0040 | 0.0000 |
| small_mlp | 1 | -0.0614 | 0.1224 | 0 | 0.0040 | 0.0040 | 0.0000 |
| pennylane_qnn | 1 | -0.0735 | 0.1623 | 1 | 0.0059 | 0.0059 | 0.0000 |
| random_forest | 1 | -0.1350 | 0.0571 | 0 | 0.0025 | 0.0025 | 0.0000 |
| xgboost | 1 | -0.1564 | 0.0442 | 0 | 0.0020 | 0.0020 | 0.0000 |

## Feature Stability Audit

Calendar-heavy selected features have low within-date cross-sectional dispersion relative to their time-series dispersion; these can produce weak cross-sectional ranking behavior.

| selected_feature | selected_for_model | selected_count | calendar_heavy_rate | mean_cross_sectional_std_by_date | mean_time_series_std_by_ticker | mean_variance_ratio | mean_missing_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| month_cos | learned_models | 4 | 1.0000 | 0.0000 | 0.6347 | 0.0000 | 0.0000 |
| weekday_cos | learned_models | 3 | 1.0000 | 0.0000 | 0.6618 | 0.0000 | 0.0000 |
| month_sin | learned_models | 1 | 1.0000 | 0.0000 | 0.6914 | 0.0000 | 0.0000 |
| relative_strength_20d | naive_momentum | 4 | 0.0000 | 0.0281 | 0.0270 | 1.0919 | 0.0000 |
| ret_1d | learned_models | 4 | 0.0000 | 0.0070 | 0.0118 | 0.3518 | 0.0000 |
| volume_ratio_20d | learned_models | 2 | 0.0000 | 0.2660 | 0.4568 | 0.3390 | 0.0000 |
| ret_20d | learned_models | 1 | 0.0000 | 0.0263 | 0.0335 | 0.6178 | 0.0000 |
| vol_20d | learned_models | 1 | 0.0000 | 0.0019 | 0.0032 | 0.3391 | 0.0000 |

## Path Dependence Audit

| model | mean_net_return | mean_alpha_vs_spy | mean_turnover | total_net_return | best_split | best_split_net_return | best_split_share_of_total | mean_net_return_ex_best_split | worst_split | worst_split_net_return | best_date | best_date_net_return | best_date_share_of_total | mean_net_return_ex_best_date | worst_date | worst_date_net_return | positive_alpha_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_svm | 0.0067 | 0.0004 | 0.6354 | 0.1072 | split_01_2019-01 | 0.0722 | 0.6730 | 0.0029 | split_00_2018-12 | -0.0017 | 2018-12-21 | 0.0465 | 0.4333 | 0.0041 | 2018-12-14 | -0.0681 | 0.5000 |
| spy_benchmark | 0.0063 | 0.0000 | 0.0000 | 0.1012 | split_01_2019-01 | 0.0608 | 0.6011 | 0.0034 | split_00_2018-12 | -0.0166 | 2018-12-21 | 0.0383 | 0.3786 | 0.0042 | 2018-12-14 | -0.0705 | 0.0000 |
| pennylane_qnn | 0.0059 | -0.0004 | 0.5521 | 0.0946 | split_01_2019-01 | 0.0716 | 0.7569 | 0.0019 | split_00_2018-12 | -0.0323 | 2019-01-23 | 0.0346 | 0.3661 | 0.0040 | 2018-12-14 | -0.0714 | 0.4375 |
| equal_weight_sector | 0.0057 | -0.0006 | 0.0313 | 0.0910 | split_01_2019-01 | 0.0565 | 0.6211 | 0.0029 | split_00_2018-12 | -0.0204 | 2018-12-31 | 0.0296 | 0.3247 | 0.0041 | 2018-12-14 | -0.0676 | 0.4375 |
| logistic_regression | 0.0054 | -0.0010 | 0.7188 | 0.0859 | split_01_2019-01 | 0.0651 | 0.7576 | 0.0017 | split_00_2018-12 | -0.0120 | 2018-12-21 | 0.0463 | 0.5389 | 0.0026 | 2018-12-14 | -0.0780 | 0.4375 |
| ridge_linear | 0.0054 | -0.0010 | 0.7188 | 0.0859 | split_01_2019-01 | 0.0651 | 0.7576 | 0.0017 | split_00_2018-12 | -0.0120 | 2018-12-21 | 0.0463 | 0.5389 | 0.0026 | 2018-12-14 | -0.0780 | 0.4375 |
| small_mlp | 0.0040 | -0.0023 | 0.7604 | 0.0642 | split_01_2019-01 | 0.0701 | 1.0917 | -0.0005 | split_00_2018-12 | -0.0387 | 2018-12-31 | 0.0295 | 0.4589 | 0.0023 | 2018-12-14 | -0.0754 | 0.3750 |
| naive_momentum | 0.0040 | -0.0023 | 0.3021 | 0.0637 | split_01_2019-01 | 0.0512 | 0.8039 | 0.0010 | split_00_2018-12 | -0.0407 | 2019-03-07 | 0.0237 | 0.3726 | 0.0027 | 2018-12-14 | -0.0581 | 0.6250 |
| random_forest | 0.0025 | -0.0038 | 0.6771 | 0.0398 | split_01_2019-01 | 0.0567 | 1.4241 | -0.0014 | split_00_2018-12 | -0.0428 | 2019-01-23 | 0.0310 | 0.7789 | 0.0006 | 2018-12-14 | -0.0723 | 0.3125 |
| xgboost | 0.0020 | -0.0044 | 0.6979 | 0.0314 | split_03_2019-03 | 0.0294 | 0.9378 | 0.0002 | split_00_2018-12 | -0.0223 | 2018-12-21 | 0.0307 | 0.9795 | 0.0000 | 2018-12-14 | -0.0682 | 0.2500 |

## Model Aggregate Summary

| model | roc_auc_mean | roc_auc_std | balanced_accuracy_mean | balanced_accuracy_std | f1_mean | f1_std | precision_at_3_mean | precision_at_3_std | rank_ic_mean | rank_ic_std | portfolio_net_return_mean_mean | portfolio_net_return_mean_std | portfolio_alpha_mean_mean | portfolio_alpha_mean_std | portfolio_sharpe_mean | portfolio_sharpe_std | portfolio_max_drawdown_mean | portfolio_max_drawdown_std | portfolio_turnover_mean_mean | portfolio_turnover_mean_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_svm | 0.5174 | 0.0582 | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.3542 | 0.1969 | 0.0551 | 0.1436 | 0.0067 | 0.0000 | 0.0004 | 0.0000 | 1.8974 | 0.0000 | -0.0681 | 0.0000 | 0.6354 | 0.0000 |
| logistic_regression | 0.4635 | 0.1007 | 0.4792 | 0.1049 | 0.4792 | 0.1049 | 0.2292 | 0.1049 | -0.0369 | 0.1232 | 0.0054 | 0.0000 | -0.0010 | 0.0000 | 1.4048 | 0.0000 | -0.0780 | 0.0000 | 0.7188 | 0.0000 |
| ridge_linear | 0.4618 | 0.0980 | 0.4792 | 0.1049 | 0.4792 | 0.1049 | 0.2292 | 0.1049 | -0.0403 | 0.1132 | 0.0054 | 0.0000 | -0.0010 | 0.0000 | 1.4048 | 0.0000 | -0.0780 | 0.0000 | 0.7188 | 0.0000 |
| naive_momentum | 0.4427 | 0.1447 | 0.4375 | 0.1049 | 0.4375 | 0.1049 | 0.2917 | 0.0833 | -0.0540 | 0.1389 | 0.0040 | 0.0000 | -0.0023 | 0.0000 | 1.4070 | 0.0000 | -0.0581 | 0.0000 | 0.3021 | 0.0000 |
| small_mlp | 0.4531 | 0.0917 | 0.4375 | 0.0798 | 0.4375 | 0.0798 | 0.2083 | 0.1076 | -0.0614 | 0.1224 | 0.0040 | 0.0000 | -0.0023 | 0.0000 | 1.1352 | 0.0000 | -0.0754 | 0.0000 | 0.7604 | 0.0000 |
| pennylane_qnn | 0.4167 | 0.0524 | 0.4444 | 0.0481 | 0.4444 | 0.0481 | 0.2292 | 0.0798 | -0.0735 | 0.1623 | 0.0059 | 0.0000 | -0.0004 | 0.0000 | 1.6117 | 0.0000 | -0.0714 | 0.0000 | 0.5521 | 0.0000 |
| random_forest | 0.4054 | 0.0660 | 0.4167 | 0.0962 | 0.4167 | 0.0962 | 0.2292 | 0.1423 | -0.1350 | 0.0571 | 0.0025 | 0.0000 | -0.0038 | 0.0000 | 0.7402 | 0.0000 | -0.0723 | 0.0000 | 0.6771 | 0.0000 |
| xgboost | 0.4115 | 0.0746 | 0.4271 | 0.0712 | 0.4325 | 0.0750 | 0.2083 | 0.1076 | -0.1564 | 0.0442 | 0.0020 | 0.0000 | -0.0044 | 0.0000 | 0.6006 | 0.0000 | -0.0682 | 0.0000 | 0.6979 | 0.0000 |

## Model Metrics

| split_id | seed | model | rows | labeled_rows | roc_auc | balanced_accuracy | f1 | precision_at_3 | rank_ic | portfolio_net_return_mean | portfolio_alpha_mean | portfolio_sharpe | portfolio_max_drawdown | portfolio_turnover_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_01_2019-01 | 42 | rbf_svm | 44 | 24 | 0.5417 | 0.5000 | 0.5000 | 0.3333 | 0.1841 | 0.0067 | 0.0004 | 1.8974 | -0.0681 | 0.6354 |
| split_01_2019-01 | 42 | logistic_regression | 44 | 24 | 0.5833 | 0.5833 | 0.5833 | 0.2500 | 0.1455 | 0.0054 | -0.0010 | 1.4048 | -0.0780 | 0.7188 |
| split_00_2018-12 | 42 | rbf_svm | 44 | 24 | 0.5417 | 0.5000 | 0.5000 | 0.5000 | 0.1318 | 0.0067 | 0.0004 | 1.8974 | -0.0681 | 0.6354 |
| split_01_2019-01 | 42 | ridge_linear | 44 | 24 | 0.5764 | 0.5833 | 0.5833 | 0.2500 | 0.1273 | 0.0054 | -0.0010 | 1.4048 | -0.0780 | 0.7188 |
| split_02_2019-02 | 42 | naive_momentum | 44 | 24 | 0.6319 | 0.5833 | 0.5833 | 0.4167 | 0.1227 | 0.0040 | -0.0023 | 1.4070 | -0.0581 | 0.3021 |
| split_01_2019-01 | 42 | small_mlp | 44 | 24 | 0.5694 | 0.5000 | 0.5000 | 0.3333 | 0.1091 | 0.0040 | -0.0023 | 1.1352 | -0.0754 | 0.7604 |
| split_02_2019-02 | 42 | pennylane_qnn | 44 | 24 | 0.4653 | 0.4167 | 0.4167 | 0.2500 | 0.0591 | 0.0059 | -0.0004 | 1.6117 | -0.0714 | 0.5521 |
| split_02_2019-02 | 42 | rbf_svm | 44 | 24 | 0.5556 | 0.5000 | 0.5000 | 0.5000 | 0.0477 | 0.0067 | 0.0004 | 1.8974 | -0.0681 | 0.6354 |
| split_01_2019-01 | 42 | pennylane_qnn | 44 | 24 | 0.4236 | 0.5000 | 0.5000 | 0.3333 | -0.0250 | 0.0059 | -0.0004 | 1.6117 | -0.0714 | 0.5521 |
| split_01_2019-01 | 42 | naive_momentum | 44 | 24 | 0.4653 | 0.4167 | 0.4167 | 0.2500 | -0.0273 | 0.0040 | -0.0023 | 1.4070 | -0.0581 | 0.3021 |
| split_01_2019-01 | 42 | random_forest | 44 | 24 | 0.3958 | 0.5000 | 0.5000 | 0.4167 | -0.0591 | 0.0025 | -0.0038 | 0.7402 | -0.0723 | 0.6771 |
| split_03_2019-03 | 42 | logistic_regression | 44 | 24 | 0.4028 | 0.5000 | 0.5000 | 0.0833 | -0.0705 | 0.0054 | -0.0010 | 1.4048 | -0.0780 | 0.7188 |
| split_03_2019-03 | 42 | ridge_linear | 44 | 24 | 0.4028 | 0.5000 | 0.5000 | 0.0833 | -0.0705 | 0.0054 | -0.0010 | 1.4048 | -0.0780 | 0.7188 |
| split_03_2019-03 | 42 | small_mlp | 44 | 24 | 0.4028 | 0.5000 | 0.5000 | 0.0833 | -0.0705 | 0.0040 | -0.0023 | 1.1352 | -0.0754 | 0.7604 |
| split_00_2018-12 | 42 | naive_momentum | 44 | 24 | 0.3819 | 0.3333 | 0.3333 | 0.2500 | -0.1045 | 0.0040 | -0.0023 | 1.4070 | -0.0581 | 0.3021 |
| split_02_2019-02 | 42 | small_mlp | 44 | 24 | 0.3611 | 0.3333 | 0.3333 | 0.2500 | -0.1045 | 0.0040 | -0.0023 | 1.1352 | -0.0754 | 0.7604 |
| split_00_2018-12 | 42 | logistic_regression | 44 | 24 | 0.5069 | 0.5000 | 0.5000 | 0.3333 | -0.1068 | 0.0054 | -0.0010 | 1.4048 | -0.0780 | 0.7188 |
| split_00_2018-12 | 42 | ridge_linear | 44 | 24 | 0.5069 | 0.5000 | 0.5000 | 0.3333 | -0.1068 | 0.0054 | -0.0010 | 1.4048 | -0.0780 | 0.7188 |
| split_02_2019-02 | 42 | ridge_linear | 44 | 24 | 0.3611 | 0.3333 | 0.3333 | 0.2500 | -0.1114 | 0.0054 | -0.0010 | 1.4048 | -0.0780 | 0.7188 |
| split_02_2019-02 | 42 | logistic_regression | 44 | 24 | 0.3611 | 0.3333 | 0.3333 | 0.2500 | -0.1159 | 0.0054 | -0.0010 | 1.4048 | -0.0780 | 0.7188 |

## Portfolio Aggregate Summary

| model | gross_return_mean | gross_return_std | net_return_mean | net_return_std | alpha_vs_spy_mean | alpha_vs_spy_std | turnover_mean | turnover_std | transaction_cost_mean | transaction_cost_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_svm | 0.0070 | 0.0251 | 0.0067 | 0.0251 | 0.0004 | 0.0071 | 0.6354 | 0.3116 | 0.0003 | 0.0002 |
| spy_benchmark | 0.0063 | 0.0252 | 0.0063 | 0.0252 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pennylane_qnn | 0.0062 | 0.0260 | 0.0059 | 0.0260 | -0.0004 | 0.0070 | 0.5521 | 0.3373 | 0.0003 | 0.0002 |
| equal_weight_sector | 0.0057 | 0.0233 | 0.0057 | 0.0233 | -0.0006 | 0.0036 | 0.0313 | 0.1250 | 0.0000 | 0.0001 |
| logistic_regression | 0.0057 | 0.0272 | 0.0054 | 0.0271 | -0.0010 | 0.0057 | 0.7188 | 0.2254 | 0.0004 | 0.0001 |
| ridge_linear | 0.0057 | 0.0272 | 0.0054 | 0.0271 | -0.0010 | 0.0057 | 0.7188 | 0.2254 | 0.0004 | 0.0001 |
| small_mlp | 0.0044 | 0.0251 | 0.0040 | 0.0251 | -0.0023 | 0.0064 | 0.7604 | 0.2105 | 0.0004 | 0.0001 |
| naive_momentum | 0.0041 | 0.0201 | 0.0040 | 0.0201 | -0.0023 | 0.0097 | 0.3021 | 0.2127 | 0.0002 | 0.0001 |
| random_forest | 0.0028 | 0.0239 | 0.0025 | 0.0239 | -0.0038 | 0.0096 | 0.6771 | 0.2315 | 0.0003 | 0.0001 |
| xgboost | 0.0023 | 0.0232 | 0.0020 | 0.0232 | -0.0044 | 0.0054 | 0.6979 | 0.2127 | 0.0003 | 0.0001 |

## Portfolio Summary

| model | net_return_mean | alpha_vs_spy_mean | turnover_mean |
| --- | --- | --- | --- |
| equal_weight_sector | 0.0057 | -0.0006 | 0.0313 |
| logistic_regression | 0.0054 | -0.0010 | 0.7188 |
| naive_momentum | 0.0040 | -0.0023 | 0.3021 |
| pennylane_qnn | 0.0059 | -0.0004 | 0.5521 |
| random_forest | 0.0025 | -0.0038 | 0.6771 |
| rbf_svm | 0.0067 | 0.0004 | 0.6354 |
| ridge_linear | 0.0054 | -0.0010 | 0.7188 |
| small_mlp | 0.0040 | -0.0023 | 0.7604 |
| spy_benchmark | 0.0063 | 0.0000 | 0.0000 |
| xgboost | 0.0020 | -0.0044 | 0.6979 |

## Quantum Diagnostics

| split_id | seed | model | n_qubits | n_layers | trainable_parameter_count | train_seconds | inference_seconds | simulation_mode | shots | shot_sensitivity_samples | shot_sensitivity_shots | shot_score_correlation | shot_mean_abs_score_diff | shot_ranking_flip_rate | epochs_ran | best_validation_loss | selected_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00_2018-12 | 42 | pennylane_qnn | 4 | 1 | 13 | 8.4332 | 0.1265 | analytic_default_qubit | analytic | 8 | 1024 | 0.9764 | 0.0197 | 0.0000 | 4 | 0.7491 | ret_1d,ret_20d,vol_20d,month_cos |
| split_01_2019-01 | 42 | pennylane_qnn | 4 | 1 | 13 | 11.3534 | 0.1264 | analytic_default_qubit | analytic | 8 | 1024 | 0.8979 | 0.0268 | 0.1481 | 5 | 0.6960 | ret_1d,volume_ratio_20d,weekday_cos,month_cos |
| split_02_2019-02 | 42 | pennylane_qnn | 4 | 1 | 13 | 10.1182 | 0.1304 | analytic_default_qubit | analytic | 8 | 1024 | -0.4807 | 0.0259 | 0.7143 | 4 | 0.6985 | ret_1d,volume_ratio_20d,weekday_cos,month_cos |
| split_03_2019-03 | 42 | pennylane_qnn | 4 | 1 | 13 | 11.0363 | 0.1280 | analytic_default_qubit | analytic | 8 | 1024 | -0.6488 | 0.0202 | 0.7692 | 4 | 0.6933 | ret_1d,weekday_cos,month_sin,month_cos |

## QNN Shot-Sensitivity Interpretation

The QNN is trained analytically, then a fixed subset of inference samples is re-evaluated with 1024-shot sampling. Large score differences, weak score correlation, or high pairwise ranking flip rates indicate sensitivity to finite-shot execution and should be treated as a limitation rather than a positive result.

## QNN Failure Audit

| split_id | seed | qnn_date_groups | constant_score_groups | mean_score_std | min_unique_scores | max_unique_scores | roc_auc | balanced_accuracy | f1 | precision_at_3 | rank_ic | train_seconds | inference_seconds | shot_score_correlation | shot_mean_abs_score_diff | shot_ranking_flip_rate | epochs_ran | best_validation_loss | selected_features | any_undefined_metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00_2018-12 | 42 | 4 | 0 | 0.1335 | 11 | 11 | 0.3611 | 0.4167 | 0.4167 | 0.1667 | -0.2545 | 8.4332 | 0.1265 | 0.9764 | 0.0197 | 0.0000 | 4 | 0.7491 | ret_1d,ret_20d,vol_20d,month_cos | False |
| split_01_2019-01 | 42 | 4 | 0 | 0.0456 | 11 | 11 | 0.4236 | 0.5000 | 0.5000 | 0.3333 | -0.0250 | 11.3534 | 0.1264 | 0.8979 | 0.0268 | 0.1481 | 5 | 0.6960 | ret_1d,volume_ratio_20d,weekday_cos,month_cos | False |
| split_02_2019-02 | 42 | 4 | 0 | 0.0010 | 11 | 11 | 0.4653 | 0.4167 | 0.4167 | 0.2500 | 0.0591 | 10.1182 | 0.1304 | -0.4807 | 0.0259 | 0.7143 | 4 | 0.6985 | ret_1d,volume_ratio_20d,weekday_cos,month_cos | False |
| split_03_2019-03 | 42 | 4 | 4 | 0.0000 | 1 | 1 | NaN | NaN | NaN | 0.1667 | NaN | 11.0363 | 0.1280 | -0.6488 | 0.0202 | 0.7692 | 4 | 0.6933 | ret_1d,weekday_cos,month_sin,month_cos | True |

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

- `results/figures/model_rank_ic.png`
- `results/figures/roc_auc_by_model.png`
- `results/figures/split_rank_ic_by_model.png`
- `results/figures/portfolio_equity.png`
- `results/figures/alpha_vs_spy_by_model.png`
- `results/figures/turnover_vs_return.png`
- `results/figures/qnn_shot_sensitivity.png`

## Limitations

This benchmark is a sensitivity study, not a trading product. Undefined metrics are reported as NaN when samples are too small, scores are constant, labels are one-class, or volatility is zero. Classical baselines may dominate, multi-seed variation may be large, and any QNN behavior should be interpreted as experimental.

Failure cases matter here: negative rank IC, high turnover, weak shot-score correlation, or classical dominance are all valid benchmark outcomes and should not be hidden.
