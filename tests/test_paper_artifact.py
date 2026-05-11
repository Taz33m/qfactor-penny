from __future__ import annotations

from pathlib import Path

import pandas as pd

LOCAL_PATH_MARKER = "/" + "Users/"


def test_paper_artifact_uses_final_failure_mode_framing():
    paper = Path("paper/qfactor_penny_paper.md")
    text = paper.read_text(encoding="utf-8")
    assert text.startswith(
        "# QFactor-Penny: A Reproducible Benchmark of Trainable Quantum Circuits "
        "for Cross-Sectional Sector Return Ranking"
    )
    assert "## When Small QNNs Fail to Rank Assets Under Walk-Forward Validation" in text
    assert "### Core Thesis" not in text
    assert "## 2. Contributions" not in text
    assert "cross-sectional sector ETF return ranking" in text
    assert "`purge_trading_days`" in text
    assert "ticker as deterministic tie-breaker" in text
    assert "not real-hardware validation" in text
    assert "only 4 walk-forward splits and one seed (`42`)" in text
    assert "no statistical-significance claim" in text
    key_sentence = (
        "Cross-sectional-aware feature selection removed the observed QNN constant-score collapse, "
        "but QNN rank IC remained negative, precision@3 stayed near random, and no net alpha versus SPY "
        "survived transaction costs."
    )
    assert text.count(key_sentence) >= 3
    assert ("final_" + "interview_brief") not in text
    assert ("Interview" + " Brief") not in text
    assert LOCAL_PATH_MARKER not in text


def test_latex_paper_artifact_exists_and_preserves_claim_discipline():
    tex = Path("paper/qfactor_penny_paper.tex")
    text = tex.read_text(encoding="utf-8")
    assert "\\title{\\textbf{\\project: A Reproducible Benchmark of Trainable Quantum Circuits" in text
    assert "\\begin{abstract}" in text
    assert "\\section{Limitations}" in text
    assert "\\author{Tazeem Mahashin\\thanks{Rensselaer Polytechnic Institute}}" in text
    assert "\\\\[1.25em]" in text
    assert "Return and alpha values are mean five-trading-day rebalance-period values, not annualized returns." in text
    assert "not real-hardware validation" in text
    assert "\\section{IBM Quantum Hardware Robustness Audit}" in text
    assert "ibm\\_rensselaer" in text
    assert "HAL error \\texttt{9604}" in text
    assert "no statistical-significance claim, no quantum-advantage claim, and no trading-edge claim" in text
    assert "3/11=0.2727" in text
    assert "qnn\\_failure\\_audit.csv" in text
    assert "\\bibliography{references}" in text
    assert LOCAL_PATH_MARKER not in text


def test_paper_metrics_match_current_variant_comparison():
    text = Path("paper/qfactor_penny_paper.md").read_text(encoding="utf-8")
    comparison = pd.read_csv("results/experimental_variant_comparison.csv")
    qnn = comparison[comparison["model"] == "pennylane_qnn"].set_index("feature_selection_mode")
    standard = qnn.loc["standard"]
    cross = qnn.loc["cross_sectional_aware"]
    assert f"{int(standard['qnn_constant_groups'])} | {int(cross['qnn_constant_groups'])}" in text
    assert f"{standard['rank_ic_mean']:.4f} | {cross['rank_ic_mean']:.4f}" in text
    assert f"{standard['precision_at_3_mean']:.4f} | {cross['precision_at_3_mean']:.4f}" in text
    assert f"{standard['alpha_vs_spy_mean']:.4f} | {cross['alpha_vs_spy_mean']:.4f}" in text
    assert f"{standard['turnover_mean']:.4f} | {cross['turnover_mean']:.4f}" in text
    assert "3 / 11 = 0.2727" in text


def test_paper_referenced_figures_exist_and_are_relative():
    text = Path("paper/qfactor_penny_paper.md").read_text(encoding="utf-8")
    figure_paths = [
        "results/figures/model_rank_ic.png",
        "results/figures/roc_auc_by_model.png",
        "results/figures/split_rank_ic_by_model.png",
        "results/figures/portfolio_equity.png",
        "results/figures/alpha_vs_spy_by_model.png",
        "results/figures/turnover_vs_return.png",
        "results/figures/qnn_shot_sensitivity.png",
        "results_cross_sectional_mvp/figures/model_rank_ic.png",
        "results_cross_sectional_mvp/figures/qnn_shot_sensitivity.png",
        "results_hardware/figures/hardware_score_scatter.png",
    ]
    for figure_path in figure_paths:
        assert figure_path in text
        assert Path(figure_path).exists()


def test_latex_paper_references_existing_figures():
    text = Path("paper/qfactor_penny_paper.tex").read_text(encoding="utf-8")
    assert "\\graphicspath{{../results/figures/}{../results_cross_sectional_mvp/figures/}{../results_hardware/figures/}}" in text
    for figure in ["model_rank_ic.png", "turnover_vs_return.png", "qnn_shot_sensitivity.png", "hardware_score_scatter.png"]:
        assert figure in text


def test_latex_model_table_contains_all_variant_rows_and_references_bib():
    text = Path("paper/qfactor_penny_paper.tex").read_text(encoding="utf-8")
    assert "Naive momentum & Cross-sectional-aware & -0.0540 & 0.2917 & -0.0023 & 0.3021" in text
    assert "Random forest & Cross-sectional-aware & -0.0386 & 0.2917 & -0.0020 & 0.7188" in text
    assert "Small MLP & Cross-sectional-aware & -0.0869 & 0.2708 & -0.0016 & 0.5521" in text
    bib = Path("paper/references.bib").read_text(encoding="utf-8")
    for key in ["white2000reality", "bailey2017pbo", "harvey2016crosssection", "lopezdeprado2018afml"]:
        assert key in bib


def test_reviewer_checklist_maps_claims_to_artifacts():
    checklist = Path("paper/reviewer_checklist.md")
    text = checklist.read_text(encoding="utf-8")
    assert "Claim/Evidence Map" in text
    assert "results/qnn_failure_audit.csv" in text
    assert "results/feature_stability_summary.csv" in text
    assert "results_cross_sectional_mvp/qnn_failure_audit.csv" in text
    assert "results/experimental_variant_comparison.csv" in text
    assert "QNN rank IC remains negative (`-0.0494`)" in text
    assert "precision@3 remains near random (`0.2708` vs random `3 / 11 = 0.2727`)" in text
    assert "1024-shot sensitivity is simulator sampling after analytic training, not real-hardware validation" in text
    assert "Frozen-QNN hardware inference used `ibm_rensselaer`, 11 samples, 100 shots" in text
    assert "IBM HAL error `9604`" in text
    assert "4 MVP splits, one seed, only a small inference-only hardware audit, no statistical-significance claim" in text
    assert "does not support a quantum-advantage or trading-edge claim" in text
    assert LOCAL_PATH_MARKER not in text
