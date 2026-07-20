#!/usr/bin/env python3
"""Staged source-only search for the pure-formula SCJKE framework.

The search chooses among a finite catalogue of fixed equations. It does not fit
classification coefficients. Target labels are evaluated only after a formula
has been frozen using cold-both folds made inside source_train.csv.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from formula_search import (
    FormulaConfig,
    FormulaEvidenceEngine,
    cold_both_split,
    make_markdown,
    metrics,
    prepare_frame,
)


def objective(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("config", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        mean_auprc=("auprc", "mean"),
        std_auroc=("auroc", "std"),
        std_auprc=("auprc", "std"),
        folds=("fold", "nunique"),
    )
    grouped["objective"] = 0.5 * (grouped.mean_auroc + grouped.mean_auprc)
    grouped["robust_objective"] = grouped.objective - 0.05 * (
        grouped.std_auroc.fillna(0) + grouped.std_auprc.fillna(0)
    )
    return grouped.sort_values(
        ["robust_objective", "mean_auroc", "mean_auprc"], ascending=False
    ).reset_index(drop=True)


def evaluate_configs(
    dataset: str,
    fold: int,
    engine: FormulaEvidenceEngine,
    labels: np.ndarray,
    configs: Sequence[FormulaConfig],
    stage: str,
) -> pd.DataFrame:
    rows = []
    for index, config in enumerate(configs, start=1):
        pred = engine.score(config)
        met = metrics(labels, pred["score"], pred["probability"])
        rows.append(
            {
                "dataset": dataset,
                "stage": stage,
                "fold": fold,
                "config": config.name,
                **met,
            }
        )
        if index % 50 == 0 or index == len(configs):
            print(f"[{dataset}] {stage} fold={fold}: {index}/{len(configs)}")
    return pd.DataFrame(rows)


def unique_configs(configs: Iterable[FormulaConfig]) -> List[FormulaConfig]:
    return list(dict.fromkeys(configs))


def stage1_kernel_configs() -> List[FormulaConfig]:
    configs = []
    for drug_kernel in ["morgan", "morgan_maccs", "multi_fp", "fp_desc", "all"]:
        for protein_kernel in ["aa3", "aa23", "aa_group", "seq_desc", "all"]:
            configs.append(
                FormulaConfig(
                    drug_kernel=drug_kernel,
                    protein_kernel=protein_kernel,
                    joint_rule="soft_and",
                    k_joint=200,
                    k_marginal=200,
                    degree_gamma=0.25,
                    evidence_mode="support_adaptive",
                    prototype_weight=0.0,
                    propensity_weight=0.0,
                )
            )
    return configs


def stage2_structural_configs(kernel_pairs: Sequence[Tuple[str, str]]) -> List[FormulaConfig]:
    configs = []
    for drug_kernel, protein_kernel in kernel_pairs:
        for joint_rule in ["product", "harmonic", "minimum", "soft_and"]:
            for k_joint in [50, 100, 200, 400]:
                for degree_gamma in [0.0, 0.25]:
                    for evidence_mode in [
                        "joint",
                        "support_adaptive",
                        "joint_plus_marginal",
                    ]:
                        configs.append(
                            FormulaConfig(
                                drug_kernel=drug_kernel,
                                protein_kernel=protein_kernel,
                                joint_rule=joint_rule,
                                k_joint=k_joint,
                                k_marginal=200,
                                degree_gamma=degree_gamma,
                                evidence_mode=evidence_mode,
                                prototype_weight=0.0,
                                propensity_weight=0.0,
                            )
                        )
    return unique_configs(configs)


def stage3_global_evidence_configs(base_configs: Sequence[FormulaConfig]) -> List[FormulaConfig]:
    configs = []
    for base in base_configs:
        for prototype_weight in [0.0, 0.10, 0.25, 0.50, 1.0]:
            for propensity_weight in [0.0, 0.10, 0.25, 0.50]:
                configs.append(
                    replace(
                        base,
                        prototype_weight=prototype_weight,
                        propensity_weight=propensity_weight,
                    )
                )
    return unique_configs(configs)


def build_fold_engine(source: pd.DataFrame, fold: int):
    reference, validation = cold_both_split(source, fold=fold, n_folds=4)
    print(
        f"fold={fold}: reference={len(reference)}, validation={len(validation)}, "
        f"validation_positive_rate={validation.Y.mean():.4f}"
    )
    return FormulaEvidenceEngine(reference, validation), validation.Y.to_numpy()


def run_dataset(dataset: str, repo_root: Path, output_dir: Path) -> Dict[str, object]:
    source = prepare_frame(
        pd.read_csv(repo_root / "datasets" / dataset / "cluster" / "source_train.csv"),
        require_label=True,
    )
    target = prepare_frame(
        pd.read_csv(repo_root / "datasets" / dataset / "cluster" / "target_test.csv"),
        require_label=True,
    )
    print(f"\n[{dataset}] source={len(source)}, target={len(target)}")

    engines: Dict[int, Tuple[FormulaEvidenceEngine, np.ndarray]] = {}
    for fold in [0, 1, 2]:
        engines[fold] = build_fold_engine(source, fold)

    all_rows = []
    config_map: Dict[str, FormulaConfig] = {}

    # Stage 1: select the two strongest fixed drug/protein multi-kernel pairs on fold 0.
    stage1 = stage1_kernel_configs()
    config_map.update({c.name: c for c in stage1})
    rows1 = evaluate_configs(dataset, 0, *engines[0], stage1, "kernel_screen")
    all_rows.append(rows1)
    rank1 = objective(rows1)
    top_kernel_names = rank1.head(2).config.tolist()
    top_kernel_pairs = [
        (config_map[name].drug_kernel, config_map[name].protein_kernel)
        for name in top_kernel_names
    ]
    print(f"[{dataset}] top kernel pairs: {top_kernel_pairs}")

    # Stage 2: search only fixed structural equations around those two kernels.
    stage2 = stage2_structural_configs(top_kernel_pairs)
    config_map.update({c.name: c for c in stage2})
    rows2_fold0 = evaluate_configs(dataset, 0, *engines[0], stage2, "structural_screen")
    all_rows.append(rows2_fold0)
    rank2_fold0 = objective(rows2_fold0)
    shortlist_names = rank2_fold0.head(12).config.tolist()
    shortlist = [config_map[name] for name in shortlist_names]
    for fold in [1, 2]:
        rows = evaluate_configs(dataset, fold, *engines[fold], shortlist, "structural_confirm")
        all_rows.append(rows)
    structural_rows = pd.concat(
        [
            rows2_fold0[rows2_fold0.config.isin(shortlist_names)],
            *[x for x in all_rows if x.stage.eq("structural_confirm").all()],
        ],
        ignore_index=True,
    )
    structural_rank = objective(structural_rows)
    top_structural_names = structural_rank.head(3).config.tolist()
    top_structural = [config_map[name] for name in top_structural_names]
    print(f"[{dataset}] confirmed structural formulas:\n{structural_rank.head(8).to_string(index=False)}")

    # Stage 3: add only closed-form class-density and entity-propensity terms.
    stage3 = stage3_global_evidence_configs(top_structural)
    config_map.update({c.name: c for c in stage3})
    stage3_rows = []
    for fold in [0, 1, 2]:
        rows = evaluate_configs(dataset, fold, *engines[fold], stage3, "global_evidence")
        stage3_rows.append(rows)
        all_rows.append(rows)
    rank3 = objective(pd.concat(stage3_rows, ignore_index=True))
    best_name = rank3.iloc[0].config
    best = config_map[best_name]
    print(f"[{dataset}] selected pure formula: {best_name}")
    print(rank3.head(10).to_string(index=False))

    # Freeze the formula before reading target labels for metrics.
    target_engine = FormulaEvidenceEngine(source, target)
    target_pred = target_engine.score(best)
    target_metrics = metrics(target.Y.to_numpy(), target_pred["score"], target_pred["probability"])
    print(f"[{dataset}] FINAL TARGET: {json.dumps(target_metrics, sort_keys=True)}")

    predictions = target[["SMILES", "Protein", "Y"]].copy()
    predictions["formula_score"] = target_pred["score"]
    predictions["p_formula"] = target_pred["probability"]
    predictions["p_joint"] = target_pred["joint_probability"]
    predictions["support"] = target_pred["support"]
    predictions["prototype_score"] = target_pred["prototype_score"]
    predictions["propensity_score"] = target_pred["propensity_score"]
    predictions.to_csv(output_dir / f"{dataset}_cluster_predictions.csv", index=False)

    pd.concat(all_rows, ignore_index=True).to_csv(
        output_dir / f"{dataset}_all_source_formula_search.csv", index=False
    )
    rank1.to_csv(output_dir / f"{dataset}_stage1_kernel_ranking.csv", index=False)
    structural_rank.to_csv(output_dir / f"{dataset}_stage2_structural_ranking.csv", index=False)
    rank3.to_csv(output_dir / f"{dataset}_stage3_final_ranking.csv", index=False)

    return {
        "dataset": dataset,
        "source_n": int(len(source)),
        "target_n": int(len(target)),
        "selection_protocol": (
            "three deterministic cold-both folds inside source_train; finite fixed-formula "
            "catalogue; target labels read only after the selected equation was frozen"
        ),
        "selected_config": asdict(best),
        "selected_config_name": best.name,
        "source_validation": rank3.iloc[0].to_dict(),
        "target": target_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("scjke_formula/results"))
    parser.add_argument("--datasets", nargs="+", default=["bindingdb", "biosnap"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [run_dataset(ds, args.repo_root, args.output_dir) for ds in args.datasets]
    (args.output_dir / "pure_formula_benchmark_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = make_markdown(summaries)
    (args.output_dir / "PURE_FORMULA_BENCHMARK_RESULTS.md").write_text(
        markdown, encoding="utf-8"
    )
    print("\n=== PURE-FORMULA FINAL RESULTS ===\n")
    print(markdown)


if __name__ == "__main__":
    main()
