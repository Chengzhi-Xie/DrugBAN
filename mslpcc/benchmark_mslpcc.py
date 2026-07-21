#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-8
SCALES = (15, 31, 63)
POOLS = ("mean", "top05", "top10", "top20", "max")
BASE_NAMES = ("hydrophobic", "charge", "hbond", "aromatic", "polar", "flexible")
COMPONENT_NAMES = BASE_NAMES + (
    "all_mean",
    "all_geom",
    "all_min",
    "core_mean",
    "core_geom",
    "hyd_arom_geom",
    "hbond_charge_mean",
    "interface_weighted",
)

# Fixed amino-acid physicochemical constants. No fitted embeddings or target labels.
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
AA = tuple(KD)
AA_INDEX = {a: i for i, a in enumerate(AA)}
AA_PROP = np.zeros((len(AA), 7), dtype=np.float32)
for aa, i in AA_INDEX.items():
    AA_PROP[i, 0] = (KD[aa] + 4.5) / 9.0
    AA_PROP[i, 1] = 1.0 if aa in "KR" else (-1.0 if aa in "DE" else (0.10 if aa == "H" else 0.0))
    AA_PROP[i, 2] = 1.0 if aa in "KRHSTYWNQ" else 0.0
    AA_PROP[i, 3] = 1.0 if aa in "DEHNQSTYC" else 0.0
    AA_PROP[i, 4] = 1.0 if aa in "FWYH" else 0.0
    AA_PROP[i, 5] = 1.0 if aa in "STNQCYHDEKR" else 0.0
    AA_PROP[i, 6] = 1.0 if aa in "GPSNQA" else 0.0


def clean_protein(seq: str) -> str:
    out = "".join(a for a in str(seq).upper() if a in AA_INDEX)
    if not out:
        raise ValueError("Protein sequence has no standard amino acids")
    return out


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def drug_vector(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    heavy = max(1, int(mol.GetNumHeavyAtoms()))
    logp = float(Crippen.MolLogP(mol))
    formal_charge = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    aromatic = float(sum(atom.GetIsAromatic() for atom in mol.GetAtoms())) / heavy
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    rot = float(Lipinski.NumRotatableBonds(mol))
    return np.asarray([
        sigmoid(logp / 2.0),
        np.clip(formal_charge / 2.0, -1.0, 1.0),
        hbd / (hbd + 3.0),
        hba / (hba + 5.0),
        np.clip(aromatic, 0.0, 1.0),
        tpsa / (tpsa + 60.0),
        rot / (rot + 5.0),
    ], dtype=np.float32)


def protein_windows(seq: str, scale: int) -> np.ndarray:
    seq = clean_protein(seq)
    arr = AA_PROP[np.fromiter((AA_INDEX[a] for a in seq), dtype=np.int32)]
    if len(arr) <= scale:
        return arr.mean(axis=0, keepdims=True)
    prefix = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float32), np.cumsum(arr, axis=0)])
    return (prefix[scale:] - prefix[:-scale]) / float(scale)


def top_fraction_mean(values: np.ndarray, fraction: float) -> np.ndarray:
    n = values.shape[0]
    k = max(1, int(math.ceil(n * fraction)))
    if k >= n:
        return values.mean(axis=0)
    return np.partition(values, n - k, axis=0)[-k:].mean(axis=0)


def complement_components(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    hyd = 1.0 - np.abs(d[0] - p[:, 0])
    charge = np.exp(-np.square((d[1] + p[:, 1]) / 0.75))
    hbond = 0.5 * (np.minimum(d[2], p[:, 3]) + np.minimum(d[3], p[:, 2]))
    aromatic = np.sqrt(np.clip(d[4] * p[:, 4], 0.0, 1.0))
    polar = 1.0 - np.abs(d[5] - p[:, 5])
    flexible = np.sqrt(np.clip(d[6] * p[:, 6], 0.0, 1.0))
    base = np.stack([hyd, charge, hbond, aromatic, polar, flexible], axis=1)
    base = np.clip(base, 0.0, 1.0)
    all_mean = base.mean(axis=1)
    all_geom = np.exp(np.log(base + 1e-6).mean(axis=1))
    all_min = base.min(axis=1)
    core = base[:, [0, 2, 3, 4]]
    core_mean = core.mean(axis=1)
    core_geom = np.exp(np.log(core + 1e-6).mean(axis=1))
    hyd_arom_geom = np.sqrt((base[:, 0] + 1e-6) * (base[:, 3] + 1e-6))
    hbond_charge_mean = 0.5 * (base[:, 1] + base[:, 2])
    weights = np.asarray([2.0, 1.0, 2.0, 1.0, 1.0, 0.5], dtype=np.float32)
    interface_weighted = (base * weights).sum(axis=1) / weights.sum()
    return np.column_stack([
        base,
        all_mean,
        all_geom,
        all_min,
        core_mean,
        core_geom,
        hyd_arom_geom,
        hbond_charge_mean,
        interface_weighted,
    ]).astype(np.float32)


def pool_components(values: np.ndarray) -> np.ndarray:
    return np.stack([
        values.mean(axis=0),
        top_fraction_mean(values, 0.05),
        top_fraction_mean(values, 0.10),
        top_fraction_mean(values, 0.20),
        values.max(axis=0),
    ], axis=0)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMILES"] = out["SMILES"].map(canonical_smiles)
    out["Protein"] = out["Protein"].map(clean_protein)
    out["Y"] = out["Y"].astype(int)
    return out.drop_duplicates(["SMILES", "Protein"]).reset_index(drop=True)


def compute_tensor(df: pd.DataFrame) -> np.ndarray:
    drugs = {s: drug_vector(s) for s in df["SMILES"].unique()}
    proteins = {
        p: {scale: protein_windows(p, scale) for scale in SCALES}
        for p in df["Protein"].unique()
    }
    tensor = np.empty((len(df), len(SCALES), len(POOLS), len(COMPONENT_NAMES)), dtype=np.float32)
    for row, (smiles, protein) in enumerate(zip(df["SMILES"], df["Protein"])):
        d = drugs[smiles]
        for si, scale in enumerate(SCALES):
            tensor[row, si] = pool_components(complement_components(d, proteins[protein][scale]))
    return tensor


def reduce_scales(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "scale_mean":
        return x.mean(axis=1)
    if mode == "scale_min":
        return x.min(axis=1)
    if mode == "scale_max":
        return x.max(axis=1)
    if mode.startswith("L"):
        idx = SCALES.index(int(mode[1:]))
        return x[:, idx]
    raise KeyError(mode)


def build_candidates(tensor: np.ndarray) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    values: List[np.ndarray] = []
    reducers = ("scale_mean", "scale_min", "scale_max", "L15", "L31", "L63")
    for pi, pool in enumerate(POOLS):
        slice_x = tensor[:, :, pi, :]
        for reducer in reducers:
            reduced = reduce_scales(slice_x, reducer)
            for ci, component in enumerate(COMPONENT_NAMES):
                score = np.clip(reduced[:, ci], 0.0, 1.0)
                name = f"{component}__{pool}__{reducer}"
                names.extend([name, name + "__inverse"])
                values.extend([score, 1.0 - score])
    # Fixed multi-pooling variants reduce sensitivity to a single top-window percentage.
    for reducer in reducers:
        reduced = reduce_scales(tensor[:, :, 1:4, :].mean(axis=2), reducer)
        for ci, component in enumerate(COMPONENT_NAMES):
            score = np.clip(reduced[:, ci], 0.0, 1.0)
            name = f"{component}__top_multiscale_pool__{reducer}"
            names.extend([name, name + "__inverse"])
            values.extend([score, 1.0 - score])
    matrix = np.column_stack(values).astype(np.float32)
    return names, matrix


def stable_fold(value: str, modulo: int = 3) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) % modulo


def metric_pair(y: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def source_validation(df: pd.DataFrame, matrix: np.ndarray) -> np.ndarray:
    drug_fold = np.asarray([stable_fold(x) for x in df["SMILES"]], dtype=int)
    protein_fold = np.asarray([stable_fold(x) for x in df["Protein"]], dtype=int)
    y = df["Y"].to_numpy(int)
    objectives = np.zeros(matrix.shape[1], dtype=np.float64)
    valid_folds = 0
    for fold in range(3):
        mask = (drug_fold == fold) & (protein_fold == fold)
        if mask.sum() < 20 or np.unique(y[mask]).size < 2:
            continue
        valid_folds += 1
        for j in range(matrix.shape[1]):
            auroc, auprc = metric_pair(y[mask], matrix[mask, j])
            objectives[j] += 0.5 * (auroc + auprc)
    if valid_folds == 0:
        raise RuntimeError("No valid source cold-both validation folds")
    return objectives / valid_folds


def target_metrics(y: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    auroc = np.empty(matrix.shape[1], dtype=np.float64)
    auprc = np.empty(matrix.shape[1], dtype=np.float64)
    for j in range(matrix.shape[1]):
        auroc[j], auprc[j] = metric_pair(y, matrix[:, j])
    return auroc, auprc


def result_for(index: int, names: List[str], auroc: np.ndarray, auprc: np.ndarray) -> Dict[str, float | str]:
    return {
        "formula": names[index],
        "auroc": float(auroc[index]),
        "auprc": float(auprc[index]),
        "mean_metric": float(0.5 * (auroc[index] + auprc[index])),
    }


def benchmark_dataset(root: Path, dataset: str) -> Dict[str, object]:
    source = prepare(pd.read_csv(root / "datasets" / dataset / "cluster" / "source_train.csv"))
    target = prepare(pd.read_csv(root / "datasets" / dataset / "cluster" / "target_test.csv"))
    combined = pd.concat([source[["SMILES", "Protein", "Y"]], target[["SMILES", "Protein", "Y"]]], ignore_index=True)
    tensor = compute_tensor(combined)
    names, matrix = build_candidates(tensor)
    source_matrix = matrix[: len(source)]
    target_matrix = matrix[len(source):]
    source_obj = source_validation(source, source_matrix)
    target_auroc, target_auprc = target_metrics(target["Y"].to_numpy(int), target_matrix)
    source_index = int(np.nanargmax(source_obj))
    oracle_mean_index = int(np.nanargmax(0.5 * (target_auroc + target_auprc)))
    oracle_min_index = int(np.nanargmax(np.minimum(target_auroc, target_auprc)))
    return {
        "dataset": dataset,
        "n_source": len(source),
        "n_target": len(target),
        "candidate_count": len(names),
        "names": names,
        "source_objective": source_obj,
        "target_auroc": target_auroc,
        "target_auprc": target_auprc,
        "source_selected": result_for(source_index, names, target_auroc, target_auprc),
        "oracle_best_mean": result_for(oracle_mean_index, names, target_auroc, target_auprc),
        "oracle_best_min_metric": result_for(oracle_min_index, names, target_auroc, target_auprc),
    }


def serializable_dataset(result: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in result.items() if k not in {"names", "source_objective", "target_auroc", "target_auprc"}}


def main() -> None:
    root = Path(".")
    bdb = benchmark_dataset(root, "bindingdb")
    bio = benchmark_dataset(root, "biosnap")
    if bdb["names"] != bio["names"]:
        raise RuntimeError("Candidate libraries differ")
    names = bdb["names"]
    bdb_source = bdb["source_objective"]
    bio_source = bio["source_objective"]
    bdb_auc, bdb_ap = bdb["target_auroc"], bdb["target_auprc"]
    bio_auc, bio_ap = bio["target_auroc"], bio["target_auprc"]

    common_source_idx = int(np.nanargmax(0.5 * (bdb_source + bio_source)))
    common_oracle_mean_idx = int(np.nanargmax((bdb_auc + bdb_ap + bio_auc + bio_ap) / 4.0))
    common_oracle_min_idx = int(np.nanargmax(np.minimum.reduce([bdb_auc, bdb_ap, bio_auc, bio_ap])))

    def common_record(index: int) -> Dict[str, object]:
        return {
            "formula": names[index],
            "bindingdb": result_for(index, names, bdb_auc, bdb_ap),
            "biosnap": result_for(index, names, bio_auc, bio_ap),
            "four_metric_mean": float((bdb_auc[index] + bdb_ap[index] + bio_auc[index] + bio_ap[index]) / 4.0),
            "worst_metric": float(min(bdb_auc[index], bdb_ap[index], bio_auc[index], bio_ap[index])),
        }

    output = {
        "method": "Multi-Scale Local Physicochemical Complementarity (MS-LPCC)",
        "constraints": {
            "predictive_model_fit": False,
            "neural_network": False,
            "protein_structure": False,
            "docking": False,
            "target_labels_used_to_compute_scores": False,
            "source_labels_used_only_for_formula_selection": True,
            "target_labels_used_for_oracle_selection": True,
        },
        "bindingdb": serializable_dataset(bdb),
        "biosnap": serializable_dataset(bio),
        "common_source_selected": common_record(common_source_idx),
        "common_oracle_best_mean": common_record(common_oracle_mean_idx),
        "common_oracle_best_worst_metric": common_record(common_oracle_min_idx),
        "interpretation": {
            "source_selected": "Configuration chosen only from source cold-both validation labels; target labels are used only for final AUROC/AUPRC evaluation.",
            "oracle": "Exploratory upper bound chosen after comparing target labels; not an unbiased test result.",
        },
    }
    out_dir = root / "mslpcc" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mslpcc_cluster_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
