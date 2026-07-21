#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer

AA_GROUPS = {
    **{aa: "A" for aa in "AGV"},
    **{aa: "B" for aa in "ILFP"},
    **{aa: "C" for aa in "YMTS"},
    **{aa: "D" for aa in "HNQW"},
    **{aa: "E" for aa in "RK"},
    **{aa: "F" for aa in "DE"},
    "C": "G",
}
EPS = 1e-8

def clean_protein(seq: str) -> str:
    seq = "".join(ch for ch in str(seq).upper() if ch in AA_GROUPS)
    if len(seq) < 3:
        raise ValueError("Protein sequence too short")
    return seq

def grouped(seq: str) -> str:
    return "".join(AA_GROUPS[a] for a in clean_protein(seq))

def canon(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)

def fps(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return (
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
        AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048),
        RDKFingerprint(mol, fpSize=2048),
        MACCSkeys.GenMACCSKeys(mol),
    )

def topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    if values.shape[1] <= k:
        return values.mean(axis=1)
    split = values.shape[1] - k
    return np.partition(values, split, axis=1)[:, -k:].mean(axis=1)

def prepare(df: pd.DataFrame, labels: bool) -> pd.DataFrame:
    out = df.copy()
    out["SMILES"] = out["SMILES"].map(canon)
    out["Protein"] = out["Protein"].map(clean_protein)
    if labels:
        out["Y"] = out["Y"].astype(int)
        out = out.drop_duplicates(["SMILES", "Protein"])
    return out.reset_index(drop=True)

def evaluate(source_path: Path, target_path: Path, batch_size: int = 48) -> dict:
    source = prepare(pd.read_csv(source_path), True)
    target = prepare(pd.read_csv(target_path), False)

    src_proteins = pd.Index(source["Protein"].unique())
    tgt_proteins = pd.Index(target["Protein"].unique())
    src_drugs = pd.Index(source["SMILES"].unique())
    tgt_drugs = pd.Index(target["SMILES"].unique())

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3,3), lowercase=False,
        binary=False, use_idf=False, norm="l2", dtype=np.float32
    )
    src_p = vectorizer.fit_transform([grouped(x) for x in src_proteins])
    tgt_p = vectorizer.transform([grouped(x) for x in tgt_proteins])

    src_fps = [fps(x) for x in src_drugs]
    tgt_fps = [fps(x) for x in tgt_drugs]

    drug_sim = np.empty((len(tgt_drugs), len(src_drugs)), dtype=np.float32)
    source_fp_columns = [[item[j] for item in src_fps] for j in range(4)]
    for qi, qfps in enumerate(tgt_fps):
        sims = []
        for fp_index in range(4):
            sims.append(np.asarray(
                DataStructs.BulkTanimotoSimilarity(
                    qfps[fp_index], source_fp_columns[fp_index]
                ), dtype=np.float32
            ))
        drug_sim[qi] = np.mean(np.stack(sims, axis=0), axis=0)

    sp_map = {v:i for i,v in enumerate(src_proteins)}
    sd_map = {v:i for i,v in enumerate(src_drugs)}
    tp_map = {v:i for i,v in enumerate(tgt_proteins)}
    td_map = {v:i for i,v in enumerate(tgt_drugs)}

    src_pair_p = np.fromiter((sp_map[v] for v in source["Protein"]), int, len(source))
    src_pair_d = np.fromiter((sd_map[v] for v in source["SMILES"]), int, len(source))
    tgt_pair_p = np.fromiter((tp_map[v] for v in target["Protein"]), int, len(target))
    tgt_pair_d = np.fromiter((td_map[v] for v in target["SMILES"]), int, len(target))
    labels = source["Y"].to_numpy(int)
    pos_mask = labels == 1
    neg_mask = labels == 0

    pred = np.empty(len(target), dtype=np.float64)
    support = np.empty(len(target), dtype=np.float64)
    for start in range(0, len(target), batch_size):
        stop = min(start + batch_size, len(target))
        idx = np.arange(start, stop)
        p_sim = (tgt_p[tgt_pair_p[idx]] @ src_p.T).toarray().astype(np.float32)
        p_pair = p_sim[:, src_pair_p]
        d_pair = drug_sim[tgt_pair_d[idx]][:, src_pair_d]
        joint = np.power(np.clip(p_pair,0,1), 1.5) * np.power(np.clip(d_pair,0,1), 0.5)
        e1 = topk_mean(joint[:,pos_mask], 200)
        e0 = topk_mean(joint[:,neg_mask], 200)
        base = (e1 + EPS) / (e1 + e0 + 2*EPS)
        sup = joint.max(axis=1)
        pred[idx] = 0.5 + sup * (base - 0.5)
        support[idx] = sup

    if "Y" not in target.columns:
        return {"n": len(target), "mean_score": float(pred.mean())}
    y = target["Y"].to_numpy(int)
    return {
        "n_source": len(source),
        "n_target": len(target),
        "auroc": float(roc_auc_score(y, pred)),
        "auprc": float(average_precision_score(y, pred)),
        "mean_score": float(pred.mean()),
        "mean_support": float(support.mean()),
    }

def main():
    root = Path(".")
    results = {}
    for dataset in ["bindingdb", "biosnap"]:
        results[dataset] = evaluate(
            root / "datasets" / dataset / "cluster" / "source_train.csv",
            root / "datasets" / dataset / "cluster" / "target_test.csv",
        )
    out = root / "formula_only_scjke" / "results" / "bindingdb_config_on_both.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
