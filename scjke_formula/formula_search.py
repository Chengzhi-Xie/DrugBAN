#!/usr/bin/env python3
"""Pure-formula DTI evidence for DrugBAN cluster-domain benchmarks.

No learned classifier, gradient optimization, regression coefficient fitting, tree
training, neural network, or target-label adaptation is used. Every score is
computed from fixed chemical/sequence kernels, source labels, and closed-form
class statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, RDKFingerprint
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

RDLogger.DisableLog("rdApp.*")
EPS = 1e-8
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
AA_GROUP_MAP = {
    **{x: "0" for x in "AGV"}, **{x: "1" for x in "ILFP"},
    **{x: "2" for x in "YMTS"}, **{x: "3" for x in "HNQW"},
    **{x: "4" for x in "RK"}, **{x: "5" for x in "DE"}, **{x: "6" for x in "C"},
}
DRUG_DESCRIPTOR_FUNCTIONS = (
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.RingCount,
    Descriptors.NumAromaticRings, Descriptors.FractionCSP3,
    Descriptors.HeavyAtomCount, Descriptors.NHOHCount, Descriptors.NOCount,
    Descriptors.NumValenceElectrons, Descriptors.MolMR, Descriptors.LabuteASA,
    Descriptors.BalabanJ, Descriptors.BertzCT, Descriptors.Chi0v,
    Descriptors.Chi1v, Descriptors.Chi2v, Descriptors.Kappa1,
    Descriptors.Kappa2, Descriptors.Kappa3, Descriptors.MaxPartialCharge,
    Descriptors.MinPartialCharge, Descriptors.NumAliphaticRings,
    Descriptors.NumSaturatedRings, Descriptors.NumHeteroatoms,
)


def clean_protein(sequence: str) -> str:
    sequence = "".join(ch for ch in str(sequence).upper() if ch in AA_SET)
    if len(sequence) < 3:
        raise ValueError("Protein sequence must contain at least three standard amino acids")
    return sequence


def grouped_protein(sequence: str) -> str:
    return "".join(AA_GROUP_MAP[ch] for ch in clean_protein(sequence))


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def prepare_frame(frame: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    required = {"SMILES", "Protein"} | ({"Y"} if require_label else set())
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = frame.copy().reset_index(drop=True)
    out["SMILES"] = out["SMILES"].map(canonical_smiles)
    out["Protein"] = out["Protein"].map(clean_protein)
    if require_label:
        out["Y"] = pd.to_numeric(out["Y"], errors="raise").astype(int)
        if set(out["Y"].unique()).difference({0, 1}):
            raise ValueError("Y must be binary")
        conflicts = out.groupby(["SMILES", "Protein"])["Y"].nunique()
        if (conflicts > 1).any():
            raise ValueError("Conflicting labels for identical SMILES/protein pairs")
        out = out.drop_duplicates(["SMILES", "Protein"]).reset_index(drop=True)
    return out


def safe_descriptor(fn, mol) -> float:
    try:
        value = float(fn(mol))
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def drug_descriptors(smiles_list: Sequence[str]) -> np.ndarray:
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        rows.append([safe_descriptor(fn, mol) for fn in DRUG_DESCRIPTOR_FUNCTIONS])
    return np.asarray(rows, dtype=np.float32)


def protein_descriptors(sequence_list: Sequence[str]) -> np.ndarray:
    rows = []
    aa_index = {aa: i for i, aa in enumerate(AA)}
    hydro, polar, positive = set("AILMFWVY"), set("STNQ"), set("KRH")
    negative, aromatic, small = set("DE"), set("FWYH"), set("AGSTCP")
    for seq in sequence_list:
        seq = clean_protein(seq); n = len(seq)
        aac = np.zeros(20, dtype=np.float32)
        for aa in seq: aac[aa_index[aa]] += 1
        aac /= n
        grouped = np.zeros(7, dtype=np.float32)
        gdpc = np.zeros((7, 7), dtype=np.float32)
        gseq = grouped_protein(seq)
        for g in gseq: grouped[int(g)] += 1
        grouped /= n
        for a, b in zip(gseq[:-1], gseq[1:]): gdpc[int(a), int(b)] += 1
        gdpc /= max(n - 1, 1)
        nz = aac[aac > 0]
        entropy = float(-(nz * np.log(nz)).sum() / np.log(20)) if len(nz) else 0.0
        global_props = np.asarray([
            np.log1p(n) / 10.0,
            sum(x in hydro for x in seq) / n,
            sum(x in polar for x in seq) / n,
            sum(x in positive for x in seq) / n,
            sum(x in negative for x in seq) / n,
            (sum(x in positive for x in seq)-sum(x in negative for x in seq))/n,
            sum(x in aromatic for x in seq) / n,
            sum(x in small for x in seq) / n,
            entropy,
        ], dtype=np.float32)
        rows.append(np.concatenate([aac, grouped, gdpc.ravel(), global_props]))
    return np.asarray(rows, dtype=np.float32)


def robust_scale_reference(reference: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(reference, axis=0)
    scale = np.maximum(np.quantile(reference, .95, axis=0)-np.quantile(reference, .05, axis=0), 1e-6)
    return (np.clip((reference-med)/scale, -8, 8).astype(np.float32),
            np.clip((query-med)/scale, -8, 8).astype(np.float32))


def rbf_l1_similarity(ref: np.ndarray, query: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    return np.exp(-(cdist(query, ref, metric="cityblock")/max(ref.shape[1], 1))/bandwidth).astype(np.float32)


def cosine_kmer_similarity(ref_seqs: Sequence[str], query_seqs: Sequence[str], k: int, grouped: bool = False) -> np.ndarray:
    ref_input = [grouped_protein(s) for s in ref_seqs] if grouped else list(ref_seqs)
    query_input = [grouped_protein(s) for s in query_seqs] if grouped else list(query_seqs)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(k, k), binary=True,
                          use_idf=False, norm="l2", lowercase=False, dtype=np.float32)
    ref = vec.fit_transform(ref_input); query = vec.transform(query_input)
    return (query @ ref.T).toarray().astype(np.float32)


def fingerprint_matrices(ref_smiles: Sequence[str], query_smiles: Sequence[str]) -> Dict[str, np.ndarray]:
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ref_mol = [Chem.MolFromSmiles(s) for s in ref_smiles]
    qry_mol = [Chem.MolFromSmiles(s) for s in query_smiles]
    ref_fp = {
        "morgan": [morgan.GetFingerprint(m) for m in ref_mol],
        "maccs": [MACCSkeys.GenMACCSKeys(m) for m in ref_mol],
        "rdk": [RDKFingerprint(m, fpSize=2048) for m in ref_mol],
    }
    qry_fp = {
        "morgan": [morgan.GetFingerprint(m) for m in qry_mol],
        "maccs": [MACCSkeys.GenMACCSKeys(m) for m in qry_mol],
        "rdk": [RDKFingerprint(m, fpSize=2048) for m in qry_mol],
    }
    return {name: np.vstack([np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fp[name]), dtype=np.float32)
                             for fp in qry_fp[name]]) for name in ref_fp}


def stable_hash_fold(value: object, n_folds: int) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % n_folds


def cold_both_split(frame: pd.DataFrame, fold: int, n_folds: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d_key = "drug_cluster" if "drug_cluster" in frame.columns else "SMILES"
    p_key = "target_cluster" if "target_cluster" in frame.columns else "Protein"
    d_hold = frame[d_key].map(lambda x: stable_hash_fold(x, n_folds) == fold)
    p_hold = frame[p_key].map(lambda x: stable_hash_fold(x, n_folds) == fold)
    val = frame[d_hold & p_hold].copy(); ref = frame[(~d_hold) & (~p_hold)].copy()
    if len(val) < 150 or val["Y"].nunique() < 2:
        raise RuntimeError(f"Cold-both fold {fold} too small: n={len(val)}")
    return ref.reset_index(drop=True), val.reset_index(drop=True)


@dataclass(frozen=True)
class FormulaConfig:
    drug_kernel: str; protein_kernel: str; joint_rule: str
    k_joint: int; k_marginal: int; degree_gamma: float
    evidence_mode: str; prototype_weight: float; propensity_weight: float
    @property
    def name(self) -> str:
        return (f"D={self.drug_kernel}|P={self.protein_kernel}|J={self.joint_rule}|"
                f"kj={self.k_joint}|km={self.k_marginal}|g={self.degree_gamma}|"
                f"mode={self.evidence_mode}|proto={self.prototype_weight}|prop={self.propensity_weight}")


@dataclass
class MetricRow:
    dataset: str; stage: str; fold: int; config: str
    n_reference: int; n_query: int; auroc: float; auprc: float; brier: float


def topk_weighted_mean(values: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    n = values.shape[1]; k = min(k, n)
    idx = np.argpartition(values, n-k, axis=1)[:, -k:]
    selected_values = np.take_along_axis(values, idx, axis=1)
    selected_weights = weights[idx]
    return ((selected_values*selected_weights).sum(axis=1)/np.maximum(selected_weights.sum(axis=1), EPS)).astype(np.float32)


def topk_similarity_average(similarity: np.ndarray, entity_values: np.ndarray, k: int) -> np.ndarray:
    n = similarity.shape[1]; k = min(k, n)
    idx = np.argpartition(similarity, n-k, axis=1)[:, -k:]
    sim = np.take_along_axis(similarity, idx, axis=1); val = entity_values[idx]
    return ((sim*val).sum(axis=1)/np.maximum(sim.sum(axis=1), EPS)).astype(np.float32)


def log_ratio(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return np.log(pos+EPS)-np.log(neg+EPS)


def class_diagonal_gaussian_score(ref_features: np.ndarray, labels: np.ndarray,
                                  query_features: np.ndarray, variance_floor: float=.20) -> np.ndarray:
    scores = []
    for c in [0, 1]:
        x = ref_features[labels == c]; mean = x.mean(axis=0); var = x.var(axis=0)+variance_floor**2
        scores.append(-.5*(((query_features-mean)**2/var)+np.log(var)).mean(axis=1))
    return (scores[1]-scores[0]).astype(np.float32)


class FormulaEvidenceEngine:
    def __init__(self, reference: pd.DataFrame, query: pd.DataFrame):
        self.reference = prepare_frame(reference, True)
        self.query = prepare_frame(query, "Y" in query.columns)
        self.labels = self.reference["Y"].to_numpy(dtype=np.int8)
        self.pos_mask = self.labels == 1; self.neg_mask = ~self.pos_mask
        self.ref_drugs = pd.Index(self.reference["SMILES"].unique())
        self.ref_proteins = pd.Index(self.reference["Protein"].unique())
        self.qry_drugs = pd.Index(self.query["SMILES"].unique())
        self.qry_proteins = pd.Index(self.query["Protein"].unique())
        rdm={x:i for i,x in enumerate(self.ref_drugs)}; rpm={x:i for i,x in enumerate(self.ref_proteins)}
        qdm={x:i for i,x in enumerate(self.qry_drugs)}; qpm={x:i for i,x in enumerate(self.qry_proteins)}
        self.ref_pair_drug=np.fromiter((rdm[x] for x in self.reference.SMILES),int,count=len(self.reference))
        self.ref_pair_protein=np.fromiter((rpm[x] for x in self.reference.Protein),int,count=len(self.reference))
        self.qry_pair_drug=np.fromiter((qdm[x] for x in self.query.SMILES),int,count=len(self.query))
        self.qry_pair_protein=np.fromiter((qpm[x] for x in self.query.Protein),int,count=len(self.query))

        self.drug_sims=fingerprint_matrices(self.ref_drugs,self.qry_drugs)
        ref_dd,qry_dd=drug_descriptors(self.ref_drugs),drug_descriptors(self.qry_drugs)
        ref_dd,qry_dd=robust_scale_reference(ref_dd,qry_dd)
        self.drug_sims["desc_rbf"]=rbf_l1_similarity(ref_dd,qry_dd)
        self.protein_sims={
            "aa3":cosine_kmer_similarity(self.ref_proteins,self.qry_proteins,3,False),
            "aa2":cosine_kmer_similarity(self.ref_proteins,self.qry_proteins,2,False),
            "group3":cosine_kmer_similarity(self.ref_proteins,self.qry_proteins,3,True),
        }
        ref_pd,qry_pd=protein_descriptors(self.ref_proteins),protein_descriptors(self.qry_proteins)
        ref_pd,qry_pd=robust_scale_reference(ref_pd,qry_pd)
        self.protein_sims["desc_rbf"]=rbf_l1_similarity(ref_pd,qry_pd)
        self.ref_pair_features=np.concatenate([ref_dd[self.ref_pair_drug],ref_pd[self.ref_pair_protein]],axis=1)
        self.qry_pair_features=np.concatenate([qry_dd[self.qry_pair_drug],qry_pd[self.qry_pair_protein]],axis=1)
        self.prototype_score=class_diagonal_gaussian_score(self.ref_pair_features,self.labels,self.qry_pair_features)
        self.drug_degree=np.bincount(self.ref_pair_drug,minlength=len(self.ref_drugs)).astype(np.float32)
        self.protein_degree=np.bincount(self.ref_pair_protein,minlength=len(self.ref_proteins)).astype(np.float32)
        drug_pos=np.bincount(self.ref_pair_drug,weights=self.labels,minlength=len(self.ref_drugs)).astype(np.float32)
        protein_pos=np.bincount(self.ref_pair_protein,weights=self.labels,minlength=len(self.ref_proteins)).astype(np.float32)
        self.drug_propensity=(drug_pos+.5)/(self.drug_degree+1)
        self.protein_propensity=(protein_pos+.5)/(self.protein_degree+1)

    def get_drug_kernel(self,name):
        s=self.drug_sims
        if name=="morgan": return s["morgan"]
        if name=="morgan_maccs": return .65*s["morgan"]+.35*s["maccs"]
        if name=="multi_fp": return .50*s["morgan"]+.25*s["maccs"]+.25*s["rdk"]
        if name=="fp_desc": return .55*s["morgan"]+.20*s["maccs"]+.25*s["desc_rbf"]
        if name=="all": return .40*s["morgan"]+.20*s["maccs"]+.15*s["rdk"]+.25*s["desc_rbf"]
        raise KeyError(name)

    def get_protein_kernel(self,name):
        s=self.protein_sims
        if name=="aa3": return s["aa3"]
        if name=="aa23": return .40*s["aa2"]+.60*s["aa3"]
        if name=="aa_group": return .55*s["aa3"]+.25*s["aa2"]+.20*s["group3"]
        if name=="seq_desc": return .50*s["aa3"]+.20*s["group3"]+.30*s["desc_rbf"]
        if name=="all": return .40*s["aa3"]+.20*s["aa2"]+.15*s["group3"]+.25*s["desc_rbf"]
        raise KeyError(name)

    @staticmethod
    def joint_similarity(drug,protein,rule):
        if rule=="product": return drug*protein
        if rule=="harmonic": return 2*drug*protein/np.maximum(drug+protein,EPS)
        if rule=="minimum": return np.minimum(drug,protein)
        if rule=="soft_and": return .5*np.minimum(drug,protein)+.5*np.sqrt(np.maximum(drug*protein,0))
        raise KeyError(rule)

    def score(self,cfg:FormulaConfig,batch_size=64):
        drug_entity=self.get_drug_kernel(cfg.drug_kernel)
        protein_entity=self.get_protein_kernel(cfg.protein_kernel)
        n=len(self.query); final=np.empty(n,np.float32); joint_p=np.empty(n,np.float32); support=np.empty(n,np.float32)
        dprop=topk_similarity_average(drug_entity[self.qry_pair_drug],self.drug_propensity,cfg.k_marginal)
        pprop=topk_similarity_average(protein_entity[self.qry_pair_protein],self.protein_propensity,cfg.k_marginal)
        prop_logit=.5*(np.log(np.clip(dprop,EPS,1-EPS)/np.clip(1-dprop,EPS,1))+
                        np.log(np.clip(pprop,EPS,1-EPS)/np.clip(1-pprop,EPS,1)))
        pair_w=(self.drug_degree[self.ref_pair_drug]*self.protein_degree[self.ref_pair_protein])**(-cfg.degree_gamma)
        for start in range(0,n,batch_size):
            stop=min(start+batch_size,n); idx=np.arange(start,stop)
            d=drug_entity[self.qry_pair_drug[idx]][:,self.ref_pair_drug]
            p=protein_entity[self.qry_pair_protein[idx]][:,self.ref_pair_protein]
            j=self.joint_similarity(d,p,cfg.joint_rule)
            e1j=topk_weighted_mean(j[:,self.pos_mask],pair_w[self.pos_mask],cfg.k_joint)
            e0j=topk_weighted_mean(j[:,self.neg_mask],pair_w[self.neg_mask],cfg.k_joint)
            e1d=topk_weighted_mean(d[:,self.pos_mask],pair_w[self.pos_mask],cfg.k_marginal)
            e0d=topk_weighted_mean(d[:,self.neg_mask],pair_w[self.neg_mask],cfg.k_marginal)
            e1p=topk_weighted_mean(p[:,self.pos_mask],pair_w[self.pos_mask],cfg.k_marginal)
            e0p=topk_weighted_mean(p[:,self.neg_mask],pair_w[self.neg_mask],cfg.k_marginal)
            lj,ld,lp=log_ratio(e1j,e0j),log_ratio(e1d,e0d),log_ratio(e1p,e0p)
            sd,sp=d.max(axis=1),p.max(axis=1)
            balance=2*np.minimum(sd,sp)/np.maximum(sd+sp,EPS)
            marginal=(sd*ld+sp*lp)/np.maximum(sd+sp,EPS)
            if cfg.evidence_mode=="joint": base=lj
            elif cfg.evidence_mode=="joint_marginal_equal": base=.50*lj+.25*ld+.25*lp
            elif cfg.evidence_mode=="support_adaptive": base=balance*lj+(1-balance)*marginal
            elif cfg.evidence_mode=="joint_plus_marginal": base=lj+.50*marginal
            else: raise KeyError(cfg.evidence_mode)
            final[idx]=base+cfg.prototype_weight*self.prototype_score[idx]+cfg.propensity_weight*prop_logit[idx]
            joint_p[idx]=(e1j+EPS)/(e1j+e0j+2*EPS); support[idx]=j.max(axis=1)
        return {"score":final,"probability":expit(final),"joint_probability":joint_p,"support":support,
                "prototype_score":self.prototype_score,"propensity_score":expit(prop_logit)}


def metrics(y,score,probability=None):
    y=np.asarray(y,int); score=np.asarray(score,float); probability=expit(score) if probability is None else probability
    return {"auroc":float(roc_auc_score(y,score)),"auprc":float(average_precision_score(y,score)),
            "brier":float(brier_score_loss(y,np.clip(probability,0,1)))}


def candidate_configs():
    configs=[]
    for dk in ["morgan","morgan_maccs","multi_fp","fp_desc","all"]:
      for pk in ["aa3","aa23","aa_group","seq_desc","all"]:
       for jr in ["product","harmonic","minimum","soft_and"]:
        for kj in [25,50,100,200,400]:
         for g in [0.,.25,.5]:
          for mode in ["joint","joint_marginal_equal","support_adaptive","joint_plus_marginal"]:
           configs.append(FormulaConfig(dk,pk,jr,kj,200,g,mode,0.,0.))
    for dk in ["fp_desc","all"]:
     for pk in ["seq_desc","all"]:
      for jr in ["harmonic","soft_and"]:
       for kj in [50,100,200,400]:
        for g in [0.,.25]:
         for mode in ["joint_marginal_equal","support_adaptive","joint_plus_marginal"]:
          for proto in [.10,.25,.50,1.0]:
           for prop in [.10,.25,.50]:
            configs.append(FormulaConfig(dk,pk,jr,kj,200,g,mode,proto,prop))
    return list(dict.fromkeys(configs))


def choose_best(df):
    g=df.groupby(["dataset","config"],as_index=False).agg(mean_auroc=("auroc","mean"),mean_auprc=("auprc","mean"),std_auroc=("auroc","std"),std_auprc=("auprc","std"),folds=("fold","nunique"))
    g["objective"]=.5*(g.mean_auroc+g.mean_auprc)
    g["robust_objective"]=g.objective-.05*(g.std_auroc.fillna(0)+g.std_auprc.fillna(0))
    return g.sort_values(["dataset","robust_objective","mean_auroc","mean_auprc"],ascending=[True,False,False,False])


def evaluate_dataset(dataset,root,configs,folds,out):
    source=prepare_frame(pd.read_csv(root/"datasets"/dataset/"cluster"/"source_train.csv"),True)
    target=prepare_frame(pd.read_csv(root/"datasets"/dataset/"cluster"/"target_test.csv"),True)
    rows=[]; print(f"[{dataset}] source={len(source)} target={len(target)} configs={len(configs)}")
    for fold in folds:
        ref,val=cold_both_split(source,fold,4); print(f"[{dataset}] fold={fold} ref={len(ref)} val={len(val)}")
        engine=FormulaEvidenceEngine(ref,val); y=val.Y.to_numpy()
        for j,cfg in enumerate(configs):
            pred=engine.score(cfg); met=metrics(y,pred["score"],pred["probability"])
            rows.append(asdict(MetricRow(dataset,"source_cold_validation",fold,cfg.name,len(ref),len(val),**met)))
            if (j+1)%500==0: print(f"[{dataset}] fold={fold} {j+1}/{len(configs)}")
    val_df=pd.DataFrame(rows); ranking=choose_best(val_df); best_name=ranking.iloc[0].config
    cfg_map={x.name:x for x in configs}; best=cfg_map[best_name]
    print(f"[{dataset}] selected: {best_name}\n{ranking.head(10).to_string(index=False)}")
    engine=FormulaEvidenceEngine(source,target); pred=engine.score(best); met=metrics(target.Y,pred["score"],pred["probability"])
    print(f"[{dataset}] TARGET {json.dumps(met,sort_keys=True)}")
    output=target[["SMILES","Protein","Y"]].copy(); output["score"]=pred["score"]; output["p_formula"]=pred["probability"]; output["p_joint"]=pred["joint_probability"]; output["support"]=pred["support"]; output["prototype_score"]=pred["prototype_score"]; output["propensity_score"]=pred["propensity_score"]
    output.to_csv(out/f"{dataset}_cluster_predictions.csv",index=False)
    val_df.to_csv(out/f"{dataset}_source_validation_all.csv",index=False); ranking.to_csv(out/f"{dataset}_source_validation_ranking.csv",index=False)
    summary={"dataset":dataset,"source_n":len(source),"target_n":len(target),"selection_protocol":"source-only deterministic cold-both folds; target labels used only after freezing formula","selected_config":asdict(best),"selected_config_name":best.name,"validation":ranking.iloc[0].to_dict(),"target":met}
    return val_df,summary


def make_markdown(summaries):
    lines=["# Pure-Formula SCJKE Cluster Benchmark Results","","No PCM, logistic regression, tree model, neural network, gradient optimization, or target-label tuning was used.","","| Dataset | AUROC | AUPRC | Brier | Selected formula |","|---|---:|---:|---:|---|"]
    for s in summaries:
        t=s["target"]; lines.append(f"| {s['dataset']} cluster | {t['auroc']:.6f} | {t['auprc']:.6f} | {t['brier']:.6f} | `{s['selected_config_name']}` |")
    lines += ["","Formula constants were selected only on deterministic cold-both splits inside `source_train.csv`; `target_test.Y` was read only after selection.",""]
    for s in summaries: lines += [f"## {s['dataset']}","","```json",json.dumps(s,indent=2,ensure_ascii=False),"```",""]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",type=Path,default=Path(".")); p.add_argument("--output-dir",type=Path,default=Path("scjke_formula/results")); p.add_argument("--datasets",nargs="+",default=["bindingdb","biosnap"]); p.add_argument("--folds",nargs="+",type=int,default=[0,1,2]); args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); configs=candidate_configs(); print("Total candidate formulas:",len(configs))
    summaries=[]; all_val=[]
    for ds in args.datasets:
        val,s=evaluate_dataset(ds,args.repo_root,configs,args.folds,args.output_dir); all_val.append(val); summaries.append(s)
    pd.concat(all_val,ignore_index=True).to_csv(args.output_dir/"all_source_validation.csv",index=False)
    (args.output_dir/"benchmark_summary.json").write_text(json.dumps(summaries,indent=2,ensure_ascii=False))
    md=make_markdown(summaries); (args.output_dir/"BENCHMARK_RESULTS.md").write_text(md); print("\n=== FINAL TARGET RESULTS ===\n"+md)

if __name__=="__main__": main()
