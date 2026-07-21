#!/usr/bin/env python3
"""Adaptive SCJKE-PCM for cold-both DTI prediction.

The model uses only source-domain protein sequences, drug SMILES, and binary
DTI labels during fitting. Query/test labels are never consumed by fit or
predict.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
AA_GROUPS = ("AGV", "ILFP", "YMTS", "HNQW", "RK", "DE", "C")
AA_GROUP_INDEX = {
    aa: group_index
    for group_index, group in enumerate(AA_GROUPS)
    for aa in group
}
HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQ")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FWYH")
SMALL = set("AGSTCP")

DRUG_DESCRIPTOR_FUNCTIONS = (
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.RingCount,
    Descriptors.NumAromaticRings,
    Descriptors.FractionCSP3,
    Descriptors.HeavyAtomCount,
    Descriptors.NHOHCount,
    Descriptors.NOCount,
    Descriptors.NumValenceElectrons,
    Descriptors.MolMR,
    Descriptors.LabuteASA,
    Descriptors.BalabanJ,
    Descriptors.BertzCT,
    Descriptors.Chi0v,
    Descriptors.Chi1v,
    Descriptors.Chi2v,
    Descriptors.Kappa1,
    Descriptors.Kappa2,
    Descriptors.Kappa3,
    Descriptors.MaxPartialCharge,
    Descriptors.MinPartialCharge,
    Descriptors.MaxAbsPartialCharge,
    Descriptors.MinAbsPartialCharge,
    Descriptors.NumAliphaticRings,
    Descriptors.NumSaturatedRings,
    Descriptors.NumHeteroatoms,
)


def _clean_protein(sequence: str) -> str:
    sequence = "".join(
        character
        for character in str(sequence).upper()
        if character in AA_INDEX
    )
    if len(sequence) < 3:
        raise ValueError("Protein sequence must contain at least three standard amino acids.")
    return sequence


def _canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def _safe_descriptor(function, molecule) -> float:
    try:
        value = float(function(molecule))
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _drug_descriptor(smiles: str) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles!r}")
    return np.asarray(
        [_safe_descriptor(function, molecule) for function in DRUG_DESCRIPTOR_FUNCTIONS],
        dtype=np.float32,
    )


def _protein_descriptor(sequence: str) -> np.ndarray:
    """AAC + dipeptide + grouped composition + global sequence properties."""
    sequence = _clean_protein(sequence)
    length = len(sequence)

    amino_acid_composition = np.zeros(20, dtype=np.float32)
    for amino_acid in sequence:
        amino_acid_composition[AA_INDEX[amino_acid]] += 1.0
    amino_acid_composition /= length

    dipeptide = np.zeros((20, 20), dtype=np.float32)
    for left, right in zip(sequence[:-1], sequence[1:]):
        dipeptide[AA_INDEX[left], AA_INDEX[right]] += 1.0
    dipeptide /= max(length - 1, 1)

    grouped_composition = np.zeros(7, dtype=np.float32)
    grouped_dipeptide = np.zeros((7, 7), dtype=np.float32)
    for amino_acid in sequence:
        grouped_composition[AA_GROUP_INDEX[amino_acid]] += 1.0
    grouped_composition /= length
    for left, right in zip(sequence[:-1], sequence[1:]):
        grouped_dipeptide[
            AA_GROUP_INDEX[left], AA_GROUP_INDEX[right]
        ] += 1.0
    grouped_dipeptide /= max(length - 1, 1)

    nonzero = amino_acid_composition[amino_acid_composition > 0]
    entropy = float(
        -(nonzero * np.log(nonzero)).sum() / np.log(20)
    ) if len(nonzero) else 0.0

    global_properties = np.asarray(
        [
            np.log1p(length) / 10.0,
            sum(amino_acid in HYDROPHOBIC for amino_acid in sequence) / length,
            sum(amino_acid in POLAR for amino_acid in sequence) / length,
            sum(amino_acid in POSITIVE for amino_acid in sequence) / length,
            sum(amino_acid in NEGATIVE for amino_acid in sequence) / length,
            (
                sum(amino_acid in POSITIVE for amino_acid in sequence)
                - sum(amino_acid in NEGATIVE for amino_acid in sequence)
            ) / length,
            sum(amino_acid in AROMATIC for amino_acid in sequence) / length,
            sum(amino_acid in SMALL for amino_acid in sequence) / length,
            entropy,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            amino_acid_composition,
            dipeptide.ravel(),
            grouped_composition,
            grouped_dipeptide.ravel(),
            global_properties,
        ]
    )


def _top_k_mean(values: np.ndarray, k: int) -> np.ndarray:
    if values.shape[1] <= k:
        return values.mean(axis=1)
    split = values.shape[1] - k
    return np.partition(values, split, axis=1)[:, -k:].mean(axis=1)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, np.clip(probabilities, 0.0, 1.0))),
    }


@dataclass
class PredictionOutput:
    p_pcm: np.ndarray
    p_scjke: np.ndarray
    p_final: np.ndarray
    support: np.ndarray
    evidence_positive: np.ndarray
    evidence_negative: np.ndarray
    pcm_weight: float


class AdaptiveSCJKEPCM:
    """Hybrid sequence/chemical evidence model for cluster-domain DTI shifts.

    PCM learns global drug and protein descriptor effects. SCJKE retrieves local
    class-conditional joint similarity evidence. Their mixture weight is based
    only on source entity sparsity:

        w_pcm = clip(0.5 * (n_unique_drugs / n_pairs
                          + n_unique_proteins / n_pairs), 0.10, 0.30)
    """

    def __init__(
        self,
        top_k: int = 200,
        protein_kmer: int = 3,
        fingerprint_bits: int = 2048,
        pcm_c: float = 0.003,
        min_pcm_weight: float = 0.10,
        max_pcm_weight: float = 0.30,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if protein_kmer < 1:
            raise ValueError("protein_kmer must be positive.")
        if fingerprint_bits < 64:
            raise ValueError("fingerprint_bits must be at least 64.")
        if not 0.0 <= min_pcm_weight <= max_pcm_weight <= 1.0:
            raise ValueError("PCM weight bounds must satisfy 0 <= min <= max <= 1.")

        self.top_k = int(top_k)
        self.protein_kmer = int(protein_kmer)
        self.fingerprint_bits = int(fingerprint_bits)
        self.pcm_c = float(pcm_c)
        self.min_pcm_weight = float(min_pcm_weight)
        self.max_pcm_weight = float(max_pcm_weight)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)

        self.fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=self.fingerprint_bits,
        )
        self.source: Optional[pd.DataFrame] = None
        self.drug_scaler: Optional[RobustScaler] = None
        self.protein_scaler: Optional[StandardScaler] = None
        self.pcm: Optional[LogisticRegression] = None
        self.protein_vectorizer: Optional[TfidfVectorizer] = None
        self.source_protein_matrix = None
        self.source_unique_proteins: Optional[pd.Index] = None
        self.source_unique_drugs: Optional[pd.Index] = None
        self.source_drug_fingerprints = None
        self.source_pair_protein_indices: Optional[np.ndarray] = None
        self.source_pair_drug_indices: Optional[np.ndarray] = None
        self.source_labels: Optional[np.ndarray] = None
        self.pcm_weight: Optional[float] = None

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame, require_label: bool) -> pd.DataFrame:
        required = {"SMILES", "Protein"}
        if require_label:
            required.add("Y")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        prepared = frame.copy().reset_index(drop=True)
        prepared["SMILES"] = prepared["SMILES"].map(_canonical_smiles)
        prepared["Protein"] = prepared["Protein"].map(_clean_protein)
        if require_label:
            prepared["Y"] = pd.to_numeric(prepared["Y"], errors="raise").astype(int)
            invalid = set(prepared["Y"].unique()).difference({0, 1})
            if invalid:
                raise ValueError(f"Y must contain only 0/1 labels; found {sorted(invalid)}")
            if prepared["Y"].nunique() != 2:
                raise ValueError("Source data must contain both DTI classes.")

            duplicate_groups = prepared.groupby(["SMILES", "Protein"])["Y"].nunique()
            if (duplicate_groups > 1).any():
                raise ValueError("Conflicting labels found for identical SMILES/protein pairs.")
            prepared = prepared.drop_duplicates(
                subset=["SMILES", "Protein"], keep="first"
            ).reset_index(drop=True)
        return prepared

    def _fingerprint(self, smiles: str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid canonical SMILES: {smiles!r}")
        return self.fingerprint_generator.GetFingerprint(molecule)

    def _fit_pcm(self, source: pd.DataFrame) -> None:
        unique_drugs = pd.Index(source["SMILES"].unique())
        unique_proteins = pd.Index(source["Protein"].unique())

        drug_descriptors = np.vstack(
            [_drug_descriptor(smiles) for smiles in unique_drugs]
        )
        protein_descriptors = np.vstack(
            [_protein_descriptor(sequence) for sequence in unique_proteins]
        )

        self.drug_scaler = RobustScaler(quantile_range=(5, 95)).fit(drug_descriptors)
        self.protein_scaler = StandardScaler().fit(protein_descriptors)
        drug_descriptors = np.nan_to_num(
            np.clip(self.drug_scaler.transform(drug_descriptors), -10, 10)
        ).astype(np.float32)
        protein_descriptors = np.nan_to_num(
            np.clip(self.protein_scaler.transform(protein_descriptors), -10, 10)
        ).astype(np.float32)

        drug_map = {smiles: index for index, smiles in enumerate(unique_drugs)}
        protein_map = {
            sequence: index for index, sequence in enumerate(unique_proteins)
        }
        source_features = np.concatenate(
            [
                drug_descriptors[
                    np.fromiter(
                        (drug_map[smiles] for smiles in source["SMILES"]),
                        dtype=int,
                        count=len(source),
                    )
                ],
                protein_descriptors[
                    np.fromiter(
                        (protein_map[sequence] for sequence in source["Protein"]),
                        dtype=int,
                        count=len(source),
                    )
                ],
            ],
            axis=1,
        )

        self.pcm = LogisticRegression(
            C=self.pcm_c,
            max_iter=1800,
            solver="saga",
            l1_ratio=1.0,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.pcm.fit(source_features, source["Y"].to_numpy(dtype=int))

    def _pcm_features(self, query: pd.DataFrame) -> np.ndarray:
        if self.drug_scaler is None or self.protein_scaler is None:
            raise RuntimeError("PCM has not been fitted.")
        unique_drugs = pd.Index(query["SMILES"].unique())
        unique_proteins = pd.Index(query["Protein"].unique())
        drug_descriptors = np.nan_to_num(
            np.clip(
                self.drug_scaler.transform(
                    np.vstack([_drug_descriptor(smiles) for smiles in unique_drugs])
                ),
                -10,
                10,
            )
        ).astype(np.float32)
        protein_descriptors = np.nan_to_num(
            np.clip(
                self.protein_scaler.transform(
                    np.vstack(
                        [_protein_descriptor(sequence) for sequence in unique_proteins]
                    )
                ),
                -10,
                10,
            )
        ).astype(np.float32)
        drug_map = {smiles: index for index, smiles in enumerate(unique_drugs)}
        protein_map = {
            sequence: index for index, sequence in enumerate(unique_proteins)
        }
        return np.concatenate(
            [
                drug_descriptors[
                    np.fromiter(
                        (drug_map[smiles] for smiles in query["SMILES"]),
                        dtype=int,
                        count=len(query),
                    )
                ],
                protein_descriptors[
                    np.fromiter(
                        (protein_map[sequence] for sequence in query["Protein"]),
                        dtype=int,
                        count=len(query),
                    )
                ],
            ],
            axis=1,
        )

    def _fit_scjke(self, source: pd.DataFrame) -> None:
        self.source_unique_proteins = pd.Index(source["Protein"].unique())
        self.source_unique_drugs = pd.Index(source["SMILES"].unique())

        self.protein_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(self.protein_kmer, self.protein_kmer),
            lowercase=False,
            binary=True,
            use_idf=False,
            norm="l2",
            dtype=np.float32,
        )
        self.source_protein_matrix = self.protein_vectorizer.fit_transform(
            self.source_unique_proteins.tolist()
        )
        self.source_drug_fingerprints = [
            self._fingerprint(smiles) for smiles in self.source_unique_drugs
        ]

        protein_map = {
            sequence: index
            for index, sequence in enumerate(self.source_unique_proteins)
        }
        drug_map = {
            smiles: index for index, smiles in enumerate(self.source_unique_drugs)
        }
        self.source_pair_protein_indices = np.fromiter(
            (protein_map[sequence] for sequence in source["Protein"]),
            dtype=int,
            count=len(source),
        )
        self.source_pair_drug_indices = np.fromiter(
            (drug_map[smiles] for smiles in source["SMILES"]),
            dtype=int,
            count=len(source),
        )
        self.source_labels = source["Y"].to_numpy(dtype=int)

    def fit(self, source_frame: pd.DataFrame) -> "AdaptiveSCJKEPCM":
        source = self._prepare_frame(source_frame, require_label=True)
        self.source = source
        self._fit_pcm(source)
        self._fit_scjke(source)

        entity_sparsity = 0.5 * (
            source["SMILES"].nunique() / len(source)
            + source["Protein"].nunique() / len(source)
        )
        self.pcm_weight = float(
            np.clip(entity_sparsity, self.min_pcm_weight, self.max_pcm_weight)
        )
        return self

    def _scjke_predict(self, query: pd.DataFrame):
        if (
            self.protein_vectorizer is None
            or self.source_protein_matrix is None
            or self.source_unique_drugs is None
            or self.source_pair_protein_indices is None
            or self.source_pair_drug_indices is None
            or self.source_labels is None
        ):
            raise RuntimeError("SCJKE has not been fitted.")

        query_unique_proteins = pd.Index(query["Protein"].unique())
        query_unique_drugs = pd.Index(query["SMILES"].unique())
        query_protein_matrix = self.protein_vectorizer.transform(
            query_unique_proteins.tolist()
        )
        query_drug_fingerprints = [
            self._fingerprint(smiles) for smiles in query_unique_drugs
        ]
        drug_similarity = np.vstack(
            [
                np.asarray(
                    DataStructs.BulkTanimotoSimilarity(
                        fingerprint, self.source_drug_fingerprints
                    ),
                    dtype=np.float32,
                )
                for fingerprint in query_drug_fingerprints
            ]
        )

        query_protein_map = {
            sequence: index
            for index, sequence in enumerate(query_unique_proteins)
        }
        query_drug_map = {
            smiles: index for index, smiles in enumerate(query_unique_drugs)
        }
        query_pair_protein_indices = np.fromiter(
            (query_protein_map[sequence] for sequence in query["Protein"]),
            dtype=int,
            count=len(query),
        )
        query_pair_drug_indices = np.fromiter(
            (query_drug_map[smiles] for smiles in query["SMILES"]),
            dtype=int,
            count=len(query),
        )

        positive_mask = self.source_labels == 1
        negative_mask = ~positive_mask
        probabilities = np.empty(len(query), dtype=np.float32)
        support = np.empty(len(query), dtype=np.float32)
        evidence_positive = np.empty(len(query), dtype=np.float32)
        evidence_negative = np.empty(len(query), dtype=np.float32)

        for start in range(0, len(query), self.batch_size):
            stop = min(start + self.batch_size, len(query))
            query_indices = np.arange(start, stop)
            protein_similarity = (
                query_protein_matrix[query_pair_protein_indices[query_indices]]
                @ self.source_protein_matrix.T
            ).toarray().astype(np.float32)
            joint_similarity = (
                protein_similarity[:, self.source_pair_protein_indices]
                * drug_similarity[query_pair_drug_indices[query_indices]][
                    :, self.source_pair_drug_indices
                ]
            )
            positive = _top_k_mean(joint_similarity[:, positive_mask], self.top_k)
            negative = _top_k_mean(joint_similarity[:, negative_mask], self.top_k)
            evidence_positive[query_indices] = positive
            evidence_negative[query_indices] = negative
            probabilities[query_indices] = (
                (positive + 1e-8) / (positive + negative + 2e-8)
            )
            support[query_indices] = joint_similarity.max(axis=1)

        return probabilities, support, evidence_positive, evidence_negative

    def predict(self, query_frame: pd.DataFrame) -> PredictionOutput:
        if self.pcm is None or self.pcm_weight is None:
            raise RuntimeError("Call fit() before predict().")
        query = self._prepare_frame(query_frame, require_label=False)
        p_pcm = self.pcm.predict_proba(self._pcm_features(query))[:, 1]
        p_scjke, support, evidence_positive, evidence_negative = self._scjke_predict(
            query
        )
        p_final = self.pcm_weight * p_pcm + (1.0 - self.pcm_weight) * p_scjke
        return PredictionOutput(
            p_pcm=np.asarray(p_pcm, dtype=float),
            p_scjke=np.asarray(p_scjke, dtype=float),
            p_final=np.asarray(p_final, dtype=float),
            support=np.asarray(support, dtype=float),
            evidence_positive=np.asarray(evidence_positive, dtype=float),
            evidence_negative=np.asarray(evidence_negative, dtype=float),
            pcm_weight=self.pcm_weight,
        )

    def predict_dataframe(
        self,
        query_frame: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        prediction = self.predict(query_frame)
        output = query_frame.copy().reset_index(drop=True)
        output["p_pcm"] = prediction.p_pcm
        output["p_scjke"] = prediction.p_scjke
        output["p_dti"] = prediction.p_final
        output["pred_label"] = (prediction.p_final >= threshold).astype(int)
        output["joint_support"] = prediction.support
        output["evidence_positive"] = prediction.evidence_positive
        output["evidence_negative"] = prediction.evidence_negative
        output["pcm_weight"] = prediction.pcm_weight
        return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Adaptive SCJKE-PCM on DrugBAN source_train and predict a "
            "cluster-domain target/query CSV."
        )
    )
    parser.add_argument("--source", required=True, help="Labeled source_train CSV.")
    parser.add_argument("--query", required=True, help="Target/query CSV.")
    parser.add_argument("--output", required=True, help="Prediction CSV path.")
    parser.add_argument("--metrics", default=None, help="Optional metrics JSON path.")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--pcm-c", type=float, default=0.003)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = pd.read_csv(args.source)
    query = pd.read_csv(args.query)

    model = AdaptiveSCJKEPCM(
        top_k=args.top_k,
        pcm_c=args.pcm_c,
        batch_size=args.batch_size,
    ).fit(source)
    prediction = model.predict_dataframe(query, threshold=args.threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_path, index=False)

    summary = {
        "source_n": int(len(model.source)),
        "query_n": int(len(query)),
        "source_unique_drugs": int(model.source["SMILES"].nunique()),
        "source_unique_proteins": int(model.source["Protein"].nunique()),
        "pcm_weight": float(model.pcm_weight),
        "top_k": int(model.top_k),
        "test_labels_used_for_fit": False,
    }
    if "Y" in query.columns:
        labels = pd.to_numeric(query["Y"], errors="raise").astype(int).to_numpy()
        summary["pcm"] = _metrics(labels, prediction["p_pcm"].to_numpy())
        summary["scjke"] = _metrics(labels, prediction["p_scjke"].to_numpy())
        summary["adaptive"] = _metrics(labels, prediction["p_dti"].to_numpy())

    if args.metrics:
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
