# Adaptive SCJKE-PCM for DrugBAN cluster-domain DTI prediction

This directory implements a source-supervised, target-label-free baseline for
DrugBAN's BindingDB and BioSNAP `cluster` splits.

## What changed from the original SCJKE formula

The original class-mean joint kernel worked in random splits but was weak when
both target drugs and target proteins were absent from `source_train`.
The optimized model uses two complementary branches:

1. **SCJKE local evidence** — protein 3-mer cosine similarity multiplied by
   Morgan/Tanimoto drug similarity, with class-conditional top-200 aggregation.
2. **PCM global evidence** — an L1-regularized logistic proteochemometric model
   trained on RDKit drug descriptors and protein sequence composition/
   dipeptide descriptors.

The final source-only score is

\[
p_{DTI}(x)=w_{PCM}p_{PCM}(x)+(1-w_{PCM})p_{SCJKE}(x).
\]

The PCM weight is computed before target prediction using only source entity
sparsity:

\[
w_{PCM}=\operatorname{clip}\left[
\frac{1}{2}\left(
\frac{N_{drug}^{unique}}{N_{source}}+
\frac{N_{protein}^{unique}}{N_{source}}
\right),0.10,0.30\right].
\]

A source set with many unique entities per DTI pair gives more weight to the
global PCM branch because local neighbor evidence is sparse. A source set with
more repeated entities gives more weight to SCJKE.

## Data use and leakage boundary

Model fitting uses only:

```text
source_train.SMILES
source_train.Protein
source_train.Y
```

Target prediction uses only:

```text
target_test.SMILES
target_test.Protein
```

`target_test.Y` is read only after prediction when the evaluation script
computes AUROC/AUPRC/Brier. Changing or removing query labels does not change
predictions; this is covered by the smoke test.

Do not use `target_train.Y` for this branch when comparing with unsupervised
DrugBAN-CDAN. Using target labels would turn the method into supervised target
adaptation.

## Reproduced DrugBAN cluster results

Source is the complete `source_train.csv`; evaluation is the complete
`target_test.csv`.

| Dataset | Source-only PCM weight | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|
| BindingDB cluster | 0.2916 | **0.6535** | **0.6130** | 0.2411 |
| BioSNAP cluster | 0.1956 | **0.6854** | **0.7334** | 0.2365 |

For comparison, the earlier global-mean SCJKE results were approximately
0.538 AUROC on BindingDB and 0.658 AUROC on BioSNAP. The optimized design thus
makes both cluster datasets distinguishable using only sequence, SMILES and
source DTI labels.

These are development results on the two provided target tests. The adaptive
weight itself does not consume target labels, but any further hyperparameter
changes after observing these metrics should be frozen and validated on a new
split or external dataset before being presented as an unbiased benchmark.

## Run

From the DrugBAN repository root:

```bash
pip install -r scjke_pcm/requirements.txt
python scjke_pcm/run_cluster_benchmarks.py --repo-root .
```

Run one dataset:

```bash
python scjke_pcm/scjke_pcm.py \
  --source datasets/bindingdb/cluster/source_train.csv \
  --query datasets/bindingdb/cluster/target_test.csv \
  --output scjke_pcm/results/bindingdb_cluster_predictions.csv \
  --metrics scjke_pcm/results/bindingdb_cluster_metrics.json
```

The query file may omit `Y`. When it contains `Y`, the script evaluates only
after all predictions have been generated.

## Output

- `p_pcm`: global proteochemometric score.
- `p_scjke`: local class-conditional joint-kernel score.
- `p_dti`: adaptive hybrid score.
- `pred_label`: thresholded class, default threshold 0.5.
- `joint_support`: maximum source-pair joint similarity.
- `evidence_positive`, `evidence_negative`: top-k class evidence.
- `pcm_weight`: source-only adaptive mixture weight.

`p_dti` is a bounded statistical score. Under strong source-target shift it
must not be interpreted as an assay-independent physical binding probability
without target-matched calibration.
