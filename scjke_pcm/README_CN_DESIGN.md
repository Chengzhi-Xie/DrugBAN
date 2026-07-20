# Adaptive SCJKE-PCM：DrugBAN cluster 跨域 DTI 二分类方案

## 1. 目标

本方案面向 DrugBAN 的 BindingDB 与 BioSNAP `cluster` 设置。当目标药物和目标蛋白都未出现在 `source_train` 中时，仅使用：

```text
source_train.SMILES
source_train.Protein
source_train.Y
```

对只提供下列输入的目标样本计算 DTI 正类分数：

```text
target_test.SMILES
target_test.Protein
```

`target_test.Y` 不参与训练、权重计算或预测，只在全部预测完成后用于计算 AUROC、AUPRC 和 Brier。

## 2. 最终结果

完整使用 DrugBAN 的 `cluster/source_train.csv -> cluster/target_test.csv`：

| 数据集 | Source | Target test | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|---:|
| BindingDB cluster | 14,928 | 1,779 | **0.6535** | **0.6130** | 0.2411 |
| BioSNAP cluster | 9,765 | 907 | **0.6854** | **0.7334** | 0.2365 |

因此，该方案能够在 cold-both/cluster 场景中，仅通过蛋白序列、药物 SMILES 和源域 DTI 标签，高于随机地区分目标 DTI 正负样本。

## 3. 总体公式

模型由两个互补分支组成：

1. `PCM`：通过药物和蛋白固定描述符学习可迁移的全局分类边界；
2. `SCJKE`：通过蛋白 3-mer 相似度与药物 Morgan-Tanimoto 相似度计算局部正负类别证据。

最终分数：

\[
p_{DTI}(x)=w_{PCM}p_{PCM}(x)+(1-w_{PCM})p_{SCJKE}(x)
\]

其中，融合权重仅根据 source_train 的实体稀疏度计算，不读取目标标签。

---

## 4. SCJKE 局部相似性分支

### 4.1 蛋白核

蛋白序列转换为归一化氨基酸 3-mer 向量：

\[
K_P(s,s_i)=\cos(\phi_{3mer}(s),\phi_{3mer}(s_i))
\]

实现配置：

```text
analyzer = char
ngram_range = (3, 3)
binary = true
use_idf = false
norm = l2
```

### 4.2 药物核

从 canonical SMILES 生成 Morgan 指纹：

```text
radius = 2
bits = 2048
```

药物相似度：

\[
K_D(m,m_i)=\operatorname{Tanimoto}(\phi_D(m),\phi_D(m_i))
\]

### 4.3 联合核

查询 DTI 对 \(x=(s,m)\) 与源域参考对 \(x_i=(s_i,m_i)\) 的联合相似度为：

\[
K_J(x,x_i)=K_P(s,s_i)K_D(m,m_i)
\]

乘积要求药物和蛋白两个方向同时相似，才能提供较强 DTI 证据。

### 4.4 类别条件 top-k

原始方案对全部类别样本求平均，在 cluster 场景会被大量低相似度样本稀释。优化后分别保留正类和负类中联合相似度最高的前 200 个样本：

\[
E_c^{(200)}(x)=\frac{1}{200}\sum_{i\in Top200_c(x)}K_J(x,x_i)
\]

SCJKE 正类分数：

\[
p_{SCJKE}(x)=\frac{E_1^{(200)}(x)+\epsilon}{E_1^{(200)}(x)+E_0^{(200)}(x)+2\epsilon}
\]

同时输出最大联合支持度：

\[
S(x)=\max_i K_J(x,x_i)
\]

`joint_support` 较低表示目标样本在源域缺少相似药物–蛋白组合，应被视为更强的 OOD 样本。

---

## 5. PCM 全局跨实体分支

SCJKE 依赖局部近邻。PCM 分支通过完全由序列和 SMILES 计算的固定描述符学习全局 DTI 分类边界，使新药物和新蛋白即使没有直接近邻也能获得预测。

### 5.1 药物描述符

从 SMILES 计算 30 个 RDKit 描述符，包括：

- 分子量、LogP、TPSA；
- 氢键供体和受体；
- 可旋转键；
- 环、芳香环、脂肪环和饱和环数量；
- FractionCSP3；
- 重原子、杂原子和 N/O 计数；
- MolMR、LabuteASA；
- BalabanJ、BertzCT；
- Chi 与 Kappa 指数；
- 部分电荷相关描述符。

记为：

\[
d(m)\in\mathbb{R}^{30}
\]

采用 RobustScaler，5%–95% 分位范围缩放，并裁剪到 `[-10,10]`。

### 5.2 蛋白描述符

仅从氨基酸序列计算：

- 20 维氨基酸组成；
- 400 维二肽组成；
- 7 维氨基酸分组组成；
- 49 维分组二肽组成；
- 9 维全局性质：长度、疏水/极性/正负电/芳香/小体积比例、净电荷组成近似和组成熵。

总维度：

\[
20+400+7+49+9=485
\]

记为：

\[
q(s)\in\mathbb{R}^{485}
\]

采用 StandardScaler 并裁剪到 `[-10,10]`。

### 5.3 PCM 分类器

拼接药物和蛋白描述符：

\[
h(x)=[\tilde d(m);\tilde q(s)]\in\mathbb{R}^{515}
\]

使用 L1 正则逻辑回归：

\[
p_{PCM}(x)=\sigma(\beta_0+\beta^Th(x))
\]

配置：

```text
solver = saga
penalty = L1
C = 0.003
class_weight = balanced
max_iter = 1800
random_state = 42
```

L1 正则用于降低高维序列描述符噪声和 source-specific 过拟合，并筛选更稳定的药物及蛋白属性。

---

## 6. 源域稀疏度自适应融合

设：

- \(N_s\)：源域 DTI 对数；
- \(N_D\)：源域唯一药物数；
- \(N_P\)：源域唯一蛋白数。

定义：

\[
w_{PCM}=\operatorname{clip}\left[\frac{1}{2}\left(\frac{N_D}{N_s}+\frac{N_P}{N_s}\right),0.10,0.30\right]
\]

解释：

- 唯一实体比例高，表示每个药物或蛋白的重复观测较少，局部近邻更稀疏，因此增加 PCM 权重；
- 实体重复度较高，局部相似性证据更充分，因此增加 SCJKE 权重。

BindingDB：

```text
N_s = 14928
N_D = 7476
N_P = 1231
w_PCM = 0.2916
```

BioSNAP：

```text
N_s = 9765
N_D = 2541
N_P = 1279
w_PCM = 0.1956
```

最终统一公式：

\[
\boxed{
p_{DTI}(x)=w_{PCM}\sigma(\beta_0+\beta^T[\tilde d(m);\tilde q(s)])+(1-w_{PCM})\frac{E_1^{(200)}(x)+\epsilon}{E_1^{(200)}(x)+E_0^{(200)}(x)+2\epsilon}
}
\]

默认分类：

\[
\hat y=1[p_{DTI}\ge 0.5]
\]

---

## 7. 分支消融结果

### BindingDB cluster

| 方法 | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
| PCM | 0.6453 | 0.6024 | 0.2372 |
| SCJKE top-200 | 0.5837 | 0.5311 | 0.2472 |
| **Adaptive SCJKE-PCM** | **0.6535** | **0.6130** | 0.2411 |

BindingDB 中药物实体较稀疏，PCM 是主要泛化来源；SCJKE 提供补充局部证据，使融合排序优于两个单独分支。

### BioSNAP cluster

| 方法 | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
| PCM | 0.6349 | 0.6184 | 0.2395 |
| SCJKE top-200 | **0.6854** | **0.7386** | 0.2374 |
| **Adaptive SCJKE-PCM** | **0.6854** | **0.7334** | **0.2365** |

BioSNAP 的局部联合相似性更有效，因此自动给予 SCJKE 更高权重。PCM 基本保留排序能力，并略微改善 Brier。

---

## 8. 数据泄漏边界

允许：

```text
fit:
  source_train.SMILES
  source_train.Protein
  source_train.Y

predict:
  target_test.SMILES
  target_test.Protein
```

禁止：

1. 使用 `target_test.Y` 训练、校准、选择 top-k 或选择融合权重；
2. 使用目标测试结果反复修改超参数后仍将结果表述为无偏测试；
3. 与无监督 DrugBAN-CDAN 比较时使用 `target_train.Y`；这会将方法变成监督目标域适配。

实现中的 `fit()` 和 `predict()` 都不读取 query 标签。即使 query CSV 带有 `Y`，也只在预测全部完成后计算指标。翻转或移除 query 标签不会改变输出概率。

---

## 9. 输出字段

| 字段 | 含义 |
|---|---|
| `p_pcm` | PCM 全局正类分数 |
| `p_scjke` | SCJKE 局部正类分数 |
| `p_dti` | 自适应融合最终分数 |
| `pred_label` | 0.5 阈值二分类结果 |
| `joint_support` | 最大源域联合相似度 |
| `evidence_positive` | 正类 top-k 联合证据 |
| `evidence_negative` | 负类 top-k 联合证据 |
| `pcm_weight` | 源域稀疏度决定的 PCM 权重 |

---

## 10. 运行

安装：

```bash
pip install -r scjke_pcm/requirements.txt
```

两个数据集完整复现：

```bash
python scjke_pcm/run_cluster_benchmarks.py --repo-root .
```

BindingDB：

```bash
python scjke_pcm/scjke_pcm.py \
  --source datasets/bindingdb/cluster/source_train.csv \
  --query datasets/bindingdb/cluster/target_test.csv \
  --output scjke_pcm/results/bindingdb_cluster_predictions.csv \
  --metrics scjke_pcm/results/bindingdb_cluster_metrics.json
```

BioSNAP：

```bash
python scjke_pcm/scjke_pcm.py \
  --source datasets/biosnap/cluster/source_train.csv \
  --query datasets/biosnap/cluster/target_test.csv \
  --output scjke_pcm/results/biosnap_cluster_predictions.csv \
  --metrics scjke_pcm/results/biosnap_cluster_metrics.json
```

query 文件可以不包含 `Y`；此时程序生成预测但不计算评估指标。

---

## 11. 与 DrugBAN 和 CP 的关系

### DrugBAN 辅助分支

建议将以下特征与 DrugBAN 融合：

```text
p_pcm
p_scjke
joint_support
evidence_positive
evidence_negative
```

例如：

\[
z_{fused}=z_{DrugBAN}+\lambda z_{SCJKE-PCM}
\]

\(\lambda\) 必须在 source validation 或预先定义的合法校准集上确定。

### Conformal Prediction

可定义：

\[
\hat p_1=p_{DTI},\quad \hat p_0=1-p_{DTI}
\]

\[
A(x,y)=1-\hat p_y(x)
\]

并将支持度用于难度感知非一致性分数：

\[
A_{support}(x,y)=\frac{1-\hat p_y(x)}{S(x)+\delta}
\]

低支持样本会更容易输出 `{0,1}`，而不是被过度自信地判为单一类别。

---

## 12. 解释边界与局限

1. `p_dti` 是统计分类分数，不是实验体系无关的真实物理结合概率；
2. PCM 当前是描述符拼接后的线性分类器，没有显式高阶药物–蛋白交互；
3. 3-mer、序列组成和 Morgan 指纹不能完整表示结合位点、构象和动力学；
4. BindingDB AUROC 0.6535 表明具备可用区分能力，但不足以替代深度 DTI 主模型；
5. 当前 top-k、L1 强度和权重边界属于开发设置，论文中应在新的嵌套 cold-both split 或外部数据集上冻结验证；
6. DTI 负样本可能包含尚未实验确认的潜在正相互作用。

## 13. 推荐项目表述

> 本方案仅使用源域蛋白序列、药物 SMILES 和 DTI 标签，通过全局 PCM 跨实体分类、局部 SCJKE 联合相似性证据和源域实体稀疏度自适应融合，在 DrugBAN 的 BindingDB 与 BioSNAP cluster 设置中分别达到 AUROC/AUPRC 0.6535/0.6130 和 0.6854/0.7334，证明在目标药物和目标蛋白均未见的 cold-both 条件下仍能高于随机地区分 DTI 正负样本。该分支最适合作为 DrugBAN 的可解释辅助证据和 domain-aware conformal prediction 的支持度来源，而不是独立替代深度 DTI 主模型。
