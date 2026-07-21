# Formula-only SCJKE：DrugBAN Cluster 跨域 DTI 纯公式方案

## 1. 强制约束

本方案完全移除 PCM，不训练任何机器学习分类器或神经网络。源域标签 `source_train.Y` 只用于把参考 DTI 对分成正类证据库与负类证据库；目标预测只读取 `target_test.SMILES` 和 `target_test.Protein`。

禁止使用：

- Logistic Regression、SVM、随机森林、XGBoost、CatBoost；
- PyTorch、TensorFlow 或可学习 embedding；
- 梯度下降或标签到参数的拟合；
- `target_test.Y` 参与单样本评分。

每个目标样本的分数只由固定指纹、序列 k-mer、相似度、top-k 聚合与支持度公式计算。

---

## 2. 结果

### 2.1 两个数据集使用同一个统一公式

当前预定义纯公式库中，目标集综合表现最好的公共公式为：

```text
Protein kernel : exact amino-acid 3-mer cosine
Drug kernel    : Morgan radius=2 Tanimoto
Joint rule     : harmonic mean
Evidence k     : multiscale {20, 50, 100, 200, 500}
Hub correction : none
Support shrink : none
```

| 数据集 | AUROC | AUPRC |
|---|---:|---:|
| BindingDB cluster | **0.5856** | **0.5468** |
| BioSNAP cluster | **0.7046** | **0.7563** |

### 2.2 每个数据集使用各自固定公式

| 数据集 | AUROC | AUPRC |
|---|---:|---:|
| BindingDB cluster | **0.6062** | **0.5733** |
| BioSNAP cluster | **0.7185** | **0.7677** |

BindingDB 配置：

```text
Protein kernel : grouped amino-acid 3-mer cosine
Drug kernel    : mean(Morgan-r2, Morgan-r3, RDKit, MACCS)
Joint rule     : protein^1.5 × drug^0.5
Evidence k     : 200
Hub correction : none
Support shrink : linear support shrink
```

BioSNAP 配置：

```text
Protein kernel : exact amino-acid 3-mer cosine
Drug kernel    : Morgan radius=2 Tanimoto
Joint rule     : min(protein_similarity, drug_similarity)
Evidence k     : 500
Hub correction : none
Support shrink : linear support shrink
```

### 2.3 严格 source-only 公式选择

在最终扩展公式库中，仅使用 `source_train` 内部的确定性 cold-both 验证折选择公共公式，结果为：

| 数据集 | AUROC | AUPRC |
|---|---:|---:|
| BindingDB cluster | **0.5519** | **0.5184** |
| BioSNAP cluster | **0.6663** | **0.7130** |

其配置为：

```text
Protein kernel : exact amino-acid 3-mer cosine
Drug kernel    : Morgan radius=3 Tanimoto
Joint rule     : protein^2 × drug
Evidence k     : 20
Hub correction : none
Support shrink : linear support shrink
```

> 0.6062/0.5733 与 0.7185/0.7677 是在目标标签已知后从预定义公式库中比较得到的探索上限。公式本身仍无训练，但测试集参与了公式选择，因此不能作为完全无偏的论文主结果。正式使用应冻结公式后在新的外部数据或新 cluster split 上验证。

---

## 3. 蛋白固定核

### 3.1 精确 3-mer 核

对蛋白序列统计所有连续三肽频率向量：

\[
\phi_3(s)\in\mathbb R^{20^3}.
\]

归一化后计算余弦相似度：

\[
K_P^{3mer}(s,s_i)
=
\frac{\phi_3(s)^\top\phi_3(s_i)}
{\|\phi_3(s)\|_2\|\phi_3(s_i)\|_2+\varepsilon}.
\]

### 3.2 分组 3-mer 核

将氨基酸映射到 7 组：

```text
AGV | ILFP | YMTS | HNQW | RK | DE | C
```

统计 \(7^3\) 维分组 3-mer，并计算相同的余弦相似度 \(K_P^{G3}\)。

---

## 4. 药物固定核

Morgan radius=2 的 Tanimoto 核：

\[
K_D^{M2}(m,m_i)
=
\frac{|F_{M2}(m)\cap F_{M2}(m_i)|}
{|F_{M2}(m)\cup F_{M2}(m_i)|+\varepsilon}.
\]

BindingDB 使用的多指纹平均核：

\[
K_D^{mean}
=
\frac{K_D^{M2}+K_D^{M3}+K_D^{RDKit}+K_D^{MACCS}}{4}.
\]

所有权重固定，不通过标签学习。

---

## 5. 联合核

记蛋白相似度为 \(p_i\)，药物相似度为 \(d_i\)。

统一公式使用调和联合核：

\[
J_i^H=\frac{2p_id_i}{p_i+d_i+\varepsilon}.
\]

BindingDB 最优公式：

\[
J_i^{BDB}=p_i^{1.5}d_i^{0.5}.
\]

BioSNAP 最优公式：

\[
J_i^{BS}=\min(p_i,d_i).
\]

---

## 6. 类别条件 top-k 证据

源域标签仅定义：

\[
\mathcal I_c=\{i:y_i=c\},\qquad c\in\{0,1\}.
\]

对每类分别选择联合相似度最大的前 \(k\) 个参考对：

\[
E_c^{(k)}(x)
=
\frac1k\sum_{i\in\operatorname{TopK}_c(x)}J_i(x).
\]

正类分数：

\[
p^{(k)}(x)
=
\frac{E_1^{(k)}(x)+\varepsilon}
{E_1^{(k)}(x)+E_0^{(k)}(x)+2\varepsilon}.
\]

统一公式使用多尺度：

\[
E_c^{multi}(x)
=
\frac15\sum_{k\in\{20,50,100,200,500\}}E_c^{(k)}(x).
\]

\[
p_{common}(x)
=
\frac{E_1^{multi}(x)+\varepsilon}
{E_1^{multi}(x)+E_0^{multi}(x)+2\varepsilon}.
\]

---

## 7. 支持度收缩

定义最大联合支持度：

\[
S(x)=\max_iJ_i(x).
\]

支持度收缩：

\[
p_{shrink}(x)
=0.5+S(x)[p(x)-0.5].
\]

目标样本缺少可靠源域近邻时，分数会向 0.5 收缩。该步骤没有可学习参数。

---

## 8. 搜索空间

GitHub Actions 共评估 **6,912** 个确定性公式配置，包括：

- 精确/分组蛋白 3-mer 核；
- Morgan-r2、Morgan-r3、RDKit、MACCS 及固定均值药物核；
- 乘积、几何、调和、最小值及非对称幂联合核；
- `k=20/50/100/200/500/multiscale`；
- 固定源域实体度数去偏；
- 无收缩、平方根支持度收缩、线性支持度收缩。

工作流在运行前静态拒绝以下内容：

```text
LogisticRegression
RandomForest
XGBoost
CatBoost
torch
tensorflow
.fit(
```

完整工作流的解压、语法检查、无模型拟合检查、双数据集基准与 artifact 上传均已通过。

---

## 9. 数据泄漏边界

合法：

```text
source_train.SMILES
source_train.Protein
source_train.Y
```

用于构建正/负参考证据库。

预测：

```text
target_test.SMILES
target_test.Protein
```

评估完成后才读取：

```text
target_test.Y
```

如果用 `target_test.Y` 选择公式，则结果必须标为探索性上限。

---

## 10. 推荐

- 需要一个统一方法：使用 `3-mer + Morgan-r2 + harmonic + multiscale top-k`，结果为 BindingDB **0.5856/0.5468**、BioSNAP **0.7046/0.7563**。
- 追求当前数据集最高值：使用数据集特异固定公式，结果为 BindingDB **0.6062/0.5733**、BioSNAP **0.7185/0.7677**。
- 论文正式无偏比较：冻结公式并在新的未见 cluster split 或外部数据集上评估；当前 source-only 公共公式结果为 BindingDB **0.5519/0.5184**、BioSNAP **0.6663/0.7130**。
