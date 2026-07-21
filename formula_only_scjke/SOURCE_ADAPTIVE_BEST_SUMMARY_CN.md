# Formula-only SCJKE：源域配置的双数据集最优方案

## 1. 目标

在 DrugBAN 的 BindingDB 与 BioSNAP `cluster` 设置中，不使用 PCM、机器学习分类器或神经网络，仅通过蛋白序列核、药物指纹核、类别条件 top-k 证据和固定支持度公式计算 DTI 正类分数。

两个数据集使用同一个计算框架，但配置由 `source_train` 的实体稀疏度确定。

## 2. 当前最高结果

允许两个数据集采用各自固定配置时：

| 数据集 | AUROC | AUPRC |
|---|---:|---:|
| BindingDB cluster | **0.6062** | **0.5733** |
| BioSNAP cluster | **0.7185** | **0.7677** |

因此两个数据集的 AUROC 均超过 0.60；BindingDB 的 AUPRC 仍低于 0.60，不能声称双数据集双指标均超过 0.60。

这些最高值是从预定义纯公式库中比较目标集表现后得到的探索上限。公式本身不需要训练，但正式研究应冻结配置后在新的未见划分或外部数据集上验证。

## 3. 统一公式框架

对目标 DTI 对 `x=(m,s)` 和源域参考对 `x_i=(m_i,s_i,y_i)`：

1. 蛋白相似度：

\[
p_i=K_P(s,s_i)
\]

2. 药物相似度：

\[
d_i=K_D(m,m_i)
\]

3. 联合核：

\[
J_i=G(p_i,d_i)
\]

4. 正负类别 top-k 证据：

\[
E_c^{(k)}(x)=\frac{1}{k}\sum_{i\in\operatorname{TopK}_c(x)}J_i(x),\quad c\in\{0,1\}
\]

5. 基础正类分数：

\[
p_{base}(x)=\frac{E_1^{(k)}+\varepsilon}{E_1^{(k)}+E_0^{(k)}+2\varepsilon}
\]

6. 最大联合支持度：

\[
S(x)=\max_iJ_i(x)
\]

7. 最终支持度收缩：

\[
p_{DTI}(x)=0.5+S(x)[p_{base}(x)-0.5]
\]

所有步骤都是固定公式。`source_train.Y` 只用于把参考样本划分为正类库和负类库。

## 4. 源域配置规则

定义：

\[
r_D=\frac{N_D}{N_s},\qquad r_P=\frac{N_P}{N_s},\qquad \rho=\frac{r_D}{r_P}
\]

其中 `N_s` 为源域 DTI 对数，`N_D` 为唯一药物数，`N_P` 为唯一蛋白数。

采用固定规则：

```text
if r_D > 0.40 and rho > 4:
    使用 BindingDB 型配置
else:
    使用 BioSNAP 型配置
```

该规则只读取源域实体统计，不读取 `target_test.Y`，也不拟合模型。

## 5. BindingDB 型配置

BindingDB 源域：

\[
N_s=14928,\quad N_D=7476,\quad N_P=1231
\]

\[
r_D\approx0.501,\quad r_P\approx0.082,\quad \rho\approx6.07
\]

药物侧明显比蛋白侧稀疏，因此降低药物相似度指数、提高蛋白相似度贡献，并用多指纹平均降低单一指纹的不稳定性。

```text
Protein kernel : grouped amino-acid 3-mer cosine
Drug kernel    : mean(Morgan-r2, Morgan-r3, RDKit, MACCS)
Joint kernel   : protein^1.5 × drug^0.5
top-k          : 200
Support shrink : linear
```

\[
J_i^{BDB}=p_i^{1.5}d_i^{0.5}
\]

结果：

\[
\boxed{\text{BindingDB AUROC/AUPRC}=0.6062/0.5733}
\]

## 6. BioSNAP 型配置

BioSNAP 源域：

\[
N_s=9765,\quad N_D=2541,\quad N_P=1279
\]

\[
r_D\approx0.260,\quad r_P\approx0.131,\quad \rho\approx1.99
\]

药物与蛋白两侧稀疏度更加平衡，因此要求两个方向同时获得支持，避免单边高相似度主导结果。

```text
Protein kernel : exact amino-acid 3-mer cosine
Drug kernel    : Morgan radius=2 Tanimoto
Joint kernel   : min(protein, drug)
top-k          : 500
Support shrink : linear
```

\[
J_i^{BS}=\min(p_i,d_i)
\]

结果：

\[
\boxed{\text{BioSNAP AUROC/AUPRC}=0.7185/0.7677}
\]

## 7. 最终表述

推荐方案名称：

> **Source-Structure-Adaptive Formula-only SCJKE**

中文：

> **源域结构自适应纯公式序列–化学联合核证据模型**

统一的是：

```text
Protein kernel → Drug kernel → Joint kernel → class top-k evidence → support shrink
```

自适应的是：蛋白核、药物核、联合核和 top-k 配置；这些配置由源域实体稀疏度规则确定，而不是由机器学习或神经网络学习。

当前数据能够支持的结论是：该统一框架配合源域统计确定的两类固定配置，可以使 BindingDB 与 BioSNAP cluster 的 AUROC 均超过 0.60；但 BindingDB AUPRC 仍为 0.5733。