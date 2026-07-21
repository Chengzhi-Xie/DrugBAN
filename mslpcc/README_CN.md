# MS-LPCC 在 DrugBAN cluster 跨域设置下的公式基准结果

## 1. 测试目标

测试 **Multi-Scale Local Physicochemical Complementarity（MS-LPCC）** 是否能够仅根据：

- 药物 SMILES；
- 蛋白质序列；
- 固定理化常数与直接计算公式；

为 BindingDB 和 BioSNAP 的 `cluster` 目标域样本生成能够区分 DTI 正负样本的连续分数。

实现约束：

- 不训练分类器或回归器；
- 不使用神经网络；
- 不使用蛋白质三维结构；
- 不使用 docking；
- 不使用目标域标签计算样本分数；
- 源域标签仅用于选择固定公式；
- 目标域标签仅用于最终 AUROC/AUPRC 评价；
- 另外报告使用目标标签比较公式后的探索性上限。

---

## 2. MS-LPCC 公式框架

对药物计算固定理化描述符：

\[
v_d=[h_d,q_d,D_d,A_d,R_d,P_d,F_d]
\]

对蛋白质采用多尺度滑动窗口：

\[
L\in\{15,31,63\}
\]

主要互补通道包括：

\[
c_h=1-|h_d-h_w|
\]

\[
c_q=\exp\left[-\left(\frac{q_d+q_w}{0.75}\right)^2\right]
\]

\[
c_{HB}=\frac12[\min(D_d,A_w)+\min(A_d,D_w)]
\]

并测试芳香性、极性、柔性以及固定组合通道。总计搜索 1008 个固定候选公式。

---

## 3. 严格源域选择结果

公式配置只通过源域 cold-both 验证标签选择，目标域标签只用于最终评价。

| 数据集 | 源域选择公式 | Target AUROC | Target AUPRC |
|---|---|---:|---:|
| BindingDB cluster | `all_geom__mean__L31` | **0.5284** | **0.4892** |
| BioSNAP cluster | `charge__top05__scale_min` | **0.5901** | **0.5622** |

### 严格统一公式

```text
charge__mean__scale_max__inverse
```

| 数据集 | AUROC | AUPRC |
|---|---:|---:|
| BindingDB cluster | **0.4893** | **0.4664** |
| BioSNAP cluster | **0.4947** | **0.4810** |

四指标平均为 **0.4828**，接近随机排序。

---

## 4. 目标标签调优后的探索性上限

以下结果通过比较 `target_test` 标签选择公式，只能作为诊断上限。

### 4.1 每个数据集分别选择 AUROC/AUPRC 平均最优公式

| 数据集 | 最优公式 | AUROC | AUPRC | 两指标平均 |
|---|---|---:|---:|---:|
| BindingDB cluster | `charge__top05__scale_mean__inverse` | **0.5771** | **0.5166** | **0.5469** |
| BioSNAP cluster | `charge__top05__scale_mean` | **0.5931** | **0.5627** | **0.5779** |

即使允许目标标签选择，两个数据集仍不能达到 AUROC 和 AUPRC 均超过 0.60。

### 4.2 双数据集统一公式：四指标平均最优

```text
charge__top_multiscale_pool__scale_mean
```

| 数据集 | AUROC | AUPRC |
|---|---:|---:|
| BindingDB cluster | **0.5459** | **0.5139** |
| BioSNAP cluster | **0.5636** | **0.5243** |

四指标平均为 **0.5369**，最差指标为 **0.5139**。

### 4.3 双数据集统一公式：最大化最差指标

```text
charge__top_multiscale_pool__scale_min
```

| 数据集 | AUROC | AUPRC |
|---|---:|---:|
| BindingDB cluster | **0.5451** | **0.5167** |
| BioSNAP cluster | **0.5596** | **0.5214** |

最差指标为 **0.5167**。

---

## 5. 结果解释

表现最好的公式几乎都来自局部电荷互补通道：

\[
c_q=\exp\left[-\left(\frac{q_d+q_w}{0.75}\right)^2\right]
\]

这说明局部电荷环境具有少量 DTI 排序信息，但：

- 疏水、氢键、芳香、极性和柔性组合没有形成稳定额外增益；
- BindingDB 与 BioSNAP 对电荷方向的偏好相反；
- 无法固定一个在两个数据集上同时有效的正负排序方向。

---

## 6. 与 Formula-only SCJKE 对比

| 方法 | BindingDB AUROC/AUPRC | BioSNAP AUROC/AUPRC |
|---|---:|---:|
| 固定 BindingDB SCJKE 公式 | 0.6062 / 0.5733 | 0.6650 / 0.7090 |
| MS-LPCC 每数据集探索上限 | 0.5771 / 0.5166 | 0.5931 / 0.5627 |
| MS-LPCC 严格源域选择 | 0.5284 / 0.4892 | 0.5901 / 0.5622 |

MS-LPCC 明显弱于源域标签条件化的 SCJKE。

---

## 7. 最终结论

> **当前 MS-LPCC 计划不能满足双数据集 DTI 跨域有效区分需求。**

主要原因：

1. 没有真实结合口袋，整条序列滑窗会产生大量与结合无关的局部高分；
2. 不含药物与具体残基之间的几何约束；
3. 简单电荷、疏水和氢键互补不是 DTI 的充分条件；
4. 两个数据集的负样本构造、蛋白家族和药物空间不同；
5. 即使从 1008 个公式中使用目标标签选择，最高结果仍低于 SCJKE。

因此不建议将 MS-LPCC 作为推动 DrugBAN 融合特征更新的主要辅助教师。它最多适合用于多通道描述、错误分析或 CP 难度分组。

---

## 8. 结果性质

- `source_selected`：源域配置选择结果；
- `oracle_best_*`：目标标签调优后的探索性上限；
- 目标标签从未用于计算单个样本的 MS-LPCC 分数；
- 所有分数由固定公式直接计算；
- 未训练分类器、回归器或神经网络。
