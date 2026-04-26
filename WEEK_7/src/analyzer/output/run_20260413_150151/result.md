# 实验结果摘要：`run_20260413_150151`

**生成时间（流水线记录）：** 2026-04-13（本地 `pipeline_summary.json` 中 `created_at_local`）  
**输出根目录：** `WEEK_5/src/output/run_20260413_150151`  
**数据来源：** `WEEK_5/src/data` 下 5 份 UTF-8 隐私政策文本（应用名即子目录名）。

---

## 1. 流水线配置（来自 `pipeline_summary.json`）

| 项目 | 取值 |
|------|------|
| 模式 | **FullPipeline**（句级分类 + 审计 + 聚类分析 + 送标队列 + review_bundle + 汇总） |
| 分类 / 审计模型 | **DeepSeek**（`provider: deepseek`），`classify_limit` / `audit_limit` 均为 **0**（全量） |
| PII few-shot | **`use_pii_shots: true`**，默认使用 `src/ref/shots.json`（摘要中 `pii_shots_json` 为 null 时表示走脚本内默认路径逻辑），**`pii_shots_max: 12`** |
| 审计前聚类 | **`pre_audit_clustering_in_prepare: true`**（写 WEEK3 CSV 前句向量 + HDBSCAN，DS 为同簇句拼接） |
| 聚类嵌入 | **`flag`** + **`BAAI/bge-small-zh-v1.5`** |
| 送标 / review 条数上限 | **`labeling_top_n: 30`**（每应用 `for_labeling.jsonl` 与 `review_bundle` 各 **30** 条，按优先级截取） |

---

## 2. 数据规模（全库合计）

| 指标 | 数值 |
|------|------|
| 应用数量 | **5** |
| 切句后总句数 | **1385** |

---

## 3. 句级审计（DeepSeek；`audit_processed.csv` 汇总）

三标签含义与 WEEK_3 任务定义一致：**incorrect**（披露缺失倾向）、**incomplete**（粗细/完整度）、**inconsistent**（冲突）。以下为各应用 **`label = 1`** 的句数及占该应用句数比例。

| 应用 | 句数 | incorrect=1 | incomplete=1 | inconsistent=1 |
|------|------|---------------|----------------|------------------|
| Blued极速版 | 192 | 51（26.6%） | 3（1.6%） | 4（2.1%） |
| LOFTER | 227 | 44（19.4%） | 5（2.2%） | 1（0.4%） |
| Soul | 347 | 87（25.1%） | 11（3.2%） | 6（1.7%） |
| 世纪佳缘 | 323 | 53（16.4%） | 2（0.6%） | 0（0%） |
| 小红书 | 296 | 109（36.8%） | 9（3.0%） | 2（0.7%） |
| **合计** | **1385** | **344（24.8%）** | **30（2.2%）** | **13（0.9%）** |

**简要解读：** 在「簇内句作 DS、当前句作 PP」及 few-shot 设定下，**incorrect 仍为主要信号**；**incomplete / inconsistent** 在多数应用中为小但非零，说明三标签维度均被模型使用。**小红书** incorrect 占比相对最高；**世纪佳缘** 未出现 inconsistent 正例。

---

## 4. 产物与下游实验

- **每应用子目录：** `sentences.jsonl`、`stats.json`、`manifest.json`、`pre_audit_cluster_summary.json`、`sentences_week3_2_2.csv`、`audit_raw.csv`、`audit_processed.csv`、`figures/`、`cluster_analysis/`（含 `sentences_cluster_full.jsonl`、`policy_vs_law_comparison.json`、UMAP 图等）、**`for_labeling.jsonl`（30 行）**、**`review_bundle.json`**。  
- **运行根目录：** `pipeline_summary.json`、`week5_aggregate_report.json`（本摘要的数据源之一）。

人工标注与将分歧写入 `shots.json` 的流程见 **`WEEK_5/README.md`** 中「送标与人工标注」及仓库内 **`pii_shots.py extract`** 说明。

---

## 5. 声明

本摘要中的比例与计数均来自 **`week5_aggregate_report.json`** 与 **`pipeline_summary.json`** 的机器汇总；审计标签为 **LLM 在给定提示与 DS/PP 构造下的自动结果**，若用于论文或对外结论，建议结合人工抽检（`review_bundle`）说明局限性。
