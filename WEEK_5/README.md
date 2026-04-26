# WEEK_5：自包含一体化工程（原 WEEK_3 审计 + WEEK_4 预处理）

本目录已**复制并固定**以下代码，可在不依赖 `WEEK_3/`、`WEEK_4/` 路径的情况下独立运行：

| 来源 | 本目录位置 |
|------|------------|
| WEEK_4 预处理、切句、分类、绘图、聚类、送标 | `WEEK_5/src/*.py`（`run_prepare.py`、`pipeline.py`、`cluster_analysis.py` 等） |
| WEEK_3 句级 Data Safety vs 隐私政策审计 | `WEEK_5/src/run_audit.py` |
| 22 项 + 个保法参考段 | `WEEK_5/ref/taxonomy/*.json` |

> 维护说明：若你在 `WEEK_4/src` 或 `WEEK_3/src/2-2` 中修复了 bug，请**手动同步**对应文件到 `WEEK_5/src`（或重新执行复制命令），本仓库未配置自动双向同步。

## 目录约定

```
WEEK_5/
  requirements.txt           # 自 WEEK_4/src 复制
  requirements-optional.txt # 聚类 / cluster_analysis 额外依赖
  ref/taxonomy/              # 22 项与法参考 JSON
  src/
    data/                    # 输入：UTF-8 隐私政策 *.txt
    output/                  # 输出根；每次运行自动创建子目录 run_yyyyMMdd_HHmmss/
    run_pipeline.ps1         # 主入口（推荐）
    run_pipeline.bat
    run_audit.py             # WEEK_3 审计
    aggregate_pipeline_report.py
    run_week5_batch.ps1      # 仅批量汇总历史 run_*
    run_week5_full_audit.ps1 # 兼容别名（-PipelineOutDir 即 -OutDir；可加 -FullPipeline）
    … 其余 Python 模块 …
```

## 环境与依赖

```powershell
cd "D:\...\Android-Privacy-Detection-Software\WEEK_5"
pip install -r requirements.txt
pip install -r requirements-optional.txt   # run_pipeline 默认写 WEEK3 CSV 前会做审计前聚类（≥2 句）；-RunClusterAnalysis 亦依赖
```

DeepSeek：`$env:DEEPSEEK_API_KEY`（句级分类与 Data Safety 审计）。聚类默认使用 **FlagEmbedding（BGE）** 本地推理（`run_pipeline.ps1` 的 `-ClusterEmbedBackend flag`、`-ClusterModel BAAI/bge-small-zh-v1.5`）；仍可从 Hugging Face 拉权重，下载慢请设 `$env:HF_ENDPOINT="https://hf-mirror.com"`（流水线在 `-RunClusterAnalysis` 且未设置时会自动使用该镜像）。若需旧路径，指定 `-ClusterEmbedBackend sentence_transformers` 与对应 `-ClusterModel`。预下载权重：`python download_bge_model.py --model BAAI/bge-small-zh-v1.5 --local-dir ref\bge-small-zh-v1.5`，再把 `--model` / `-ClusterModel` 指向该目录。微调请安装 `pip install -U "FlagEmbedding[finetune]"` 并参考 FlagEmbedding 官方 `examples/finetune/embedder`。

## 一键运行（推荐）

在 **`WEEK_5/src`** 下执行；**不指定 `-OutDir` 时**，自动创建 **`output\run_<时间戳>\`**：

```powershell
cd "D:\...\Android-Privacy-Detection-Software\WEEK_5\src"
$env:DEEPSEEK_API_KEY = "sk-..."

# 等价于：classify + 审计 + 聚类 + for_labeling + review_bundle；LabelingTopN 默认 200；存在 ref\shots.json 时自动 few-shot
# run_pipeline 在写 sentences_week3_2_2.csv 前固定做句向量+HDBSCAN（文档 ≥2 句时），DS=同簇其它句、PP=当前句
.\run_pipeline.ps1 -FullPipeline
# 需要全量送标（不设 200 上限）时显式加大，例如：.\run_pipeline.ps1 -FullPipeline -LabelingTopN 999999
```

与上面等价的分步写法（便于单独开关某步）：

```powershell
.\run_pipeline.ps1 `
  -PrepareMode classify `
  -Provider deepseek `
  -RunAudit `
  -RunClusterAnalysis `
  -ExportLabelingQueue `
  -ExportReviewJson `
  -LabelingTopN 200
```

完成后在同一次运行目录下会有：

- 各应用子文件夹：`sentences.jsonl`、`stats.json`、`sentences_week3_2_2.csv`、可选 `audit_*.csv`、`cluster_analysis/`、`for_labeling.jsonl` 等  
- **`pipeline_summary.json`**  
- **`week5_aggregate_report.json`**（脚本末尾默认执行 `aggregate_pipeline_report.py`；若只要中间产物可加 **`-SkipAggregate`**）

## 送标与人工标注流程

流水线在开启 **`-ExportLabelingQueue`** / **`-ExportReviewJson`**（**`-FullPipeline`** 已包含）时，会在每个应用子目录下写出 **`for_labeling.jsonl`** 与 **`review_bundle.json`**，用于「优先送标哪些句」以及「在同一份 JSON 里对照 AI 与人工结论」。

### 1. 送标队列：`for_labeling.jsonl`

- **内容**：从当前 run 的 **`sentences.jsonl`** 按规则排序后的若干行，每行一条 JSON（一句）。字段与 `sentences.jsonl` 基本一致，并额外包含 **`labeling_queue_rank`**（0 起）、**`labeling_priority_score`**（见下）。
- **排序意图**（`labeling_queue.py`）：**模型判为 PII 相关且带 keyword_hint 的句排在最前**；其次考虑 **`pii_related`** 与 **`keyword_hint`** 的组合；再以 **较低 `confidence`** 作为不确定性的辅助排序，便于优先请人看「难例 / 分歧风险」。
- **条数**：由 **`-LabelingTopN`** 控制（**`-FullPipeline`** 下默认 **200**；句数不足则少于 200）。需要全量送标时把该参数设为足够大（如 `999999`）。
- **用途**：可导入自研标注工具、表格或脚本；**不要求**与 `review_bundle` 同时存在，但二者通常来自同一次 `run_prepare`。

### 2. 人工标注载体：`review_bundle.json`

- **生成时机**：与 `for_labeling.jsonl` 一起在 **`run_prepare.py`** 阶段写出（路径见该应用目录下的 **`review_bundle.json`**）。
- **结构概要**：顶层含 **`schema_version`**、**`kind`**、**`items`** 数组。每条 **`items[]`** 含稳定 **`id`**（格式 **`{doc_id}#{sent_index}`**）、**`text`**、**`ai`**（含 `pii_related`、`confidence`、`raw_model_output` 等）以及 **`human`**。
- **初始状态**：**`human.pii_related`** 为 **`null`**，**`human.reviewed_at`** 等为空；标注人补全后用于与 **`ai`** 对比、算一致率等。

### 3. 推荐标注方式

**方式 A — 直接编辑总包（适合人数少、条数不多）**

1. 打开某应用下的 **`review_bundle.json`**。  
2. 对需要标注的条目，在 **`human`** 内填写：  
   - **`pii_related`**：人工判断句是否与 PII/个人信息处理实质相关，填 **`true`** 或 **`false`**；  
   - 可选 **`notes`**、**`reviewer_id`**；  
   - 完成该条后填写 **`reviewed_at`**（建议 **ISO8601** 时间字符串，如 `2026-04-13T12:00:00+08:00`）。  
3. 保存后执行校验与统计（见下文 **`review_store.py validate` / `stats`**）。

**方式 B — 拆分编辑再合并（适合分工只改「人工」块）**

在 **`WEEK_5/src`** 下：

```powershell
cd "D:\...\Android-Privacy-Detection-Software\WEEK_5\src"
# 从已有总包拆出三份 JSON（句子列表、仅 AI、仅 human）
python review_store.py split --bundle "..\output\run_xxx\AppName\review_bundle.json" --out-dir "D:\work\review_split"
# 编辑 out-dir 中的 human_evaluations.json（保持每条 id 与结构一致）
python review_store.py merge --bundle "..\output\run_xxx\AppName\review_bundle.json" --human "D:\work\review_split\human_evaluations.json" --out "..\output\run_xxx\AppName\review_bundle_merged.json"
```

若需**从送标队列 JSONL 重新生成**总包（例如想换子集或修正上游行）：

```powershell
python review_store.py init --from-jsonl "..\output\run_xxx\AppName\for_labeling.jsonl" --out "D:\work\review_bundle.json" --note "manual_init"
```

### 4. `review_store.py` 子命令一览

| 子命令 | 作用 |
|--------|------|
| **`init`** | 从 **`for_labeling.jsonl`** 或 **`sentences.jsonl`** 构建 **`review_bundle.json`**。 |
| **`validate`** | 检查 **`human.pii_related`** 类型、与 **`reviewed_at`** 等一致性；失败时非 0 退出。 |
| **`stats`** | 输出已标数量、与 AI 一致/分歧等汇总（便于实验报告）。 |
| **`split`** | 拆成 **`sentences_for_review.json`**、**`ai_evaluations.json`**、**`human_evaluations.json`** 等，便于只传人工文件。 |
| **`merge`** | 将编辑后的 **`human_evaluations.json`**（或兼容列表结构）合并回总包。 |

校验与统计示例：

```powershell
python review_store.py validate --bundle "..\output\run_xxx\AppName\review_bundle.json"
python review_store.py stats --bundle "..\output\run_xxx\AppName\review_bundle.json"
```

### 5. 与流水线的关系说明

- **送标与总包生成**在 **`run_prepare`** 中完成；**`run_pipeline.ps1`** 通过 **`-ExportLabelingQueue` / `-ExportReviewJson`**（或 **`-FullPipeline`**）打开即可。  
- **`run_audit.py`**、**`cluster_analysis.py`** 不修改 `review_bundle`；若重跑流水线产生新目录，需在**新 run** 下重新导出或自行拷贝/合并人工结果。  
- **`human.pii_related`** 的语义应与分类阶段 **`pii_related`** 一致（句级是否属个人信息处理相关表述），便于 **`stats`** 做 AI–人 对比。

指定输出路径：

```powershell
.\run_pipeline.ps1 -OutDir "D:\work\my_run" -PrepareMode classify -Provider mock
```

## 批处理：仅对已有 `output\run_*` 重新汇总

```powershell
cd "D:\...\Android-Privacy-Detection-Software\WEEK_5\src"
.\run_week5_batch.ps1 -MaxRuns 20
```

## 实验五流程说明（简述）

1. **WEEK_4 段**：`data\*.txt` → 切句 →（可选）`pii_related` 分类 → WEEK_3 格式 CSV → 可选送标导出。  
2. **WEEK_3 段**：`-RunAudit` 时对每行调用 `run_audit.py`。经 `run_pipeline` 写出 WEEK3 CSV 时 **固定**在写表前做聚类（句数 ≥2）：**DS 列为同簇其它句拼接**，**PP 列为当前句**；仅 0～1 句时无法聚类，回退为与 `all_data_150.csv` 类似的 **空 JSON 占位 DS**。单独跑 `run_prepare.py --write-week3-csv` 时行为相同。  
3. **WEEK_5 段**：默认写 **`week5_aggregate_report.json`**，汇总各应用句量与 audit 正例数等。

## 与旧 WEEK_4 脚本的关系

仓库中 **`WEEK_4/src/run_full_pipeline.ps1`** 仍可单独使用（输出在 `WEEK_4/out/…`）。**自包含、固定 `src/data` 与 `src/output`** 的一体化流程请以 **`WEEK_5/src/run_pipeline.ps1`** 为准。

## 从上游目录重新同步 Python（可选）

在仓库根目录 PowerShell 示例（**会覆盖** `WEEK_5/src` 中与 WEEK_4 同名的模块；请勿覆盖你已改动的 `run_pipeline.ps1` / `aggregate_pipeline_report.py` 除非你有意为之）：

```powershell
$R = "D:\...\Android-Privacy-Detection-Software"
Copy-Item "$R\WEEK_4\src\*.py" "$R\WEEK_5\src\" -Force
Copy-Item "$R\WEEK_3\src\2-2\run_audit.py" "$R\WEEK_5\src\" -Force
Copy-Item "$R\WEEK_4\ref\taxonomy\*" "$R\WEEK_5\ref\taxonomy\" -Force
Copy-Item "$R\WEEK_4\src\requirements.txt" "$R\WEEK_5\requirements.txt" -Force
Copy-Item "$R\WEEK_4\requirements-optional.txt" "$R\WEEK_5\requirements-optional.txt" -Force
```
