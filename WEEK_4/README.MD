# Lab 2-3 实验过程手册：隐私政策切句与「个人信息利用相关」句筛选

本实验对应课程讲义《LLM-based Privacy Policy Audit》（Lab2-3）中的**文本分类**思路，并衔接后续**数据标注**与**比较器打分**流程。参考材料：

- `ref/system_design_expr_lab2_3_handout.pdf`：分类与聚类 pipeline 概念（提示工程、嵌入、降维、聚类）。
- `ref/privacy_policy/policies_huaWeiAppMarket/Soul.txt`：示例隐私政策全文（中文 App 商店抓取文本）。

## 1. 实验目的

1. 将单份隐私政策**切分为句子**（或语义上可独立标注的最小单元），保留在原文中的顺序与可追溯下标。
2. 对每个句子判断其是否与**个人信息的处理与利用**相关（收集、使用、存储、共享、转让、委托处理、跨境、安全、权利响应、未成年人等），输出可供人工校对与批量导入标注工具的格式。
3. 为后续环节准备**高价值子集**：优先将模型判为「相关」的句子（及可选的规则命中句）送入人工标注，再交给比较器与 Data Safety 等来源对齐打分。

## 2. 与讲义知识点的对应关系

| 讲义主题 | 在本实验中的体现 |
|----------|------------------|
| 文本分类 | 句子级二分类：是否与「个人信息利用」相关；通过提示词约束 JSON 输出。 |
| 嵌入与聚类（拓展） | 对句子编码后做降维与聚类，用于发现主题分布或抽检难例，**非必做**（本仓库提供 `sentence_cluster.py`）。 |

## 2.5 代码完整工作流与模块

下列顺序对应仓库中的**实际调用关系**，便于从「政策文本」一直走到「图表 / 人工 JSON / 可选聚类」。

### 端到端数据流

1. **输入**：UTF-8 隐私政策 `.txt`（`ref/privacy_policy/...` 或 `src/data/*.txt`）。
2. **切句**（`sentence_split.split_policy_text`）：由 `pipeline.build_rows_for_text` 在逐句循环前调用，可选 `--max-chars` 合并过长碎片。
3. **规则弱信号**（`keyword_hint.keyword_hint`）：每句写入布尔字段 `keyword_hint`。
4. **可选 LLM 二分类**（`llm_related.classify_sentence`）：`--mode classify` 且 `--provider mock|ollama|deepseek` 时写入 `pii_related`、`confidence`、`raw_model_output`；`split-only` 时 `pii_related` 与 `confidence` 为 `null`。
5. **主产物写出**（`run_prepare.py` / `batch_export_week3.py`）：`sentences.jsonl`（或批量 `all_sentences.jsonl`）、`stats.json`、`manifest.json`（或 `batch_stats.json`）；可选 `sentences_week3_2_2.csv`（或 `all_sentences_week3_2_2.csv`）。
6. **送标排序**（`labeling_queue`）：按 `pii_related` 与 `keyword_hint` **联合得分**排序，可选 `--labeling-top-n` 截断；`--export-labeling-queue` 写出 `for_labeling.jsonl`。
7. **本地评审 JSON**（`review_store`）：`--export-review-json` 写出 `review_bundle.json`（每条含 `text`、`ai`、空 `human`）；也可用 `review_store.py init|split|merge|validate|stats` 维护人工标签。
8. **可选聚类**（`sentence_cluster.py`，依赖见 `WEEK_4/requirements-optional.txt`）：对 `sentences.jsonl` 句向量 → UMAP → HDBSCAN，辅助主题与噪声难例；输出 `sentences_clustered.jsonl`、`cluster_summary.json`、可选 `umap_hdbscan.png`。
9. **可视化**（`plot_experiment.py`）：读取某次实验目录下的 `stats.json` 与可选 `sentences.jsonl`，生成 `figures/` 下 PNG 与 `summary_metrics.csv`；可叠加 WEEK_3 `audit_processed.csv`。
10. **一键批跑**（`run_full_pipeline.ps1` / `.bat`）：对 `data/*.txt` 逐个调用 `run_prepare.py`（默认 `split-only` + WEEK_3 CSV + 出图），可选 `classify`、WEEK_3 `run_audit`。

### 入口脚本一览

| 脚本 | 作用 |
|------|------|
| `run_prepare.py` | 单文档：切句、可选分类、WEEK_3 CSV、送标、评审包、统计与 manifest |
| `batch_export_week3.py` | 多文档合并：`all_sentences.jsonl`、批量 CSV、批量统计；可选送标与评审包 |
| `labeling_queue.py` | 从已有 JSONL 单独生成 `for_labeling.jsonl` |
| `review_store.py` | `review_bundle` 初始化 / 拆分为三 JSON / 合并人工 / 校验 / 与 AI 对比统计 |
| `sentence_cluster.py` | 句向量 + UMAP + HDBSCAN（可选拓展） |
| `plot_experiment.py` | 实验目录图表与汇总 CSV |

更细的目录结构、DeepSeek 分步命令与一键脚本参数见 **`WEEK_4/WORKFLOW.md`**。

## 3. 数据与目录约定

- 原始政策文本：UTF-8 `.txt`，一行或多行均可；示例见 `ref/privacy_policy/policies_huaWeiAppMarket/`。
- 本仓库提供的脚本默认从**文件路径**读取，不修改 `ref/` 下原始文件。
- 建议输出目录：`WEEK_4/out/<run_name>/`，与原始数据分离。

## 4. 环境准备

在 `WEEK_4/src` 下安装依赖（含可选 Ollama 调用所需的最小包）：

```powershell
cd "D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\WEEK_4\src"
pip install -r requirements.txt
```

句向量聚类（`sentence_cluster.py`）需额外安装：

```powershell
pip install -r "..\requirements-optional.txt"
```

若使用 Ollama 本地模型（与 WEEK_3 习惯一致），请提前确认服务与模型名称，例如：

```powershell
ollama list
```

## 5. 实验步骤

### 5.1 仅切句（不调用模型）

用于检查切分效果、词表长度分布，或作为人工标注的空白模板。

```powershell
python run_prepare.py `
  --input "..\ref\privacy_policy\policies_huaWeiAppMarket\Soul.txt" `
  --output-dir "..\out\soul_split_only" `
  --mode split-only
```

### 5.2 规则弱信号 `keyword_hint`

每条句子都会在 `sentences.jsonl` 与 `stats.json` 中附带 `keyword_hint`（是否命中与「个人信息处理」常见表述相关的词干）。它**不替代**模型或人工标签，仅用于粗统计与排序。`--keyword-hint` 参数已弃用（行为与默认一致）。

### 5.3 与 WEEK_3 `2-2` 相同的 CSV 列 + 统计文件

列名与 `WEEK_3/src/2-2/data/example.csv` 一致：`app_id,app_pkg,app_name,category_id,data_safety_content,privacy_policy_content,result`。

- 每个**句子**占一行；`privacy_policy_content` 为该句全文；`data_safety_content` 为占位用的空 Data Safety 结构字符串（与 2-2 中 Python 字典字面量风格一致）；`result` 留空供你后续填入模型或人工结论。
- `app_id` 默认按 `app_id_start + sent_index` 生成，保证同一次导出内主键不重复。
- `stats.json` 汇总：
  - **与「个人信息处理」相关的模型判断**：`pii_related.ratio_of_total`（仅当**全部句子**都已分类且无 `null` 时给出，否则为 `null`）；`pii_related.ratio_of_labeled` 为已标注子集上的比例。
  - **关键词粗占比**：`keyword_hint.ratio_of_total`（始终可算）。

单份政策示例（含 WEEK_3 格式 CSV）：

```powershell
python run_prepare.py `
  --input "..\ref\privacy_policy\policies_huaWeiAppMarket\Soul.txt" `
  --output-dir "..\out\soul_week3" `
  --mode split-only `
  --write-week3-csv `
  --app-pkg "com.soulapp" `
  --app-name "Soul" `
  --category-id 1
```

若需得到**可靠的「隐私相关句占比」**，请使用 `classify`（`mock` / `ollama`）对**全部句子**跑完，再读 `stats.json` 中的 `pii_related.ratio_of_total`。

### 5.4 批量：目录内所有 `.txt`

合并为一个 `all_sentences.jsonl`、一个 `all_sentences_week3_2_2.csv`，以及按文件拆分的 `batch_stats.json`（含汇总与 `by_file`）：

```powershell
python batch_export_week3.py `
  --input-dir "..\ref\privacy_policy\policies_huaWeiAppMarket" `
  --output-dir "..\out\all_policies" `
  --glob "*.txt" `
  --max-files 0
```

`--max-files N` 可用于先试跑前 N 个文件（按文件名排序）。大批量下建议保持 `split-only`，再对筛选子集做 `classify`。

可选与单文档一致：**送标队列**与**评审包**（与 `--labeling-top-n` 共用截断逻辑）：

```powershell
python batch_export_week3.py `
  --input-dir "..\src\data" `
  --output-dir "..\out\batch_labeled" `
  --mode classify `
  --provider mock `
  --export-labeling-queue `
  --export-review-json `
  --labeling-top-n 300
```

### 5.5 切句 + LLM 二分类（Ollama）

对每句调用本地聊天接口，要求**仅返回 JSON**（字段见下节）。大批量时请配合 `--limit` 做冒烟测试。

```powershell
python run_prepare.py `
  --input "..\ref\privacy_policy\policies_huaWeiAppMarket\Soul.txt" `
  --output-dir "..\out\soul_ollama" `
  --mode classify `
  --provider ollama `
  --ollama-model qwen3.5:9b `
  --limit 30
```

### 5.6 离线联调（mock）

不访问网络与 Ollama，用于流水线调试：

```powershell
python run_prepare.py `
  --input "..\ref\privacy_policy\policies_huaWeiAppMarket\Soul.txt" `
  --output-dir "..\out\soul_mock" `
  --mode classify `
  --provider mock
```

### 5.7 工程化扩展：送标队列、本地评审 JSON、句向量聚类

**送标队列 `for_labeling.jsonl`**（`labeling_queue.py` 或 `run_prepare.py` / `batch_export_week3.py` 加 `--export-labeling-queue`）：在 `pii_related` 与 `keyword_hint` 联合排序后导出（可选 `--labeling-top-n`），每行附带 `labeling_queue_rank`、`labeling_priority_score`。

**评审包 `review_bundle.json`**（`--export-review-json` 或 `review_store.py init`）：单文件同时保存送标句全文、**AI** 块（`pii_related` / `confidence` / `raw_model_output`）与待填 **human** 块（`pii_related`、`notes`、`reviewer_id`、`reviewed_at`）。拆分与合并：

```powershell
python review_store.py split --bundle "..\out\my_run\review_bundle.json" --out-dir "..\out\my_run\split"
python review_store.py merge --bundle "..\out\my_run\review_bundle.json" --human "..\out\my_run\split\human_evaluations.json"
python review_store.py validate --bundle "..\out\my_run\review_bundle.json"
python review_store.py stats --bundle "..\out\my_run\review_bundle.json"
```

单文档示例（与送标同序、同 Top-N）：

```powershell
python run_prepare.py `
  --input "..\ref\privacy_policy\policies_huaWeiAppMarket\Soul.txt" `
  --output-dir "..\out\soul_review" `
  --mode classify `
  --provider mock `
  --export-labeling-queue `
  --export-review-json `
  --labeling-top-n 200
```

**句向量 + UMAP + HDBSCAN**（讲义拓展，非必做）：安装 `WEEK_4/requirements-optional.txt` 后，对已有 `sentences.jsonl` 运行。

- **默认 `local`**：使用 `sentence-transformers`，首次会从 **huggingface.co** 拉取模型。若出现 `Connection to huggingface.co timed out`，多为网络无法直连 HF；可在**当前终端**设置镜像后重试（PowerShell 示例）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
pip install -r "..\requirements-optional.txt"
python sentence_cluster.py --input "..\out\soul_review\sentences.jsonl" --output-dir "..\out\soul_review\cluster"
```

- **`deepseek-chat` 与聚类的区别**：句子分类已用 `deepseek-chat`（对话接口）；**聚类需要的是「句向量」**，不能直接把 `deepseek-chat` 当嵌入模型用。若希望**不走 Hugging Face 下载**，可改用云端嵌入接口（与对话是不同 API；默认请求 `POST https://api.deepseek.com/v1/embeddings`，模型名用 `--deepseek-embedding-model`，**以你控制台/官方文档为准**；若返回 404/模型不存在，说明当前账号或产品线未开放该接口，请仍使用 `local` + 镜像）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
pip install -r "..\requirements-optional.txt"
python sentence_cluster.py `
  --embed-backend deepseek-api `
  --input "..\out\pipeline_20260410_201539\Soul\sentences.jsonl" `
  --output-dir "..\out\pipeline_20260410_201539\Soul\cluster" `
  --batch-size 16
```

默认本地多语言句向量模型可在 `--model` 覆盖；输出含 `cluster_id`、`is_cluster_noise`、`cluster_summary.json` 与 `umap_hdbscan.png`（可用 `--no-plot` 关闭图）。`cluster_summary.json` 中会记录 `embed_backend` 与所用嵌入模型名。

**强化版：`cluster_analysis.py`（22 项 + audit 三标签 + 个保法参考向量对比）**：在已安装 `requirements-optional.txt` 的前提下，对单份 `sentences.jsonl` 生成：

- 与 `ref/taxonomy/pipl_22_items.json`、`pipl_law_reference_segments.json` 对齐的 **22 维余弦最近邻**（字段 `taxonomy_22_id`、`taxonomy_22_scores` 等）；
- 可选合并 **`audit_processed.csv`**（须与 JSONL **等长且同序**）得到 `incorrect` / `incomplete` / `inconsistent` 与 `audit_priority`（`inconsistent` > `incomplete` > `incorrect`）；
- UMAP 图：`umap_audit_priority.png`、`umap_taxonomy_22.png`，以及 HDBSCAN 的 `umap_hdbscan.png`；
- `policy_vs_law_comparison.json`：按项统计「被判为该项的句子」与对应法参考段的平均相似度及交叉矩阵（用于观察**政策句嵌入**与**个保法语境参考段**是否对齐）。

```powershell
cd "D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\WEEK_4\src"
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 若 HF 超时
pip install -r "..\requirements-optional.txt"
python cluster_analysis.py `
  --sentences-jsonl "..\out\pipeline_20260406_201540\Soul\sentences.jsonl" `
  --audit-processed "..\out\pipeline_20260406_194301\Soul\audit_processed.csv" `
  --output-dir "..\out\pipeline_20260406_201540\Soul\cluster_analysis"
```

无 `audit_processed.csv` 时可省略 `--audit-processed`；法条参考文本可在 `ref/taxonomy/` 内按课程口径自行修订后再跑。

### 5.8 与 WEEK_3 `run_audit.py` 联用（Soul_week3 等句子表）

`WEEK_3/src/2-2/run_audit.py` 的 `audit` 子命令只要求 CSV 中存在列 **`data_safety_content`** 与 **`privacy_policy_content`**。`sentences_week3_2_2.csv` 列名与 `WEEK_3/src/2-2/data/example.csv` 一致，因此可直接作为 `--input-csv` 使用。

在 `WEEK_3/src/2-2` 目录下示例（路径按你本机仓库根目录调整）：

```powershell
cd "D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\WEEK_3\src\2-2"

python run_audit.py audit `
  --input-csv "..\..\..\WEEK_4\out\soul_week3\sentences_week3_2_2.csv" `
  --output-csv "results\soul_week3_audit_raw.csv" `
  --provider mock `
  --limit 10 `
  --log-file "results\soul_week3_audit.log"

python run_audit.py postprocess `
  --input-csv "results\soul_week3_audit_raw.csv" `
  --output-csv "results\soul_week3_audit_processed.csv" `
  --log-file "results\soul_week3_postprocess.log"
```

将 `mock` 换成 `ollama` / `deepseek` 等与 WEEK_3 README 一致；去掉 `--limit` 即处理全部句子（行数多、耗时与费用需自行评估）。

**语义说明**：`run_audit` 的设计是「**整段** Data Safety 声明 vs **整段** 隐私政策」做三类冲突标签。句子级 CSV 中每一行的 `privacy_policy_content` 仅为**单句**，而 `data_safety_content` 多为占位空声明；模型实际在做「空/占位 DS 与该句 PP 是否不一致」的局部判断，**不等价**于「Soul 全量政策 vs 真实 Play Data Safety」的应用级审计。若你已有该应用在商店的 **Data Safety 原始文本或 JSON**，更合理的做法是：**单独做一行**，`privacy_policy_content` 填**完整政策**（或你从句子拼回的全文），`data_safety_content` 填真实 DS。

**`evaluate` / `full` 子命令**：需要与输入行一一对应的 **`incorrect` / `incomplete` / `inconsistent` 金标准 CSV**（参见 `data/*_groundtruth.csv`）。句子级数据若无金标准，可只用 `audit` + `postprocess` 得到预测列，不必强行跑 `evaluate`。

## 6. 输出格式说明（服务标注与比较器）

默认写入 `output-dir`（批量时在输出目录根下多为 `all_sentences.jsonl`、`batch_stats.json`）：

| 文件 | 说明 |
|------|------|
| `sentences.jsonl` | 每行一个 JSON 对象，核心字段见下。 |
| `sentences_week3_2_2.csv` | 与 WEEK_3 `2-2` 数据表同列的句子级 CSV（需加 `--write-week3-csv`）。 |
| `stats.json` | 句子总数、`pii_related` 与 `keyword_hint` 计数及占比。 |
| `manifest.json` | 本次运行的参数摘要与 `stats_summary`；若启用送标/评审包则含 `labeling_queue`、`review_bundle` 路径摘要。 |
| `for_labeling.jsonl` | 可选：送标排序后的句子（`--export-labeling-queue`）。 |
| `review_bundle.json` | 可选：送标句 + AI + 待填 human（`--export-review-json`）。 |

`sentences.jsonl` 单条记录字段：

- `doc_id`：文档标识（默认取输入文件名去扩展名）。
- `sent_index`：从 0 开始的句序。
- `text`：句子正文。
- `pii_related`：`true` / `false` / `null`（`split-only` 模式下为 `null`）。
- `confidence`：0–1 浮点，mock 为占位；真实模型可在提示中要求自评或留空。
- `keyword_hint`：布尔，由规则词表给出，用于粗统计。
- `raw_model_output`：模型原始字符串（便于审计与复现）；`split-only` 时为空字符串。

人工标注推荐直接维护 **`review_bundle.json`** 中每条 `human` 字段（或与 `review_store.py split` 得到的 `human_evaluations.json`），再 `validate` / `stats` 与 AI 对比；亦可在 JSONL 上并行增加字段后自行对齐主键 `doc_id` + `sent_index`。

## 7. 评价与质量控制建议

1. **抽样复核**：对每个 `pii_related` 取值随机抽样，记录与人工一致率。
2. **边界句**：定义细则——例如纯法律援引、无具体处理行为的口号句是否算「相关」；建议在 README 旁维护一份 10～20 条的**标注指南**（课程作业可单独提交）。
3. **长句**：若单句过长，可在标注工具中再拆子句；切分函数中预留 `max_chars` 参数供实验对比。

## 8. 常见问题

- **切句过碎或过粗**：可调整 `sentence_split.py` 中的断句规则，或引入专业中文分句库（本框架为可替换模块）。
- **Ollama 超时**：增大超时、减小 batch（当前实现为逐句）、或对长文档先用 `keyword_hint` 预筛再送模型。
- **JSON 解析失败**：检查模型是否遵守提示；可将 `raw_model_output` 导出做错误分析。

## 9. 如何保证句子可与《个人信息保护法》对齐？

> 说明：日常表述中的「个信法」多指《中华人民共和国个人信息保护法》（PIPL，业内也常简称「个保法」）。**自动切句 + 自动分类并不能法律效力上「等同」法条适用**；可复现的「对齐」指：在方法论上能把某句政策表述映射到明确的法定义务类型或条款编号，并留下审计痕迹。建议做法如下。

1. **先定义「对齐」粒度**  
   区分三层：  
   - **话题级**（如：告知同意、最小必要、敏感个人信息、共享/委托、跨境、自动化决策、权利响应、未成年人、安全事件、法律责任）；  
   - **义务级**（可对应到法条中的具体义务表述）；  
   - **证据级**（该句是否构成满足某条的前置条件，如「单独同意」是否出现）。  
   课程实验通常做到话题级或义务级即可。

2. **建立法条—话题对照表（taxonomy）**  
   用官方文本固定术语（如「个人信息处理者」「敏感个人信息」「单独同意」「跨境提供」），做一张表：话题标签 → 法条范围（可写「第×条至第×条」或「第×条第×款」）。后续所有标注只允许使用表内标签，避免同义反复导致无法对照。

3. **切句尽量保持「单一可判断命题」**  
   若一句同时涵盖「收集目的」与「第三方共享」，拆句或拆标注单元，否则难以对应到单一义务。可用 `max_chars` 与人工二次切分配合。

4. **在数据里显式保存可追溯字段**  
   本流水线已提供 `doc_id`、`sent_index`；若写入 WEEK_3 格式 CSV，请同时保留 `sentences.jsonl` 或在 `result` 列写入与之一致的 id，便于回溯原文位置与政策版本（建议在文件名或单独字段记录**政策更新日期**）。

5. **人机协同：模型出候选，人定法条**  
   LLM 可输出「可能涉及的条款编号」作**候选**；最终法条对齐应由人工（或法学助教）按对照表勾选。比较器打分应使用**人审后的标签**，模型预测仅作辅助。

6. **版本与地域**  
   法条修订或监管口径变化时，应固定所用法律版本日期，并重新跑标注或至少做差异审查。

## 10. 可视化实验结果与统计

### 10.1 WEEK_4 切句 / 关键词 / PII 分类结果

依赖：`WEEK_4/src` 下安装 `matplotlib`（已写入 `requirements.txt`）。

对包含 `stats.json` 与（可选）`sentences.jsonl` 的实验目录生成柱状图、句长直方图，并导出 `summary_metrics.csv`：

```powershell
cd "D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\WEEK_4\src"
pip install -r requirements.txt

python plot_experiment.py `
  --experiment-dir "..\..\WEEK_3\src\2-2\data\soul_week3"
```

默认输出目录为 `<experiment-dir>\figures\`：

| 文件 | 含义 |
|------|------|
| `stats_counts.png` | `keyword_hint` 与 `pii_related`（含未标注 `null`）句数对比 |
| `sentence_length_hist.png` | 各句字符数分布（存在 `sentences.jsonl` 时） |
| `summary_metrics.csv` | 总句数、各类占比等汇总表，便于贴进实验报告 |

若已用 WEEK_3 `run_audit.py postprocess` 得到带 `incorrect` / `incomplete` / `inconsistent` 列的 CSV，可叠加「预测为 1」的句数柱状图：

```powershell
python plot_experiment.py `
  --experiment-dir "..\..\WEEK_3\src\2-2\data\soul_week3" `
  --audit-processed-csv "..\..\WEEK_3\src\2-2\results\soul_week3_audit_processed.csv"
```

会额外生成 `audit_positive_counts.png`。

**说明**：若从未跑过 `classify`，`pii_related` 会全部为 `null`，右图仅反映未标注状态；可先对全量句跑 `run_prepare.py --mode classify` 再绘图。

### 10.2 WEEK_3 与金标准对比时的混淆矩阵（需 groundtruth）

当你有与预测行对齐的 `*_groundtruth.csv`（格式同 `WEEK_3/src/2-2/data/all_data_150_groundtruth.csv`）时，使用 `run_audit.py evaluate` 并指定 `--figures-dir`，可得到各标签的混淆矩阵 PNG 与指标 CSV：

```powershell
cd "D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\WEEK_3\src\2-2"

python run_audit.py evaluate `
  --prediction-csv "results\your_processed.csv" `
  --groundtruth-csv "data\your_groundtruth.csv" `
  --metrics-output-csv "results\your_metrics.csv" `
  --figures-dir "results\figures" `
  --log-file "results\evaluate.log"
```

输出包括 `results\figures\cm_incorrect.png` 等及 `your_metrics.csv` 中的 precision / recall / F1。

## 11. 完整工作流一键执行

将隐私政策 **UTF-8** 文本放入 **`WEEK_4/src/data/*.txt`**，在 `WEEK_4/src` 下执行：

```powershell
cd "D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\WEEK_4\src"
.\run_full_pipeline.ps1
```

或 CMD：

```bat
cd /d D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\WEEK_4\src
run_full_pipeline.bat
```

默认：`split-only` + 导出 WEEK_3 格式 CSV + `plot_experiment` 图表；输出在 `WEEK_4/out/pipeline_<时间戳>/`，并生成 `pipeline_summary.json`。

常用参数示例：`.\run_full_pipeline.ps1 -PrepareMode classify -Provider mock`；`.\run_full_pipeline.ps1 -RunAudit -AuditLimit 30 -Provider mock`（会调用 `WEEK_3/src/2-2/run_audit.py`）。

**实验五扩展（WEEK_5 衔接）**：在流水线末尾增加 **聚类/22 项对比**（`-RunClusterAnalysis`，依赖 `requirements-optional.txt`）与 **送标导出**（`-ExportLabelingQueue` / `-ExportReviewJson`、`-LabelingTopN`）时，例如：

```powershell
.\run_full_pipeline.ps1 -PrepareMode classify -Provider deepseek -RunAudit -RunClusterAnalysis -ExportLabelingQueue -LabelingTopN 300 -OutDir "..\out\exp5_run"
```

**自包含一体化副本**：`WEEK_5` 目录内已复制 WEEK_4/WEEK_3 相关脚本，固定使用 **`WEEK_5/src/data`** 输入与 **`WEEK_5/src/output/run_*`** 输出；入口为 **`WEEK_5/src/run_pipeline.ps1`**（说明见 `WEEK_5/README.md`）。若仍使用本目录 `run_full_pipeline.ps1`，也可在生成结果后单独对 `out\pipeline_*` 调用 `WEEK_5\src\aggregate_pipeline_report.py` 生成 **`week5_aggregate_report.json`**。

使用 **DeepSeek Chat API** 时：先设置 `$env:DEEPSEEK_API_KEY`，再 `-Provider deepseek`（默认模型 `deepseek-chat`，可用 `-DeepSeekModel` 覆盖）。完整分步命令见 **`WEEK_4/WORKFLOW.md`** 中「使用 DeepSeek Chat API 的全流程」。

**流程说明、目录结构与注意事项**见同目录 **`WEEK_4/WORKFLOW.md`**。

## 12. 学术诚信与数据使用

`ref/privacy_policy/` 下文本来自公开应用商店页面抓取，仅用于课程实验；请勿将完整原文与标注结果用于商业用途或违反平台条款的场景。
