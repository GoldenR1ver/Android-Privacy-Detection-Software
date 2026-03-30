# Android隐私分析程序实验手册

## 1. 实验定位与目标

本实验面向 Android 平台的数据合规审计，核心任务是利用大语言模型（LLM）自动比对：

- Google Play 的 `Data Safety` 声明信息
- App 的 `Privacy Policy` 隐私政策文本

并输出三类风险标签：

- `incorrect`：Data Safety 未声明，但隐私政策提到（错误/遗漏声明）
- `incomplete`：Data Safety 有声明，但细节不完整
- `inconsistent`：两者都提到，但描述冲突

你之前的 `slide` 分析了社交类 App 隐私政策合规重点（基于《个保法》22项标准），本实验可以把这些重点作为“结果解释框架”，将模型输出转化为可落地的合规结论。

---

## 2. 目录与文件说明（基于当前工程）

本手册默认你在 `WEEK 3` 目录执行实验。

- `reference/system_design_expr_lab2_1_handout.pdf`：LLM基础与本地调用准备
- `reference/system_design_expr_lab2_2_handout.pdf`：隐私审计实验流程
- `reference/Week2PPT/slide.pdf`、`slide.tex`：你们既有隐私政策分析报告
- `src/2-2/policy_audit.ipynb`：主实验代码（Notebook）
- `src/2-2/requirements.txt`：依赖包
- `src/2-2/data/example.csv`：小样本输入
- `src/2-2/data/example_groudtruth.csv`：小样本标签（注意拼写是 groudtruth）
- `src/2-2/data/all_data_150.csv`：完整数据集
- `src/2-2/data/all_data_150_groundtruth.csv`：完整标签集
- `src/2-2/results/`：输出目录

---

## 3. 实验总流程（先看全局）

1. 环境准备（Conda + requirements）
2. 运行 `policy_audit.ipynb` 的 Step 1（定义 LLM handler）
3. Step 2（批处理 CSV，生成原始结果）
4. Step 3（解析 JSON 字段，生成可评估结果）
5. Step 4（计算混淆矩阵 + Precision/Recall/F1/Accuracy）
6. Step 5（可视化）
7. 将结果映射到你们 `slide` 的22项合规维度，写实验结论

---

## 4. 环境准备（Windows + PowerShell）

### 4.1 创建并激活 Conda 环境

在 `WEEK 3/src/2-2` 下执行：

```powershell
conda create -n android_privacy_lab python=3.10 -y
conda activate android_privacy_lab
```

### 4.2 安装依赖

```powershell
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- transformers
- numpy
- jupyter
- bitsandbytes
- torch
- pandas
- scikit-learn
- matplotlib

### 4.3 启动 Notebook

```powershell
jupyter notebook
```

打开：`src/2-2/policy_audit.ipynb`

### 4.4 代理与网络（按需）

Notebook 中已示例：

```python
os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"] = "http://127.0.0.1:10809"
```

如你本机无需代理，请注释掉这两行。

---

## 5. 核心实验步骤（逐步执行）

## Step 0：确认实验输入格式

`example.csv` / `all_data_150.csv` 应至少包含：

- 样本标识字段（如 id）
- `Data Safety` 文本字段
- `Privacy Policy` 文本字段
- `result` 字段（用于写回模型输出）

当前 Notebook 在循环时按列下标取 `row[4]` 和 `row[5]`，说明输入 CSV 列顺序与现有代码绑定。  
若你改过 CSV 列顺序，务必同步修改 `loop_csv` 内取值逻辑。

---

## Step 1：定义 LLM Handler（Notebook 已给）

主要包括 3 个函数：

1. `remove_empty_lines(content)`：清洗模型输出
2. `ask_gpt(data_safety_content, privacy_policy_content)`：构造 prompt 并调用模型
3. `loop_csv(input_csv_path, output_csv_path)`：逐行读取样本，写入预测结果

模型默认：

- `microsoft/Phi-3-mini-4k-instruct`
- 使用 `transformers.pipeline("text-generation")`
- 设定 `max_new_tokens=500`, `do_sample=False`

### Step 1执行建议

- 先运行 1~2 行样本（`example.csv`）验证流程
- 确认输出是纯 JSON 字符串，格式如：

```json
{"incorrect": 1, "incomplete": 1, "inconsistent": 0}
```

---

## Step 2：跑审计任务（先小样本，再全量）

### 2.1 小样本验证

Notebook 示例：

```python
input_csv_path = "data/example.csv"
output_csv_path = "results/example_results.csv"
HANDLER().loop_csv(input_csv_path, output_csv_path)
```

### 2.2 全量数据（150样本）

将路径改为：

```python
input_csv_path = "data/all_data_150.csv"
output_csv_path = "results/all_data_150_results.csv"
HANDLER().loop_csv(input_csv_path, output_csv_path)
```

### 2.3 运行监控要点

- 每条样本应打印 `Run times <id>`
- 失败通常来自：
  - 模型输出非 JSON
  - 输入文本超长或格式异常
  - 网络/模型下载问题

---

## Step 3：结果解析与结构化保存

目标：把 `result` 列里的 JSON 字符串拆成三列标签。

你可以复用 Notebook 的逻辑，并将文件名替换为全量版本：

```python
df = pd.read_csv("results/all_data_150_results.csv")
...
df.to_csv("results/all_data_150_results_processed.csv", index=False)
```

### Step 3检查清单

- 处理后文件必须包含：
  - `incorrect`
  - `incomplete`
  - `inconsistent`
- 取值必须是 0/1
- 样本行数与输入一致

---

## Step 4：计算评估指标

与标注文件对齐后评估：

- 预测：`results/all_data_150_results_processed.csv`
- 真值：`data/all_data_150_groundtruth.csv`

按字段分别评估：

- 混淆矩阵（confusion matrix）
- Precision
- Recall
- F1
- Accuracy

### 建议补充

Notebook 当前在“单一标签分布”时可能出现 sklearn warning（如某类全是 0）。  
可在 `precision_score/recall_score/f1_score` 增加 `zero_division=0`，保证流程稳定。

---

## Step 5：可视化

复用 Notebook 的 `plot_confusion_matrix`，分别绘制三类标签的混淆矩阵图。  
建议将图保存到 `results/figures/` 便于写报告。

可加一段保存代码：

```python
plt.savefig("results/figures/cm_incorrect.png", dpi=200, bbox_inches="tight")
```

---

## 6. 将实验结果接入你们 slide 的分析逻辑

你们 `slide` 的主结论可作为“解释层”，把模型输出变成合规判断。

## 6.1 映射关系建议

- `incorrect = 1`  
  对应“声明缺失风险”，可优先关联：
  - 第2项：处理目的/方式/类型/保存期限
  - 第9项：敏感信息告知
  - 第13项：自动化决策
  - 第20项：同意机制

- `incomplete = 1`  
  对应“声明粒度不足”，可关联：
  - 保存期限未细化
  - 接收方信息不完整
  - 用户权利行使路径不完整

- `inconsistent = 1`  
  对应“文本冲突风险”，可重点检查：
  - 跨境传输（第10项）
  - 第三方共享/单独同意（第23条相关）
  - 平台承诺与实际功能描述冲突

## 6.2 与你们已有发现对齐

你们已识别的行业共性短板（保存期限、敏感信息、自动化决策、同意机制），可在实验报告中写成：

- LLM审计是否能稳定检出这些高风险项
- 哪一类风险最容易误判（例如 inconsistent 常见困难）
- 人工复核是否必要（结论：必要，尤其是复杂法条语义）

---

## 7. 实验报告写作模板（可直接套用）

## 7.1 实验目的

- 评估 LLM 在 Android 隐私声明一致性审计中的有效性
- 比较模型输出与人工标签一致性
- 分析其在合规场景中的可用性与局限

## 7.2 实验设置

- 模型：Phi-3-mini-4k-instruct
- 数据集：example + all_data_150
- 标签：incorrect/incomplete/inconsistent
- 指标：Precision/Recall/F1/Accuracy

## 7.3 实验结果

- 用表格列出三类标签四项指标
- 附混淆矩阵图
- 给出典型误判案例（至少 3 个）

## 7.4 结果分析（结合22项标准）

- 模型在哪些法条维度表现最好
- 哪些维度需要人工二次审计
- 与你们 Week2 隐私政策研究结论是否一致

## 7.5 局限与改进

- Prompt 对输出稳定性影响大
- 模型可能 hallucination
- 建议加入 few-shot 示例、结构化 schema 校验、多模型投票

---

## 8. 常见问题与排错

## 8.1 模型输出不是 JSON

- 在 prompt 最后强制要求“only output JSON”
- 增加后处理：正则抽取 `{...}` 再 `json.loads`

## 8.2 CSV 解析报错

- 检查编码统一为 UTF-8
- 检查是否存在换行破坏字段
- 检查 `result` 列是否存在空值

## 8.3 指标全 0 或 warning

- 样本可能标签极不均衡
- 先看混淆矩阵，再解释指标
- 使用 `zero_division=0`

## 8.4 跑全量太慢

- 先抽样调试 prompt
- 降低 `max_new_tokens`
- 分批处理 CSV 并合并结果

---

## 9. 一次完整实验的最小执行清单

1. `conda create` + `pip install -r requirements.txt`
2. 跑 `example.csv` 并产出 `example_results_processed.csv`
3. 跑 `all_data_150.csv` 并产出 `all_data_150_results_processed.csv`
4. 与 `all_data_150_groundtruth.csv` 对齐评估
5. 导出 3 张混淆矩阵图 + 指标表
6. 按你们 22 项合规框架撰写结论

---

## 10. 建议的最终产出目录

- `results/all_data_150_results.csv`
- `results/all_data_150_results_processed.csv`
- `results/figures/cm_incorrect.png`
- `results/figures/cm_incomplete.png`
- `results/figures/cm_inconsistent.png`
- `results/metrics_summary.csv`
- `report/lab2_android_privacy_audit_report.md`（或 pdf）

---

## 11. 结论建议（可直接放报告摘要）

本实验验证了：LLM 能较高效地筛查 Android `Data Safety` 与 `Privacy Policy` 的不一致问题，适合作为“合规初筛器”。  
但在复杂法律语义（尤其跨境传输、单独同意、自动化决策等）上仍需人工复核。  
结合你们既有的22项《个保法》分析框架，可形成“模型初筛 + 人工审计”的工程化流程，用于社交类 App 的隐私合规治理。

