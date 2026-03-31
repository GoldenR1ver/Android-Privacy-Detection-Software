# run_audit.py 工程结构详细介绍（结合具体代码）

本文用于配合 `PPT/slide.tex` 讲解 `WEEK_3/src/2-2/run_audit.py` 的代码实现。  
重点回答五个问题：

1. Provider 层如何统一 `infer()` 接口
2. Audit 层如何逐行读取 CSV 并写 `result`
3. Postprocess 层如何解析模型文本并产出三标签
4. Evaluate 层如何计算 P/R/F1/Accuracy 并画混淆矩阵
5. 如何通过 `--resume / --resume-from` 实现断点续跑

---

## 1) 总体架构：命令分发 + 四阶段流水线

脚本入口是 `main()`，通过 `build_parser()` 注册四个子命令：

- `audit`：只做推理，输出原始 `result`
- `postprocess`：将 `result` 解析为三标签列
- `evaluate`：对比真值并计算指标
- `full`：顺序执行 `audit -> postprocess -> evaluate`

这意味着系统既可端到端跑，也可分阶段调试，适合实验场景和工程排错场景。

---

## 2) Prompt 设计：模板固定 + 动态填充

代码中 prompt 由两部分组成：

- `SYSTEM_PROMPT`：角色约束（隐私一致性分析专家）
- `USER_PROMPT_TEMPLATE`：任务说明、三标签定义、JSON 输出约束、输入占位符

输入填充方式：

- `user_prompt = USER_PROMPT_TEMPLATE.format(data_safety=..., privacy_policy=...)`

其中最关键的是输出约束：

- 明确要求 `Return JSON only`
- 固定字段：`incorrect / incomplete / inconsistent`

这样做的工程意义是：减少后处理复杂度，提高评估阶段稳定性。

---

## 3) Provider 层：统一 `infer()` 契约

### 3.1 抽象层

- `class BaseProvider` 定义统一接口：`infer(data_safety, privacy_policy) -> Dict[str, int]`

只要实现这个接口，上层 `run_audit()` 完全不用关心后端细节。

### 3.2 三个实现类

1. `MockProvider`
   - 规则基线，不调用模型/API
   - 用关键词和空字段规则快速给出三标签
   - 适合本地联调、冒烟测试、无网环境

2. `DeepSeekProvider`
   - 使用 HTTP 请求调用 chat completions
   - `messages` 中传 `system + user`
   - `temperature=0` 增强复现性
   - 对 API key 做空值和非法字符校验

3. `LocalHFProvider`
   - 使用 `transformers` 本地模型
   - 优先 `apply_chat_template`，失败回退字符串拼接
   - 不依赖外部 API，适合离线实验

### 3.3 工厂方法

- `make_provider(args)` 根据 `--provider` 创建实例
- 对应关系：`mock | local | deepseek`

---

## 4) Audit 层：逐行推理并写入 `result`

核心函数：`run_audit(input_csv, output_csv, provider, limit, logger, log_every, resume, resume_from)`

### 4.1 主流程

1. 用 `csv.DictReader` 逐行读取输入 CSV
2. 取两列文本：
   - `data_safety_content`
   - `privacy_policy_content`
3. 调用 `provider.infer(data_safety, privacy_policy)`
4. 将推理结果写入 `row["result"]`（JSON 字符串）
5. 通过 `csv.DictWriter` 写回输出 CSV

### 4.2 关键代码机制

- 若输出字段不存在 `result`，自动追加字段名
- `log_every` 控制日志粒度，便于长任务观测
- `limit` 支持只跑前 N 条样本，适合快速验证

这层的定位是“样本调度器 + 推理执行器”，不做复杂清洗，保证职责单一。

---

## 5) Postprocess 层：把模型文本变成标准三标签

核心函数：`postprocess_results(input_csv, output_csv, logger)`

### 5.1 解析函数 `extract_json_dict(text)`

为了处理模型不稳定输出，代码按以下顺序容错：

1. 去掉可能的 ```json 代码块包裹
2. 尝试 `json.loads` 直接解析
3. 用正则提取 `{...}` 片段再解析
4. 再失败时尝试 `ast.literal_eval`
5. 仍失败则回退默认值：
   - `{"incorrect": 0, "incomplete": 0, "inconsistent": 0}`

### 5.2 标准化函数 `normalize_prediction(d)`

- 强制只保留三个字段
- 所有值转成 `int`

### 5.3 输出结构

- 删除原始 `result` 字段
- 写入三列：`incorrect / incomplete / inconsistent`

这层保证了 Evaluate 阶段输入“始终合法、结构固定”。

---

## 6) Evaluate 层：逐标签二分类评估

核心函数：`evaluate_results(prediction_csv, groundtruth_csv, metrics_output_csv, logger, figures_dir=None)`

### 6.1 评估方式

对每个标签分别做二分类计算：

- Precision
- Recall
- F1
- Accuracy
- 混淆矩阵（TN/FP/FN/TP）

标签集合固定为：

- `labels = ["incorrect", "incomplete", "inconsistent"]`

### 6.2 输出结果

- 指标 CSV：每个标签一行，包含 P/R/F1/Acc 和 TN/FP/FN/TP
- 图像（可选）：每个标签一张混淆矩阵 PNG

这一步给出的不是“单一分数”，而是可解释的诊断信息。

---

## 7) 可恢复运行：`--resume / --resume-from`

续跑逻辑写在 `run_audit()` 中，核心变量是 `start_row`。

### 7.1 两种模式

1. `--resume-from N`
   - 直接指定从第 N 行开始（1-based）
   - 用于精准重跑某一段数据

2. `--resume`
   - 自动读取 `output_csv` 已完成行数
   - `start_row = completed + 1`

### 7.2 关键实现点

- `count_data_rows(output_csv)`：统计输出文件已完成样本
- 写入模式自动切换：
  - 新跑：`w`
  - 续跑：`a`
- 日志记录起止位置，便于排查中断点

### 7.3 工程价值

- API 超时/网络波动后可恢复
- 大数据集可分段执行
- 降低重跑成本，提升实验可复现性

---

## 8) 如何在汇报中讲这五层（可直接念）

> 我们把 `run_audit.py` 拆成五个模块：Provider、Audit、Postprocess、Evaluate 和可恢复运行。  
> Provider 通过统一 `infer()` 接口把不同模型后端封装起来；Audit 层逐行读取 CSV 并把推理结果写入 `result`；Postprocess 层把模型文本容错解析成标准三标签；Evaluate 层按标签输出 P/R/F1/Accuracy 和混淆矩阵。  
> 另外 `--resume/--resume-from` 支持断点续跑，保证大任务中断后可继续执行，满足工程稳定性和可复现要求。

---

## 9) 本地测试集“通过”判定（建议口径）

建议定义三条“通过”标准：

1. **流程通过**：`full` 命令完整执行，原始结果、后处理结果、指标 CSV、图像文件全部生成
2. **数据通过**：预测结果成功标准化为三标签，且可与 groundtruth 对齐
3. **评估通过**：指标值与混淆矩阵可生成、可解释，能够定位误检/漏检

按这个口径汇报，可以体现项目是“工程闭环”，而不仅是一次性实验脚本。

