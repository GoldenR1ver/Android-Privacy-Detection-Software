# Android Privacy Audit 使用说明（含 Ollama + Qwen3.5:9b）

本目录提供命令行脚本 `run_audit.py`，用于比较 Android Data Safety 与 Privacy Policy 的一致性，输出三类标签：

- `incorrect`
- `incomplete`
- `inconsistent`

当前支持的模型后端：

- `mock`：本地快速测试，无需模型
- `local`：本地 HuggingFace 模型
- `deepseek`：DeepSeek API
- `ollama`：本机 Ollama 服务（已支持，适合你当前的 Qwen3.5:9b）

---

## 1. 环境准备

建议在 `WEEK_3/src/2-2` 目录下运行命令。

```powershell
cd "D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\WEEK_3\src\2-2"
```

如需安装依赖：

```powershell
pip install -r requirements.txt
```

---

## 2. Ollama 模型检查（Qwen3.5:9b）

先确认 Ollama 服务和模型可用：

```powershell
ollama list
ollama run qwen3.5:9b "你好，请只输出 JSON: {\"ok\":1}"
```

如果你本地模型名称不是 `qwen3.5:9b`，后续将 `--ollama-model` 替换为实际名称即可。

---

## 3. 快速冒烟测试（推荐先跑 5 条）

```powershell
python run_audit.py audit `
  --provider ollama `
  --ollama-model qwen3.5:9b `
  --input-csv data/example.csv `
  --output-csv results/example_results_ollama_raw.csv `
  --limit 5 `
  --log-file results/run_audit_ollama.log
```

说明：

- `--provider ollama`：启用 Ollama 后端
- `--ollama-base-url` 默认是 `http://127.0.0.1:11434/api/chat`，通常无需修改
- `--limit 5` 便于先验证流程是否跑通

---

## 4. 分步完整流程

### 4.1 audit：模型推理，生成原始结果

```powershell
python run_audit.py audit `
  --provider ollama `
  --ollama-model qwen3.5:9b `
  --input-csv data/all_data_150.csv `
  --output-csv results/all_data_150_results_ollama_raw.csv `
  --log-file results/run_audit_ollama_all.log
```

### 4.2 postprocess：把 `result` JSON 拆成三列标签

```powershell
python run_audit.py postprocess `
  --input-csv results/all_data_150_results_ollama_raw.csv `
  --output-csv results/all_data_150_results_ollama_processed.csv
```

### 4.3 evaluate：和标注集对比并输出指标

```powershell
python run_audit.py evaluate `
  --prediction-csv results/all_data_150_results_ollama_processed.csv `
  --groundtruth-csv data/all_data_150_groundtruth.csv `
  --metrics-output-csv results/metrics_summary_all_data_150_ollama.csv `
  --figures-dir results/figures_ollama_all_data_150
```

---

## 5. 一条命令跑完整流程（full）

```powershell
python run_audit.py full `
  --provider ollama `
  --ollama-model qwen3.5:9b `
  --input-csv data/all_data_150.csv `
  --groundtruth-csv data/all_data_150_groundtruth.csv `
  --raw-output-csv results/all_data_150_results_ollama_raw.csv `
  --processed-output-csv results/all_data_150_results_ollama_processed.csv `
  --metrics-output-csv results/metrics_summary_all_data_150_ollama.csv `
  --figures-dir results/figures_ollama_all_data_150
```

---

## 6. 断点续跑（长任务推荐）

`audit` 和 `full` 支持续跑：

- `--resume`：自动从已有输出 CSV 的下一行继续
- `--resume-from N`：手动指定从输入 CSV 的第 N 行开始（1-based）

示例：

```powershell
python run_audit.py audit `
  --provider ollama `
  --ollama-model qwen3.5:9b `
  --input-csv data/all_data_150.csv `
  --output-csv results/all_data_150_results_ollama_raw.csv `
  --resume
```

---

## 7. 常见问题

1) **报错连接不到 Ollama**
- 确认 Ollama 服务已启动
- 默认地址为 `http://127.0.0.1:11434`
- 可手动指定：

```powershell
--ollama-base-url http://127.0.0.1:11434/api/chat
```

2) **输出不是严格 JSON**
- 脚本已内置 JSON 提取与兜底逻辑
- 建议先用 `--limit 5` 验证输出稳定性，再跑全量
3) **速度较慢**
- 先小样本测试
- 降低并发外部负载
- 必要时可换更小模型做对照实验

---

## 8. 参数帮助

查看总帮助：

```powershell
python run_audit.py --help
```

查看 `audit` 子命令帮助：

```powershell
python run_audit.py audit -h
```
