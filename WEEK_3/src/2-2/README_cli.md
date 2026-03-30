# CLI实验说明（无需 `conda activate`）

本目录新增 `run_audit.py`，用于替代 Notebook，支持三种模型后端：

- `local`：本地 HuggingFace 模型（默认 `microsoft/Phi-3-mini-4k-instruct`）
- `deepseek`：API 调用 `deepseek-reasoner`
- `mock`：本地测试专用（无需模型与API）

---

## 1) 不使用 `conda activate` 的运行方式

你可以直接使用：

```powershell
conda run -n <你的环境名> python run_audit.py ...
```

例如：

```powershell
conda run -n android_privacy_lab python run_audit.py --help
```

---

## 2) 全流程命令（audit + postprocess + evaluate）

```powershell
conda run -n android_privacy_lab python run_audit.py full `
  --provider mock `
  --input-csv data/all_data_150.csv `
  --groundtruth-csv data/all_data_150_groundtruth.csv `
  --raw-output-csv results/all_data_150_results_cli.csv `
  --processed-output-csv results/all_data_150_results_processed_cli.csv `
  --metrics-output-csv results/metrics_summary_all_data_150_cli.csv `
  --figures-dir results/figures_cli_all_data_150
```

---

## 3) 本地模型模式（`local`）

```powershell
conda run -n android_privacy_lab python run_audit.py full `
  --provider local `
  --local-model-id microsoft/Phi-3-mini-4k-instruct `
  --max-new-tokens 256 `
  --input-csv data/all_data_150.csv `
  --groundtruth-csv data/all_data_150_groundtruth.csv `
  --raw-output-csv results/all_data_150_results_local.csv `
  --processed-output-csv results/all_data_150_results_processed_local.csv `
  --metrics-output-csv results/metrics_summary_all_data_150_local.csv `
  --figures-dir results/figures_local_all_data_150
```

### 使用其他本地模型（例如 Qwen）

只需要替换 `--local-model-id`。例如：

```powershell
conda run -n android_privacy_lab python run_audit.py full `
  --provider local `
  --local-model-id Qwen/Qwen2.5-7B-Instruct `
  --max-new-tokens 256 `
  --input-csv data/example.csv `
  --groundtruth-csv data/example_groudtruth.csv `
  --raw-output-csv results/example_results_qwen.csv `
  --processed-output-csv results/example_results_processed_qwen.csv `
  --metrics-output-csv results/metrics_summary_example_qwen.csv `
  --figures-dir results/figures_qwen_example
```

建议流程：

1. 先用 `example.csv` 小样本验证模型可加载、输出为 JSON。
2. 再跑 `all_data_150.csv` 全量。
3. 如果显存不足，优先换更小模型，或降低 `--max-new-tokens`。

---

## 4) DeepSeek API 模式（`deepseek-reasoner`）

先设置 API Key（当前 PowerShell 窗口）：

```powershell
$env:DEEPSEEK_API_KEY="你的key"
```

然后运行：

```powershell
conda run -n android_privacy_lab python run_audit.py full `
  --provider deepseek `
  --deepseek-model deepseek-reasoner `
  --input-csv data/all_data_150.csv `
  --groundtruth-csv data/all_data_150_groundtruth.csv `
  --raw-output-csv results/all_data_150_results_deepseek.csv `
  --processed-output-csv results/all_data_150_results_processed_deepseek.csv `
  --metrics-output-csv results/metrics_summary_all_data_150_deepseek.csv `
  --figures-dir results/figures_deepseek_all_data_150
```

如需自定义接口地址（OpenAI兼容网关），可追加：

```powershell
--base-url https://api.deepseek.com/chat/completions
```

---

## 5) 分步命令

仅审计：

```powershell
conda run -n android_privacy_lab python run_audit.py audit --provider mock --input-csv data/example.csv --output-csv results/example_results_cli.csv
```

仅后处理：

```powershell
conda run -n android_privacy_lab python run_audit.py postprocess --input-csv results/example_results_cli.csv --output-csv results/example_results_processed_cli.csv
```

仅评估：

```powershell
conda run -n android_privacy_lab python run_audit.py evaluate --prediction-csv results/example_results_processed_cli.csv --groundtruth-csv data/example_groudtruth.csv --metrics-output-csv results/metrics_summary_example_cli.csv --figures-dir results/figures_cli
```

---

## 6) 自动化测试

```powershell
conda run -n android_privacy_lab python -m unittest discover -s tests -p "test_*.py"
```

