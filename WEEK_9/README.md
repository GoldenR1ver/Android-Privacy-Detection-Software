# WEEK_9：APP 隐私政策中“声明将获取”的个人信息列表

本周目标：基于前面实验（优先复用 `WEEK_7/src/analyzer/output/run_*` 结果），自动汇总各 APP 隐私政策中声明会收集/处理的个人信息项。

## 目录结构

```
WEEK_9/
  README.md
  src/
    extract_declared_personal_info.py
  output/
    declared_personal_info_report.md
    declared_personal_info_items.csv
    declared_personal_info_items.json
    app_declared_personal_info.csv
```

## 处理逻辑

`extract_declared_personal_info.py` 会：

1. 自动找到 `WEEK_7/src/analyzer/output` 下最新的 `run_*`（也可手动指定）。
2. 优先读取每个 APP 的 `cluster_analysis/sentences_taxonomy_22.jsonl`，若不存在则回退到 `sentences.jsonl`。
3. 仅保留 `pii_related == true` 的句子。
4. 从两类来源抽取“信息项”：
   - 结构化字段 `target_data`（如果存在）
   - 句子文本中的关键词模式（手机号、身份证、位置、设备标识等）
5. 进行同义词归一化，输出全局清单与按 APP 清单。

## 运行方式

在仓库根目录执行：

```powershell
python WEEK_9/src/extract_declared_personal_info.py
```

手动指定输入 run 目录：

```powershell
python WEEK_9/src/extract_declared_personal_info.py `
  --input-run-dir "D:/dev/Android-Privacy-Detection-Software/WEEK_7/src/analyzer/output/run_20260427_001723"
```

## 结果文件说明

- `output/declared_personal_info_report.md`：面向汇报的 Markdown 摘要。
- `output/declared_personal_info_items.csv`：全局信息项表（覆盖 APP 数、命中句数、示例句）。
- `output/declared_personal_info_items.json`：同上 JSON 结构。
- `output/app_declared_personal_info.csv`：按 APP 展开的信息项。

## 注意事项

- 结果是“规则+既有实验字段”的抽取汇总，适合课程实验统计与对比，不等价于法律意见。
- 若需更高准确率，建议在 `review_bundle.json` 的人工复核后再二次汇总。
