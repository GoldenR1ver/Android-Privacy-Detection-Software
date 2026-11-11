# oldsrc — 历史 Lab3 实验产物（只读）

`oldsrc/` 是课程前几周的实验代码与产出，保留用于**历史证据导入**和结果追溯。它不是当前工具的主执行入口。

## 1. 定位

| | `oldsrc/` | `src/`（主路径） |
| --- | --- | --- |
| 角色 | 历史 Lab3 实验代码、LLM 审计、动态测试产物 | 证据驱动 + 规则引擎 + 法律知识库 |
| 使用方式 | 只读 `analyzer/output/`，经 `import-legacy` 导入 | `python -m privacy_audit.cli ...` |
| 不宜 | 再拿 oldsrc 代码跑主流程 | 用旧标签/旧报告直接下结论 |

## 2. 目录

```text
oldsrc/analyzer/
├── data/                 # 旧版审计数据、review_bundle
├── dynamic_testing/      # 旧动态测试：test_cases、per_app 结果、scoring
├── output/               # 历次运行产物（主要数据源）
│   ├── run_20260426_200525/
│   ├── run_20260426_203140/
│   ├── run_20260427_001723/    # Lab3 JSON、audit_processed.csv、test_cases.json
│   └── run_20260427_001723z/   # 各 App 动态测试 Markdown 摘要
├── reviews/
└── ref/                  # 旧 taxonomy
```

`run_20260427_001723/` 下每个 App 目录通常包含：

- `audit_processed.csv`：政策审计可测项；
- 旧版审计 JSON；
- `dynamic_testing/per_app/<App>/*.json`：动态测试步骤与结果。

## 3. 如何被当前工具使用

### 3.1 导入派生证据

```powershell
cd project/src
python -m privacy_audit.cli import-legacy --apps-file input/apps_audit_batch.txt --rebuild-bundles
```

会生成：

- `output/static/legacy/<slug>/static_evidence.json`（从审计可测项派生）；
- `output/dynamic/legacy/<App>/dynamic_evidence.json`（从 test_cases/结果派生）；
- `output/evidence/<App>.json`（可选重建）。

### 3.2 知情权合并

`audit-disclosure` / `build-evidence` 的 `--legacy-run-dir`、`--legacy-md-run-dir` 会读取：

- `dynamic_testing/per_app/<App>/*.json`：PASS 步骤关键词；
- `run_*z/<App>/*.md`：动态测试 Markdown 摘要；
- `audit_processed.csv`：政策审计可测项。

这些证据会被标记为 `legacy` / `oldsrc_lab3` / `oldsrc_audit`，属于**弱证据**，汇报时应与真机静态/动态证据分开。

### 3.3 旧人工标注

```powershell
python -m privacy_audit.cli import-legacy-claims --review-bundle-file oldsrc/analyzer/data/main/review_bundle.json --output-file output/stage1/claims_from_review_bundle.json
```

## 4. 注意事项

- `oldsrc` 目录可能包含旧依赖、缓存和未完成实验，不要在当前主流程中 import 旧模块；
- 旧报告里的结论不能直接当作当前结论，只能作为证据线索；
- 如果旧数据路径变化，用 `--legacy-run-dir` / `--legacy-md-run-dir` 显式指定；
- 旧目录中的 `__pycache__`、临时文件不属于数据源。
