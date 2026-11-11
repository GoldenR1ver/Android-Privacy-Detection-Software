# privacy_audit — 主审计包

`privacy_audit` 是隐私检测工具的核心 Python 包。它把「政策声明 → 取证 → 证据汇聚 → 规则判定 → 报告」串成可复现的命令行流程。

## 1. 分层

```text
CLI (cli.py)
  │
  ├─ workflows/       编排：一次审计/一次取证/一批任务
  │    ├─ disclosure_audit.py     知情权
  │    ├─ consent_audit.py        同意权
  │    ├─ access_audit.py         访问权
  │    ├─ static_scan.py          静态取证
  │    ├─ dynamic_capture.py      动态取证
  │    └─ batch_runner.py         批量调度
  │
  ├─ analyzers/       产生原始证据
  │    ├─ policy/     政策声明解析
  │    ├─ static/     APK 权限/API/SDK
  │    ├─ dynamic/    Appium UI + Frida 运行时
  │    └─ network/    mitmproxy 流量
  │
  ├─ ingestion/       把多源证据合并成统一结构
  │    ├─ evidence_loader.py
  │    ├─ legacy_importer.py
  │    └─ ...
  │
  ├─ engine/          声明 vs 观测的规则判定
  ├─ reporting/       JSON / Excel 输出
  ├─ config/          PII 分类、规则、标签映射
  ├─ domain/          EvidenceItem / RightClaim / VerificationResult
  └─ utils/           adb、日志、重试、UI 层级、屏幕抓取
```

## 2. 命令与模块对应

| CLI 命令 | 工作流入口 | 主要输出 |
| --- | --- | --- |
| `audit-disclosure` | `workflows.disclosure_audit.run_disclosure_audit` | `<App>_disclosure_report.json` |
| `audit-consent` | `workflows.consent_audit.run_consent_audit` | `<App>_consent_result.json` |
| `audit-access` | `workflows.access_audit.run_access_audit` | `<App>_access_result.json` |
| `scan-static` | `workflows.static_scan.run_static_scan(_directory)` | `static_evidence.json` |
| `capture-dynamic` | `workflows.dynamic_capture.run_dynamic_capture` | `dynamic_evidence.json` |
| `batch` | `workflows.batch_runner.run_batch_job` | `batch_jobs/<job_id>/` |
| `build-evidence` | `ingestion.evidence_loader.build_and_export_evidence_bundles` | `evidence/<App>.json` |
| `import-legacy` | `ingestion.legacy_importer.import_legacy_pipeline` | `static/legacy/`、`dynamic/legacy/` |

## 3. 知情权数据流

```text
input/pii_text/<App>.txt
        ↓ policy_parser
declared_pii
        ↓
evidence_loader.build_merged_evidence
  static | dynamic | legacy | legacy_cases | legacy_audit | knowledge_base
        ↓
observed_pii + provenance + feasible_evidence
        ↓ engine.compare_engine
verdicts
        ↓ reporting.disclosure_report
output/disclosure/<App>_disclosure_report.json
```

## 4. 报告关键字段

| 字段 | 含义 |
| --- | --- |
| `audit` | `disclosure` / `consent` / `access` |
| `claims.declared_pii` | 政策声明的 PII 类型 |
| `evidence.observed_pii` | 合并后的观测 PII 类型 |
| `evidence.sources` | 每个来源贡献的 PII 类型 |
| `evidence.provenance` | 证据文件的可追溯路径 |
| `verdicts[]` | 每个 PII 类型的 `status`、`risk_level`、`message` |
| `summary.status_counts` | 三种 verdict 的计数 |
| `sdk_disclosure` | 政策声明 SDK vs APK 内嵌 SDK |
| `feasible_evidence` | 知识库给出的可引用法律证据 |

## 5. 扩展点

- **新增 PII 类型**：改 `config/personal_info.yaml`（别名、权限、API、hook、网络键）。
- **新增判定规则**：改 `rules/decision_rules.yaml`；规则键为 `declared and observed` 等。
- **新增法律条目/映射**：改 `privacy_knowledgebase/input.md`，再运行 `kb-build`。
- **新增取证源**：在 `ingestion/evidence_loader.py` 的 `sources` 中增加来源，并在 `build_merged_evidence` 中合并。
- **新增报告格式**：在 `reporting/` 增加 writer，并在工作流末尾调用。

## 6. 子模块文档

- [analyzers](analyzers/README.md)
- [ingestion](ingestion/README.md)
- [engine](engine/README.md)
- [workflows](workflows/README.md)
- [reporting](reporting/README.md)
- [config](config/README.md)
- [utils](utils/README.md)
