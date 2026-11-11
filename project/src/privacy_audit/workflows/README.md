# workflows — 工作流与批量编排

`workflows` 是**编排层**：把取证、汇聚、判定、报告串成一次可执行的审计任务。每个 CLI 命令最终都落到这里的一个入口函数。

## 1. 模块总览

| 模块 | 入口 | 对应命令 | 作用 |
| --- | --- | --- | --- |
| `disclosure_audit.py` | `run_disclosure_audit` | `audit-disclosure` | 知情权：政策声明 vs 观测证据 |
| `consent_audit.py` | `run_consent_audit` | `audit-consent` | 同意权：开关前后 Frida/mitm 对比 |
| `access_audit.py` | `run_access_audit` | `audit-access` | 访问权：声明/运行时/副本覆盖 |
| `static_scan.py` | `run_static_scan`、`run_static_scan_directory` | `scan-static` | APK 静态取证 |
| `dynamic_capture.py` | `run_dynamic_capture` | `capture-dynamic` | Appium + Frida + mitm 动态取证 |
| `capture_track.py` | `resolve_capture_tracks` | —（内部） | 根据 tier/probe 选择动态轨道 |
| `batch_runner.py` | `run_batch_job` | `batch` | 重试、续跑、日志、manifest |
| `batch_report_paths.py` | `resolve_batch_report_path` | `batch-status` / `compare-verdicts` | 定位批量报告（兼容旧文件名） |
| `excel_export.py` | `export_excel_evidence` | `export-excel` | PMA 风格 Excel |
| `verdict_compare.py` | `compare_batch_verdicts` | `compare-verdicts` | 两次知情权结果差异 |
| `tier_sync.py` | `sync_frida_tier_from_probe_dir` | `scripts/sync_frida_tier_from_probe.py` | 把探针结果写回 `frida_tier.json` |

## 2. 知情权工作流

```text
政策文本
  ↓ policy_parser.parse_declared_pii
declared_pii
  ↓ evidence_loader.build_merged_evidence
evidence（observed_pii / sources / provenance / knowledge_base）
  ↓ engine.compare_claims_with_evidence
verdicts
  ↓ engine.tracker_disclosure（可选）
sdk_disclosure
  ↓ reporting.disclosure_report
output/disclosure/<App>_disclosure_report.json
```

关键参数：

- `--evidence-file`：直接使用已构建的证据包；
- `--evidence-dir`：现场从静态/动态/历史目录合并；
- `--static-evidence-dir`：静态证据根目录；
- `--rules-file`：判定规则；
- `--taxonomy-file`：PII 分类。

## 3. 同意权工作流

```text
before_off.json ─┐
                 ├─ _extract_runtime_hits（Frida 事件）
after_off.json ──┘
                 ├─ _extract_network_counts（mitm tracker/PII keys）
                 ↓
           comparisons[] + track3_network
                 ↓
output/consent/<App>_consent_result.json
```

判定：

| verdict | 含义 |
| --- | --- |
| `effective_off` | 关闭后该 PII 类型不再出现 |
| `still_collecting_after_off` | 关闭后仍出现 |
| `risk_increased_after_off` | 关闭后反而新增 |
| `insufficient_observation` | 前后都没有可用观测 |

网络轨另有 `still_transmitting_after_off`、`tracker_stopped_after_off` 等。

## 4. 访问权工作流

```text
政策文本 ──→ declared_pii
动态证据 ──→ runtime_observed_pii（Frida + mitm）
副本 JSON ─→ copy_keys（扁平化字段名）
                 ↓
copy_covered_pii / missing_from_copy / manual_copy_audit
                 ↓
output/access/<App>_access_result.json
```

重要字段：

- `coverage_rate`：运行时观测类型中，副本字段可匹配的比例；
- `coverage_rate_applicable`：运行时观测为空时为 `false`，不能解读为「副本未覆盖」；
- `manual_copy_audit[]`：字段级人工审核建议。

## 5. 批量编排

`batch_runner.py` 的 `run_batch_job` 提供：

- 每 App 任务状态：`pending` / `running` / `success` / `failed` / `skipped`；
- 失败重试（`--retry-count`、指数退避）；
- 断点续跑（`--resume`，跳过已成功 App）；
- 强制重跑（`--force-rerun`）；
- 每 App 日志与批量日志；
- 报告归档与 manifest。

### 5.1 目录结构

```text
output/<audit>/
├── batch_jobs/<job_id>/
│   ├── manifest.json
│   ├── summary.json
│   ├── logs/
│   │   ├── batch.log
│   │   └── <App>.log
│   └── reports/<App>/<App>_<audit>_report.json
├── batch_<mode>_<job_id>_summary.json
└── <App>_<audit>_report.json
```

### 5.2 manifest 字段

| 字段 | 含义 |
| --- | --- |
| `job_id` / `mode` / `apps` | 任务标识与范围 |
| `tasks.<App>.status` | 任务状态 |
| `tasks.<App>.attempts` | 已尝试次数 |
| `tasks.<App>.last_error` | 最近一次错误 |
| `tasks.<App>.report_path` | 报告路径 |
| `updated_at_utc` | 更新时间 |

### 5.3 三种模式的输入

| 模式 | 输入参数 |
| --- | --- |
| `disclosure` | `--policy-dir`、`--evidence-dir`、`--static-evidence-dir`、`--legacy-run-dir` |
| `consent` | `--before-dir`、`--after-dir` |
| `access` | `--policy-dir`、`--dynamic-dir`、`--copy-dir` |

## 6. 报告路径解析

`batch_report_paths.resolve_batch_report_path` 先找新文件名：

```text
<App>_disclosure_report.json
```

再回退到旧实验名：

```text
<App>_lab4_1_report.json
```

因此 `compare-verdicts` 可以比较新旧两批报告。

## 7. 动态轨道选择

`capture_track.resolve_capture_tracks` 读取：

- `input/frida_tier.json`：每 App 的 `tier`、`java_bridge`、`preferred_dynamic_tracks`；
- `output/frida_probe/<slug>.json`：探针结果。

返回：

```json
{
  "dynamic_track_requested": "auto",
  "dynamic_track_resolved": "mitm",
  "enable_frida": false,
  "enable_mitm": true,
  "mitm_primary": true
}
```

规则：`auto` / `consent` 默认 mitm；`frida` 显式启用 Frida；`ui` 只做 UI 验证。

## 8. 扩展新工作流

1. 在 `workflows/` 新建模块，提供 `run_xxx(...) -> dict`；
2. 在 `cli.py` 注册子命令和 handler；
3. 复用 `evidence_loader`、`engine`、`reporting`，不要在 workflow 里重复实现规则；
4. 补一个 `tests/unit` 或 `tests/integration` 用例。
