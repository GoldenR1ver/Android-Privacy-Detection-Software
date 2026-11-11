# reporting — 报告导出

`reporting` 把工作流产生的 dict 写成可交付的 JSON/Excel 文件。它不改变判定结果，只做结构整形与字段裁剪。

## 1. 模块

| 模块 | 入口 | 输出 |
| --- | --- | --- |
| `report_exporter.py` | `export_json_report(data, path)` | 任意 JSON 报告 |
| `disclosure_report.py` | `finalize_disclosure_report(result)` | 知情权报告整形 |
| `excel_writer.py` | `write_excel(...)` / `write_excel_bundle(...)` | PMA 风格 Excel |
| `html_renderer.py` | `render_html(...)` | 简单 HTML 渲染 |

## 2. 知情权报告整形

`disclosure_report.finalize_disclosure_report` 做两件事：

1. 把知识库可行证据提升为独立章节 `feasible_evidence`；
2. 裁剪 `evidence.knowledge_base`，避免同一份大列表在报告中重复出现。

报告顶层结构：

```json
{
  "app_name": "Keep",
  "audit": "disclosure",
  "claims": {"declared_pii": []},
  "evidence": {
    "observed_pii": [],
    "sources": {},
    "provenance": {},
    "source": "merged:static+dynamic+...",
    "knowledge_base": {"feasible_count": 26}
  },
  "verdicts": [],
  "summary": {"status_counts": {}},
  "verdict_basis": {},
  "sdk_disclosure": {},
  "feasible_evidence": {
    "feasible_count": 26,
    "high_confidence_count": 18,
    "legal_matrix": {},
    "items": []
  }
}
```

## 3. Excel 报告

`export-excel` 生成 PMA 风格三列证据：

| 列 | 来源 |
| --- | --- |
| `app_name` | App 名 |
| `pii_type` | PII 类型 |
| `api_name` | Frida 事件或静态 API |
| `stack_summary` | Frida 调用栈摘要 |
| `declared_in_policy` | 是否在政策中声明 |
| `evidence_level` | 证据强度（如 static/dynamic/probe） |

```powershell
python -m privacy_audit.cli export-excel --app-name Keep --output-dir output/excel
```

输出：`output/excel/Keep_pma_evidence.xlsx`

## 4. 阅读顺序建议

**批量汇报**：

1. `batch_<mode>_<job_id>_summary.json` → `summary.success_count`、`reports[].summary`
2. `batch_jobs/<job_id>/manifest.json` → 失败 App 与错误
3. 单 App 报告 `<App>_<audit>_report.json` → `verdicts[]`
4. 证据包 `output/evidence/<App>.json` → `provenance`、`sources`、`feasible_evidence`

**单 App 深挖**：

1. `summary.status_counts`
2. `verdicts[]` 的 `status` / `risk_level` / `message`
3. `evidence.sources` 区分技术证据与知识库推断
4. `feasible_evidence.items[]` 的 `legal_item_ids` / `confidence`

## 5. 扩展新格式

1. 在 `reporting/` 增加 writer（HTML、PDF 等）；
2. 保持输入为工作流返回的 dict，不重新计算 verdict；
3. 在 CLI 增加导出命令，或在 workflow 末尾调用；
4. 用 `output/evidence/<App>.json` 作为稳定输入，避免依赖中间状态。
