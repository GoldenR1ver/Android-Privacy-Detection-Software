# privacy_knowledgebase — 法律知识库与可行证据

把 Android 技术信号（权限、API、Frida 事件、SDK、网络字段）映射到**法律个人信息矩阵**（PIPL / CCPA / GDPR 对照），再桥接到 `privacy_audit/config/personal_info.yaml` 的审计类型，为报告提供可引用的 `feasible_evidence`。

## 1. 它解决什么问题

`compare_engine` 能回答「声明了没有、观测到了没有」，但很难回答：

- 这个技术信号对应哪条法律定义的个人信息？
- 为什么把 `getImei` 归到「设备信息」而不是「身份标识」？
- 报告里的高风险结论，能否引用具体法律条目？

知识库把这些问题变成结构化字段：

```json
{
  "signal_type": "api",
  "signal_value": "imei",
  "legal_item_ids": ["A_04"],
  "audit_pii_types": ["imei", "android_id"],
  "legal_labels": ["在线标识符"],
  "confidence": 0.88,
  "feasible_for_verdict": true,
  "rationale": "Androguard API hit bucket.",
  "source": "static.androguard.api_hits"
}
```

## 2. 目录

```text
privacy_knowledgebase/
├── input.md                      # 法律个人信息矩阵（源文件）
├── build_kb.py                   # 生成 knowledge_base/*.json
├── kb_store.py                   # 加载与查询
├── signal_inferencer.py          # 从 static/dynamic/文本推断信号
├── enricher.py                   # 写入 evidence.knowledge_base / feasible_evidence
├── legal_pdf_ingest.py           # 可选：从 reference 法律 PDF 提取附录
└── knowledge_base/
    ├── categories.json           # A–H 类条目 + 别名索引
    ├── aliases.json
    ├── sensitivity.json
    ├── android_mapping.json      # permission/api/sdk/network 规则
    ├── risk_rules.json
    └── audit_taxonomy_bridge.json # 法律条目 <-> audit pii_type
```

## 3. 构建

修改 `input.md` 后执行：

```powershell
cd project/src
python -m privacy_audit.cli kb-build
```

可选：从法律参考 PDF 生成附录（需要本地 PDF 与解析依赖）：

```powershell
python -m privacy_audit.cli kb-build --ingest-pdfs --reference-dir ../reference/法律参考文件
```

构建会同时把 JSON 镜像到 `output/knowledge_base/`，便于报告引用。

## 4. 与证据管线集成

`ingestion/evidence_loader.build_merged_evidence()` 在合并 static/dynamic/legacy 后默认调用 `enricher.enrich_merged_evidence()`：

| 输出字段 | 含义 |
| --- | --- |
| `sources.knowledge_base` | 高置信 KB 推断的 audit PII 类型 |
| `knowledge_base.feasible_evidence[]` | 每条可行证据：信号、法律条目、置信度、法理标签、来源 |
| `knowledge_base.legal_matrix` | 涉及条目的中文 label、敏感级别、风险桶 |
| `knowledge_base.observed_audit_types` | 合并后的 audit 类型 |
| `knowledge_base.legal_item_ids` | 涉及的法律条目 ID |

关闭知识库：

```powershell
python -m privacy_audit.cli build-evidence --apps Keep --no-knowledge-base
```

调整并入 `observed_pii` 的置信度阈值：

```powershell
python -m privacy_audit.cli build-evidence --apps Keep --kb-min-confidence 0.8
```

## 5. 法律矩阵维护原则

- 主表是 `input.md`；**不要**在矩阵里擅自添加 PIPL 未列的敏感项。
- 分类原则见 `reference/法律参考文件/README.md`。
- `categories.json` 的 A–H 类与 `input.md` 的章节对应；新增条目要同步更新：
  - `categories.json`：条目与别名；
  - `android_mapping.json`：permission/api/sdk/network 规则；
  - `audit_taxonomy_bridge.json`：法律条目 → audit `pii_type`。
- 改完后运行 `kb-build`，再跑单 App 审计检查 `feasible_evidence` 是否合理。

## 6. 阅读 feasible_evidence

汇报时建议这样用：

1. **verdict 仍由 `compare_engine` 产生**：`declared_pii` vs `observed_pii`；
2. `feasible_evidence` 用来解释「为什么观测到该类型、对应哪条法律」；
3. 注意 `confidence` 与 `feasible_for_verdict`：
   - 高置信且 `feasible_for_verdict=true` 才适合并入判定；
   - 低置信条目只作为线索，不应单独支撑高风险结论。

## 7. 相关命令

```powershell
# 重建知识库
python -m privacy_audit.cli kb-build

# 导入历史数据并重建证据包（默认含 KB）
python -m privacy_audit.cli import-legacy --apps-file input/apps_audit_batch.txt --rebuild-bundles

# 查看单 App 的可行证据
# 打开 output/evidence/<App>.json 的 knowledge_base.feasible_evidence
```
