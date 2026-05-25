# Privacy Knowledge Base

为 Lab4 AI 审计提供 **可引用的可行证据（feasible evidence）**：将 Android 技术信号（权限、API、Frida、SDK）映射到 **法律个人信息矩阵**（`input.md` / PIPL·CCPA·GDPR 对照表），再桥接到 `privacy_audit` 的 `personal_info.yaml` 审计类型。

## 法律矩阵来源

- 主表：`input.md`（禁止在矩阵中擅自添加 PIPL 未列敏感项）
- 参考说明：`reference/法律参考文件/README.md`
- 生成命令：`python -m privacy_audit.cli kb-build`

## 目录结构

```text
privacy_knowledgebase/
├── input.md                      # 法律个人信息矩阵（源）
├── process_input.py              # 生成 knowledge_base/*.json
├── kb_store.py                   # 加载与查询
├── signal_inferencer.py          # 从 static/dynamic/文本推断信号
├── enricher.py                   # 写入 evidence.knowledge_base / feasible_evidence
└── knowledge_base/
    ├── categories.json           # A–H 类条目 + 别名索引
    ├── aliases.json
    ├── sensitivity.json
    ├── android_mapping.json      # permission/api/sdk/network 规则
    ├── risk_rules.json
    └── audit_taxonomy_bridge.json  # 法律条目 <-> audit pii_type
```

## 与证据管线集成

`privacy_audit.ingestion.evidence_loader.build_merged_evidence()` 在合并 static/dynamic/legacy 后默认调用 KB 增强，输出字段：

| 字段 | 含义 |
|------|------|
| `sources.knowledge_base` | 高置信度 KB 推断的 audit pii_type |
| `knowledge_base.feasible_evidence[]` | 每条可行证据（信号、法律条目、置信度、法理标签） |
| `knowledge_base.legal_matrix` | 涉及法律条目的 label / 敏感级别 / 风险桶 |

## 推荐命令

```powershell
cd project/src

# 1. 由 input.md 重建知识库 JSON
python -m privacy_audit.cli kb-build

# 2. oldsrc 导入 + 重建证据包（已含 KB）
python -m privacy_audit.cli import-oldsrc-evidence --apps-file input/apps_lab4_batch.txt --rebuild-bundles

# 3. Lab4-1 batch
python -m privacy_audit.cli batch --apps-file input/apps_lab4_batch.txt --mode lab4-1 ...
```

禁用 KB：`--no-knowledge-base`  
调整置信度阈值：`--kb-min-confidence 0.8`

## 阅读 feasible_evidence

每条记录示例：

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

汇报时：**verdict** 仍由 `compare_engine` 基于 `declared_pii` vs `observed_pii` 产生；**feasible_evidence** 用于向 AI/人工解释「为何观测到该类型」。
