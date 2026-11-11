# Knowledge Base Output

本目录由 `../build_kb.py` 从 `../input.md` 生成，**不要手工修改**。

| 文件 | 内容 |
| --- | --- |
| `categories.json` | 法律分类（A–H）与条目、别名索引 |
| `aliases.json` | 别名 → 条目 ID |
| `sensitivity.json` | 条目 ID → 敏感级别 |
| `android_mapping.json` | permission / API / SDK / network 信号 → 法律条目 |
| `risk_rules.json` | 条目 ID → 风险桶 |
| `audit_taxonomy_bridge.json` | 法律条目 ↔ `privacy_audit` 的 `pii_type` |
| `README.md` | 本文件 |

## 重建

```powershell
cd project/src
python -m privacy_audit.cli kb-build
```

或直接运行生成脚本：

```powershell
python -c "from privacy_knowledgebase.build_kb import generate_outputs; from pathlib import Path; generate_outputs(Path('privacy_knowledgebase/input.md'), Path('privacy_knowledgebase/knowledge_base'))"
```

重建后 `output/knowledge_base/` 会同步镜像，报告的 `feasible_evidence` 会引用新条目。
