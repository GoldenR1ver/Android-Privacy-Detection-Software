# engine — 判定引擎

`engine` 负责把「政策声明」与「观测证据」比较成可解释的 verdict。它不产生证据，只做规则判断与结果整形。

## 1. 模块

| 模块 | 入口 | 作用 |
| --- | --- | --- |
| `compare_engine.py` | `compare_claims_with_evidence(...)` | 知情权核心：声明 vs 观测 |
| `kb_verdict.py` | `finalize_observed_layers(...)`、`apply_kb_verdict_adjustments(...)` | 把知识库推断并入观测层，并调整 verdict |
| `tracker_disclosure.py` | `compare_embedded_trackers_with_policy(...)` | 政策声明的第三方 SDK vs APK 内嵌 SDK |
| `rule_loader.py` | `load_decision_rules(...)` | 读取 `rules/decision_rules.yaml` |
| `normalizer.py` | `normalize_evidence(...)` | 旧结构兼容/归一化 |
| `scorer.py` | `score_verification(...)` | 汇总评分辅助 |

## 2. 判定矩阵

`rules/decision_rules.yaml` 定义三种情况：

| 条件键 | verdict | 风险 | 含义 |
| --- | --- | --- | --- |
| `declared and observed` | `consistent` | low | 政策声明且证据观测到 |
| `not_declared and observed` | `inconsistent_high_risk` | high | 证据观测到但政策未声明 |
| `declared and not_observed` | `declared_not_observed` | medium | 政策声明但当前证据未观测到 |

> `declared_not_observed` **不等于**「没有采集」。它只表示当前静态/动态/历史证据没有覆盖到，需要补证或标注观测边界。

默认规则内置在 `compare_engine._default_rule_map()`；如果 `rules/decision_rules.yaml` 存在且可读，则以文件为准。

## 3. observed 的三层

知识库开启时，证据包会区分三层：

| 字段 | 含义 |
| --- | --- |
| `observed_pii_technical` | 静态/动态/历史直接观测到的类型 |
| `observed_pii_kb` | 仅由知识库信号推断出的类型 |
| `observed_pii_for_verdict` | 最终参与判定的类型（默认是 technical；KB 高置信命中可并入） |

对应逻辑在 `kb_verdict.finalize_observed_layers()`。

## 4. 知识库对 verdict 的调整

`apply_kb_verdict_adjustments` 可能产生：

- `kb_supported_consistent`：政策声明 + KB 证据支持，但技术证据不足；
- `kb_supported`：KB 支持该类型，但技术层未直接观测；
- 其他带 `kb_influenced: true` 的行。

报告中通过 `verdict_basis` 解释：

```json
{
  "use_knowledge_base": true,
  "observed_for_verdict": ["imei", "location"],
  "observed_technical": ["imei"],
  "observed_kb_only": ["location"],
  "kb_supported_types": ["location"],
  "kb_declared_support_types": [],
  "kb_influenced_verdict_count": 1
}
```

## 5. SDK 披露比对

`tracker_disclosure.py` 独立于 PII 类型比对，回答：

- APK 内嵌了哪些第三方 SDK/tracker？
- 隐私政策是否提到该公司/该 SDK/第三方共享章节？

输出 `sdk_disclosure`：

```json
{
  "has_third_party_sdk_section": true,
  "embedded_count": 7,
  "declared_in_policy_count": 3,
  "undeclared_count": 4,
  "declared_trackers": ["WeChat SDK"],
  "undeclared_trackers": ["Google Ads"],
  "risk_level": "medium",
  "trackers": [
    {
      "name": "Google Ads",
      "categories": ["ads"],
      "matched_signatures": ["com.google.android.gms.ads"],
      "declared_in_policy": false,
      "evidence_level": "S1_static_embedded"
    }
  ]
}
```

这些结果会以 `third_party_sdk` 类型的 verdict 合并进知情权报告。

## 6. 如何新增/修改规则

1. 编辑 `privacy_audit/rules/decision_rules.yaml`（主文件）；
2. 保持 `when` 键与 `compare_engine` 中的判断条件一致；
3. 运行单 App 审计验证：

```powershell
python -m privacy_audit.cli audit-disclosure --app-name Keep --policy-file input/pii_text/Keep.txt --evidence-file output/evidence/Keep.json --rules-file privacy_audit/rules/decision_rules.yaml --output-dir output/disclosure
```

4. 若规则影响批量结果，建议用 `compare-verdicts` 对比修改前后的报告目录。

## 7. 阅读顺序

1. 先看 `summary.status_counts`：三类结论各多少；
2. 再看 `verdicts[]`：每个 PII 类型的 `status`、`message`、`evidence_fields`；
3. 最后看 `feasible_evidence.items[]`：知识库给出的法律解释与置信度。

相关文档：[ingestion](../ingestion/README.md)、[config](../config/README.md)、[knowledgebase](../../privacy_knowledgebase/README.md)。
