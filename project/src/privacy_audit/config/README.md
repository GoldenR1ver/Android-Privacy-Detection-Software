# config — 配置与 PII 分类

`config` 保存「什么算个人信息」以及「如何判定」的声明式配置。改这里通常不需要改 Python 代码。

## 1. 文件

| 文件 | 作用 |
| --- | --- |
| `personal_info.yaml` | PII 类型、别名、政策关键词、权限、API、Frida hook、网络键 |
| `../rules/decision_rules.yaml` | 知情权判定矩阵（主文件） |
| `decision_rules.yaml` | 旧镜像，已弃用；请改 `../rules/decision_rules.yaml` |
| `legacy_label_mapping.yaml` | 旧版人工标签 → 新 PII 类型的映射 |
| `settings.py` | 运行路径/默认值 |
| `trace_config.py` | `privacy_trace_methods.txt` 与 `personal_info.yaml` 的双向同步 |

## 2. `personal_info.yaml` 结构

每个顶层键是一个 `pii_type`：

```yaml
location:
  aliases: ["location", "gps", "地理位置", "位置信息"]
  policy_keywords: ["地理位置", "位置信息", "定位"]
  permissions: ["ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION"]
  apis: ["getLastKnownLocation", "requestLocationUpdates"]
  hooks: ["hook_location.js"]
  api_targets:
    - {clazz: "android.location.LocationManager", method: "getLastKnownLocation"}
    - {clazz: "android.location.LocationManager", method: "requestLocationUpdates"}
  network_keys: ["lat", "lng", "latitude", "longitude", "location"]
  related_rights: ["informed", "consent", "access", "deletion"]
  sensitivity: ["sensitive"]
```

| 字段 | 用途 |
| --- | --- |
| `aliases` | 政策解析、副本字段匹配、报告展示的别名 |
| `policy_keywords` | 政策文本中用于识别该类信息的关键词 |
| `permissions` | 静态扫描 Android 权限 |
| `apis` | 静态扫描 API 方法名/签名 |
| `hooks` | 动态取证加载的 Frida 脚本 |
| `api_targets` | `hook_papi.js` 的数据驱动目标（类名 + 方法） |
| `network_keys` | mitm 抓包时在 URL/body 中匹配的字段键 |
| `related_rights` | 该 PII 涉及哪些权利 |
| `sensitivity` | `normal` / `sensitive` |

## 3. 新增一种 PII 类型

1. 在 `personal_info.yaml` 增加条目；
2. 如有对应 hook，在 `analyzers/dynamic/hooks/` 增加 JS，并在 `hooks:` 中引用；
3. 如需静态扫描，补 `permissions` / `apis`；
4. 如需网络匹配，补 `network_keys`；
5. 运行单 App 验证：

```powershell
python -m privacy_audit.cli scan-static --apk-file input/apk/keep.apk --output-dir output/static
python -m privacy_audit.cli capture-dynamic --app-name Keep --app-package com.gotokeep.keep --output-dir output/dynamic
python -m privacy_audit.cli build-evidence --apps Keep --output-dir output/evidence
python -m privacy_audit.cli audit-disclosure --app-name Keep --policy-file input/pii_text/Keep.txt --evidence-file output/evidence/Keep.json --output-dir output/disclosure
```

## 4. 修改判定规则

编辑 `privacy_audit/rules/decision_rules.yaml`：

```yaml
rules:
  - when: "declared and observed"
    verdict: "consistent"
    risk_level: "low"
    message: "Policy declared and runtime/static evidence observed."
    evidence_fields: ["claims.declared_pii", "evidence.observed_pii"]
```

`when` 目前支持：

- `declared and observed`
- `not_declared and observed`
- `declared and not_observed`

修改后用 `compare-verdicts` 对比前后两批报告，避免误改导致批量结论漂移。

## 5. 旧标签映射

`legacy_label_mapping.yaml` 用于 `import-legacy-claims`：

- 键：旧标签（人工标注/模型标签）；
- 值：对应的 PII 类型列表。

未知标签可用 `refine-label-mapping` 聚类并生成建议，再人工确认后写回映射。

## 6. `privacy_trace_methods.txt`

`input/privacy_trace_methods.txt` 是 PMA 风格的 API 清单。`trace_config.py` 可以：

- 从 YAML 生成 TXT；
- 从 TXT 合并回 YAML；
- 检查两者一致性。

```powershell
python scripts/sync_trace_config.py --check
```
