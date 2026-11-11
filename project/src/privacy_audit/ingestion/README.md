# ingestion — 证据汇聚与历史导入

`ingestion` 负责把不同来源、不同目录、不同命名习惯的证据，整理成 `engine` 可以直接消费的统一结构。

## 1. 核心入口

| 模块 | 入口 | 作用 |
| --- | --- | --- |
| `evidence_loader.py` | `build_merged_evidence(...)` | 合并静态/动态/历史/知识库证据 |
| `evidence_loader.py` | `build_and_export_evidence_bundles(...)` | 为一批 App 生成 `output/evidence/<App>.json` |
| `legacy_importer.py` | `import_legacy_pipeline(...)` | 导入 oldsrc Lab3 数据，生成派生静态/动态证据 |
| `legacy_bridge.py` | `import_claims_from_review_bundle(...)` | 旧人工标注 → 统一 claim |
| `unknown_label_refiner.py` | `refine_unknown_labels(...)` | 未知标签聚类与映射建议 |
| `output_mirror.py` | `mirror_provenance_artifacts(...)` | 把历史文件镜像到 `output/`，保证 provenance 可读 |
| `apk_registry.py` | `load_apk_list(...)` | 加载 APK 目录 |
| `policy_loader.py` | `load_policy_text(...)` | 读取政策文本 |

## 2. 证据来源

`build_merged_evidence` 输出 `sources`，每个键代表一类来源：

| 来源键 | 含义 | 新路径 | 兼容旧路径 |
| --- | --- | --- | --- |
| `static` | APK 静态 | `output/static/<slug>/static_evidence.json` | `output/stage2_static_samples/`、`output/stage2_static_apk/` |
| `dynamic` | 设备动态 | `output/dynamic/<App>/dynamic_evidence.json` | `output/stage3_dynamic_real/`、`output/stage3_dynamic_samples/` |
| `legacy` | Lab3 动态测试 PASS 步骤关键词 | `output/stage3_legacy/<App>/` | 同左 |
| `oldsrc_lab3` | `test_cases.json` + 用例结果 | `output/dynamic/legacy/<App>/` | `output/stage3_oldsrc_derived/` |
| `oldsrc_audit` | 政策审计可测项 | `output/stage2_oldsrc_audit/<App>/audit_processed.csv` | 同左 |
| `knowledge_base` | 法律矩阵 + Android 映射 | `privacy_knowledgebase/knowledge_base/` | 同左 |
| `bundle` | 已生成的扁平证据包 | `output/evidence/<App>.json` | `output/lab4_1_evidence/<App>.json` |

> 旧路径仍会被扫描；新运行默认写入新路径。路径解析在 `_static_roots()`、`_dynamic_roots()`、`resolve_evidence_bundle_path()` 中完成。

## 3. 合并后的证据结构

```json
{
  "app_name": "Keep",
  "observed_pii": ["imei", "location", "phone_number"],
  "source": "merged:static+dynamic+legacy+knowledge_base",
  "sources": {
    "static": ["imei", "location"],
    "dynamic": ["phone_number"],
    "legacy": [],
    "oldsrc_lab3": [],
    "oldsrc_audit": [],
    "knowledge_base": []
  },
  "provenance": {
    "static": "output/static/keep/static_evidence.json",
    "dynamic": "output/dynamic/Keep/dynamic_evidence.json"
  },
  "embedded_trackers": [],
  "knowledge_base": {
    "feasible_evidence": [],
    "legal_matrix": {},
    "observed_audit_types": []
  }
}
```

关键点：

- `observed_pii` 是给人看的合并结果；
- `sources` 用于解释「这个类型是谁提供的」；
- `provenance` 用于从报告回到原始文件；
- `knowledge_base.feasible_evidence` 用于向 AI/人工解释「为什么观测到该类型」。

## 4. 路径解析规则

### 4.1 静态

在 `static_dir`（默认 `output/static`）及其子目录 `apk/`、`samples/`、`legacy/` 中查找：

```text
<root>/<App>/static_evidence.json
<root>/<app.lower()>/static_evidence.json
<root>/<slug>/static_evidence.json
```

`slug` 映射见 `_slug_for_app()`（如 `Keep → keep`、`小红书 → xiaohongshu`）。

### 4.2 动态

在 `evidence_dir`（默认 `output`）下扫描：

```text
output/dynamic/<App>/dynamic_evidence.json
output/dynamic/samples/<App>/dynamic_evidence.json
output/dynamic/legacy/<App>/dynamic_evidence.json
output/stage3_dynamic_real/<App>/dynamic_evidence.json
output/stage3_dynamic_samples/<App>/dynamic_evidence.json
output/stage3_oldsrc_derived/<App>/dynamic_evidence.json
```

### 4.3 证据包

优先使用显式 `--evidence-file`；否则在 `output/evidence/` 与 `output/lab4_1_evidence/` 中查找 `<App>.json`。

## 5. 历史数据导入

`import-legacy` 做三件事：

1. 从 `oldsrc/analyzer/output/run_*/` 读取 `test_cases.json`、每 App 结果、`audit_processed.csv`；
2. 生成 `output/static/legacy/` 与 `output/dynamic/legacy/` 下的派生证据；
3. （可选）重建 `output/evidence/<App>.json`。

```powershell
python -m privacy_audit.cli import-legacy --apps-file input/apps_audit_batch.txt --rebuild-bundles
```

注意：

- `oldsrc` 只作为**历史数据源**，不作为主执行入口；
- 派生证据属于弱证据（审计/用例推断），汇报时应与真机静态/动态证据区分。

## 6. 输出镜像

`output_mirror.py` 把历史文件复制到 `output/` 下，确保报告中的 provenance 路径始终可读：

| 目标目录 | 内容 |
| --- | --- |
| `output/stage3_legacy/` | Lab3 每 App 动态 JSON |
| `output/stage2_legacy_md/` | Lab3 Markdown 摘要 |
| `output/stage2_oldsrc_audit/` | `audit_processed.csv` |
| `output/oldsrc_shared/` | 共享 `test_cases.json` |
| `output/knowledge_base/` | 知识库 JSON 镜像 |

这些目录名保留 `stage2/stage3` 是为了兼容历史 provenance；新流程不需要手动维护它们。
