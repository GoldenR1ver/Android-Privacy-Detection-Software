# Android 隐私检测系统（Lab4）

在 Android 应用上自动比对**隐私政策声明**与**多源技术证据**（静态 APK 扫描、动态运行时、历史 Lab3 数据、法律知识库推断），输出 Lab4 三项权利（知情权 / 同意权 / 访问权）的结构化审计报告，并为 AI 审计提供**可引用的可行证据（feasible evidence）**。

- **主代码目录**：`project/src/`（包名 `privacy_audit`、`privacy_knowledgebase`）
- **历史实验数据**：`project/oldsrc/analyzer/output/`（通过导入命令复用，不跑 oldsrc 主流程）
- **设计文档**：`project/reference/`

---

## 功能概览

| 模块 | 说明 |
|------|------|
| **Lab4-1** | 知情权：声明 PII vs 观测证据 → `consistent` / `inconsistent_high_risk` / `declared_not_observed` |
| **Lab4-2** | 同意权：开关关闭前后 Frida 事件对比 |
| **Lab4-3** | 访问权：声明 / 运行时 / 副本字段覆盖比对 |
| **Stage2** | Exodus + Androguard 静态分析 → `static_evidence.json` |
| **Stage3** | Appium + Frida 动态取证 → `dynamic_evidence.json` |
| **Knowledge Base** | 法律个人信息矩阵 + Android 信号映射 → `feasible_evidence[]` |
| **Batch** | 多 App 批处理、重试、断点续跑、manifest 与日志归档 |

---

## 项目结构

```text
project/
├── README.md                          # 本文件
├── reference/
│   ├── 开发计划.md
│   ├── plan.md
│   └── 法律参考文件/                  # PIPL/CCPA/GDPR 参考材料说明（矩阵源见 input.md）
├── oldsrc/
│   └── analyzer/output/
│       ├── run_20260427_001723/       # Lab3 JSON、audit_processed.csv、test_cases.json
│       └── run_20260427_001723z/      # 各 App 动态测试 Markdown 摘要
├── utils/                             # Frida、Androguard、Exodus 等
├── tools/frida/                       # frida-server（与 PC 端 Frida 版本一致）
└── src/                               # ★ 所有 CLI 命令在此目录执行
    ├── privacy_audit/                 # 主审计包
    │   ├── cli.py
    │   ├── config/                    # personal_info.yaml、legacy 映射
    │   ├── ingestion/
    │   │   ├── evidence_loader.py     # 多源证据合并 + KB 增强
    │   │   └── oldsrc_evidence_importer.py
    │   ├── analyzers/                 # 政策 / 静态 / 动态
    │   ├── engine/compare_engine.py
    │   ├── workflows/                   # lab4_*、stage2/3、batch
    │   └── rules/decision_rules.yaml
    ├── privacy_knowledgebase/           # 法律知识库 + 可行证据
    │   ├── input.md                   # 法律个人信息矩阵（PIPL/CCPA/GDPR 对照）
    │   ├── process_input.py           # 生成 knowledge_base/*.json
    │   ├── kb_store.py / enricher.py / signal_inferencer.py
    │   └── knowledge_base/            # categories、android_mapping、audit_taxonomy_bridge
    ├── input/
    │   ├── pii_text/                  # <App名>.txt 隐私政策
    │   ├── apps_lab4_batch.txt        # 当前批量 9 个 App（已排除 Soul）
    │   └── apk/                      # stage2 用 APK（需自行准备）
    └── output/
        ├── stage2_static_samples/     # 真 APK 扫描（Keep / 支付宝 / 小红书）
        ├── stage2_oldsrc_derived/     # 自 oldsrc 衍生的静态证据
        ├── stage3_dynamic_real/       # 真机动态（Keep）
        ├── stage3_dynamic_samples/
        ├── stage3_oldsrc_derived/
        ├── lab4_1_evidence/           # 扁平合并证据包 <App>.json
        └── lab4_1/                    # 报告 + batch_jobs/
```

### 数据流（Lab4-1）

```text
input/pii_text/<App>.txt
        ↓ policy_parser
    declared_pii
        ↓
┌───────────────────────────────────────────────────────────┐
│ evidence_loader.build_merged_evidence                     │
│   static │ dynamic │ legacy │ oldsrc_lab3 │ oldsrc_audit │
│   + privacy_knowledgebase.enricher (默认开启)              │
└───────────────────────────────────────────────────────────┘
        ↓
    observed_pii + feasible_evidence + provenance
        ↓ compare_engine
    verdicts → output/lab4_1/<App>_lab4_1_report.json
```

---

## 环境要求

- **Python** 3.10+（建议虚拟环境）
- 依赖：androguard、PyYAML 等（按项目 `requirements.txt` 或本地环境安装）
- **Stage3 真机/模拟器（可选）**
  - Android 模拟器（如 MuMu）+ `adb connect`
  - Appium Server（默认 `http://127.0.0.1:4723`）
  - 设备内 `frida-server`（架构与版本需匹配，见 `tools/frida/`）

---

## 快速开始

```powershell
cd D:\LEARNING_RESOURCE\AndroidPDS\gitRepository\Android-Privacy-Detection-Software\project\src

python -m privacy_audit.cli --help
```

### 运行日志（排查 stage3 卡住）

CLI 默认**实时输出**进度到终端；`stage3-dynamic` 另写入 `output/.../<App>/stage3_run.log`。

```powershell
# 推荐：加 -v，并确认带 --real-run
python -m privacy_audit.cli -v stage3-dynamic `
  --app-name Keep `
  --app-package com.gotokeep.keep `
  --real-run `
  ...
```

启动前请单独开终端运行 **Appium Server**（`appium` 或 Appium Desktop）。若未启动，旧版会在 `webdriver.Remote()` 处阻塞约 1–2 分钟且无输出；现已增加 **Appium /status 预检**，数秒内会打印 `appium: FAIL` 并跳过会话创建。

真机 `attach` 前请在 MuMu **打开并登录 Keep**，保持前台。

**Frida 与 Appium 同步（`--real-run`）**：先 attach 并加载 hook → **默认 5s 手动窗口**（`--frida-grace-sec`）→ Appium 自动导航（此期间 Frida 持续采集）→ 结束后 **3s 尾采**（`--frida-tail-sec`）。报告里看 `summary.frida_runtime_events`（非 `hook_loaded` 计数）。

---

## 推荐完整流程（Lab4-1 批量）

当前批量列表：`input/apps_lab4_batch.txt`（9 个 App：Keep、小红书、支付宝、LOFTER、智联招聘、世纪佳缘、好大夫医生版、东方财富、小荷健康）。

### 步骤 0：重建法律知识库（修改 `input.md` 后执行）

法律矩阵源文件：`privacy_knowledgebase/input.md`（须遵守 `reference/法律参考文件/` 中的分类原则，勿擅自添加 PIPL 未列敏感项）。

```powershell
python -m privacy_audit.cli kb-build
```

### 步骤 1：从 oldsrc 导入并重建证据包

写入 `stage2_oldsrc_derived`、`stage3_oldsrc_derived`，并生成 `output/lab4_1_evidence/<App>.json`（**默认含 KB 可行证据**）。

```powershell
python -m privacy_audit.cli import-oldsrc-evidence `
  --apps-file input/apps_lab4_batch.txt `
  --rebuild-bundles
```

可选参数：

- `--no-knowledge-base`：关闭法律知识库增强
- `--kb-min-confidence 0.72`：仅将置信度 ≥ 阈值的 KB 信号并入 `observed_pii`

### 步骤 2：批量 Lab4-1

```powershell
python -m privacy_audit.cli batch `
  --apps-file input/apps_lab4_batch.txt `
  --mode lab4-1 `
  --policy-dir input/pii_text `
  --evidence-dir output/lab4_1_evidence `
  --legacy-run-dir ..\oldsrc\analyzer\output\run_20260427_001723 `
  --legacy-md-run-dir ..\oldsrc\analyzer\output\run_20260427_001723z `
  --output-dir output/lab4_1 `
  --job-id keep_batch_004 `
  --retry-count 2
```

### 步骤 3：查看 batch 状态

```powershell
python -m privacy_audit.cli batch-status `
  --output-dir output/lab4_1 `
  --job-id keep_batch_004
```

### 一键命令（步骤 0 → 1 → 2）

```powershell
python -m privacy_audit.cli kb-build; python -m privacy_audit.cli import-oldsrc-evidence --apps-file input/apps_lab4_batch.txt --rebuild-bundles; python -m privacy_audit.cli batch --apps-file input/apps_lab4_batch.txt --mode lab4-1 --policy-dir input/pii_text --evidence-dir output/lab4_1_evidence --legacy-run-dir ..\oldsrc\analyzer\output\run_20260427_001723 --legacy-md-run-dir ..\oldsrc\analyzer\output\run_20260427_001723z --output-dir output/lab4_1 --job-id keep_batch_004 --retry-count 2
```

---

## CLI 命令参考

| 命令 | 用途 |
|------|------|
| `kb-build` | 由 `input.md` 生成 `knowledge_base/*.json` |
| `import-oldsrc-evidence` | oldsrc → derived 目录；`--rebuild-bundles` 生成 `lab4_1_evidence` |
| `build-evidence-bundles` | 仅合并证据（不跑 oldsrc 导入） |
| `lab4-1` | 单 App 知情权审计 |
| `lab4-2` / `lab4-3` | 同意权 / 访问权 |
| `batch` | 批量 lab4-1 / lab4-2 / lab4-3 |
| `batch-status` | 查看 manifest、任务成功/失败 |
| `stage2-static` | APK 静态扫描 |
| `stage3-dynamic` | Appium + Frida（`--real-run`） |
| `bridge-stage1` | legacy `review_bundle` 桥接 |
| `refine-stage1-mapping` | 未知标签聚类与映射建议 |

### 单 App 示例（Keep）

```powershell
python -m privacy_audit.cli lab4-1 `
  --app-name Keep `
  --policy-file input/pii_text/Keep.txt `
  --evidence-file output/lab4_1_evidence/Keep.json `
  --output-dir output/lab4_1
```

### Stage2 / Stage3

```powershell
# 静态（APK 放入 input/apk/）
python -m privacy_audit.cli stage2-static --apk-dir input/apk --output-dir output/stage2_static_samples

# 动态真机（先 adb、frida-server、Appium）
python -m privacy_audit.cli stage3-dynamic `
  --app-name Keep `
  --app-package com.gotokeep.keep `
  --pii-text-dir input/pii_text `
  --stage2-evidence-file output/stage2_static_samples/keep/static_evidence.json `
  --output-dir output/stage3_dynamic_real `
  --real-run `
  --frida-mode attach
```

Stage3 常用参数：`--frida-mode attach`（默认，保留登录态）、`--frida-mode spawn`（冷启动）、不加 `--appium-reset` 即 **noReset**。

支付宝示例（在 `project/src` 下）：

```powershell
python -m privacy_audit.cli -v stage3-dynamic `
  --app-name 支付宝 `
  --app-package com.eg.android.AlipayGphone `
  --pii-text-dir input/pii_text `
  --output-dir output/stage3_dynamic_real `
  --real-run
```

或便捷模块入口（自动读 `input/app_packages.json`）：

```powershell
python -m privacy_audit.workflows.stage3_dynamic_pipeline --app 支付宝 --real-run -v
```

---

## Privacy Knowledge Base

法律知识库将 Android 技术信号映射到 **法律个人信息矩阵**（A–H 类，见 `categories.json`），再通过 `audit_taxonomy_bridge.json` 桥接到审计用 `personal_info.yaml` 类型。

重建证据或跑 batch 时**默认启用** KB，在证据包中写入：

| 字段 | 说明 |
|------|------|
| `sources.knowledge_base` | 高置信 KB 推断的 audit PII 类型 |
| `knowledge_base.feasible_evidence[]` | 每条可行证据：信号类型、法律条目 ID、置信度、法理标签、`source` |
| `knowledge_base.legal_matrix` | 涉及条目的中文 label、敏感级别、风险桶 |

详细说明：[src/privacy_knowledgebase/README.md](src/privacy_knowledgebase/README.md)

---

## 证据来源说明

实现：`privacy_audit/ingestion/evidence_loader.py`

| 来源键 | 含义 | 典型路径 |
|--------|------|----------|
| `static` | APK 静态（Exodus + Androguard） | `stage2_static_samples/`、`stage2_oldsrc_derived/` |
| `dynamic` | Frida / Appium 动态 | `stage3_dynamic_real/`、`stage3_dynamic_samples/`、`stage3_oldsrc_derived/` |
| `legacy` | Lab3 动态测试 PASS 步骤关键词 | `oldsrc/.../per_app/<App>/*.json` |
| `oldsrc_lab3` | `test_cases.json` + 用例结果 | `dynamic_testing/test_cases.json` |
| `oldsrc_audit` | 政策审计可测项 | `<App>/audit_processed.csv` |
| `knowledge_base` | 法律矩阵 + Android 映射规则 | `privacy_knowledgebase/knowledge_base/` |
| `bundle` | 已生成的扁平包（避免重复解析时用） | `lab4_1_evidence/<App>.json` |

**证据强度（汇报时务必区分）：**

| 强度 | 来源示例 |
|------|----------|
| 强 | Keep 的 stage2 APK + stage3 真机 Frida |
| 中 | 小红书 / 支付宝 的 static + dynamic 样本 |
| 弱（审计/用例推断） | `oldsrc_audit`、`oldsrc_lab3`；不宜单独作为「已采集」结论 |

---

## 实验结果文件与阅读顺序

**正式汇报建议以最新 batch job 为准**（如 `keep_batch_004`），勿混用旧 job 覆盖后的根目录报告。

### 优先阅读

| 文件 | 用途 |
|------|------|
| `output/lab4_1/batch_lab4-1_<job_id>_summary.json` | 全批汇总：每 App 的 `summary.status_counts` |
| `output/lab4_1/<App>_lab4_1_report.json` | 单 App：`claims`、`evidence`、`verdicts` |
| `output/lab4_1_evidence/<App>.json` | 合并证据 + `provenance` + `feasible_evidence` |

### 工程归档

```text
output/lab4_1/batch_jobs/<job_id>/
├── manifest.json
├── summary.json
├── logs/{App}.log
└── reports/{App}/{App}_lab4_1_report.json
```

### 阅读顺序

1. **总览**：`batch_*_summary.json` → `summary.success_count`、`reports[].summary`（`declared_count` / `observed_count` / `consistent` 等）
2. **单 App 结论**：`<App>_lab4_1_report.json` → `verdicts[]`
3. **证据溯源**：`lab4_1_evidence/<App>.json` → `provenance`、`sources`、`knowledge_base.feasible_evidence`
4. **底层细节**：`stage3_dynamic_real/`、`oldsrc/.../access_result.json` 等

### Verdict 含义

| status | 含义 |
|--------|------|
| `consistent` | 政策声明且证据观测到 |
| `inconsistent_high_risk` | 证据有、政策未声明 |
| `declared_not_observed` | 政策声明但当前证据未观测到（≠ 未采集） |

### 迭代参考（Lab4-1 batch）

| Job | 说明 |
|-----|------|
| `keep_batch_001` | 3 App，证据路径未对齐，`observed` 多为 0 |
| `keep_batch_002` | 10 App，合并证据后结论有意义 |
| `keep_batch_003` | 9 App + oldsrc 审计导入，当前推荐基线 |
| `keep_batch_004` | 启用 privacy_knowledgebase 后的版本 |

---

## 配置与规则

| 文件 | 作用 |
|------|------|
| `privacy_audit/config/personal_info.yaml` | 审计用 PII 类型、别名、API、权限 |
| `privacy_audit/rules/decision_rules.yaml` | Lab4-1 判定规则 |
| `privacy_knowledgebase/input.md` | 法律个人信息矩阵（源） |
| `privacy_knowledgebase/knowledge_base/*.json` | 机器可读矩阵与 Android 映射 |
| `privacy_audit/config/legacy_label_mapping.yaml` | 旧版标签映射 |

---

## 添加新 App

1. 在 `input/pii_text/` 添加 `<App名>.txt`
2. （可选）`input/apk/` + `stage2-static`
3. （可选）`stage3-dynamic --real-run`
4. 若 oldsrc 有数据，加入 `apps_lab4_batch.txt`
5. `kb-build`（若改过矩阵）→ `import-oldsrc-evidence --rebuild-bundles` → `batch`

---

## 与 oldsrc 的关系

| | oldsrc | src（本仓库主路径） |
|---|--------|---------------------|
| 角色 | 历史实验产出与 LLM 审计 | 证据驱动 + 规则引擎 + 法律知识库 |
| 使用方式 | 只读 `analyzer/output/`，经 `import-oldsrc-evidence` 导入 | `python -m privacy_audit.cli ...` |
| 不宜 | 再作为主执行入口 | 用 oldsrc 代码跑主流程 |

---

## 测试

```powershell
cd src
python -m pytest tests/ -q
```

---

## 进一步阅读

- [reference/开发计划.md](reference/开发计划.md) — 阶段验收标准
- [reference/plan.md](reference/plan.md) — 补充设计
- [reference/法律参考文件/README.md](reference/法律参考文件/README.md) — 法律参考与矩阵维护
- [src/privacy_knowledgebase/README.md](src/privacy_knowledgebase/README.md) — 知识库与 feasible evidence
- [oldsrc/analyzer/dynamic_testing/README.md](../oldsrc/analyzer/dynamic_testing/README.md) — 历史动态测试

---

## 常见问题

**Q: `ModuleNotFoundError: No module named 'privacy_audit'`**  
A: 必须在 `project/src` 下执行命令。

**Q: batch 全是 `declared_not_observed`**  
A: 先执行 `import-oldsrc-evidence --rebuild-bundles`，batch 使用 `--evidence-dir output/lab4_1_evidence`。

**Q: 报告里看不到 `feasible_evidence`**  
A: 查看 `output/lab4_1_evidence/<App>.json` 的 `knowledge_base` 字段；确保证据包重建时未加 `--no-knowledge-base`。

**Q: Stage3 无论指定哪个 App 都打开 Keep**  
A: 旧版在 Appium 带 `appPackage` 建会话失败时会**回退到无包名会话**，自动化会绑定当前前台应用（若刚测过 Keep 就会一直操作 Keep）。日志里若有 `session_created_fallback_no_package` 即属此情况。现已改为：先用 adb 启动目标包、重试带包名会话、校验 `foreground_package`，不再无包名回退。跑前可先手动关掉 Keep，或看日志 `adb_foreground_before=` / `foreground_package=`。

**Q: `python -m privacy_audit.workflows.stage3_dynamic_pipeline` 秒退、模拟器无操作**  
A: 旧版该模块**没有** `main` 入口，带参数也会立即退出。请改用 `python -m privacy_audit.cli -v stage3-dynamic --app-name 支付宝 --app-package com.eg.android.AlipayGphone --real-run ...`；或已支持便捷入口：`python -m privacy_audit.workflows.stage3_dynamic_pipeline --app 支付宝 --real-run`（勿只写 `--dry-run false` 却不加 `--real-run` 时确认日志里 `dry_run=False`）。

**Q: `stage3-dynamic` 跑很久终端无输出、模拟器无反应**  
A: ① 必须在 `project\src` 下执行；② 加 `-v` 看实时日志；③ 先启动 Appium（`http://127.0.0.1:4723/status` 可访问）；④ `--real-run` 才会驱动 UI；⑤ attach 前 Keep 需在前台。日志见 `output/stage3_dynamic_real/Keep/stage3_run.log`。Frida 与 Appium 已同步：前 5s 可手动操作，随后 Appium 导航时 Frida 同时记录。

**Q: Frida 只有 `hook_loaded`、无运行时事件**  
A: 确认使用 `--real-run`；在 grace 窗口或 Appium 进权限/定位相关页时触发 API；查看 `registered_hooks` 是否含 `hook_phone.js` 等新脚本；`summary.frida_runtime_events` 应 > 0 才有 Lab4-2 对比价值。

**Q: `probe_hook.py` / attach 报 `refused to load frida-agent`（pid 为 `com.gotokeep.keep`）**  
A: MuMu 上常见：应 attach 进程名 **`Keep`**（`adb shell pidof com.gotokeep.keep` 的**第一个** PID），不要只 attach 子进程 `com.gotokeep.keep`。先启动 Keep，再运行 `python scripts/probe_hook.py`（无参数时会按推荐顺序逐个尝试）。PC 与模拟器 `frida-server` 版本需一致（如 17.9.10）。

**Q: PowerShell 多行命令报错**  
A: 行末用反引号 `` ` `` 续行，或合并为单行并用 `;` 分隔。

**Q: 课程要求 ≥10 个 App**  
A: 当前列表 9 个（Soul 已排除）；可在 `apps_lab4_batch.txt` 增加 oldsrc 中已有数据的 App 后重跑。

**Q: 如何关闭法律知识库**  
A: `import-oldsrc-evidence`、`build-evidence-bundles`、`batch` 均可加 `--no-knowledge-base`。
