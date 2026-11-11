# analyzers — 取证分析器

`analyzers` 负责从四类来源产生原始证据。它**不判断合规**，只回答「我们看到了什么」。

```text
analyzers/
├── policy/    隐私政策 → 声明了哪些 PII
├── static/    APK → 权限、API、内嵌 SDK
├── dynamic/   设备 → UI 步骤、Frida 运行时事件
└── network/   流量 → tracker 域名、PII 字段键
```

---

## 1. policy — 政策声明解析

**入口**：`policy/policy_parser.py`

| 函数 | 作用 |
| --- | --- |
| `load_personal_info_taxonomy(yaml)` | 读取 `config/personal_info.yaml`，得到 PII 类型及别名/关键词 |
| `parse_declared_pii(text, taxonomy)` | 按别名与 `policy_keywords` 匹配政策文本，输出 `declared_pii` |

特点：

- 纯规则匹配，不调用 LLM，结果可复现；
- 同一 PII 类型命中一次即计入；
- 政策文本越长，召回越依赖 `policy_keywords` 的维护质量。

新增声明词：在 `personal_info.yaml` 对应类型的 `aliases` / `policy_keywords` 中追加。

---

## 2. static — APK 静态取证

**入口**：`workflows/static_scan.py`，内部调用：

| 模块 | 作用 | 输出字段 |
| --- | --- | --- |
| `static/exodus_adapter.py` | 权限与 tracker 初筛 | `exodus.permissions`、`exodus.pii_permission_hits`、`exodus.trackers` |
| `static/androguard_scanner.py` | 扫描与 PII 相关的 API 调用 | `androguard.api_hits` |
| `static/tracker_dex_scanner.py` | DEX 类名匹配 SDK/tracker 签名 | `embedded_trackers[]` |
| `static/tracker_blocklist_loader.py` | 加载 `input/tracker_blocklist/` | 域名/签名规则 |

静态证据文件结构：

```json
{
  "app_name": "keep",
  "apk_path": "input/apk/keep.apk",
  "observed_pii": ["imei", "location"],
  "exodus": {"permissions": [], "pii_permission_hits": {}, "trackers": []},
  "androguard": {"api_hits": {}},
  "embedded_trackers": [{"name": "Google Ads", "matched_signatures": []}],
  "evidence_items": [],
  "summary": {}
}
```

使用建议：

- 批量初筛加 `--fast`，跳过完整 Androguard `AnalyzeAPK`；
- 需要 SDK 披露比对时，确保 `embedded_trackers` 非空；
- Android 10+ 上 IMEI 常不可读，静态命中不等于运行时采集。

---

## 3. dynamic — 设备动态取证

**入口**：`workflows/dynamic_capture.py`，核心协作模块：

### 3.1 UI 自动化

| 模块 | 职责 |
| --- | --- |
| `dynamic/appium_runner.py` | 创建 Appium 会话、按政策路径导航、截图、keyword checks |
| `dynamic/element_tap.py` | 稳健点击：多选择器、滚动、重试、超时 |
| `dynamic/sidebar_detector.py` | 识别侧边栏/抽屉状态 |
| `dynamic/ui_hierarchy.py` | 解析页面层级，定位可点击元素 |
| `dynamic/navigation_path_builder.py` | 把政策文本/目标拆成导航 token |
| `dynamic/navigation_bridge_store.py` | 读写 `input/navigation_bridges.json` |
| `dynamic/llm_path_assistant.py` | 导航失败时用 DeepSeek 建议中间桥（只建议，不裁决） |
| `dynamic/app_launch.py` | adb 启动/关闭目标包 |

### 3.2 Frida 运行时

| 模块 | 职责 |
| --- | --- |
| `dynamic/frida_runner.py` | attach/spawn、进程选择、hook 加载、事件收集 |
| `dynamic/frida_sync.py` | 与 Appium 同步的采集会话（grace/tail 窗口） |
| `dynamic/hook_loader.py` | 按 PII 类型加载并包装 JS hook |
| `dynamic/frida_reactor.py` | 事件过滤与反应逻辑 |
| `dynamic/hooks/*.js` | 具体 hook：IMEI、位置、通讯录、剪贴板、OAID、OkHttp 等 |

`dynamic/hooks/` 中的脚本：

| 脚本 | 目标 |
| --- | --- |
| `hook_papi.js` | 数据驱动的通用 Java API 监控 |
| `hook_imei.js` | `TelephonyManager` 设备标识 |
| `hook_android_id.js` | `Settings.Secure.ANDROID_ID` |
| `hook_oaid.js` | OAID SDK |
| `hook_location.js` | `LocationManager` / `Location` |
| `hook_contacts.js` | `ContentResolver` 通讯录查询 |
| `hook_clipboard.js` | 剪贴板读取 |
| `hook_mac_ip.js` | Wi-Fi MAC / IP |
| `hook_phone.js` | 手机号/运营商信息 |
| `hook_network_okhttp.js` | OkHttp 请求体中的 PII 字段 |
| `hook_common.js` | 公共工具函数 |

动态证据文件结构（节选）：

```json
{
  "app_name": "Keep",
  "ui_result": {"steps": [], "screenshots": [], "keyword_checks": []},
  "frida_result": {"events": [], "mode": "real", "error": ""},
  "network_result": {"tracker_hits": [], "pii_key_hits": [], "summary": {}},
  "summary": {"frida_runtime_events": 3, "mitm_flow_count": 42}
}
```

### 3.3 采集轨道

`workflows/capture_track.py` 根据 `input/frida_tier.json` 与探针结果选择：

| 轨道 | 何时使用 |
| --- | --- |
| `mitm` | 默认；无 Java 桥、强反注入、只想看外传 |
| `frida` | Tier-1 可注入 App；需要 API 级证据 |
| `mitm+frida` | 同时需要 API 与网络证据 |
| `ui` | 只做页面/开关验证，不采集运行时 |

---

## 4. network — 网络取证

| 模块 | 职责 |
| --- | --- |
| `network/mitm_runner.py` | 启动/停止 mitmdump，设置 adb 代理，录制 flows |
| `network/mitm_addon.py` | mitmproxy addon：把请求写入 flows 文件 |
| `network/tracker_matcher.py` | 请求 host 匹配 tracker 域名库 |
| `network/traffic_checker.py` | 汇总 `network_result`：tracker hits、PII key hits、summary |

输出字段：

- `tracker_hits[]`：命中的 tracker 公司、host、时间戳；
- `pii_key_hits[]`：URL/body 中出现的 `network_keys` 字段；
- `summary.flow_count`、`summary.tracker_hit_count`、`summary.pii_key_hit_count`。

限制：HTTPS + 证书锁定可能导致 `mitm_blocked`；此时报告应标注为「网络轨不可用」，不能据此判定未外传。

---

## 5. 证据如何进入汇总层

各分析器只产生自己的 JSON；`ingestion/evidence_loader.py` 负责：

1. 按 App 名/别名找到静态、动态、历史证据；
2. 从 `androguard.api_hits`、`exodus.pii_permission_hits`、Frida 事件、网络字段中提取 `observed_pii`；
3. 合并为 `sources` 与 `provenance`；
4. 调用知识库 `enricher` 生成 `feasible_evidence`。

下一步阅读：[ingestion](../ingestion/README.md)、[engine](../engine/README.md)。
