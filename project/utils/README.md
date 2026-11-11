# utils — 开源参考仓库

`project/utils/` 存放与本项目相关的开源参考实现。它们**不是**主流程的运行时依赖；主流程使用的是 Python 包（`androguard`、`frida` 等）和本项目 `src/privacy_audit` 的适配器。

## 1. 目录

| 目录 | 作用 | 本项目如何使用 |
| --- | --- | --- |
| `androguard/` | APK/DEX 静态分析 | `analyzers/static/androguard_scanner.py` 参考/调用 |
| `exodus/` | APK 隐私/tracker 初筛 | `analyzers/static/exodus_adapter.py` 参考其输出格式 |
| `frida/` | Frida 运行时 | `analyzers/dynamic/frida_runner.py` |
| `awesome-frida/` | Frida 脚本集合 | hook 写法参考 |
| `frida-android-hooks/` | Android hook 集合 | hook 模板参考 |
| `tracker-control-android/` | tracker 域名库 + DEX 扫描思路 | `tracker_blocklist/`、`tracker_dex_scanner.py` 的设计参考 |
| `PrivacyMonitorAndroid/` | 编译期 ASM 插桩 + Excel | 报告列与 API 清单的设计参考 |
| `PAPIMonitor/` | 运行时 API 监控 | `hook_papi.js` 的设计参考 |
| `objection/`、`frida-tools/`、`frida-trace/`、`frida_dump/`、`rekit/` | 本地参考工具 | 按需手动使用 |

## 2. 使用边界

- **不直接移植** Android App（如 TrackerControl 的 VPN App）到 Python 管线；
- **不把** PrivacyMonitorAndroid 的 Gradle 插件接入本流水线（它要求源码重编译）；
- 需要新增 hook/扫描逻辑时，优先在 `src/privacy_audit` 内实现适配器，参考这些仓库的思路。

## 3. Git 与许可

- 部分目录（`TrackerControl`、`PrivacyMonitorAndroid`、`objection` 等）在 `.gitignore` 中，属于本地参考克隆，不随仓库提交；
- 使用/分发这些代码时请遵守各自 LICENSE；
- 本项目只保留必要的接口与配置，不复制其大体积资源。
