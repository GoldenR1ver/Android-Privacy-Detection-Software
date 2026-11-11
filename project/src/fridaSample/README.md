# fridaSample — 韭研公社 Frida Hook 实验

独立沙盒，用于在 **韭研公社**（`com.jiucaigongshe`）上尝试 Hook 关键 SDK 与 Android 隐私 API，不改动 `privacy_audit` 主流程。

## 目标 App

| 字段 | 值 |
|------|-----|
| 应用名 | 韭研公社 |
| 包名 | `com.jiucaigongshe` |
| 别名 | 韭菜公社 |

已知网络/SDK 线索（来自 Lab4 mitm / 政策）：

- **友盟 Umeng**：`aaid.umeng.com`
- **腾讯 X5/TBS**：`log.tbs.qq.com`
- **推送**：可能存在 `:push` / `pushcore` 子进程

配置见 `target/jiuyangongshe.json`。

## 目录结构

```
fridaSample/
├── target/jiuyangongshe.json   # 目标 App 元数据与默认 hook 列表
├── hooks/                      # 本模块专用 hook 脚本
│   ├── hook_java_bridge_probe.js
│   ├── hook_sdk_umeng.js
│   ├── hook_sdk_tencent_x5.js
│   └── hook_sdk_push.js
├── run_hook.py                 # Python 运行器
└── run_hook.ps1                # Windows 快捷脚本
```

Hook 事件格式与主工程一致：

```json
{
  "pii_type": "sdk_info",
  "hook_file": "hook_sdk_umeng.js",
  "api_name": "Umeng.DeviceConfig.getDeviceId",
  "args_summary": "...",
  "stack_summary": "..."
}
```

## 前置条件

1. 已 root 的 Android 设备 / 模拟器，**frida-server** 与 PC 端 `frida` 版本一致（见项目 `tools/frida/`）。
2. `adb devices` 可见设备。
3. 在 **`project/src`** 目录执行命令。
4. 韭研公社已安装；attach 模式需 App 在前台或后台运行。

## 快速开始

```powershell
# 进入 src 目录
cd project\src

# 列出可用 hook
python fridaSample/run_hook.py --list-hooks

# attach 模式（默认 hook：SDK 探针）
python fridaSample/run_hook.py

# 冷启动 spawn（韭研公社 tier 推荐）
python fridaSample/run_hook.py --mode spawn --wait-sec 30

# 追加主工程 PII hook（android_id / okhttp / imei 等）
python fridaSample/run_hook.py --with-project-pii --wait-sec 40

# 只测友盟 + 网络
python fridaSample/run_hook.py --hooks hook_sdk_umeng.js hook_network_okhttp.js --use-project-hooks

# Windows 快捷方式
.\fridaSample\run_hook.ps1 -Mode spawn -WaitSec 30
```

注入成功后，在 App 内浏览首页、发帖、打开 WebView 页、触发推送权限等，以产生 SDK / API 调用。

## 输出

事件 JSON 写入 `output/frida_sample/com_jiucaigongshe_<timestamp>.json`，包含：

- `runtime_event_count`：实际 API/SDK 命中数
- `runtime_hits`：精简命中列表
- `events`：完整 Frida message 载荷

## 与主工程的关系

| 组件 | 说明 |
|------|------|
| `privacy_audit/.../hooks/` | 生产级 PII hook（IMEI / OkHttp / PAPI 等），可通过 `--use-project-hooks` 引用 |
| `scripts/probe_hook.py` | Java 桥探针；`--profile jiuyan` 专用于韭研公社 attach/spawn 诊断 |
| `privacy_audit` stage3-dynamic | 正式 Lab4 动态取证管线 |

本目录适合 **快速迭代 hook 脚本**；验证稳定后可把新 target 合并进 `personal_info.yaml` 的 `api_targets` 或主工程 `hooks/`。

## 常见问题

**Java bridge 无响应 / `java_never_ready`**

- 试 `--mode spawn`
- 运行 `python scripts/probe_hook.py --profile jiuyan --mode both`

**只有 `hook_loaded`、无 runtime 事件**

- 延长 `--wait-sec`，并在 App 内实际操作（刷新、进设置、打开 H5 页）
- 加 `--with-project-pii` 扩大 API 覆盖面

**attach 找不到进程**

```bash
adb shell monkey -p com.jiucaigongshe -c android.intent.category.LAUNCHER 1
adb shell pidof com.jiucaigongshe
```
