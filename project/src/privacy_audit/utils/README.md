# utils — 工具函数

`utils` 提供与具体审计逻辑无关的基础设施：adb、日志、重试、路径、UI 解析、屏幕抓取。

## 1. 模块

| 模块 | 主要函数 | 用途 |
| --- | --- | --- |
| `adb_device.py` | `list_connected_devices`、`resolve_adb_serial`、`adb_run` | adb 设备发现与命令执行 |
| `runtime_preflight.py` | `check_adb_devices`、`check_appium_server`、`check_frida_usb`、`run_capture_preflight` | 动态取证前置检查 |
| `logger.py` | `configure_cli_logging`、`setup_file_logger`、`get_logger` | 终端/文件日志 |
| `progress_log.py` | `log_step`、`log_countdown` | 长步骤进度 |
| `retry.py` | `run_with_retry` | 指数退避重试 |
| `path_utils.py` | `output_root`、`output_provenance`、`as_output_path` | output 相对路径与 provenance |
| `ui_hierarchy.py` | `parse_page_source`、`analyze_sidebar_state`、`summarize_clickable_labels` | Appium 页面层级解析 |
| `screenGrab/` | `capture_emulator_screen`、`capture_with_active_driver` | 截图 + UI XML 抓取 |
| `time_utils.py` | `utc_now_iso` | UTC 时间戳 |

## 2. 动态取证前置检查

`run_capture_preflight` 会检查：

1. `adb devices` 是否有设备；
2. Appium `/status` 是否可访问；
3. `frida-ps -U` 是否能列出设备进程。

```powershell
python -m privacy_audit.cli capture-dynamic --app-name Keep --app-package com.gotokeep.keep --real-run --skip-preflight
```

- 不加 `--skip-preflight`：失败会告警但仍继续；
- 加 `--skip-preflight`：完全跳过检查，适合已知环境正常时。

## 3. 截图工具

`screenGrab` 可独立使用：

```powershell
python -m privacy_audit.utils.screenGrab --label Keep_privacy_page
```

输出目录默认在 `privacy_audit/utils/screenGrab/output/<timestamp>_<label>/`，包含 `screenshot.png`、`ui.xml`、`summary.json`。

## 4. 路径与 provenance

报告中的证据路径统一写成 `output/...` 形式：

```python
from privacy_audit.utils.path_utils import output_provenance
output_provenance("dynamic", "Keep", "dynamic_evidence.json")
# -> "output/dynamic/Keep/dynamic_evidence.json"
```

新增证据来源时，请同时写入 `provenance`，保证报告可追溯。

## 5. 日志

CLI 默认实时输出；`capture-dynamic` 还会写入：

```text
output/dynamic/<App>/capture_run.log
```

建议排查动态问题时加 `-v`：

```powershell
python -m privacy_audit.cli -v capture-dynamic --app-name Keep --app-package com.gotokeep.keep --real-run
```
