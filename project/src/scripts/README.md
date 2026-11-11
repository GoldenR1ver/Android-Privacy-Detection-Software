# scripts — 辅助脚本

`scripts` 是**辅助工具层**，用于批量调度、Frida 探针、配置同步和副本规范化。日常单 App 验证优先使用 `privacy_audit.cli`；只有重复性/实验性任务才需要脚本。

所有脚本都在 `project/src` 下执行：

```powershell
cd project/src
python scripts/<script>.py --help
```

## 1. 批量实验编排

| 脚本 | 用途 |
| --- | --- |
| `run_privacy_pipeline_batch.py` | 全流程：静态 → 动态 → 知情权 → 同意权 → Excel |
| `run_full_experiment_batch.py` | 带阶段开关的完整实验批处理 |
| `run_static_scan_batch.py` | APK 批量静态扫描 + 知情权批处理 + 统计 |
| `run_dynamic_capture_batch.py` | `--dynamic-track auto` 批量动态取证 |
| `run_network_capture_batch.py` | mitm 网络轨批量取证 |
| `run_consent_ui_batch.py` | 同意权 before/after UI 采集与对比 |
| `run_consent_network_batch.py` | 同意权 mitm 网络轨批量对比 |

常用示例：

```powershell
# 单 App 全流程
python scripts/run_privacy_pipeline_batch.py --apps 钓鱼人 --phase all --real-run

# 只跑静态 + 动态 + 知情权
python scripts/run_privacy_pipeline_batch.py --phase static,dynamic,disclosure --real-run

# 同意权 before/after
python scripts/run_consent_ui_batch.py --phase before --app 小荷健康
python scripts/run_consent_ui_batch.py --phase after --app 小荷健康
python scripts/run_consent_ui_batch.py --phase compare
```

## 2. Frida 探针

| 脚本 | 用途 |
| --- | --- |
| `probe_hook.py` | 多模式注入探针：attach/spawn、`java-wait-only`、hook 加载验证 |
| `probe_frida_attach.py` | 只验证 attach 与进程选择 |
| `probe_spawn.py` | 只验证 spawn 冷启动 |
| `sync_frida_tier_from_probe.py` | 把探针结果写回 `input/frida_tier.json` |

```powershell
python scripts/probe_hook.py --profile keep --java-wait-only --hook-wait-sec 45
python scripts/sync_frida_tier_from_probe.py
```

探针输出：

- `output/frida_probe/<slug>.json`：`java_bridge`、`preferred_dynamic_tracks`；
- `input/frida_tier.json`：每 App 的 `tier` 与动态轨道偏好。

## 3. 证据与报告

| 脚本 | 用途 |
| --- | --- |
| `audit_outputs.py` | 汇总三项审计产物与证据计数 |
| `compare_consent_ui.py` | 同意权 UI before/after 差异 |
| `compare_dynamic_runs.py` | 比较两次动态取证目录 |
| `generate_app_track_matrix.py` | 生成 App × 轨道覆盖矩阵 |
| `normalize_user_copy.py` | 把 `input/user_copy/` 的原始副本规范化为 JSON |

```powershell
# 规范化用户副本
python scripts/normalize_user_copy.py

# 生成覆盖矩阵
python scripts/generate_app_track_matrix.py --output output/app_track_matrix.json
```

## 4. 配置同步

| 脚本 | 用途 |
| --- | --- |
| `sync_trace_config.py` | `privacy_trace_methods.txt` ↔ `personal_info.yaml` |
| `sync_tracker_blocklist.py` | 同步 tracker 域名/SDK 签名库 |

```powershell
python scripts/sync_trace_config.py --check
python scripts/sync_tracker_blocklist.py
```

## 5. 使用建议

- 先用 CLI 跑通单 App，再交给脚本批量；
- 脚本默认会在 `project/src` 下找 `input/`、`output/`；
- 批量前确认 App 列表文件（如 `input/apps_audit_batch.txt`）与包名映射；
- 脚本的中间产物仍写入 `output/`，可用 `batch-status` 或 `audit_outputs.py` 汇总。
