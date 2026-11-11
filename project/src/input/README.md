# input — 输入数据

`input` 存放所有**手工准备**的输入：政策文本、APK、App 列表、用户副本和配置。

## 1. 目录结构

```text
input/
├── pii_text/                  # <App>.txt 隐私政策纯文本
├── app_packages.json          # App 名 → 包名（及 code_prefixes）
├── apps_audit_batch.txt       # 当前批量 App 列表
├── apps_consent_four.txt      # 同意权小批量列表
├── apk/                       # 待扫描 APK
├── consent_before/            # 同意权：开关前动态证据 <App>.json
├── consent_after/             # 同意权：开关后动态证据 <App>.json
├── access_copy/               # 访问权：用户副本 <App>.json
├── access_runtime/            # 访问权：运行时证据 <App>.json
├── user_copy/                 # 原始副本（格式可不统一）
├── user_copy_normalized/      # normalize_user_copy.py 生成的扁平 JSON
├── frida_tier.json            # 每 App 的 Frida 等级与轨道偏好
├── navigation_bridges.json    # 学习到的 UI 导航桥
├── privacy_trace_methods.txt  # PMA 风格 API 清单
└── tracker_blocklist/         # tracker 域名 + SDK 签名
```

## 2. 新增一个 App

1. **政策文本**：`input/pii_text/<App名>.txt`（UTF-8）。
2. **包名映射**：在 `input/app_packages.json` 增加：

```json
{
  "App名": {
    "package": "com.example.app",
    "code_prefixes": ["com.example"]
  }
}
```

3. **加入批量列表**：把 App 名加到 `input/apps_audit_batch.txt`（一行一个）。
4. **（可选）APK**：放到 `input/apk/`，然后 `scan-static`。
5. **（可选）动态**：配置 `frida_tier.json` 的 tier/轨道，运行 `capture-dynamic`。
6. **（可选）访问权**：把副本放到 `access_copy/<App>.json`，运行时证据放到 `access_runtime/<App>.json`。

## 3. 命名约定

- App 名必须与 `pii_text/<App>.txt`、`app_packages.json`、报告文件名一致；
- 静态证据目录使用 slug（如 `Keep → keep`、`小红书 → xiaohongshu`），映射见 `privacy_audit/ingestion/evidence_loader.py` 的 `_slug_for_app()`；
- 批量输入文件使用 `<App>.json` 扁平命名。

## 4. 用户副本

`user_copy/` 允许格式不统一（子目录、`.txt`、`.json`）。先规范化：

```powershell
python scripts/normalize_user_copy.py
```

输出到 `user_copy_normalized/<App>.json`，键为导出字段名，例如：

```json
{
  "手机号": "138****0000",
  "昵称": "demo",
  "位置信息": {"城市": "上海"}
}
```

`audit-access` 的 `--copy-file` 必须是**单个 JSON 对象**（不是数组）。

## 5. 配置说明

### `frida_tier.json`

```json
{
  "apps": {
    "小荷健康": {
      "tier": "tier1",
      "java_bridge": "ok",
      "preferred_dynamic_tracks": ["frida_java", "frida_network", "ui"],
      "enable_frida_consent": true
    },
    "支付宝": {
      "tier": "tier3",
      "java_bridge": "refused",
      "preferred_dynamic_tracks": ["mitm", "ui"],
      "enable_frida_consent": false
    }
  }
}
```

- `tier1`：可注入，优先 Frida；
- `tier2`：部分可用；
- `tier3`：强防护，mitm/UI 为主。

### `navigation_bridges.json`

记录「从当前页面到目标页面」的中间点击路径，格式由 `navigation_bridge_store.py` 维护，可手工修补。

### `tracker_blocklist/`

- `trackers_domains.json`：tracker 域名 → 公司/用途；
- `sdk_signatures.json`：DEX 类名前缀 → SDK 名称。

## 6. 旧目录

历史目录（`lab4_2_before/`、`lab4_3_copy/` 等）已重命名为 `consent_*` / `access_*`；如果旧数据仍在旧目录，命令显式指定旧路径即可，或重新运行脚本同步。
