# tools — 外部工具

当前只有 Frida 相关二进制。

## 1. frida-server

`tools/frida/` 存放与 PC 端 Frida 版本一致的 `frida-server`：

```text
tools/frida/
├── frida-server-17.9.10-android-x86_64
└── frida-server-17.9.10-android-x86_64.xz
```

使用步骤（模拟器/真机）：

1. 确认 PC 端版本：

```powershell
frida --version
```

2. 把对应架构的 `frida-server` 推送到设备：

```powershell
adb push tools/frida/frida-server-17.9.10-android-x86_64 /data/local/tmp/frida-server
adb shell chmod 755 /data/local/tmp/frida-server
```

3. 启动（需要 root/模拟器权限）：

```powershell
adb shell "/data/local/tmp/frida-server &"
```

4. 验证：

```powershell
frida-ps -U
```

## 2. 版本匹配

PC 端 `frida` Python 包与设备端 `frida-server` **必须版本一致**（例如都为 17.9.10）。版本不一致时常见错误：

- `unable to connect to remote frida-server`
- `refused to load frida-agent`
- hook 加载后无事件

## 3. 注意

- `.xz` 压缩包与二进制在 `.gitignore` 中被排除，需要本地准备；
- `frida-server` 属于安全测试工具，仅在你有权限的设备/模拟器上使用；
- 如果 App 有反注入，参考 `input/frida_tier.json` 与 `scripts/probe_hook.py`，不要假设所有 App 都能注入。
