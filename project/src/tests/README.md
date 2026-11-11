# tests — 测试

测试全部在 `project/src` 下运行：

```powershell
cd project/src
python -m pytest tests -q
```

## 1. 分层

| 目录 | 内容 | 是否需要设备 |
| --- | --- | --- |
| `unit/` | 纯函数、路径解析、hook 加载、mitm 解析、规则判定 | 否 |
| `integration/` | 跨模块证据流、报告整形 | 否 |
| `e2e/` | 端到端实验流程 | 可能需要设备/模拟器 |

## 2. 常用命令

```powershell
# 全部单元测试
python -m pytest tests/unit -q

# 单个文件
python -m pytest tests/unit/test_access_audit_mitm.py -q

# 显示失败详情
python -m pytest tests -q -x -vv
```

## 3. 覆盖重点

- `test_access_audit_mitm.py`：访问权的 Frida/mitm 运行时字段
- `test_network_r3.py`：mitm 证据与同意权网络轨
- `test_tracker_dex_scanner.py`：SDK 签名扫描与披露比对
- `test_frida_*`：attach/spawn 候选、code prefixes、hook 加载
- `test_appium_paths.py`：导航路径构建与 sidebar 识别
- `test_trace_config.py`：API 清单与 YAML 同步
- `test_excel_writer.py`：Excel 导出
- `test_smoke.py`：包导入与最小闭环

## 4. 新增测试建议

- 业务规则改动：补 `tests/unit`，使用小 JSON fixture，不依赖真实设备；
- 路径/兼容改动：覆盖新旧目录两种布局；
- 报告字段改动：断言 `report_path`、`summary.status_counts`、`provenance`。
