# src 目录索引

这里是隐私检测工具的运行根目录。所有 CLI 命令都应在 `project/src` 下执行。

```powershell
cd project/src
python -m privacy_audit.cli --help
```

## 包与目录

| 路径 | 职责 | 文档 |
| --- | --- | --- |
| `privacy_audit/` | 主审计包：CLI、取证、汇聚、判定、工作流、报告 | [README](privacy_audit/README.md) |
| `privacy_knowledgebase/` | 法律个人信息矩阵、Android 信号映射、可行证据 | [README](privacy_knowledgebase/README.md) |
| `scripts/` | 批量调度、Frida 探针、配置同步、副本规范化 | [README](scripts/README.md) |
| `input/` | 政策文本、APK、App 列表、用户副本、tracker 库 | [README](input/README.md) |
| `output/` | 静态/动态证据、审计报告、批量任务归档 | [README](output/README.md) |
| `tests/` | 单元、集成、E2E 测试 | [README](tests/README.md) |
| `fridaSample/` | Frida 样例脚本与目标配置 | [README](fridaSample/README.md) |

## 主包分层

```text
privacy_audit/
├── cli.py            # 命令解析与调度
├── analyzers/        # 取证：政策 / 静态 / 动态 / 网络
├── ingestion/        # 证据聚合、历史导入、路径解析
├── engine/           # 声明-证据比对、规则、SDK 披露
├── workflows/        # 三项审计 + 取证 + 批量编排
├── reporting/        # JSON / Excel 报告
├── config/           # PII 分类、判定规则、标签映射
├── domain/           # 统一数据模型
└── utils/            # adb、日志、重试、UI 层级等
```

## 快速入口

- 首次使用：先看 [主 README](../README.md#6-五分钟快速开始)。
- 想加一种个人信息类型：看 [config/README](privacy_audit/config/README.md)。
- 想改判定规则：看 [engine/README](privacy_audit/engine/README.md)。
- 想理解证据包字段：看 [ingestion/README](privacy_audit/ingestion/README.md)。
- 想批量跑实验：看 [workflows/README](privacy_audit/workflows/README.md) 与 [scripts/README](scripts/README.md)。
