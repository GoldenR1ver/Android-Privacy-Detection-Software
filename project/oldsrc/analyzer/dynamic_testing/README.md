# Part 2：基于 Appium 的动态测试与可用性评估

> 对应 Lab3 实验要求：自动化验证 APP 用户隐私权力的执行方式真实性，并进行可用性评价。

## 环境要求

### 1. Python 依赖

```bash
pip install Appium-Python-Client selenium
```

### 2. Appium Server（需 Node.js）

```bash
npm install -g appium
appium driver install uiautomator2
```

### 3. Android SDK Platform-Tools（ADB）

下载 [platform-tools](https://developer.android.com/tools/releases/platform-tools)，将 `adb.exe` 所在目录加入系统 PATH。

### 4. MuMu 模拟器

安装 [MuMu模拟器](https://mumu.163.com/)，启动后确保 ADB 能连接：

```bash
# MuMu 6 默认端口 7555
adb connect 127.0.0.1:7555

# MuMu 12 / MuMu X 默认端口 16384
adb connect 127.0.0.1:16384

# 验证连接
adb devices
```

> 如果端口不同，修改 `config.py` 中的 `MUMU_ADB_PORT`。

### 5. APP 安装（自动安装 APK）

测试前需要在 MuMu 模拟器中安装待测 APP。系统支持**自动安装**：

1. 创建 APK 存放目录（如 `data/apk/`）
2. 将 APK 文件放入目录，**文件名与 APP 名称一致**（如 `Keep.apk`、`支付宝.apk`）
3. 在 `config.py` 中设置 `APK_DIR`：

```python
from pathlib import Path
APK_DIR = Path("data/apk")
```

启动测试时，系统会：
1. 检查 APP 是否已安装在模拟器中
2. 如果未安装，自动从 `APK_DIR` 中找对应 APK 并执行 `adb install`
3. 如果安装失败或找不到 APK，记录错误并跳过该 APP

> **注意**：APK 需自行从 [APKPure](https://apkpure.com/) / [APKMirror](https://www.apkmirror.com/) 等渠道下载。

> 如果 `APK_DIR` 为 `None`（默认），则跳过自动安装，需手动在模拟器中安装 APP。

## 目录结构

```
src/analyzer/
├── dynamic_testing/          # Part 2 主包
│   ├── models.py             # 数据模型
│   ├── config.py             # 配置（APP列表、包名、MuMu端口）
│   ├── test_case_generator.py
│   ├── driver_manager.py     # Appium 会话管理
│   ├── ui_automator.py       # UI 元素查找
│   ├── navigator.py          # 中文路径解析
│   ├── channels/             # 渠道层
│   │   ├── app_ui_channel.py
│   │   ├── web_channel.py
│   │   └── manual_channel.py
│   ├── services/             # 权利测试服务
│   │   ├── consent_service.py
│   │   ├── access_service.py
│   │   └── deletion_service.py
│   ├── scoring/              # 评分
│   │   ├── criteria.py
│   │   └── usability_scorer.py
│   ├── reporting/            # 报告
│   │   └── report_generator.py
│   └── runner.py             # 编排器
└── run_dynamic_test.py       # CLI 入口
```

## 使用方法

### 完整流程

```bash
cd src/analyzer

# 一键执行：发现 → 测试 → 报告
python run_dynamic_test.py all --pipeline-root output/run_20260427_001723
```

### 分步执行

```bash
# 阶段1：生成测试用例（不需要设备）
python run_dynamic_test.py discover --pipeline-root output/run_20260427_001723

# 阶段2：执行测试（需要连接设备）
python run_dynamic_test.py run --pipeline-root output/run_20260427_001723

# 阶段3：生成报告（不需要设备）
python run_dynamic_test.py report --pipeline-root output/run_20260427_001723
```

### 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--pipeline-root` | Part 1 输出目录（必填） | `output/run_20260427_001723` |
| `--apps` | 指定测试的 APP（逗号分隔） | `--apps Keep,Soul,支付宝` |
| `--rights` | 指定权利类型 | `--rights consent,deletion` |
| `--limit` | 每APP每种权利最多测几条 | `--limit 5` |
| `--dry-run` | **无设备调试模式**，生成模拟结果 | `--dry-run` |
| `--resume` | 断点续跑，跳过已完成的 APP | `--resume` |
| `--verbose` / `-v` | 详细日志 | `-v` |

### 示例

```bash
# 调试模式：无设备，仅测试3个APP各5条
python run_dynamic_test.py all \
    --pipeline-root output/run_20260427_001723 \
    --apps Keep,Soul,支付宝 \
    --limit 5 \
    --dry-run \
    -v

# 真实测试：连接MuMu，仅测同意权
python run_dynamic_test.py all \
    --pipeline-root output/run_20260427_001723 \
    --rights consent \
    --limit 10

# 断点续跑
python run_dynamic_test.py run \
    --pipeline-root output/run_20260427_001723 \
    --resume
```

## 输出

所有输出在 `{pipeline_root}/dynamic_testing/` 目录下：

```
dynamic_testing/
├── manifest.json                  # 运行元数据
├── test_cases.json                # 全部测试用例
├── test_cases_app_ui_only.json    # 仅 app_ui 渠道用例
├── results.json                   # 全部测试结果
├── aggregate_report.json          # 聚合报告（重点查看）
├── usability_scores.csv           # 可用性分数表（Excel可打开）
└── per_app/                       # 单APP报告
    └── Keep/
        ├── consent_result.json
        ├── access_result.json
        ├── deletion_result.json
        └── screenshots/           # 测试截图证据
```

### 聚合报告字段说明

```json
{
  "overall_pass_rate": 0.667,       // 总体通过率
  "overall_avg_score": 63.8,        // 综合可用性得分(0-100)
  "by_right": {                     // 按权利类型
    "consent": {
      "avg_score": 55.3,            // 平均可用性得分
      "avg_findability": 2.73,      // 可寻性(1-5)
      "avg_functionality": 2.67,    // 功能性(1-5)
      "avg_efficiency": 2.67,       // 效率(1-5)
      "avg_consistency": 3.2        // 一致性(1-5)
    }
  },
  "by_app": { "Keep": {...} },      // 按APP
  "app_ranking": [...]              // APP排名
}
```

## 评分体系

四项维度，每项 1-5 分，加权归一化到 0-100：

| 维度 | 权重 | 评价标准 |
|------|------|---------|
| 可寻性 | 30% | 需要几次点击/滚动才能找到目标功能 |
| 功能性 | 35% | 功能是否按隐私政策描述正常工作 |
| 效率 | 20% | 完成操作需要多少步骤 |
| 一致性 | 15% | UI标签是否与隐私政策描述匹配 |

## 测试渠道策略

| 渠道 | 自动化方式 | 覆盖率 |
|------|-----------|--------|
| `app_ui` | Appium UIAutomator2 全自动 | 60-70% |
| `web` | Selenium 网页检测 | 10-15% |
| `email`/`phone` | 记录地址，标记人工复核 | 5-10% |
| `customer_service` | 半自动（验证入口存在） | 10-15% |
| `mail`/`offline`/`unknown` | 跳过并说明原因 | <5% |
