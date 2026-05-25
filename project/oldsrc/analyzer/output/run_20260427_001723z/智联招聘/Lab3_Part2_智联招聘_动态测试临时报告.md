# Lab3 Part2 智联招聘 动态测试临时报告（Access）

- 生成时间：2026-05-09
- 数据来源：`output/run_20260427_001723/dynamic_testing/per_app/智联招聘/access_result.json`
- 测试命令：`python run_dynamic_test.py all --pipeline-root output/run_20260427_001723 --apps 小红书,Keep,支付宝,LOFTER,智联招聘,好大夫医生版,小荷健康,东方财富,世纪佳缘 --rights access --limit 10 -v`

## 总体结果

- 用例总数：10
- PASS：0 | FAIL：4 | SKIP：6 | ERROR：0
- 通过率：0.0%
- 平均可用性分：23.4/100

## 样例明细（统一格式）

- FAIL：`智联招聘-access-0278`（未进入隐私中心，导航未达目标页）
- FAIL：`智联招聘-access-0280`（未找到导出入口）
- SKIP：`智联招聘-access-0067`（渠道信息不足，需人工核验）
- SKIP：`智联招聘-access-0236`（需人工确认网页渠道）

## 结论

- 智联招聘当前自动化覆盖受入口可达性和渠道类型限制较大。
