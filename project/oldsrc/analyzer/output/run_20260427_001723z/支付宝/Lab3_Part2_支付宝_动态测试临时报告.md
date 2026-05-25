# Lab3 Part2 支付宝 动态测试临时报告（Access）

- 生成时间：2026-05-09
- 数据来源：`output/run_20260427_001723/dynamic_testing/per_app/支付宝/access_result.json`
- 测试命令：`python run_dynamic_test.py all --pipeline-root output/run_20260427_001723 --apps 小红书,Keep,支付宝,LOFTER,智联招聘,好大夫医生版,小荷健康,东方财富,世纪佳缘 --rights access --limit 10 -v`

## 总体结果

- 用例总数：3
- PASS：0 | FAIL：2 | SKIP：1 | ERROR：0
- 通过率：0.0%
- 平均可用性分：16.3/100

## 样例明细（统一格式）

- FAIL：`支付宝-access-0180`（未找到应用内入口，无法继续路径）
- FAIL：`支付宝-access-0174`（未找到身份/账单/账户相关入口）
- SKIP：`支付宝-access-0009`（渠道信息不足，需人工核验）

## 结论

- 当前失败主要集中在首步入口命中不足，自动化验证覆盖受限。
