# Lab3 Part2 东方财富 动态测试临时报告（Access）

- 生成时间：2026-05-09
- 数据来源：`output/run_20260427_001723/dynamic_testing/per_app/东方财富/access_result.json`
- 测试命令：`python run_dynamic_test.py all --pipeline-root output/run_20260427_001723 --apps 小红书,Keep,支付宝,LOFTER,智联招聘,好大夫医生版,小荷健康,东方财富,世纪佳缘 --rights access --limit 10 -v`

## 总体结果

- 用例总数：1
- PASS：0 | FAIL：0 | SKIP：1 | ERROR：0
- 通过率：0.0%
- 平均可用性分：9.0/100

## 样例明细（统一格式）

- SKIP：`东方财富-access-0012`（渠道描述抽象，需人工核验）

## 结论

- 东方财富本轮未形成可自动验证链路，建议补充人工核验闭环。
