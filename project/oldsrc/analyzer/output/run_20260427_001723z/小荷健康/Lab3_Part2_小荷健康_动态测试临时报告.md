# Lab3 Part2 小荷健康 动态测试临时报告（Access）

- 生成时间：2026-05-09
- 数据来源：`output/run_20260427_001723/dynamic_testing/per_app/小荷健康/access_result.json`
- 测试命令：`python run_dynamic_test.py all --pipeline-root output/run_20260427_001723 --apps 小红书,Keep,支付宝,LOFTER,智联招聘,好大夫医生版,小荷健康,东方财富,世纪佳缘 --rights access --limit 10 -v`

## 总体结果

- 用例总数：1
- PASS：0 | FAIL：0 | SKIP：1 | ERROR：0
- 通过率：0.0%
- 平均可用性分：35.0/100

## 样例明细（统一格式）

- SKIP：`小荷健康-access-0244`（客服/联系渠道类验证，需人工核验）

## 结论

- 本轮小荷健康为人工渠道验证样本，自动化结果暂不充分。
