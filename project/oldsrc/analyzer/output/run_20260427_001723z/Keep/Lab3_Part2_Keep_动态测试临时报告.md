# Lab3 Part2 Keep 动态测试临时报告（Access）

- 生成时间：2026-05-09
- 数据来源：`output/run_20260427_001723/dynamic_testing/per_app/Keep/access_result.json`
- 测试命令：`python run_dynamic_test.py all --pipeline-root output/run_20260427_001723 --apps 小红书,Keep,支付宝,LOFTER,智联招聘,好大夫医生版,小荷健康,东方财富,世纪佳缘 --rights access --limit 10 -v`

## 总体结果

- 用例总数：9
- PASS：2 | FAIL：4 | SKIP：3 | ERROR：0
- 通过率：22.2%
- 平均可用性分：40.3/100

## 样例明细（统一格式）

- PASS：`Keep-access-0231`（我 -> 设置 -> 账号与安全）
- PASS：`Keep-access-0233`（我 -> 设置 -> 消息与提醒）
- FAIL：`Keep-access-0232`（已进入隐私页，但行权入口不满足预期）
- FAIL：`Keep-access-0229`（未找到访问/更正相关入口）
- SKIP：`Keep-access-0015`（未知渠道，需人工核验）
- SKIP：`Keep-access-0239`（邮件渠道，需人工核验）

## 结论

- Keep 已出现稳定可达链路，但复杂路径与非自动化渠道仍是主要限制项。
