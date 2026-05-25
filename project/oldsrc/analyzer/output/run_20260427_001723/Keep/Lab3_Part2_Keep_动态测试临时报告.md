# Lab3 Part2 Keep 动态测试临时报告（Access）

- 生成时间：2026-05-09 00:05
- 数据来源：`output/run_20260427_001723/dynamic_testing/per_app/Keep/access_result.json`
- 测试命令：`python run_dynamic_test.py all --pipeline-root output/run_20260427_001723 --apps 小红书,Keep,支付宝 --rights access --limit 10 -v`

## 总体结果

- 用例总数：9
- PASS：2 | FAIL：4 | SKIP：3 | ERROR：0
- 通过率：22.2%
- 平均可用性分：42.6/100

## 通过用例（关键）

- `Keep-access-0231`：PASS（`我 -> 设置 -> 账号与安全`）
- `Keep-access-0233`：PASS（`我 -> 设置 -> 消息与提醒`）

## 失败与跳过概览

- 失败样例：
  - `Keep-access-0232`：进入隐私页，但功能判定未通过。
  - `Keep-access-0230`：进入设置后未命中 `个人资料`，最终落在第三方 SDK 列表页。
  - `Keep-access-0234`：路径命中后末屏无有效文本，无法完成验证。
- 跳过样例：
  - `Keep-access-0015`：未知渠道（`unknown`），需人工核验。
  - `Keep-access-0239`：邮件渠道（`email`），需人工发信验证。
  - `Keep-access-0293`：客服渠道（`customer_service`），需人工客服链路验证。

## 结论

- 本次 Keep Access 较之前有提升（出现可稳定通过的设置链路）。
- 仍需继续提高对“路径抽象词/政策句子”到真实菜单节点的映射能力，并分离自动化与人工渠道统计结论。
