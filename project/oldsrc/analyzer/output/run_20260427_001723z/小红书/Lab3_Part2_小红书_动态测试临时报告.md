# Lab3 Part2 小红书 动态测试临时报告（Access）

- 生成时间：2026-05-09
- 数据来源：`output/run_20260427_001723/dynamic_testing/per_app/小红书/access_result.json`
- 测试命令：`python run_dynamic_test.py all --pipeline-root output/run_20260427_001723 --apps 小红书,Keep,支付宝,LOFTER,智联招聘,好大夫医生版,小荷健康,东方财富,世纪佳缘 --rights access --limit 10 -v`

## 总体结果

- 用例总数：4
- PASS：2 | FAIL：2 | SKIP：0 | ERROR：0
- 通过率：50.0%
- 平均可用性分：61.5/100

## 样例明细（统一格式）

- PASS：`小红书-access-0256`（隐私设置页面可直接访问）
- PASS：`小红书-access-0254`（我 -> 编辑资料，可查看并修改资料）
- FAIL：`小红书-access-0257`（未找到笔记/收藏/赞过完整入口链路）
- FAIL：`小红书-access-0255`（导航未达目标页面）

## 结论

- 小红书入口可达性已有提升，但深层路径稳定性仍需继续优化。
