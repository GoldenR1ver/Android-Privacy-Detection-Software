# Lab4-2 / Lab4-3 汇报讲稿（2026-06-12 · 15 App）

## 1. 开场（1 min）

各位好。本次汇报 **Lab4-2 同意权** 与 **Lab4-3 访问权**，样本扩展为 **15 款 App**。动态验证策略升级为 **mitm 网络主轨 + Appium UI**，完成 before/after 隐私开关对比。

---

## 2. 方法（2 min）

- Stage3：`--dynamic-track lab42 --real-run`，同步 Appium 导航与 mitmproxy 抓包
- BEFORE：开关开启 → 真机手动关闭隐私开关 → AFTER
- **主判定 track3_network**：Tracker / PII 网络字段 before vs after
- 辅助：UI 对比（关键词、tap 稳定性）
- Lab4-1 受 mitm 代理影响，本汇报以 **Lab4-2/3** 为主

---

## 3. Lab4-2 结果（4 min）

**总览**：15 App 全部完成；**pass 3**（Keep、支付宝、韭研公社）；**review_needed 12**；**7 款**关开关后 Tracker 仍传输。

**强案例（汇报重点）**：

- **钓鱼人**：Umeng、Bugly、百度广告关后仍传（82→83 条流）
- **好大夫医生版**：Umeng×6、百度、Alipay SDK 不变
- **快看漫画**：Umeng/百度/个推，before 162 / after 206 条
- **世纪佳缘**：百度广告、Bugly 仍传

**正面对照**：**Keep** — 关后 PII 网络字段（location/mac 等）停止。

**弱证据**：支付宝、韭研公社流量极少（1 条），pass 但不宜过度解读。

（展示 fig_lab42_status、fig_lab42_tracker_leaks、fig_lab42_flows）

---

## 4. Lab4-3（2 min）

- **9/15** App 已有 `lab4_3_result.json`（副本 + manual_copy_audit）
- 流程：用户导出个人信息副本 → 与政策 declared_pii 字段映射 → 人工审核
- **待补 6 款**：杉果游戏、小黑盒、韭研公社、咪噜游戏、钓鱼人、快看漫画 副本导出

（展示 fig_lab43_copy_vs_declared）

---

## 5. 局限与后续（1 min）

- mitm 代理影响 Lab4-1 动态观测，不作为本次主结论
- 支付宝 H5 容器仍限制 Appium 导航
- 后续：补 Lab4-3 副本、低流量 App 加长 mitm 窗口

---

## 6. 收尾

完成 15 App Lab4-2 全量实验，网络证据支撑「关隐私开关后仍有 SDK 传输」结论。感谢聆听，欢迎提问。
