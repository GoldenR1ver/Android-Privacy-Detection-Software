# Lab4 系统设计与实验汇报 — 口述稿

> 对应 `lab4_slide.tex` / `系统设计Lab4.pptx`  
> 数据批次：`exp_stage3_20260525` · 详见 `src/output/result/result_20260525.md`

---

## 1. 开场（约 30 秒）

各位老师、同学好。我们是「我的刀盾队」。今天汇报的是 **Android 隐私检测系统 Lab4 设计与实验**，核心是把隐私政策里的三项用户权利——**知情权、同意权、访问权**——做成可复现、可溯源的自动化审计流水线，并在 9 款主流 App 上跑通批量实验。

---

## 2. 背景与目标（约 1 分钟）

早期开发计划里，系统更偏向 **LLM 直接给结论**。经过阶段 0 到阶段 1 的衔接，我们现在的实现是 **证据优先**：先把政策声明、静态 APK、真机动态、历史 oldsrc、法律知识库合并成证据包，再用规则引擎出 verdict，每条都能追到 `sources`。

Lab4 三个维度：  
- **Lab4-1 知情权**：政策写的 PII 类型，和技术观测是否一致——这是我们**主结论**来源。  
- **Lab4-2 同意权**：关隐私开关前后，运行时是否还采集——本批**没有有效实证**。  
- **Lab4-3 访问权**：声明、运行时和「副本」字段是否对得上——程序 **9/9 成功**，但覆盖率都是 0，要说明数据语义。

---

## 3. 系统架构（约 1.5 分钟）

请看架构图：输入是 `input/pii_text` 下的政策文本，可选 APK；**Stage2** 用 Exodus 和 Androguard 出静态证据；**Stage3** 用 Appium 按政策路径点 UI，同时挂 Frida；然后 **evidence_loader** 把 static、dynamic、oldsrc、legacy 合并，**privacy_knowledgebase** 补法律矩阵里的 `feasible_evidence`；最后 **lab4-1/2/3-audit** 和 **batch** 出报告。

统一命令是 `python -m privacy_audit.cli`，工作目录在 `project/src`。这和 README、程序管线文档一致，但和最早「以 LLM 为主」的 plan 已经有明显分工变化。

---

## 4. 实验设计（约 1 分钟）

环境是 Windows + MuMu + Appium 4723 + Frida。样本 9 个 App，列表在 `apps_lab4_batch.txt`。Stage3 产物在 `stage3_dynamic_real/<App>/`，每个都有 `dynamic_evidence.json`。

**必须主动说明的一点**：9 个 App 的 `frida_runtime_events` 都是 0，只有 hook 加载登记。所以 Lab4-1 里「观测到的类型」主要来自静态、oldsrc、KB，**不能**在答辩里说成「这次 Frida 全抓到了」。

UI 适配上：世纪佳缘底部 Tab 是带空格的「 我 」，要点最右边 bounds；小荷设置是右上角**最右侧**齿轮，左边是扫一扫；支付宝要精确「我的」。

---

## 5. Lab4-1 结果（约 2 分钟）

批量 **9/9 成功**。全样本 verdict 合计：**consistent 90 条**，**inconsistent_high_risk 41 条**，**declared_not_observed 6 条**。

图表上能看到：很多 App **观测类型数大于声明类型数**，堆叠图里红色「高风险不一致」不少——含义是：技术或知识库侧有信号，但政策没写全。这是合规风险点，也是我们规则引擎要突出的。

分 App 表：Keep、小红书等声明约 11–12 类，观测约 18 类，一致约 10 条、高风险约 8 条。答辩可以挑一个 App 打开 `*_lab4_1_report.json`，指一条 `inconsistent_high_risk` 的 `sources` 字段。

---

## 6. Lab4-2 / Lab4-3（约 1 分钟）

**Lab4-2**：设计是对比开关前后 Frida 命中；实际 after 侧多是 oldsrc 推导，又没有 runtime 事件，所以 **9/9 failed**，不是 App 合规，是**我们证据不足**。

**Lab4-3**：batch 命令用 `--dynamic-dir`，不是 `--dynamic-evidence-dir`。流水线 9/9 成功，但 `coverage_rate` 全是 0，`lab4_3_copy` 来自 oldsrc 的可用性测试 JSON，**不是**用户导出的个人信息副本。汇报要说清：**程序成功 ≠ 访问权合规**。

---

## 7. 总结（约 30 秒）

我们完成了 Lab4 工程化流水线和 9 App 知情权审计主实验；UI 自动化和分 App 适配已落地。待加强的是 Frida 运行时、同意权双采、访问权真实副本。详细数字和复现命令在 `result_20260525.md`。谢谢大家。
