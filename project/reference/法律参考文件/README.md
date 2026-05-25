# 法律参考文件说明

本目录用于存放 **PIPL / CCPA / GDPR** 对照与个人信息分类的原始参考材料（法规摘录、对照表、课程讲义等）。

## 与知识库的关系

机器可读的法律个人信息矩阵由以下流程生成，**矩阵内容以 `src/privacy_knowledgebase/input.md` 为准**（该文件已按三部法律列标注 I/E 覆盖与敏感级别）：

```text
reference/法律参考文件/   （人工参考、PDF、条文摘录，可继续补充）
        ↓ 对照整理
src/privacy_knowledgebase/input.md
        ↓ python process_input.py
src/privacy_knowledgebase/knowledge_base/*.json
```

审计运行时 **不直接读取** 本目录下的非结构化文件，而是读取 `knowledge_base/` 下已结构化的 JSON。

## 矩阵遵守原则

1. **敏感个人信息（B 类）** 以 PIPL 第 28 条列举及 `input.md` 中 ⚠️ 标记为准。  
2. **PIPL 未单独列出的类别**（如种族、政治观点）在矩阵中已标注删除/不适用，不得自行加回检测项。  
3. **账户凭证（A_06）** 仅 CCPA 明确为敏感项，PIPL/GDPR 场景下按 `input.md` 备注处理。  
4. **H 类（假名化/匿名化）** 一般不视为「收集个人信息」的观测目标，默认不参与 Lab4-1 采集比对。

请将新增法律材料放入本目录后，运行：

```powershell
cd project/src
python -m privacy_audit.cli kb-build --ingest-pdfs
```

`--ingest-pdfs` 会将 PDF 文本追加到 `input.md` 的「附录：法律参考文件自动摘录」章节（**不自动改写** A–H 矩阵表），再生成 `knowledge_base/*.json` 并镜像到 `output/knowledge_base/`。
