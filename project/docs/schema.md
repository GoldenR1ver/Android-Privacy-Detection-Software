# Unified Data Schema (Stage 0)

## Goal

Define a unified schema so every final conclusion can be traced back to at least one concrete evidence field, not only natural language notes.

This schema is the baseline for:
- Lab4-1 (informed right authenticity)
- Later extension to Lab4-2 / Lab4-3

---

## Core Objects

## 1) EvidenceItem

Atomic evidence from one source (`policy/static/dynamic/network/human_review`).

Required fields:
- `evidence_id`: globally unique id
- `app_name`: app name
- `right_type`: `informed|consent|access|correction|deletion`
- `pii_type`: normalized pii key (e.g., `imei`, `location`)
- `source`: evidence source
- `signal`: short machine-readable signal name
- `value`: key observation content
- `timestamp_utc`: evidence time in UTC

Optional fields:
- `confidence` (0~1), `artifact_path`, `metadata`

Example:
```json
{
  "evidence_id": "e1",
  "app_name": "Keep",
  "right_type": "access",
  "pii_type": "location",
  "source": "dynamic",
  "signal": "frida_api_hit",
  "value": "LocationManager.getLastKnownLocation",
  "confidence": 0.92,
  "artifact_path": "output/per_app/Keep/screenshots/xxx.png",
  "timestamp_utc": "2026-05-23T06:30:00+00:00",
  "metadata": {
    "case_id": "Keep-access-0232"
  }
}
```

## 2) RightClaim

Policy-side claim item produced by policy parser.

Required fields:
- `claim_id`, `app_name`, `right_type`, `pii_type`, `declared`, `claim_text`, `policy_doc_id`

Optional fields:
- `sentence_index`, `confidence`, `metadata`

Example:
```json
{
  "claim_id": "c1",
  "app_name": "Keep",
  "right_type": "informed",
  "pii_type": "location",
  "declared": true,
  "claim_text": "我们会收集位置信息用于附近推荐",
  "policy_doc_id": "Keep",
  "sentence_index": 143,
  "confidence": 1.0
}
```

## 3) VerificationResult

Final judgement item from compare engine.

Required fields:
- `result_id`, `app_name`, `right_type`, `pii_type`, `verdict`, `risk_level`, `reason`, `evidence_ids`

Optional fields:
- `generated_at_utc`, `metadata`

Example:
```json
{
  "result_id": "r1",
  "app_name": "Keep",
  "right_type": "informed",
  "pii_type": "location",
  "verdict": "inconsistent_high_risk",
  "risk_level": "high",
  "reason": "Observed location API but no corresponding declaration.",
  "evidence_ids": ["e1", "e2"],
  "generated_at_utc": "2026-05-23T06:31:00+00:00"
}
```

---

## Legacy Inputs Mapping (oldsrc -> new schema)

## A) `oldsrc/.../dynamic_testing/results.json`

Useful legacy fields:
- `case_id`, `app_name`, `right_type`, `status`, `steps_taken`, `notes`, `screenshot_paths`

Mapping:
- one case can generate multiple `EvidenceItem`
- `status` + `notes` -> `metadata` / candidate reason
- `screenshot_paths` -> `artifact_path`

## B) `oldsrc/.../dynamic_testing/test_cases.json`

Useful legacy fields:
- `dynamic_test_goal`, `target_data`, `sentence`, `path_text`, `channel`

Mapping:
- `sentence` -> `RightClaim.claim_text` candidate
- `target_data` helps normalize `pii_type`
- `channel/path_text` -> `EvidenceItem.metadata`

## C) `oldsrc/analyzer/data/main/review_bundle.json` (human labeled)

Useful legacy fields:
- `doc_id`, `sent_index`, `text`, `ai.pii_related`, `human.pii_related`, `human.notes`

Mapping:
- each reviewed sentence can generate:
  - `RightClaim` candidate (from `text`)
  - `EvidenceItem` with `source=human_review`, signal=`manual_label`
- `human.pii_related` has higher priority than `ai.pii_related`

---

## Minimum Acceptance Rule (Stage 0)

For each `VerificationResult`, `evidence_ids` must contain at least 1 item.

This ensures every conclusion is traceable and satisfies the Stage 0 acceptance requirement.

---

## Related Files

- Domain models: `src/privacy_audit/domain/models.py`
- Enums: `src/privacy_audit/domain/enums.py`
- PII taxonomy: `src/privacy_audit/config/personal_info.yaml`
