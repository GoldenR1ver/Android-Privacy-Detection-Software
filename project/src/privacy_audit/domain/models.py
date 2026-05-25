"""Unified domain models for Lab4 pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from privacy_audit.domain.enums import EvidenceSource, RightType, Verdict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvidenceItem:
    """Atomic evidence record from one source."""

    app_name: str
    right_type: RightType
    pii_type: str
    source: EvidenceSource
    signal: str
    value: str
    confidence: float = 1.0
    artifact_path: str = ""
    timestamp_utc: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RightClaim:
    """Policy-side claim for a privacy right and pii type."""

    app_name: str
    right_type: RightType
    pii_type: str
    declared: bool
    claim_text: str
    policy_doc_id: str
    sentence_index: int | None = None
    confidence: float = 1.0
    claim_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Final verification item produced by compare engine."""

    app_name: str
    right_type: RightType
    pii_type: str
    verdict: Verdict
    risk_level: str
    reason: str
    evidence_ids: list[str] = field(default_factory=list)
    generated_at_utc: str = field(default_factory=_utc_now_iso)
    result_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
