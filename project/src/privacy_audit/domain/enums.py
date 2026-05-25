"""Domain enums used by unified schema."""

from enum import Enum


class RightType(str, Enum):
    INFORMED = "informed"
    CONSENT = "consent"
    ACCESS = "access"
    CORRECTION = "correction"
    DELETION = "deletion"


class EvidenceSource(str, Enum):
    POLICY = "policy"
    STATIC = "static"
    DYNAMIC = "dynamic"
    NETWORK = "network"
    HUMAN_REVIEW = "human_review"


class Verdict(str, Enum):
    CONSISTENT = "consistent"
    INCONSISTENT_HIGH_RISK = "inconsistent_high_risk"
    DECLARED_NOT_OBSERVED = "declared_not_observed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
