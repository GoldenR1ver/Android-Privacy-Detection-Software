"""Scoring placeholder."""


def score_verification(result: dict) -> dict:
    """Attach score and risk level to comparison result."""
    result = dict(result)
    result["score"] = 0
    result["risk_level"] = "unknown"
    return result
