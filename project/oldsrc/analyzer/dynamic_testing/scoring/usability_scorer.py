"""Usability scorer — aggregates per-test scores into per-app and cross-app summaries."""

from __future__ import annotations

import statistics
from typing import List, Dict

from ..models import TestResult, TestStatus
from .criteria import composite_score


class UsabilityScorer:
    """Scores a collection of TestResults and produces aggregated scores."""

    def __init__(self, weights: dict | None = None):
        self.weights = weights  # use config defaults if None

    def score_result(self, result: TestResult) -> float:
        """Compute the composite usability score for a single test result."""
        score = composite_score(
            findability=result.findability_score,
            functionality=result.functionality_score,
            efficiency=result.efficiency_score,
            consistency=result.consistency_score,
            weights=self.weights,
        )
        result.usability_score = score
        return score

    def score_all(self, results: List[TestResult]) -> List[TestResult]:
        """Compute composite scores for all results (mutates in place)."""
        for r in results:
            self.score_result(r)
        return results

    def aggregate_by_right(
        self, results: List[TestResult]
    ) -> Dict[str, dict]:
        """Aggregate scores by right_type (consent, access, deletion).

        Returns dict: right_type → {avg, median, min, max, count, pass_rate, ...}
        """
        buckets: dict[str, list[TestResult]] = {"consent": [], "access": [], "deletion": []}
        for r in results:
            if r.right_type in buckets:
                buckets[r.right_type].append(r)

        summary = {}
        for right_type, items in buckets.items():
            if not items:
                summary[right_type] = {"count": 0}
                continue

            passed = sum(1 for r in items if r.status == TestStatus.PASS)
            scores = [r.usability_score for r in items if r.usability_score > 0]

            summary[right_type] = {
                "count": len(items),
                "passed": passed,
                "failed": sum(1 for r in items if r.status == TestStatus.FAIL),
                "error": sum(1 for r in items if r.status == TestStatus.ERROR),
                "skipped": sum(1 for r in items if r.status == TestStatus.SKIP),
                "pass_rate": round(passed / len(items), 3) if items else 0,
                "avg_score": round(statistics.mean(scores), 1) if scores else 0,
                "median_score": round(statistics.median(scores), 1) if scores else 0,
                "min_score": round(min(scores), 1) if scores else 0,
                "max_score": round(max(scores), 1) if scores else 0,
                "avg_findability": round(statistics.mean([r.findability_score for r in items]), 2),
                "avg_functionality": round(statistics.mean([r.functionality_score for r in items]), 2),
                "avg_efficiency": round(statistics.mean([r.efficiency_score for r in items]), 2),
                "avg_consistency": round(statistics.mean([r.consistency_score for r in items]), 2),
            }
        return summary

    def aggregate_by_app(
        self, results: List[TestResult]
    ) -> Dict[str, dict]:
        """Aggregate scores by app_name."""
        buckets: dict[str, list[TestResult]] = {}
        for r in results:
            buckets.setdefault(r.app_name, []).append(r)

        summary = {}
        for app, items in buckets.items():
            passed = sum(1 for r in items if r.status == TestStatus.PASS)
            scores = [r.usability_score for r in items if r.usability_score > 0]

            summary[app] = {
                "count": len(items),
                "passed": passed,
                "failed": sum(1 for r in items if r.status == TestStatus.FAIL),
                "error": sum(1 for r in items if r.status == TestStatus.ERROR),
                "skipped": sum(1 for r in items if r.status == TestStatus.SKIP),
                "pass_rate": round(passed / len(items), 3) if items else 0,
                "avg_score": round(statistics.mean(scores), 1) if scores else 0,
                "median_score": round(statistics.median(scores), 1) if scores else 0,
                "min_score": round(min(scores), 1) if scores else 0,
                "max_score": round(max(scores), 1) if scores else 0,
                "per_right": self.aggregate_by_right(items),
            }
        return summary

    def overall_summary(self, results: List[TestResult]) -> dict:
        """Overall stats across all results."""
        if not results:
            return {"total": 0}

        passed = sum(1 for r in results if r.status == TestStatus.PASS)
        scores = [r.usability_score for r in results if r.usability_score > 0]

        return {
            "total": len(results),
            "passed": passed,
            "failed": sum(1 for r in results if r.status == TestStatus.FAIL),
            "error": sum(1 for r in results if r.status == TestStatus.ERROR),
            "skipped": sum(1 for r in results if r.status == TestStatus.SKIP),
            "overall_pass_rate": round(passed / len(results), 3),
            "overall_avg_score": round(statistics.mean(scores), 1) if scores else 0,
            "by_right": self.aggregate_by_right(results),
            "by_app": self.aggregate_by_app(results),
        }
