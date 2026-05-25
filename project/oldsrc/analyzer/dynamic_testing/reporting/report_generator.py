"""Report generator — produces per-app JSON reports and aggregate summaries."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict

from ..models import TestResult, TestStatus
from ..scoring.usability_scorer import UsabilityScorer
from ..config import AUTOMATED_CHANNELS

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates test reports: per-app JSON, aggregate JSON, usability CSV."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.scorer = UsabilityScorer()

    def generate(
        self,
        results: list[TestResult],
        apps: list[str] | None = None,
    ) -> dict:
        """Generate all reports. Returns aggregate summary dict."""
        # Score all results
        self.scorer.score_all(results)

        # Per-app reports
        self._generate_per_app_reports(results)

        # Aggregate report
        aggregate = self._generate_aggregate_report(results, apps or [])

        # CSV
        self._generate_usability_csv(results)

        # Manifest
        self._generate_manifest(results, aggregate)

        return aggregate

    def _generate_per_app_reports(self, results: list[TestResult]) -> None:
        """Write per-app, per-right JSON reports."""
        per_app_dir = self.output_root / "per_app"
        per_app_dir.mkdir(parents=True, exist_ok=True)

        # Group by app → right_type
        groups: dict[str, dict[str, list[TestResult]]] = {}
        for r in results:
            groups.setdefault(r.app_name, {}).setdefault(r.right_type, []).append(r)

        for app_name, rights in groups.items():
            app_dir = per_app_dir / app_name
            app_dir.mkdir(parents=True, exist_ok=True)

            for right_type, items in rights.items():
                passed = sum(1 for r in items if r.status == TestStatus.PASS)
                scores = [r.usability_score for r in items]

                report = {
                    "app_name": app_name,
                    "right_type": right_type,
                    "test_cases": len(items),
                    "passed": passed,
                    "failed": sum(1 for r in items if r.status == TestStatus.FAIL),
                    "error": sum(1 for r in items if r.status == TestStatus.ERROR),
                    "skipped": sum(1 for r in items if r.status == TestStatus.SKIP),
                    "pass_rate": round(passed / len(items), 3) if items else 0,
                    "avg_findability": round(
                        sum(r.findability_score for r in items) / len(items), 2
                    ) if items else 0,
                    "avg_functionality": round(
                        sum(r.functionality_score for r in items) / len(items), 2
                    ) if items else 0,
                    "avg_efficiency": round(
                        sum(r.efficiency_score for r in items) / len(items), 2
                    ) if items else 0,
                    "avg_consistency": round(
                        sum(r.consistency_score for r in items) / len(items), 2
                    ) if items else 0,
                    "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
                    "results": [r.to_dict() for r in items],
                }

                filepath = app_dir / f"{right_type}_result.json"
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)

    def _generate_aggregate_report(
        self, results: list[TestResult], apps: list[str]
    ) -> dict:
        """Generate cross-app aggregate report."""
        overall = self.scorer.overall_summary(results)

        # App ranking by avg score
        app_ranking = sorted(
            overall.get("by_app", {}).items(),
            key=lambda x: x[1].get("avg_score", 0),
            reverse=True,
        )
        overall["app_ranking"] = [
            {"rank": i + 1, "app": app, "score": info["avg_score"]}
            for i, (app, info) in enumerate(app_ranking)
        ]

        # Channel distribution
        channels: dict[str, int] = {}
        for r in results:
            channels[r.channel] = channels.get(r.channel, 0) + 1
        overall["channel_distribution"] = channels
        overall["failure_reason_breakdown"] = self._build_failure_reason_breakdown(results)
        overall["automated_summary"] = self._build_automated_summary(results)
        overall["non_automated_summary"] = self._build_non_automated_summary(results)

        filepath = self.output_root / "aggregate_report.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)

        return overall

    def _generate_usability_csv(self, results: list[TestResult]) -> None:
        """Export a flat CSV for spreadsheet analysis."""
        filepath = self.output_root / "usability_scores.csv"
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "case_id", "app_name", "right_type", "channel", "status",
                "findability", "functionality", "efficiency", "consistency",
                "usability_score", "duration_sec", "notes",
            ])
            for r in results:
                writer.writerow([
                    r.case_id, r.app_name, r.right_type, r.channel,
                    r.status.value, r.findability_score, r.functionality_score,
                    r.efficiency_score, r.consistency_score,
                    r.usability_score, r.duration_sec, r.notes,
                ])

    def _generate_manifest(self, results: list[TestResult], aggregate: dict) -> None:
        """Write a lightweight manifest with run metadata."""
        automated = aggregate.get("automated_summary", {})
        manifest = {
            "total_test_cases": len(results),
            "apps_tested": len(set(r.app_name for r in results)),
            "overall_pass_rate": aggregate.get("overall_pass_rate", 0),
            "overall_avg_score": aggregate.get("overall_avg_score", 0),
            "automated_pass_rate": automated.get("pass_rate", 0),
            "automated_avg_score": automated.get("avg_score", 0),
        }
        filepath = self.output_root / "manifest.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _build_automated_summary(self, results: list[TestResult]) -> dict:
        """Summarize app_ui automated test performance separately."""
        automated = [r for r in results if r.channel in AUTOMATED_CHANNELS]
        if not automated:
            return {"total": 0}
        passed = sum(1 for r in automated if r.status == TestStatus.PASS)
        scores = [r.usability_score for r in automated if r.usability_score > 0]
        return {
            "total": len(automated),
            "passed": passed,
            "failed": sum(1 for r in automated if r.status == TestStatus.FAIL),
            "error": sum(1 for r in automated if r.status == TestStatus.ERROR),
            "skipped": sum(1 for r in automated if r.status == TestStatus.SKIP),
            "pass_rate": round(passed / len(automated), 3),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        }

    def _build_non_automated_summary(self, results: list[TestResult]) -> dict:
        """Summarize manual / semi-manual channels separately."""
        non_auto = [r for r in results if r.channel not in AUTOMATED_CHANNELS]
        if not non_auto:
            return {"total": 0}
        by_channel: dict[str, dict] = {}
        for r in non_auto:
            bucket = by_channel.setdefault(
                r.channel,
                {"count": 0, "skipped": 0, "pass": 0, "fail": 0, "error": 0},
            )
            bucket["count"] += 1
            if r.status == TestStatus.SKIP:
                bucket["skipped"] += 1
            elif r.status == TestStatus.PASS:
                bucket["pass"] += 1
            elif r.status == TestStatus.FAIL:
                bucket["fail"] += 1
            else:
                bucket["error"] += 1
        return {
            "total": len(non_auto),
            "by_channel": by_channel,
        }

    def _build_failure_reason_breakdown(self, results: list[TestResult]) -> dict:
        """Summarize layered failure reasons encoded in notes."""
        failed = [r for r in results if r.status == TestStatus.FAIL]
        if not failed:
            return {"total_failed": 0}

        buckets = {
            "product_capability_gap": 0,
            "navigation_not_reached": 0,
            "feature_missing_or_not_hit": 0,
            "other": 0,
        }
        for r in failed:
            notes = (r.notes or "")
            if "归因: 产品缺口" in notes:
                buckets["product_capability_gap"] += 1
            elif "归因: 导航未达目标页" in notes:
                buckets["navigation_not_reached"] += 1
            elif "归因: 功能缺失或未命中导出入口" in notes:
                buckets["feature_missing_or_not_hit"] += 1
            else:
                buckets["other"] += 1
        buckets["total_failed"] = len(failed)
        return buckets
