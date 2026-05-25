#!/usr/bin/env python3
"""Part 2 CLI: Automated Dynamic Testing and Usability Evaluation.

Usage:
    python run_dynamic_test.py discover --pipeline-root <path>    # Generate test cases
    python run_dynamic_test.py run --pipeline-root <path>         # Execute tests
    python run_dynamic_test.py report --pipeline-root <path>      # Generate reports
    python run_dynamic_test.py all --pipeline-root <path>         # Full pipeline

Example:
    python run_dynamic_test.py all \\
        --pipeline-root src/analyzer/output/run_20260427_001723 \\
        --apps Keep,Soul,支付宝 \\
        --limit 5 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path so we can import dynamic_testing
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamic_testing.runner import DynamicTestRunner
from dynamic_testing.config import TARGET_APPS, DEFAULT_LIMIT_PER_APP_PER_RIGHT


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Part 2: Appium Dynamic Testing & Usability Evaluation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── discover ──────────────────────────────────────────────────────
    p_discover = sub.add_parser("discover", help="Generate test cases from Part 1 output")
    p_discover.add_argument(
        "--pipeline-root", required=True,
        help="Path to Part 1 run output directory",
    )
    p_discover.add_argument(
        "--apps", default=None,
        help="Comma-separated target apps (default: all 12 target apps)",
    )
    p_discover.add_argument(
        "--rights", default=None,
        help="Comma-separated: consent,access,deletion (default: all)",
    )
    p_discover.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT_PER_APP_PER_RIGHT,
        help=f"Max test cases per app per right (default: {DEFAULT_LIMIT_PER_APP_PER_RIGHT})",
    )

    # ── run ────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Execute generated test cases")
    p_run.add_argument(
        "--pipeline-root", required=True,
        help="Path to Part 1 run output directory",
    )
    p_run.add_argument(
        "--apps", default=None,
        help="Comma-separated target apps to test",
    )
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="Mock mode — no device needed, generates simulated results",
    )
    p_run.add_argument(
        "--resume", action="store_true",
        help="Skip apps that already have results",
    )

    # ── report ─────────────────────────────────────────────────────────
    p_report = sub.add_parser("report", help="Generate reports from test results")
    p_report.add_argument(
        "--pipeline-root", required=True,
        help="Path to Part 1 run output directory",
    )
    p_report.add_argument(
        "--apps", default=None,
        help="Comma-separated target apps",
    )

    # ── all ────────────────────────────────────────────────────────────
    p_all = sub.add_parser("all", help="Full pipeline: discover + run + report")
    p_all.add_argument(
        "--pipeline-root", required=True,
        help="Path to Part 1 run output directory",
    )
    p_all.add_argument(
        "--apps", default=None,
        help="Comma-separated target apps (default: all 12 target apps)",
    )
    p_all.add_argument(
        "--rights", default=None,
        help="Comma-separated: consent,access,deletion (default: all)",
    )
    p_all.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT_PER_APP_PER_RIGHT,
        help=f"Max test cases per app per right (default: {DEFAULT_LIMIT_PER_APP_PER_RIGHT})",
    )
    p_all.add_argument(
        "--dry-run", action="store_true",
        help="Mock mode — no device needed",
    )
    p_all.add_argument(
        "--resume", action="store_true",
        help="Skip apps that already have results",
    )
    p_all.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    return parser


def parse_list(arg: str | None) -> list[str] | None:
    """Parse a comma-separated string into a list."""
    if not arg:
        return None
    return [s.strip() for s in arg.split(",") if s.strip()]


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=getattr(args, "verbose", False))
    logger = logging.getLogger("run_dynamic_test")

    pipeline_root = Path(args.pipeline_root)
    if not pipeline_root.exists():
        logger.error("Pipeline root does not exist: %s", pipeline_root)
        sys.exit(1)

    runner = DynamicTestRunner(
        pipeline_root=pipeline_root,
        dry_run=getattr(args, "dry_run", False),
    )

    if args.command == "discover":
        cases = runner.discover(
            apps=parse_list(args.apps),
            rights=parse_list(args.rights),
            limit=args.limit,
        )
        logger.info("Discovery complete. %d test cases generated.", len(cases))
        return

    if args.command == "run":
        # Load test cases from discover phase
        cases_path = runner.output_dir / "test_cases.json"
        if not cases_path.exists():
            logger.error(
                "No test_cases.json found. Run 'discover' first:\n"
                "  python run_dynamic_test.py discover --pipeline-root %s",
                pipeline_root,
            )
            sys.exit(1)

        import json
        with open(cases_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        from dynamic_testing.models import TestCase
        cases = [
            TestCase(
                case_id=tc["case_id"],
                app_name=tc["app_name"],
                app_pkg=tc["app_pkg"],
                right_type=tc["right_type"],
                channel=tc["channel"],
                path_text=tc["path_text"],
                dynamic_test_goal=tc["dynamic_test_goal"],
                usability_risk=tc["usability_risk"],
                target_data=tc["target_data"],
                sentence=tc["sentence"],
                priority=tc["priority"],
                requires_login=tc.get("requires_login", True),
                access_copy_type=tc.get("access_copy_type", "unknown"),
            )
            for tc in data["test_cases"]
        ]

        results = runner.execute(
            test_cases=cases,
            apps=parse_list(args.apps),
            resume=args.resume,
        )
        logger.info("Execution complete. %d results.", len(results))
        return

    if args.command == "report":
        # Load existing results
        results_path = runner.output_dir / "results.json"
        if not results_path.exists():
            logger.error(
                "No results.json found. Run 'run' first:\n"
                "  python run_dynamic_test.py run --pipeline-root %s",
                pipeline_root,
            )
            sys.exit(1)

        import json
        from dynamic_testing.models import TestResult, TestStatus
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = [
            TestResult(
                case_id=r["case_id"],
                app_name=r["app_name"],
                right_type=r["right_type"],
                channel=r["channel"],
                status=TestStatus(r["status"]),
                findability_score=r["findability_score"],
                functionality_score=r["functionality_score"],
                efficiency_score=r["efficiency_score"],
                consistency_score=r["consistency_score"],
                usability_score=r.get("usability_score", 0.0),
                steps_taken=r.get("steps_taken", []),
                error_message=r.get("error_message", ""),
                screenshot_paths=r.get("screenshot_paths", []),
                duration_sec=r.get("duration_sec", 0.0),
                notes=r.get("notes", ""),
            )
            for r in data
        ]

        aggregate = runner.report(results, apps=parse_list(args.apps))
        logger.info(
            "Report complete. Overall: %d cases, avg_score=%.1f",
            aggregate.get("total", 0),
            aggregate.get("overall_avg_score", 0),
        )
        return

    if args.command == "all":
        aggregate = runner.run_all(
            apps=parse_list(args.apps),
            rights=parse_list(args.rights),
            limit=args.limit,
            resume=args.resume,
        )
        logger.info("Full pipeline complete!")
        logger.info(
            "Total: %d cases | Pass rate: %.1f%% | Avg score: %.1f/100",
            aggregate.get("total", 0),
            aggregate.get("overall_pass_rate", 0) * 100,
            aggregate.get("overall_avg_score", 0),
        )
        return


if __name__ == "__main__":
    main()
