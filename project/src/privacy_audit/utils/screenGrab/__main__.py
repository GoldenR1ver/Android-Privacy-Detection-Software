"""CLI: python -m privacy_audit.utils.screenGrab --label keep_profile"""

from __future__ import annotations

import argparse
import json
import sys

from privacy_audit.utils.screenGrab.capture import capture_emulator_screen, default_output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture emulator UI XML, screenshot, and element summary for debugging.",
    )
    parser.add_argument("--label", default="capture", help="Output file name prefix tag")
    parser.add_argument(
        "--mode",
        choices=("auto", "appium", "adb"),
        default="auto",
        help="auto: Appium if --app-package set, else adb",
    )
    parser.add_argument("--app-package", default="", help="App package for Appium activate")
    parser.add_argument("--appium-url", default="http://127.0.0.1:4723")
    parser.add_argument("--device-name", default="Android")
    parser.add_argument("--adb-device", default="", help="adb -s SERIAL (optional)")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output directory (default: utils/screenGrab/output)",
    )
    args = parser.parse_args()

    out = default_output_dir() if not args.output_dir else __import__("pathlib").Path(args.output_dir)
    result = capture_emulator_screen(
        label=args.label,
        output_dir=out,
        mode=args.mode,
        appium_server_url=args.appium_url,
        app_package=args.app_package,
        device_name=args.device_name,
        adb_device=args.adb_device or None,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "report"}, ensure_ascii=False, indent=2))
    if result.get("ok"):
        print(f"\nSaved under: {result.get('capture_dir')}", file=sys.stderr)
        return 0
    err = result.get("report", {}).get("capture", {}).get("error", "unknown")
    print(f"\nCapture failed: {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
