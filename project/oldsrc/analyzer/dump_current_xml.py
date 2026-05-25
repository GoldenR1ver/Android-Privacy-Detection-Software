"""One-click XML dump for current emulator screen.

Usage:
    python dump_current_xml.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_SERIAL = "127.0.0.1:16384"
REMOTE_XML_PATH = "/sdcard/window_dump.xml"


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump current Android UI XML by adb.")
    parser.add_argument(
        "--serial",
        default=DEFAULT_SERIAL,
        help=f"adb device serial (default: {DEFAULT_SERIAL})",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_xml = temp_dir / f"ui_dump_{ts}.xml"

    print(f"[1/3] adb connect {args.serial}")
    connect = run_cmd(["adb", "connect", args.serial])
    if connect.returncode != 0:
        print(connect.stderr.strip() or connect.stdout.strip())
        print("连接模拟器失败，请确认 adb 可用且端口正确。")
        return 1

    print(f"[2/3] dump xml -> {REMOTE_XML_PATH}")
    dump = run_cmd(
        ["adb", "-s", args.serial, "shell", "uiautomator", "dump", REMOTE_XML_PATH]
    )
    if dump.returncode != 0:
        print(dump.stderr.strip() or dump.stdout.strip())
        print("抓取失败：uiautomator dump 执行失败。")
        return 1

    print(f"[3/3] pull xml -> {local_xml}")
    pull = run_cmd(["adb", "-s", args.serial, "pull", REMOTE_XML_PATH, str(local_xml)])
    if pull.returncode != 0:
        print(pull.stderr.strip() or pull.stdout.strip())
        print("抓取失败：无法拉取 XML 到本地。")
        return 1

    print(f"完成：XML 已保存到 {local_xml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
