"""mitmproxy traffic capture for the dynamic network track."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from privacy_audit.utils.adb_device import adb_base_cmd, resolve_adb_serial

logger = logging.getLogger("privacy_audit.mitm")

ADDON_PATH = Path(__file__).resolve().parent / "mitm_addon.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_flow_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize HAR / mitm / manual flow dict into a common schema."""
    host = str(raw.get("host") or "")
    if not host and raw.get("url"):
        parsed = urlparse(str(raw["url"]))
        host = parsed.hostname or ""
        path = parsed.path or ""
        if parsed.query:
            path = f"{path}?{parsed.query}"
    else:
        path = str(raw.get("path") or "")
    return {
        "timestamp_utc": str(raw.get("timestamp_utc") or _utc_now()),
        "host": host.lower().strip(),
        "path": path,
        "method": str(raw.get("method") or "GET").upper(),
        "scheme": str(raw.get("scheme") or "https"),
        "request_body": str(raw.get("request_body") or raw.get("body") or "")[:8000],
        "response_status": int(raw.get("response_status") or raw.get("status") or 0),
    }


def load_flows_from_file(path: Path) -> list[dict[str, Any]]:
    """Load flows from JSON bundle or HAR 1.2."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("flows"), list):
        return [normalize_flow_record(row) for row in data["flows"] if isinstance(row, dict)]
    if isinstance(data, dict) and isinstance(data.get("log"), dict):
        entries = data["log"].get("entries") or []
        rows: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            req = entry.get("request") or {}
            url = str(req.get("url") or "")
            parsed = urlparse(url)
            post = req.get("postData") or {}
            body = str(post.get("text") or "")
            rows.append(
                normalize_flow_record(
                    {
                        "timestamp_utc": entry.get("startedDateTime", ""),
                        "host": parsed.hostname or "",
                        "path": (parsed.path or "") + (f"?{parsed.query}" if parsed.query else ""),
                        "method": req.get("method", "GET"),
                        "scheme": parsed.scheme or "https",
                        "request_body": body,
                        "response_status": (entry.get("response") or {}).get("status", 0),
                    }
                )
            )
        return rows
    if isinstance(data, list):
        return [normalize_flow_record(row) for row in data if isinstance(row, dict)]
    return []


def _find_mitmdump() -> str | None:
    return shutil.which("mitmdump") or shutil.which("mitmproxy")


def set_adb_http_proxy(proxy_host: str, port: int, *, adb_serial: str | None = None) -> bool:
    """Point emulator/device HTTP proxy at mitm listener."""
    target = f"{proxy_host}:{port}"
    serial = resolve_adb_serial(explicit=adb_serial)
    try:
        proc = subprocess.run(
            [
                *adb_base_cmd(serial=serial),
                "shell",
                "settings",
                "put",
                "global",
                "http_proxy",
                target,
            ],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
        ok = proc.returncode == 0
        logger.info("adb http_proxy=%s serial=%s ok=%s", target, serial or "(default)", ok)
        if not ok:
            err = (proc.stderr or proc.stdout or "").strip()
            if err:
                logger.warning("adb http_proxy failed: %s", err[:300])
        return ok
    except Exception as exc:
        logger.warning("adb http_proxy failed: %s", exc)
        return False


def clear_adb_http_proxy(*, adb_serial: str | None = None) -> bool:
    try:
        proc = subprocess.run(
            [
                *adb_base_cmd(serial=adb_serial),
                "shell",
                "settings",
                "put",
                "global",
                "http_proxy",
                ":0",
            ],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


class MitmCaptureSession:
    """Start mitmdump, optionally set adb proxy, collect flows JSON on stop."""

    def __init__(
        self,
        flows_out: Path,
        *,
        listen_host: str = "0.0.0.0",
        listen_port: int = 8080,
        adb_proxy_host: str = "10.0.2.2",
        dry_run: bool = False,
        replay_file: Path | None = None,
        set_device_proxy: bool = True,
        adb_serial: str | None = None,
    ) -> None:
        self.flows_out = Path(flows_out).resolve()
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.adb_proxy_host = adb_proxy_host
        self.dry_run = dry_run
        self.replay_file = replay_file
        self.set_device_proxy = set_device_proxy
        self.adb_serial = adb_serial
        self._proc: subprocess.Popen[str] | None = None
        self._started_at: float | None = None
        self._proxy_set_ok = True

    def start(self) -> bool:
        if self.dry_run:
            logger.info("MitmCaptureSession dry_run=True (no mitmdump)")
            self._started_at = time.time()
            return True
        if self.replay_file is not None and self.replay_file.exists():
            logger.info("MitmCaptureSession replay from %s", self.replay_file)
            self._started_at = time.time()
            return True

        mitmdump = _find_mitmdump()
        if mitmdump is None:
            logger.warning("mitmdump not found on PATH; network capture unavailable")
            return False

        self.flows_out.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PRIVACY_AUDIT_MITM_OUT"] = str(self.flows_out)
        cmd = [
            mitmdump,
            "--listen-host",
            self.listen_host,
            "--listen-port",
            str(self.listen_port),
            "-s",
            str(ADDON_PATH),
            "--set",
            "block_global=false",
            "--quiet",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            time.sleep(1.5)
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() if self._proc.stderr else "")[:500]
                logger.error("mitmdump exited early: %s", err)
                self._proc = None
                return False
            if self.set_device_proxy:
                self._proxy_set_ok = set_adb_http_proxy(
                    self.adb_proxy_host,
                    self.listen_port,
                    adb_serial=self.adb_serial,
                )
            self._started_at = time.time()
            logger.info("mitmdump started port=%s out=%s", self.listen_port, self.flows_out)
            return True
        except Exception as exc:
            logger.error("mitmdump start failed: %s", exc)
            self._proc = None
            return False

    def stop(self) -> dict[str, Any]:
        duration = 0.0
        if self._started_at is not None:
            duration = time.time() - self._started_at

        if self.set_device_proxy and not self.dry_run:
            clear_adb_http_proxy(adb_serial=self.adb_serial)

        stderr_tail = ""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=8)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            if self._proc.stderr:
                stderr_tail = self._proc.stderr.read()[-800:]
            self._proc = None

        if not self.dry_run and self.replay_file is None:
            for _ in range(24):
                if self.flows_out.exists():
                    break
                time.sleep(0.25)

        flows: list[dict[str, Any]] = []
        source = "none"
        if self.replay_file is not None and self.replay_file.exists():
            flows = load_flows_from_file(self.replay_file)
            source = f"replay:{self.replay_file}"
        elif self.flows_out.exists():
            flows = load_flows_from_file(self.flows_out)
            source = str(self.flows_out)
        elif self.dry_run:
            source = "dry_run"
            flows = []

        pinning_hint = ""
        if not self.dry_run and not self._proxy_set_ok:
            pinning_hint = "adb_proxy_set_failed"
        elif not self.dry_run and duration >= 8.0 and not flows and self.replay_file is None:
            if self.flows_out.exists():
                pinning_hint = "no_traffic"
            else:
                pinning_hint = "flows_file_missing"
        if "certificate" in stderr_tail.lower() or "ssl" in stderr_tail.lower():
            pinning_hint = "tls_error"

        return {
            "adapter": "mitm_runner",
            "started_at_utc": _utc_now(),
            "duration_sec": round(duration, 2),
            "listen_port": self.listen_port,
            "flows_source": source,
            "flows": flows,
            "flow_count": len(flows),
            "unique_hosts": sorted({f.get("host", "") for f in flows if f.get("host")}),
            "mitm_blocked": pinning_hint or None,
            "stderr_tail": stderr_tail,
            "dry_run": self.dry_run,
        }


def request_query_text(flow: dict[str, Any]) -> str:
    """Join path query string and body for keyword scan."""
    path = str(flow.get("path") or "")
    query = ""
    if "?" in path:
        query = path.split("?", 1)[1]
    body = str(flow.get("request_body") or "")
    if query:
        try:
            flat = []
            for values in parse_qs(query, keep_blank_values=True).values():
                flat.extend(values)
            query = "&".join(flat)
        except Exception:
            pass
    return f"{query}\n{body}".lower()
