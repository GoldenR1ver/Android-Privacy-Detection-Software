"""
mitmproxy addon: record HTTP(S) flows to JSON for privacy audit (R3).

Run via mitmdump:
  mitmdump -p 8080 -s privacy_audit/analyzers/network/mitm_addon.py
Env:
  PRIVACY_AUDIT_MITM_OUT — absolute output JSON path (default mitm_flows.json in cwd)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mitmproxy import ctx  # type: ignore[import-untyped]


def write_flows_payload(out_path: str | Path, records: list[dict[str, Any]]) -> None:
    """Atomically persist flow records (safe for incremental flush during capture)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "flows": records, "flow_count": len(records)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class FlowRecorder:
    """mitmproxy addon that dumps host/path/body snippets after each response."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        raw = os.environ.get("PRIVACY_AUDIT_MITM_OUT", "mitm_flows.json")
        self.out_path = str(Path(raw).resolve())

    def running(self) -> None:
        """Create output file early so dynamic capture can detect a live recorder."""
        self._flush()

    def response(self, flow: Any) -> None:
        if flow is None or flow.request is None:
            return
        req = flow.request
        host = str(getattr(req, "host", "") or "")
        path = str(getattr(req, "path", "") or "")
        method = str(getattr(req, "method", "") or "")
        scheme = str(getattr(req, "scheme", "") or "https")
        body = ""
        try:
            if req.content:
                body = req.get_text(strict=False) or ""
        except Exception:
            body = ""
        resp = getattr(flow, "response", None)
        status = int(getattr(resp, "status_code", 0) or 0) if resp else 0
        self.records.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "host": host,
                "path": path,
                "method": method,
                "scheme": scheme,
                "request_body": (body or "")[:8000],
                "response_status": status,
            }
        )
        self._flush()

    def done(self) -> None:
        self._flush()
        ctx.log.info(f"FlowRecorder wrote {len(self.records)} flows -> {self.out_path}")

    def _flush(self) -> None:
        try:
            write_flows_payload(self.out_path, self.records)
        except Exception as exc:
            ctx.log.error(f"FlowRecorder write failed: {exc}")


addons = [FlowRecorder()]
