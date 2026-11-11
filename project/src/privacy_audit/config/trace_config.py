"""PMA-style privacy trace method list ↔ personal_info.yaml api_targets sync."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_TRACE_LINE_RE = re.compile(
    r"^-tracemethod\s+([^\s#]+)(?:\s+#\s*pii_type\s*=\s*(\w+))?",
    re.IGNORECASE,
)
_COMMENT_PII_RE = re.compile(r"#\s*pii_type\s*=\s*(\w+)", re.IGNORECASE)


def default_trace_methods_file() -> Path:
    return Path(__file__).resolve().parents[2] / "input" / "privacy_trace_methods.txt"


def default_taxonomy_file() -> Path:
    return Path(__file__).resolve().parent / "personal_info.yaml"


def clazz_to_pma_path(clazz: str) -> str:
    """``android.provider.Settings$Secure`` → ``android/provider/Settings$Secure``."""
    return clazz.replace(".", "/")


def pma_path_to_clazz(path: str) -> str:
    """``android/telephony/TelephonyManager`` → ``android.telephony.TelephonyManager``."""
    return path.replace("/", ".")


def target_to_pma_line(pii_type: str, target: dict[str, Any]) -> str | None:
    if target.get("enumerate"):
        return None
    clazz = str(target.get("clazz") or "").strip()
    method = str(target.get("method") or "").strip()
    if not clazz or not method:
        return None
    pma = f"{clazz_to_pma_path(clazz)}.{method}"
    return f"-tracemethod {pma}  # pii_type={pii_type}"


def pma_line_to_target(line: str, default_pii: str = "") -> dict[str, Any] | None:
    match = _TRACE_LINE_RE.match(line.strip())
    if not match:
        return None
    raw_path = match.group(1).strip()
    pii_type = (match.group(2) or default_pii or "").strip()
    comment_match = _COMMENT_PII_RE.search(line)
    if comment_match:
        pii_type = comment_match.group(1)
    if "." not in raw_path:
        return None
    clazz_part, method = raw_path.rsplit(".", 1)
    clazz = pma_path_to_clazz(clazz_part)
    row: dict[str, Any] = {"clazz": clazz, "method": method}
    if pii_type:
        row["pii_type"] = pii_type
    return row


def parse_trace_methods_txt(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-keeppackage") or stripped.startswith("-keepclass"):
            continue
        row = pma_line_to_target(stripped)
        if row:
            rows.append(row)
    return rows


def load_taxonomy_api_targets(taxonomy_file: Path) -> dict[str, list[dict[str, Any]]]:
    from privacy_audit.analyzers.policy.policy_parser import load_personal_info_taxonomy

    taxonomy = load_personal_info_taxonomy(taxonomy_file)
    out: dict[str, list[dict[str, Any]]] = {}
    for pii_type, meta in taxonomy.items():
        if not isinstance(meta, dict):
            continue
        targets = meta.get("api_targets") or []
        if isinstance(targets, list) and targets:
            out[str(pii_type)] = [t for t in targets if isinstance(t, dict)]
    return out


def _target_key(target: dict[str, Any]) -> str:
    if target.get("enumerate"):
        return f"enumerate:{target.get('enumerate')}"
    return f"{target.get('clazz', '')}#{target.get('method', '')}"


def build_txt_from_yaml(taxonomy_file: Path) -> str:
    api_map = load_taxonomy_api_targets(taxonomy_file)
    lines = [
        "# PrivacyMonitorAndroid-style API trace list (R1)",
        "# Format: -tracemethod <package/Class.method>  # pii_type=<taxonomy_key>",
        "# Sync: python scripts/sync_trace_config.py --check",
        "#       python scripts/sync_trace_config.py --write-txt",
        "",
        "-keeppackage privacy_audit/hooks",
        "",
    ]
    for pii_type in sorted(api_map.keys()):
        lines.append(f"# --- {pii_type} ---")
        for target in api_map[pii_type]:
            if target.get("enumerate"):
                pattern = str(target.get("enumerate", ""))
                lines.append(f"# enumerate {pattern}  # pii_type={pii_type}")
                continue
            line = target_to_pma_line(pii_type, target)
            if line:
                lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def check_trace_yaml_consistency(
    trace_file: Path,
    taxonomy_file: Path,
) -> dict[str, Any]:
    """Compare explicit -tracemethod entries with yaml api_targets (clazz+method)."""
    txt_rows = parse_trace_methods_txt(trace_file)
    yaml_map = load_taxonomy_api_targets(taxonomy_file)

    yaml_pairs: dict[str, str] = {}
    for pii_type, targets in yaml_map.items():
        for target in targets:
            if target.get("enumerate"):
                continue
            key = _target_key(target)
            yaml_pairs[key] = pii_type

    txt_pairs: dict[str, str] = {}
    for row in txt_rows:
        key = _target_key(row)
        txt_pairs[key] = str(row.get("pii_type") or "")

    missing_in_txt = sorted(set(yaml_pairs.keys()) - set(txt_pairs.keys()))
    extra_in_txt = sorted(set(txt_pairs.keys()) - set(yaml_pairs.keys()))
    pii_mismatch = [
        {"key": key, "txt": txt_pairs[key], "yaml": yaml_pairs[key]}
        for key in sorted(set(txt_pairs.keys()) & set(yaml_pairs.keys()))
        if txt_pairs[key] and yaml_pairs[key] and txt_pairs[key] != yaml_pairs[key]
    ]

    return {
        "consistent": not missing_in_txt and not extra_in_txt and not pii_mismatch,
        "trace_method_count": len(txt_rows),
        "yaml_explicit_target_count": len(yaml_pairs),
        "missing_in_txt": missing_in_txt,
        "extra_in_txt": extra_in_txt,
        "pii_type_mismatch": pii_mismatch,
    }


def merge_txt_into_yaml(
    trace_file: Path,
    taxonomy_file: Path,
    *,
    write: bool = False,
) -> dict[str, Any]:
    """Add txt-only clazz+method targets into taxonomy api_targets."""
    import yaml

    txt_rows = parse_trace_methods_txt(trace_file)
    raw = yaml.safe_load(taxonomy_file.read_text(encoding="utf-8")) or {}
    added = 0
    for row in txt_rows:
        pii_type = str(row.get("pii_type") or "").strip()
        if not pii_type or pii_type not in raw:
            continue
        entry = raw[pii_type]
        if not isinstance(entry, dict):
            continue
        targets = list(entry.get("api_targets") or [])
        key = _target_key(row)
        existing = {_target_key(t) for t in targets if isinstance(t, dict)}
        if key in existing:
            continue
        targets.append({"clazz": row["clazz"], "method": row["method"]})
        entry["api_targets"] = targets
        added += 1
    if write and added:
        taxonomy_file.write_text(
            yaml.dump(raw, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    return {"added_to_yaml": added, "trace_rows": len(txt_rows)}
