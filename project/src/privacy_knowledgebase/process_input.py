from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SECTION_RE = re.compile(r"^##\s+([A-H])\.\s*(.+)$")
FULLWIDTH_PAREN_RE = re.compile(r"^(.+?)（(.+?)）$")
ASCII_PAREN_RE = re.compile(r"^(.+?)\((.+?)\)$")

CATEGORY_META = {
    "A": {"id": "identifiers", "display_name_zh": "标识符", "display_name_en": "Identifiers", "sensitive": False},
    "B": {"id": "sensitive_personal_information", "display_name_zh": "敏感个人信息", "display_name_en": "Special Categories", "sensitive": True},
    "C": {"id": "digital_activity", "display_name_zh": "数字活动信息", "display_name_en": "Digital Activity", "sensitive": False},
    "D": {"id": "commercial_professional", "display_name_zh": "商业与职业信息", "display_name_en": "Commercial & Professional", "sensitive": False},
    "E": {"id": "physical_sensory", "display_name_zh": "生理与感官信息", "display_name_en": "Physical & Sensory", "sensitive": False},
    "F": {"id": "profiling_inferences", "display_name_zh": "画像与推论", "display_name_en": "Profiling & Inferences", "sensitive": False},
    "G": {"id": "household_information", "display_name_zh": "家庭信息", "display_name_en": "Household Information", "sensitive": False},
    "H": {"id": "special_processing_states", "display_name_zh": "特殊处理状态", "display_name_en": "Special Processing States", "sensitive": False},
}

ANDROID_RULES = [
    {
        "type": "permission",
        "match": [
            "android.permission.ACCESS_FINE_LOCATION",
            "android.permission.ACCESS_COARSE_LOCATION",
        ],
        "collects": ["B_10", "B_11"],
        "collects_categories": ["location"],
        "confidence": 0.95,
        "rationale": "位置权限通常对应精确地理位置和行踪轨迹。",
    },
    {
        "type": "permission",
        "match": ["android.permission.READ_CONTACTS"],
        "collects": ["A_03", "G_01"],
        "collects_categories": ["contact", "household"],
        "confidence": 0.9,
        "rationale": "联系人权限可接触联系方式和家庭成员关系。",
    },
    {
        "type": "permission",
        "match": ["android.permission.READ_PHONE_STATE"],
        "collects": ["A_04", "A_02"],
        "collects_categories": ["identity", "online_identifier"],
        "confidence": 0.9,
        "rationale": "电话状态权限常见于设备标识和在线标识符收集。",
    },
    {
        "type": "permission",
        "match": ["android.permission.CAMERA"],
        "collects": ["B_01", "E_02"],
        "collects_categories": ["biometric", "sensory"],
        "confidence": 0.9,
        "rationale": "相机权限可用于面部、生物识别和图像采集。",
    },
    {
        "type": "permission",
        "match": ["android.permission.RECORD_AUDIO"],
        "collects": ["B_01", "E_02"],
        "collects_categories": ["biometric", "sensory"],
        "confidence": 0.9,
        "rationale": "录音权限可用于声纹与音频内容采集。",
    },
    {
        "type": "permission",
        "match": ["android.permission.READ_CALENDAR"],
        "collects": ["D_05"],
        "collects_categories": ["schedule"],
        "confidence": 0.8,
        "rationale": "日历权限通常接触日程和职业活动。",
    },
    {
        "type": "permission",
        "match": ["android.permission.READ_SMS"],
        "collects": ["B_13"],
        "collects_categories": ["communication"],
        "confidence": 0.85,
        "rationale": "短信权限直接触及私人通信内容。",
    },
    {
        "type": "permission",
        "match": ["android.permission.BODY_SENSORS"],
        "collects": ["B_03"],
        "collects_categories": ["health"],
        "confidence": 0.95,
        "rationale": "身体传感器权限对应健康和生理数据。",
    },
    {
        "type": "api",
        "match": ["LocationManager", "FusedLocationProviderClient", "LocationManager.getLastKnownLocation"],
        "collects": ["B_10", "B_11"],
        "collects_categories": ["location"],
        "confidence": 0.9,
        "rationale": "定位 API 显示地理位置与轨迹采集。",
    },
    {
        "type": "api",
        "match": ["TelephonyManager", "TelephonyManager.getDeviceId", "TelephonyManager.getImei"],
        "collects": ["A_02", "A_04"],
        "collects_categories": ["identity", "online_identifier"],
        "confidence": 0.95,
        "rationale": "电话管理 API 常用于设备标识与身份识别。",
    },
    {
        "type": "api",
        "match": ["AccountManager"],
        "collects": ["A_05"],
        "collects_categories": ["account"],
        "confidence": 0.9,
        "rationale": "账号管理 API 对应账户身份信息。",
    },
    {
        "type": "api",
        "match": ["ContactsContract"],
        "collects": ["A_03"],
        "collects_categories": ["contact"],
        "confidence": 0.9,
        "rationale": "联系人合同 API 对应联系方式。",
    },
    {
        "type": "api",
        "match": ["CameraX", "CameraManager", "FaceDetector"],
        "collects": ["B_01", "E_02"],
        "collects_categories": ["biometric", "sensory"],
        "confidence": 0.85,
        "rationale": "相机与人脸检测 API 对应图像和生物识别处理。",
    },
    {
        "type": "api",
        "match": ["BluetoothAdapter"],
        "collects": ["C_04"],
        "collects_categories": ["device"],
        "confidence": 0.75,
        "rationale": "蓝牙相关 API 常与设备信息和周边设备交互有关。",
    },
    {
        "type": "api",
        "match": ["MediaRecorder"],
        "collects": ["B_13", "E_02"],
        "collects_categories": ["sensory", "communication"],
        "confidence": 0.8,
        "rationale": "录制媒体 API 可触及音频与通信内容。",
    },
    {
        "type": "sdk",
        "match": ["com.google.firebase.analytics", "Firebase Analytics"],
        "collects": ["C_01", "C_02", "C_03", "F_01", "F_02"],
        "collects_categories": ["behavior", "profile"],
        "confidence": 0.7,
        "rationale": "分析 SDK 通常用于行为分析和画像。",
    },
    {
        "type": "sdk",
        "match": ["AppsFlyer"],
        "collects": ["A_04", "F_01", "F_02"],
        "collects_categories": ["advertising", "profile"],
        "confidence": 0.75,
        "rationale": "归因 SDK 常涉及广告标识与画像。",
    },
    {
        "type": "sdk",
        "match": ["Facebook SDK"],
        "collects": ["F_01", "F_02"],
        "collects_categories": ["profile"],
        "confidence": 0.7,
        "rationale": "社交 SDK 常与画像和用户关系数据相关。",
    },
    {
        "type": "sdk",
        "match": ["Bugly"],
        "collects": ["C_04", "A_04"],
        "collects_categories": ["device"],
        "confidence": 0.65,
        "rationale": "崩溃分析 SDK 常收集设备和运行环境信息。",
    },
    {
        "type": "sdk",
        "match": ["Umeng"],
        "collects": ["F_01", "F_02", "C_01", "C_02"],
        "collects_categories": ["profile", "behavior"],
        "confidence": 0.7,
        "rationale": "统计 SDK 常用于画像与行为分析。",
    },
    {
        "type": "network",
        "match": ["imei", "deviceid", "oaid", "android_id", "adid", "lat", "lng", "userid", "cookie", "token", "email", "phone"],
        "collects": ["A_02", "A_03", "A_04", "B_10", "C_04"],
        "collects_categories": ["identity", "contact", "location", "online_identifier"],
        "confidence": 0.6,
        "rationale": "网络参数关键词可直接提示身份、联系方式、位置和在线标识符。",
        "pattern_type": "keyword",
    },
    {
        "type": "permission",
        "match": ["android.permission.READ_PHONE_NUMBERS", "android.permission.READ_CALL_LOG"],
        "collects": ["A_03"],
        "collects_categories": ["contact"],
        "confidence": 0.88,
        "rationale": "读取号码/通话记录权限与联系方式相关。",
    },
    {
        "type": "api",
        "match": ["getLine1Number", "PhoneNumberUtils", "READ_PHONE_NUMBERS"],
        "collects": ["A_03"],
        "collects_categories": ["contact"],
        "confidence": 0.9,
        "rationale": "电话号码读取 API。",
    },
    {
        "type": "api",
        "match": ["Settings.Secure.ANDROID_ID", "ANDROID_ID", "getString"],
        "collects": ["A_04"],
        "collects_categories": ["online_identifier"],
        "confidence": 0.85,
        "rationale": "Android ID 读取相关 API/常量。",
    },
    {
        "type": "api",
        "match": ["ClipboardManager", "getPrimaryClip"],
        "collects": ["B_13"],
        "collects_categories": ["communication"],
        "confidence": 0.85,
        "rationale": "剪贴板访问 API。",
    },
    {
        "type": "api",
        "match": ["WifiInfo.getIpAddress", "InetAddress", "getHostAddress"],
        "collects": ["A_04"],
        "collects_categories": ["online_identifier"],
        "confidence": 0.8,
        "rationale": "IP 地址获取 API。",
    },
    {
        "type": "api",
        "match": ["WifiInfo.getMacAddress", "BluetoothAdapter.getAddress"],
        "collects": ["C_04"],
        "collects_categories": ["device"],
        "confidence": 0.8,
        "rationale": "MAC/蓝牙地址 API。",
    },
    {
        "type": "sdk",
        "match": ["sensorsdata", "SensorsAnalytics", "神策"],
        "collects": ["F_01", "C_01", "C_03"],
        "collects_categories": ["profile", "behavior"],
        "confidence": 0.72,
        "rationale": "行为分析 SDK。",
    },
    {
        "type": "sdk",
        "match": ["TalkingData", "talkingdata"],
        "collects": ["A_04", "F_01"],
        "collects_categories": ["online_identifier", "profile"],
        "confidence": 0.7,
        "rationale": "统计/归因 SDK。",
    },
]


def split_title(text: str) -> tuple[str, str | None]:
    text = text.strip()
    match = FULLWIDTH_PAREN_RE.match(text) or ASCII_PAREN_RE.match(text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, None


def strip_outer_pipes(line: str) -> str:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return value


def split_cells(line: str) -> list[str]:
    return [cell.strip() for cell in strip_outer_pipes(line).split("|")]


def is_separator_row(line: str) -> bool:
    stripped = strip_outer_pipes(line).replace(" ", "")
    if not stripped:
        return False
    return all(ch in "-:|" for ch in stripped)


def normalize_label(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^[⚠️\s]+", "", value)
    return value.strip()


def normalize_sensitivity(text: str | None) -> dict[str, str | None]:
    raw = text.strip() if text else None
    if not raw:
        return {"raw": None, "normalized": None, "conditional_sensitive": False}

    if "敏感" in raw:
        normalized = "sensitive"
    elif "一般" in raw:
        normalized = "general"
    elif "低" in raw:
        normalized = "low"
    else:
        normalized = "unknown"

    conditional_sensitive = "仅在" in raw or "条件" in raw or "照片" in raw
    if raw == "一般（照片仅在用于生物识别时为敏感）":
        normalized = "general"
        conditional_sensitive = True

    return {"raw": raw, "normalized": normalized, "conditional_sensitive": conditional_sensitive}


def parse_coverage(text: str | None) -> dict[str, str | None]:
    raw = text.strip() if text else None
    if not raw or raw in {"-", "–", "—"}:
        return {"status": None}

    status_match = re.match(r"^([IE])(?:（([^）]+)）|\(([^)]+)\))?$", raw)
    if status_match:
        return {"status": status_match.group(1)}

    return {"status": None}


def split_alias_candidates(text: str) -> list[str]:
    cleaned = normalize_label(text)
    if not cleaned:
        return []

    parts = [cleaned]
    base = re.sub(r"[（(][^）)]+[）)]$", "", cleaned).strip()
    if base and base not in parts:
        parts.append(base)

    for chunk in re.split(r"[、，,/+；;]+", cleaned):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.isascii() and len(chunk) < 3:
            continue
        if chunk[0] in "与和及或的":
            continue
        if chunk not in parts:
            parts.append(chunk)
    return parts


def split_examples(text: str | None) -> list[str]:
    if not text:
        return []

    cleaned = text.strip()
    if not cleaned or cleaned in {"-", "–", "—"}:
        return []

    if "的" in cleaned:
        return [cleaned]

    parts = [part.strip() for part in re.split(r"[、，]+", cleaned)]
    result = [part for part in parts if part]
    if not result and cleaned:
        result = [cleaned]
    return result


def slugify(text: str) -> str:
    value = normalize_label(text).lower()
    value = re.sub(r"[（(].*?[）)]", "", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "item"


def row_to_dict(headers: list[str], cells: list[str]) -> dict[str, str]:
    values = cells[:]
    if len(values) < len(headers):
        values.extend([""] * (len(headers) - len(values)))
    elif len(values) > len(headers):
        values = values[: len(headers) - 1] + [" | ".join(values[len(headers) - 1 :])]
    return {headers[index]: values[index].strip() for index in range(len(headers))}


def build_item_record(row: dict[str, Any], section_id: str, section_title_zh: str, section_title_en: str | None) -> dict[str, Any]:
    sensitivity = row["sensitivity"]

    if row["id"] == "B_13":
        sensitivity = {**sensitivity, "conditional_sensitive": True}

    sensitivity_output = {
        "normalized": sensitivity.get("normalized"),
        "conditional_sensitive": sensitivity.get("conditional_sensitive", False),
    }

    return {
        "id": row["id"],
        "slug": slugify(row["label"]),
        "label": row["label"],
        "warning": row["warning"],
        "examples": row.get("examples", []),
        "aliases": split_alias_candidates(row["label"])[1:] + row.get("examples", []),
        "sensitivity": sensitivity_output,
        "source": {
            "section_id": section_id,
            "section_title_zh": section_title_zh,
            "section_title_en": section_title_en,
        },
    }


@dataclass
class ParsedRow:
    id: str
    section_id: str
    section_title_zh: str
    section_title_en: str | None
    label: str
    warning: bool
    examples: list[str] = field(default_factory=list)
    columns: dict[str, str] = field(default_factory=dict)
    pipl: dict[str, str | None] = field(default_factory=dict)
    ccpa: dict[str, str | None] = field(default_factory=dict)
    gdpr: dict[str, str | None] = field(default_factory=dict)
    sensitivity: dict[str, str | None] = field(default_factory=dict)


def parse_input_markdown(source_path: Path) -> dict[str, Any]:
    lines = source_path.read_text(encoding="utf-8").splitlines()

    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None
    pending_header: list[str] | None = None
    row_counter_by_section: dict[str, int] = {}

    for raw_line in lines:
        line = raw_line.rstrip()
        section_match = SECTION_RE.match(line)
        if section_match:
            section_id = section_match.group(1)
            title_zh, title_en = split_title(section_match.group(2))
            current_section = {
                "id": section_id,
                "title_zh": title_zh,
                "title_en": title_en,
                "headers": [],
                "rows": [],
            }
            sections.append(current_section)
            pending_header = None
            row_counter_by_section.setdefault(section_id, 0)
            continue

        if current_section is None:
            continue

        if not line.strip():
            continue

        if not line.lstrip().startswith("|"):
            continue

        if is_separator_row(line):
            continue

        cells = split_cells(line)
        if pending_header is None:
            pending_header = cells
            current_section["headers"] = cells
            continue

        headers = pending_header
        row = row_to_dict(headers, cells)
        row_counter_by_section[current_section["id"]] += 1
        row_id = f"{current_section['id']}_{row_counter_by_section[current_section['id']]:02d}"

        first_header = headers[0] if headers else ""
        label = normalize_label(row.get(first_header, "")) if first_header else ""
        warning = row.get(first_header, "").strip().startswith("⚠️")

        example_column = headers[1] if len(headers) > 1 else None
        sensitivity_column = next((header for header in headers if "敏感级别" in header), None)
        pipl_column = next((header for header in headers if "PIPL" in header), None)
        ccpa_column = next((header for header in headers if "CCPA" in header), None)
        gdpr_column = next((header for header in headers if "GDPR" in header), None)
        examples = split_examples(row.get(example_column, "")) if example_column else []
        sensitivity = normalize_sensitivity(row.get(sensitivity_column)) if sensitivity_column else {"raw": None, "normalized": None}

        parsed_row = ParsedRow(
            id=row_id,
            section_id=current_section["id"],
            section_title_zh=current_section["title_zh"],
            section_title_en=current_section["title_en"],
            label=label,
            warning=warning,
            examples=examples,
            columns=row,
            pipl=parse_coverage(row.get(pipl_column)) if pipl_column else {"raw": None, "status": None, "detail": None},
            ccpa=parse_coverage(row.get(ccpa_column)) if ccpa_column else {"raw": None, "status": None, "detail": None},
            gdpr=parse_coverage(row.get(gdpr_column)) if gdpr_column else {"raw": None, "status": None, "detail": None},
            sensitivity=sensitivity,
        )

        current_section["rows"].append(
            {
                "id": parsed_row.id,
                "label": parsed_row.label,
                "warning": parsed_row.warning,
                "examples": parsed_row.examples,
                "columns": parsed_row.columns,
                "sensitivity": parsed_row.sensitivity,
            }
        )

    return {"sections": sections}


def build_aliases(parsed: dict[str, Any]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}

    for section in parsed["sections"]:
        for row in section["rows"]:
            candidates = split_alias_candidates(row["label"])
            for example in row.get("examples", []):
                candidates.extend(split_alias_candidates(example))

            for candidate in candidates:
                bucket = aliases.setdefault(candidate, [])
                if row["id"] not in bucket:
                    bucket.append(row["id"])

    return aliases


def build_sensitivity(parsed: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    sensitivity: dict[str, dict[str, str | None]] = {}
    for section in parsed["sections"]:
        for row in section["rows"]:
            sensitivity[row["id"]] = row["sensitivity"]
    return sensitivity


def build_android_mapping() -> list[dict[str, Any]]:
    return ANDROID_RULES


def build_risk_rules() -> dict[str, list[str]]:
    return {
        "HIGH": ["B_01", "B_02", "B_03", "B_09", "B_10", "B_11", "B_12"],
        "MEDIUM": ["A_01", "A_02", "A_03", "A_04", "A_05", "A_06", "B_13", "C_01", "C_02", "C_03", "D_01", "D_02", "D_03", "D_04", "D_05", "D_06", "F_01", "F_02", "G_01"],
        "LOW": ["C_04", "E_01", "E_02", "H_01", "H_02", "H_03", "H_04"],
    }


def build_categories(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    categories: list[dict[str, Any]] = []

    for section in parsed["sections"]:
        meta = CATEGORY_META.get(section["id"], {})
        items = [build_item_record(row, section["id"], section["title_zh"], section["title_en"]) for row in section["rows"]]

        categories.append(
            {
                "id": meta.get("id", section["id"].lower()),
                "section_id": section["id"],
                "display_name": {
                    "zh": section["title_zh"],
                    "en": meta.get("display_name_en") or section["title_en"],
                },
                "sensitive": meta.get("sensitive", False),
                "items": items,
            }
        )

    return categories


def build_index(categories: list[dict[str, Any]]) -> dict[str, Any]:
    by_item_id: dict[str, dict[str, str]] = {}
    by_alias: dict[str, list[str]] = defaultdict(list)

    for category in categories:
        for item in category["items"]:
            by_item_id[item["id"]] = {
                "category_id": category["id"],
                "category_section_id": category["section_id"],
                "label": item["label"],
                "slug": item["slug"],
            }
            for alias in item["aliases"]:
                if item["id"] not in by_alias[alias]:
                    by_alias[alias].append(item["id"])

    return {"by_item_id": by_item_id, "by_alias": dict(by_alias)}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Knowledge Base Output",
                "",
                "This directory is generated from input.md.",
                "",
                "- categories.json: normalized legal categories and items",
                "- aliases.json: alias to row id mapping",
                "- sensitivity.json: normalized sensitivity levels by row id",
                "- android_mapping.json: Android evidence mapping rules",
                "- risk_rules.json: risk buckets by item id",
                "",
                "Run process_input.py again after updating input.md.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def generate_outputs(input_path: Path, output_dir: Path) -> dict[str, Path]:
    parsed = parse_input_markdown(input_path)
    categories = build_categories(parsed)
    index = build_index(categories)
    aliases = build_aliases(parsed)
    sensitivity = build_sensitivity(parsed)

    output_dir.mkdir(parents=True, exist_ok=True)

    categories_path = output_dir / "categories.json"
    aliases_path = output_dir / "aliases.json"
    sensitivity_path = output_dir / "sensitivity.json"
    android_mapping_path = output_dir / "android_mapping.json"
    risk_rules_path = output_dir / "risk_rules.json"
    readme_path = output_dir / "README.md"

    write_json(
        categories_path,
        {
            "schema_version": "2.0",
            "source": {
                "input_file": input_path.name,
                "section_count": len(parsed["sections"]),
                "row_count": sum(len(section["rows"]) for section in parsed["sections"]),
            },
            "categories": categories,
            "index": index,
        },
    )
    write_json(aliases_path, {"schema_version": "2.0", "aliases": aliases})
    write_json(sensitivity_path, sensitivity)
    write_json(android_mapping_path, {"schema_version": "1.0", "rules": build_android_mapping()})
    write_json(risk_rules_path, {"schema_version": "1.0", "buckets": build_risk_rules()})
    write_readme(readme_path)

    return {
        "categories": categories_path,
        "aliases": aliases_path,
        "sensitivity": sensitivity_path,
        "android_mapping": android_mapping_path,
        "risk_rules": risk_rules_path,
        "readme": readme_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse input.md into structured knowledge-base JSON files.")
    parser.add_argument("--input", default="input.md", help="Path to the markdown source file.")
    parser.add_argument("--output-dir", default="knowledge_base", help="Directory for generated JSON files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    outputs = generate_outputs(input_path, output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())