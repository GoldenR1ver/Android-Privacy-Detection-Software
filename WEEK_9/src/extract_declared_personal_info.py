#!/usr/bin/env python3
"""从既有实验结果中提取 APP 隐私政策“声明将获取”的个人信息项。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

APP_TO_INDUSTRY = {
    "360借条": "金融",
    "东方财富": "金融",
    "中国工商银行": "金融",
    "京东金融": "金融",
    "平安证券": "金融",
    "支付宝": "金融",
    "Hi 医生": "健康",
    "Keep": "健康",
    "京东健康": "健康",
    "好大夫医生版": "健康",
    "小荷健康": "健康",
    "心脏健康研究": "健康",
    "51Job": "求职",
    "58同城": "求职",
    "BOSS": "求职",
    "BOSS直聘": "求职",
    "店长直聘": "求职",
    "智联招聘": "求职",
    "Soul": "社交",
    "小红书": "社交",
    "LOFTER": "社交",
    "Blued极速版": "社交",
    "世纪佳缘": "社交",
}

SYNONYMS = {
    "手机号": "手机号码",
    "联系电话": "手机号码",
    "电话号码": "手机号码",
    "手机": "手机号码",
    "身份证号": "身份证号码",
    "身份证件号码": "身份证号码",
    "身份证": "身份证号码",
    "住址": "地址信息",
    "通信地址": "地址信息",
    "位置信息": "地理位置信息",
    "定位信息": "地理位置信息",
    "位置": "地理位置信息",
    "通讯录": "联系人信息",
    "联系人": "联系人信息",
    "设备标识符": "设备标识信息",
    "设备识别码": "设备标识信息",
    "设备信息": "设备标识信息",
    "oaid": "设备标识信息",
    "android id": "设备标识信息",
    "imei": "设备标识信息",
    "imsi": "设备标识信息",
    "mac地址": "MAC地址",
    "ip": "IP地址",
    "人脸": "人脸信息",
    "指纹": "指纹信息",
    "声纹": "声纹信息",
    "银行卡": "银行卡信息",
    "银行账户": "银行卡信息",
    "账户": "账户信息",
    "账号": "账户信息",
    "订单": "订单信息",
    "交易": "交易信息",
    "搜索": "搜索记录",
    "浏览": "浏览记录",
}

GENERIC_TERMS = {
    "个人信息",
    "个人敏感信息",
    "敏感信息",
    "相关信息",
    "必要信息",
    "上述信息",
    "该等信息",
    "用户信息",
    "数据",
    "内容",
}

PATTERN_TO_CANONICAL = [
    (re.compile(r"手机号码|手机号|联系电话|电话号码"), "手机号码"),
    (re.compile(r"身份证号?码?|身份证件号码"), "身份证号码"),
    (re.compile(r"真实姓名|姓名"), "姓名"),
    (re.compile(r"昵称"), "昵称"),
    (re.compile(r"头像"), "头像"),
    (re.compile(r"性别"), "性别"),
    (re.compile(r"生日|出生日期"), "出生日期"),
    (re.compile(r"年龄"), "年龄"),
    (re.compile(r"位置信息|定位信息|地理位置"), "地理位置信息"),
    (re.compile(r"通讯录|联系人"), "联系人信息"),
    (re.compile(r"设备标识符|设备识别码|OAID|Android\s*ID|IMEI|IMSI"), "设备标识信息"),
    (re.compile(r"MAC地址"), "MAC地址"),
    (re.compile(r"IP地址|\bIP\b"), "IP地址"),
    (re.compile(r"相册|照片|图片"), "相册/图片"),
    (re.compile(r"摄像头"), "摄像头权限"),
    (re.compile(r"麦克风"), "麦克风权限"),
    (re.compile(r"通话记录"), "通话记录"),
    (re.compile(r"短信"), "短信"),
    (re.compile(r"剪贴板"), "剪贴板"),
    (re.compile(r"人脸"), "人脸信息"),
    (re.compile(r"指纹"), "指纹信息"),
    (re.compile(r"声纹"), "声纹信息"),
    (re.compile(r"银行卡|银行账户"), "银行卡信息"),
    (re.compile(r"账户|账号"), "账户信息"),
    (re.compile(r"订单"), "订单信息"),
    (re.compile(r"交易"), "交易信息"),
    (re.compile(r"日志"), "日志信息"),
    (re.compile(r"浏览"), "浏览记录"),
    (re.compile(r"搜索"), "搜索记录"),
]

COLLECTION_VERB_RE = re.compile(
    r"收集|获取|采集|处理|保存|存储|上传|读取|使用|访问|提供"
)

NEGATION_PHRASE_RE = re.compile(
    r"不收集|不会收集|不获取|不会获取|不采集|不会采集|不使用|不会使用|"
    r"不读取|不会读取|不访问|不会访问|不提供|不会提供|无须收集|无需收集"
)

TERM_KEYWORD_RE = re.compile(
    r"信息|号码|账号|账户|地址|位置|记录|权限|照片|图片|视频|音频|指纹|人脸|声纹|"
    r"银行卡|身份证|日志|Cookie|IDFA|IDFV|OAID|IMEI|IMSI|MAC|IP|SSID|BSSID"
)


def normalize_item(raw: str) -> str:
    item = raw.strip().strip("，。；;：:、,.()（）[]【】\"'“”‘’")
    if not item:
        return ""
    lowered = item.lower()
    if lowered in SYNONYMS:
        return SYNONYMS[lowered]
    if item in SYNONYMS:
        return SYNONYMS[item]
    return item


def split_target_data(value: str) -> Iterable[str]:
    for part in re.split(r"[、，,；;。/\\]|(?:以及|及|和|与|或)", value):
        part = normalize_item(part)
        if part:
            yield part


def is_plausible_item(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if token in GENERIC_TERMS:
        return False
    if len(token) > 16:
        return False
    if token.startswith(("上述", "前述", "相关", "其他", "该", "此", "用于", "包括", "如", "可能")):
        return False
    if token.endswith(("服务", "功能", "行为", "情况", "内容", "数据")) and len(token) > 8:
        return False
    if not TERM_KEYWORD_RE.search(token):
        return False
    return True


def has_collection_signal(record: Dict[str, object]) -> bool:
    text = str(record.get("text", "") or "")
    target_data = str(record.get("target_data", "") or "")
    merged = f"{text} {target_data}"
    return bool(COLLECTION_VERB_RE.search(merged))


def is_negated_context(text: str, match_start: int, match_end: int) -> bool:
    left = max(0, match_start - 12)
    right = min(len(text), match_end + 8)
    context = text[left:right]
    return bool(NEGATION_PHRASE_RE.search(context))


def is_positive_collection_context(text: str, match_start: int, match_end: int) -> bool:
    left = max(0, match_start - 24)
    right = min(len(text), match_end + 24)
    context = text[left:right]
    if not COLLECTION_VERB_RE.search(context):
        return False
    if NEGATION_PHRASE_RE.search(context):
        return False
    return True


def extract_items_from_record(record: Dict[str, object]) -> Set[str]:
    if not has_collection_signal(record):
        return set()

    found: Set[str] = set()

    target_data = str(record.get("target_data", "") or "")
    text = str(record.get("text", "") or "")
    for pattern, canonical in PATTERN_TO_CANONICAL:
        text_matches = list(pattern.finditer(text))
        if text_matches:
            has_positive_text = any(
                (not is_negated_context(text, m.start(), m.end()))
                and is_positive_collection_context(text, m.start(), m.end())
                for m in text_matches
            )
            if has_positive_text:
                found.add(canonical)
            continue

        if pattern.search(target_data) and not NEGATION_PHRASE_RE.search(target_data):
            found.add(canonical)

    return {normalize_item(x) for x in found if x and x not in GENERIC_TERMS}


def discover_latest_run(search_root: Path) -> Path:
    runs = [p for p in search_root.glob("run_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"未在 {search_root} 下找到 run_* 目录")
    runs.sort(key=lambda p: p.name)
    return runs[-1]


def find_jsonl_for_app(app_dir: Path) -> Path | None:
    p1 = app_dir / "cluster_analysis" / "sentences_taxonomy_22.jsonl"
    if p1.is_file():
        return p1
    p2 = app_dir / "sentences.jsonl"
    if p2.is_file():
        return p2
    return None


def load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def load_taxonomy_title_map(path: Path) -> Dict[int, str]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return {}
    out: Dict[int, str] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        title = row.get("title")
        if isinstance(rid, int) and isinstance(title, str):
            out[rid] = title
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="提取隐私政策声明将获取的个人信息项")
    parser.add_argument(
        "--search-root",
        default=str(Path(__file__).resolve().parents[2] / "WEEK_7" / "src" / "analyzer" / "output"),
        help="存放 run_* 输出的目录",
    )
    parser.add_argument(
        "--input-run-dir",
        default="",
        help="指定 run_yyyyMMdd_HHmmss 目录；为空时自动取 search-root 下最新 run_*",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "output"),
        help="输出目录",
    )
    parser.add_argument(
        "--taxonomy-items-json",
        default=str(Path(__file__).resolve().parents[2] / "WEEK_7" / "src" / "analyzer" / "ref" / "taxonomy" / "pipl_22_items.json"),
        help="taxonomy 22 项定义 JSON",
    )
    args = parser.parse_args()

    search_root = Path(args.search_root).resolve()
    run_dir = Path(args.input_run_dir).resolve() if args.input_run_dir else discover_latest_run(search_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_title_map = load_taxonomy_title_map(Path(args.taxonomy_items_json).resolve())

    app_to_items: Dict[str, Set[str]] = defaultdict(set)
    item_to_apps: Dict[str, Set[str]] = defaultdict(set)
    item_sentence_hits: Dict[str, int] = defaultdict(int)
    item_examples: Dict[str, List[str]] = defaultdict(list)
    industry_to_apps: Dict[str, Set[str]] = defaultdict(set)
    industry_to_items: Dict[str, Set[str]] = defaultdict(set)
    industry_item_hits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    item_taxonomy_hits: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    app_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
    app_dirs.sort(key=lambda p: p.name)

    for app_dir in app_dirs:
        jsonl_path = find_jsonl_for_app(app_dir)
        if not jsonl_path:
            continue

        app_name = app_dir.name
        for rec in load_jsonl(jsonl_path):
            if rec.get("pii_related") is not True:
                continue

            items = extract_items_from_record(rec)
            if not items:
                continue

            text = str(rec.get("text", "") or "").strip()
            taxonomy_id = rec.get("taxonomy_22_id")
            taxonomy_id = taxonomy_id if isinstance(taxonomy_id, int) else None
            for item in items:
                app_to_items[app_name].add(item)
                item_to_apps[item].add(app_name)
                item_sentence_hits[item] += 1
                if text and len(item_examples[item]) < 3:
                    item_examples[item].append(text)
                if isinstance(taxonomy_id, int) and taxonomy_id > 0:
                    item_taxonomy_hits[item][taxonomy_id] += 1

                industry = APP_TO_INDUSTRY.get(app_name, "未分组")
                industry_to_apps[industry].add(app_name)
                industry_to_items[industry].add(item)
                industry_item_hits[industry][item] += 1

    sorted_items = sorted(item_to_apps.keys(), key=lambda x: (-len(item_to_apps[x]), x))

    csv_path = output_dir / "declared_personal_info_items.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "apps_count", "apps", "sentence_hits", "example_count", "examples"])
        for item in sorted_items:
            apps = sorted(item_to_apps[item])
            examples = item_examples[item]
            writer.writerow(
                [
                    item,
                    len(apps),
                    " | ".join(apps),
                    item_sentence_hits[item],
                    len(examples),
                    " || ".join(examples),
                ]
            )

    json_path = output_dir / "declared_personal_info_items.json"
    json_payload = []
    for item in sorted_items:
        json_payload.append(
            {
                "item": item,
                "apps_count": len(item_to_apps[item]),
                "apps": sorted(item_to_apps[item]),
                "sentence_hits": item_sentence_hits[item],
                "examples": item_examples[item],
            }
        )
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    app_csv_path = output_dir / "app_declared_personal_info.csv"
    with app_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["app", "item_count", "items"])
        for app in sorted(app_to_items.keys()):
            items = sorted(app_to_items[app])
            writer.writerow([app, len(items), " | ".join(items)])

    industry_csv_path = output_dir / "industry_declared_personal_info.csv"
    with industry_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["industry", "app_count", "apps", "item_count", "items"])
        for industry in sorted(industry_to_apps.keys()):
            apps = sorted(industry_to_apps[industry])
            items = sorted(industry_to_items[industry])
            writer.writerow(
                [
                    industry,
                    len(apps),
                    " | ".join(apps),
                    len(items),
                    " | ".join(items),
                ]
            )

    industry_md_path = output_dir / "industry_declared_personal_info_report.md"
    ind_lines: List[str] = []
    ind_lines.append("# 按行业分组的个人信息声明清单（WEEK_9）")
    ind_lines.append("")
    for industry in sorted(industry_to_apps.keys()):
        apps = sorted(industry_to_apps[industry])
        items = sorted(industry_to_items[industry])
        ind_lines.append(f"## {industry}")
        ind_lines.append(f"- APP 数量: {len(apps)}")
        ind_lines.append(f"- APP 列表: {'、'.join(apps)}")
        ind_lines.append(f"- 信息项数量: {len(items)}")
        top_items = sorted(items, key=lambda x: (-industry_item_hits[industry][x], x))[:12]
        ind_lines.append("- 高频信息项: " + "、".join(top_items))
        ind_lines.append("")
    industry_md_path.write_text("\n".join(ind_lines), encoding="utf-8")

    tax_csv_path = output_dir / "item_taxonomy_alignment.csv"
    with tax_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "item",
                "dominant_taxonomy_id",
                "dominant_taxonomy_title",
                "dominant_hits",
                "all_taxonomy_distribution",
            ]
        )
        for item in sorted_items:
            dist = item_taxonomy_hits.get(item, {})
            if not dist:
                writer.writerow([item, "", "", 0, ""])
                continue
            ordered = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
            top_id, top_hits = ordered[0]
            dist_text = " | ".join(
                f"{tid}:{taxonomy_title_map.get(tid, '')}:{hits}" for tid, hits in ordered
            )
            writer.writerow(
                [
                    item,
                    top_id,
                    taxonomy_title_map.get(top_id, ""),
                    top_hits,
                    dist_text,
                ]
            )

    tax_json_path = output_dir / "item_taxonomy_alignment.json"
    tax_payload = []
    for item in sorted_items:
        dist = item_taxonomy_hits.get(item, {})
        ordered = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
        tax_payload.append(
            {
                "item": item,
                "dominant_taxonomy": {
                    "id": ordered[0][0] if ordered else None,
                    "title": taxonomy_title_map.get(ordered[0][0], "") if ordered else "",
                    "hits": ordered[0][1] if ordered else 0,
                },
                "taxonomy_distribution": [
                    {
                        "id": tid,
                        "title": taxonomy_title_map.get(tid, ""),
                        "hits": hits,
                    }
                    for tid, hits in ordered
                ],
            }
        )
    tax_json_path.write_text(json.dumps(tax_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    tax_md_path = output_dir / "item_taxonomy_alignment_report.md"
    tax_lines: List[str] = []
    tax_lines.append("# 个人信息项与 Taxonomy 对齐（WEEK_9）")
    tax_lines.append("")
    tax_lines.append("说明：对每个信息项，统计其命中句子在 taxonomy_22_id 上的分布，并给出主对齐项。")
    tax_lines.append("注意：该结果是句子共现统计，不等同于法律语义上的严格归属。")
    tax_lines.append("")
    for item in sorted_items:
        dist = item_taxonomy_hits.get(item, {})
        ordered = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
        if not ordered:
            tax_lines.append(f"- {item}: 未命中 taxonomy 标签")
            continue
        top_id, top_hits = ordered[0]
        top_title = taxonomy_title_map.get(top_id, "")
        top3 = "；".join(
            f"{tid}-{taxonomy_title_map.get(tid, '')}({hits})" for tid, hits in ordered[:3]
        )
        tax_lines.append(
            f"- {item}: 主对齐 {top_id}-{top_title}（{top_hits}）; Top3={top3}"
        )
    tax_md_path.write_text("\n".join(tax_lines), encoding="utf-8")

    md_path = output_dir / "declared_personal_info_report.md"
    lines: List[str] = []
    lines.append("# APP 隐私政策声明获取的个人信息项（WEEK_9）")
    lines.append("")
    lines.append(f"- 数据来源目录: `{run_dir}`")
    lines.append(f"- 覆盖 APP 数量: {len(app_to_items)}")
    lines.append(f"- 抽取出的信息项数量: {len(sorted_items)}")
    lines.append("")
    lines.append("## 全局高频信息项")
    lines.append("")
    for item in sorted_items[:30]:
        apps = sorted(item_to_apps[item])
        lines.append(
            f"- {item}: 覆盖 {len(apps)} 个 APP, 句子命中 {item_sentence_hits[item]} 次"
        )
    lines.append("")
    lines.append("## 各 APP 声明信息项")
    lines.append("")
    for app in sorted(app_to_items.keys()):
        items = sorted(app_to_items[app])
        lines.append(f"### {app}")
        if not items:
            lines.append("- 未识别到明确项")
        else:
            lines.append("- " + "、".join(items))
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("[DONE] 输出文件：")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {app_csv_path}")
    print(f"- {md_path}")
    print(f"- {industry_csv_path}")
    print(f"- {industry_md_path}")
    print(f"- {tax_csv_path}")
    print(f"- {tax_json_path}")
    print(f"- {tax_md_path}")


if __name__ == "__main__":
    main()
