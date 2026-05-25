"""Parse Android UI hierarchy (Appium page_source XML) for element-level checks."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

# Classes often used for drawer / overflow controls.
_DRAWER_LAYOUT_HINTS = ("DrawerLayout", "NavigationView", "SlidingPaneLayout")
_TRIGGER_CLASS_HINTS = ("ImageButton", "ImageView", "ActionMenuItemView")
_MENU_ENTRY_LABELS = ("设置", "账号与安全", "个人资料", "隐私设置", "消息与提醒")
_TRIGGER_TEXT_OR_DESC = (
    "更多",
    "菜单",
    "menu",
    "drawer",
    "navigation",
    "打开导航",
    "打开侧边栏",
    "侧边栏",
)
# Keep: drawer content uses wide panel (~87% screen); menu row text is not clickable itself.
_KEEP_DRAWER_PANEL_IDS = ("view_main_slide", "listItems")
_KEEP_SIDEBAR_TRIGGER_IDS = ("viewToolbarSidebar",)
_DRAWER_PANEL_X_MAX_RATIO = 0.9


@dataclass
class UiElement:
    tag: str
    text: str
    content_desc: str
    resource_id: str
    clazz: str
    bounds: tuple[int, int, int, int] | None
    clickable: bool
    displayed: bool
    enabled: bool

    @property
    def label(self) -> str:
        return (self.text or self.content_desc or self.resource_id).strip()

    @property
    def center(self) -> tuple[float, float] | None:
        if not self.bounds:
            return None
        x1, y1, x2, y2 = self.bounds
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class SidebarUiState:
    """Element-level sidebar / drawer assessment from one hierarchy snapshot."""

    drawer_open: bool = False
    trigger_present: bool = False
    menu_entries_visible: list[str] = field(default_factory=list)
    drawer_layout_seen: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def sidebar_action_needed(self) -> bool:
        """Need to tap trigger: menu not open yet but path expects sidebar."""
        return self.trigger_present and not self.drawer_open

    @property
    def sidebar_step_satisfied(self) -> bool:
        """Sidebar navigation step goal met (drawer shows settings entries)."""
        return self.drawer_open or bool(self.menu_entries_visible)


def parse_bounds(raw: str) -> tuple[int, int, int, int] | None:
    match = _BOUNDS_RE.search(raw or "")
    if not match:
        return None
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def _attr(node: ET.Element, name: str, default: str = "") -> str:
    return node.attrib.get(name, default) or ""


def _bool_attr(node: ET.Element, name: str) -> bool:
    return _attr(node, name, "false").lower() == "true"


def parse_page_source(page_source: str) -> list[UiElement]:
    if not page_source or not page_source.strip():
        return []
    try:
        root = ET.fromstring(page_source)
    except ET.ParseError:
        return []

    elements: list[UiElement] = []
    for node in root.iter():
        clazz = _attr(node, "class")
        elements.append(
            UiElement(
                tag=node.tag,
                text=_attr(node, "text"),
                content_desc=_attr(node, "content-desc"),
                resource_id=_attr(node, "resource-id"),
                clazz=clazz,
                bounds=parse_bounds(_attr(node, "bounds")),
                clickable=_bool_attr(node, "clickable"),
                displayed=_bool_attr(node, "displayed"),
                enabled=_bool_attr(node, "enabled"),
            )
        )
    return elements


def _screen_size_from_elements(elements: list[UiElement]) -> tuple[int, int]:
    max_x = 0
    max_y = 0
    for el in elements:
        if el.bounds:
            max_x = max(max_x, el.bounds[2])
            max_y = max(max_y, el.bounds[3])
    return max(max_x, 1080), max(max_y, 1920)


def _in_top_band(el: UiElement, height: int, ratio: float = 0.22) -> bool:
    center = el.center
    return center is not None and center[1] <= height * ratio


def _in_left_band(el: UiElement, width: int, ratio: float = 0.38) -> bool:
    center = el.center
    return center is not None and center[0] <= width * ratio


def _find_drawer_panel_bounds(elements: list[UiElement]) -> tuple[int, int, int, int] | None:
    """Keep slide menu container (view_main_slide) when drawer is open."""
    for el in elements:
        if not el.displayed or not el.bounds:
            continue
        if any(rid in el.resource_id for rid in _KEEP_DRAWER_PANEL_IDS):
            if "view_main_slide" in el.resource_id:
                return el.bounds
    for el in elements:
        if not el.displayed or not el.bounds:
            continue
        if "listItems" in el.resource_id:
            return el.bounds
    return None


def _point_in_bounds(x: float, y: float, bounds: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def _in_drawer_panel_region(
    el: UiElement,
    width: int,
    panel_bounds: tuple[int, int, int, int] | None,
) -> bool:
    center = el.center
    if center is None:
        return False
    if panel_bounds and _point_in_bounds(center[0], center[1], panel_bounds):
        return True
    # Keep menu labels sit around x=123..842 on 1080px — wider than old 38% left band.
    return center[0] <= width * _DRAWER_PANEL_X_MAX_RATIO


def _label_matches_menu_entry(label: str) -> str | None:
    normalized = label.strip()
    if not normalized:
        return None
    lower = normalized.lower()
    for entry in _MENU_ENTRY_LABELS:
        if entry in normalized or entry.lower() in lower:
            return entry
    return None


def _is_keep_sidebar_trigger(el: UiElement) -> bool:
    if not el.clickable or not el.displayed:
        return False
    if any(rid in el.resource_id for rid in _KEEP_SIDEBAR_TRIGGER_IDS):
        return True
    desc = el.content_desc.strip()
    return desc in ("打开侧边栏", "更多", "菜单")


def _is_trigger_candidate(el: UiElement, width: int, height: int) -> bool:
    if _is_keep_sidebar_trigger(el):
        return True
    if not el.clickable or not el.displayed:
        return False
    if not _in_top_band(el, height):
        return False
    center = el.center
    if center is None:
        return False
    if center[0] > width * 0.32:
        return False
    label = (el.text + " " + el.content_desc).lower()
    if any(k in label for k in _TRIGGER_TEXT_OR_DESC):
        return True
    if any(h in el.clazz for h in _TRIGGER_CLASS_HINTS):
        if el.bounds:
            w = el.bounds[2] - el.bounds[0]
            h_px = el.bounds[3] - el.bounds[1]
            if 0 < w <= width * 0.2 and 0 < h_px <= height * 0.15:
                return True
    return False


def analyze_sidebar_state(
    page_source: str,
    *,
    screen_width: int | None = None,
    screen_height: int | None = None,
) -> SidebarUiState:
    """
    Distinguish drawer **open** vs **trigger visible** using XML nodes.

    - drawer_open: left panel lists 设置 / 账号与安全 etc.
    - trigger_present: top-left clickable hamburger / 更多 (drawer closed).
    """
    elements = parse_page_source(page_source)
    state = SidebarUiState(evidence={"element_count": len(elements)})
    if not elements:
        return state

    width, height = screen_width or 0, screen_height or 0
    if width <= 0 or height <= 0:
        width, height = _screen_size_from_elements(elements)

    menu_hits: list[str] = []
    trigger_hits: list[str] = []
    drawer_nodes = 0
    panel_bounds = _find_drawer_panel_bounds(elements)
    panel_wide = False
    if panel_bounds:
        panel_wide = (panel_bounds[2] - panel_bounds[0]) >= int(width * 0.5)

    for el in elements:
        if any(h in el.clazz for h in _DRAWER_LAYOUT_HINTS):
            drawer_nodes += 1

        entry = _label_matches_menu_entry(el.text or el.label)
        if entry and el.displayed and _in_drawer_panel_region(el, width, panel_bounds):
            if entry not in menu_hits:
                menu_hits.append(entry)

        if _is_trigger_candidate(el, width, height):
            trigger_hits.append(el.content_desc or el.resource_id or el.clazz)

    state.menu_entries_visible = menu_hits
    state.drawer_layout_seen = drawer_nodes > 0
    state.drawer_open = bool(
        menu_hits
        or (panel_wide and panel_bounds is not None)
        or (panel_bounds is not None and "设置" in menu_hits)
    )
    if len(menu_hits) >= 1 and panel_bounds is not None:
        state.drawer_open = True

    # Drawer open: ignore hamburger still in tree (Keep keeps it clickable).
    state.trigger_present = bool(trigger_hits) and not state.drawer_open

    state.evidence.update(
        {
            "menu_entries": menu_hits,
            "trigger_hits": trigger_hits[:6],
            "drawer_layout_nodes": drawer_nodes,
            "drawer_panel_bounds": list(panel_bounds) if panel_bounds else None,
            "drawer_panel_wide": panel_wide,
            "screen_width": width,
            "screen_height": height,
        }
    )
    return state


def summarize_clickable_labels(
    page_source: str,
    *,
    max_items: int = 40,
) -> list[dict[str, str]]:
    """Compact list for screenGrab JSON (debugging)."""
    rows: list[dict[str, str]] = []
    for el in parse_page_source(page_source):
        if not el.clickable or not el.displayed:
            continue
        label = el.label
        if not label:
            continue
        rows.append(
            {
                "text": el.text,
                "content_desc": el.content_desc,
                "resource_id": el.resource_id,
                "class": el.clazz,
                "bounds": str(el.bounds) if el.bounds else "",
            }
        )
        if len(rows) >= max_items:
            break
    return rows
