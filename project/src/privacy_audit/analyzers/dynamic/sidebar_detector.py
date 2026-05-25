"""Sidebar / drawer detection via UI hierarchy XML (not substring heuristics alone)."""

from __future__ import annotations

from typing import Any

from privacy_audit.utils.ui_hierarchy import SidebarUiState, analyze_sidebar_state


def read_page_source(driver: Any) -> str:
    try:
        return driver.page_source or ""
    except Exception:
        return ""


def get_sidebar_ui_state(driver: Any) -> SidebarUiState:
    source = read_page_source(driver)
    width, height = 0, 0
    try:
        size = driver.get_window_size()
        width = int(size.get("width", 0))
        height = int(size.get("height", 0))
    except Exception:
        pass
    return analyze_sidebar_state(source, screen_width=width, screen_height=height)


def detect_sidebar_trigger_on_screen(driver: Any, appium_by: Any = None) -> bool:
    """True when hamburger / 更多 trigger appears clickable (drawer likely closed)."""
    _ = appium_by
    return get_sidebar_ui_state(driver).trigger_present


def detect_drawer_open(driver: Any, appium_by: Any = None) -> bool:
    """True when drawer menu entries (e.g. 设置) are visible in the left panel."""
    _ = appium_by
    return get_sidebar_ui_state(driver).drawer_open


def detect_sidebar_on_screen(driver: Any, appium_by: Any = None) -> bool:
    """Legacy name: True if either trigger or open drawer is detected."""
    state = get_sidebar_ui_state(driver)
    return state.trigger_present or state.drawer_open


def format_sidebar_state_step(state: SidebarUiState) -> str:
    return (
        f"xml_drawer_open={state.drawer_open} "
        f"xml_trigger={state.trigger_present} "
        f"menu={','.join(state.menu_entries_visible) or '-'}"
    )
