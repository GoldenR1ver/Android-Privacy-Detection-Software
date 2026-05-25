"""Chinese path_text parser → Appium UI action executor.

Parses navigation path strings from Part 1 output like:
    "右上角「我」-「设置」-「账号与安全」"
into executable tap sequences on the Android device.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .ui_automator import UIAutomator
from .config import UI_TEXT_ALIASES, APP_LOCATOR_HINTS, APP_PARALLEL_PATH_GROUPS

logger = logging.getLogger(__name__)


@dataclass
class NavStep:
    """A single navigation step parsed from path_text."""

    action: str  # tap | tap_any | scroll_to | back | check
    target: str  # element text to find, or URL, or instruction
    location: str = ""  # top_right | top_left | bottom | center | ""
    alternatives: list[str] = field(default_factory=list)  # for tap_any


class Navigator:
    """Parses Chinese navigation path_text and executes via UIAutomator."""

    # Path separator characters
    SEP_PATTERN = re.compile(r"\s*[-→>/>→]\s*")

    # Extract text within Chinese quotes 「」 or "" or 《》
    QUOTE_PATTERN = re.compile(r"[「《]([^」》]+)[」》]|\"([^\"]+)\"")

    # Location markers in Chinese
    LOCATION_MARKERS = {
        "右上角": "top_right",
        "左上角": "top_left",
        "右下角": "bottom_right",
        "左下角": "bottom_left",
        "顶部": "top",
        "底部": "bottom",
        "页首": "top",
        "页尾": "bottom",
    }

    # Annotation patterns to strip (like "（可选）", "（以实际界面为准）")
    ANNOTATION_PATTERN = re.compile(r"[（(][^）)]*[）)]")

    def __init__(self, ui: UIAutomator, app_name: str = ""):
        self.ui = ui
        self.app_name = app_name

    def set_app_context(self, app_name: str) -> None:
        """Set current app name so app-specific locator hints can be applied."""
        self.app_name = app_name or ""

    def parse_path(self, path_text: str) -> List[NavStep]:
        """Parse a Chinese navigation path string into NavStep list.

        Example:
            "右上角「我」-「设置」-「账号与安全」"
            → [NavStep(tap, "我", top_right),
               NavStep(tap, "设置"),
               NavStep(tap, "账号与安全")]
        """
        if not path_text or not path_text.strip():
            return []

        # Clean annotations like "（可选）"
        cleaned = self.ANNOTATION_PATTERN.sub("", path_text).strip()

        # Split by common separators
        segments = self.SEP_PATTERN.split(cleaned)
        segments = [s.strip() for s in segments if s.strip()]

        steps: list[NavStep] = []

        for seg in segments:
            step = self._parse_segment(seg)
            if step:
                steps.append(step)

        return self._merge_parallel_tap_steps(steps)

    def _parse_segment(self, segment: str) -> Optional[NavStep]:
        """Parse a single segment of a navigation path."""
        # Check for URL pattern
        if re.match(r"^https?://", segment):
            return NavStep(action="check", target=segment)

        # Check for email pattern
        if re.match(r"^[\w.+-]+@[\w-]+\.[\w.-]+$", segment):
            return NavStep(action="check", target=segment)

        # Check for location markers
        location = ""
        text_part = segment
        for marker, loc_name in self.LOCATION_MARKERS.items():
            if marker in segment:
                location = loc_name
                text_part = segment.replace(marker, "").strip()
                break

        # Extract quoted text — use that as the primary target
        quoted_texts = self.QUOTE_PATTERN.findall(text_part)
        if quoted_texts:
            # quoted_texts is list of tuples from regex groups; each has two groups
            for groups in quoted_texts:
                target = groups[0] or groups[1]
                if target.strip():
                    return NavStep(action="tap", target=target.strip(), location=location)

            # Fallback: join all captured texts
            all_texts = []
            for groups in quoted_texts:
                t = groups[0] or groups[1]
                if t.strip():
                    all_texts.append(t.strip())
            return NavStep(action="tap", target=" - ".join(all_texts), location=location)

        # No quotes found — use the raw segment text
        if len(text_part) <= 20 and text_part:
            action = "tap"
            if any(kw in text_part for kw in ["联系", "客服", "邮件", "电话", "邮箱"]):
                action = "check"
            return NavStep(action=action, target=text_part, location=location)

        # Generic long instruction — treat as scroll+check
        if text_part:
            return NavStep(action="scroll_to", target=text_part)

        return None

    def execute(self, driver, steps: List[NavStep]) -> tuple[bool, List[str]]:
        """Execute parsed nav steps sequentially.

        Returns:
            (success: bool, step_log: List[str]) — whether all steps completed.
        """
        step_log: list[str] = []
        if not steps:
            step_log.append("No navigation steps to execute")
            return True, step_log

        for i, step in enumerate(steps):
            step_desc = f"[{i+1}/{len(steps)}] {step.action}: {step.target or step.location}"
            logger.debug("Executing %s", step_desc)

            if step.action == "tap":
                ok, msg = self._execute_tap(step)
                step_log.append(msg)
                if not ok:
                    return False, step_log
            elif step.action == "tap_any":
                before_sig = self._get_page_signature()
                ok_any = False
                chosen = ""
                for candidate in step.alternatives:
                    ok, msg = self._execute_tap(
                        NavStep(action="tap", target=candidate, location=step.location)
                    )
                    if not ok:
                        step_log.append(f"tap_any候选失败: {candidate}")
                        continue
                    changed = self._page_changed(before_sig)
                    if changed:
                        ok_any = True
                        chosen = candidate
                        step_log.append(f"tap_any命中: {msg}（页面已变化）")
                        break
                    step_log.append(f"tap_any候选无跳页: {candidate}，继续尝试下一个")
                if not ok_any:
                    labels = " / ".join(step.alternatives or [step.target])
                    return False, step_log + [f"tap_any → 「{labels}」：均未触发有效页面变化"]
                if chosen:
                    logger.debug("tap_any picked candidate: %s", chosen)
            elif step.action == "scroll_to":
                ok, msg = self._execute_scroll_to(step)
                step_log.append(msg)
                if not ok:
                    return False, step_log
            elif step.action == "check":
                step_log.append(f"check → {step.target}: 记录（非自动化验证）")
            elif step.action == "back":
                self.ui.back()
                step_log.append("执行返回操作")

            time.sleep(0.5)  # Short wait for UI transitions

        return True, step_log

    def _execute_tap(self, step: NavStep) -> tuple[bool, str]:
        """Try to find and tap the target element.

        Uses multiple strategies in order of reliability:
        1. XPath exact text on clickable element (best for bottom nav tabs)
        2. XPath exact text → try click, then try clicking parent
        3. UiAutomator exact text
        4. UiAutomator partial text
        5. Scroll to find
        6. Content description
        """
        target = step.target
        target_candidates = self._expand_target_candidates(target)
        app_hints = self._get_app_hint(target)

        # Keep-specific pattern: "设置" may be inside top-left hamburger menu.
        if app_hints.get("requires_menu_open"):
            opened, open_msg = self._open_app_menu_if_needed()
            if opened:
                logger.debug("Menu opened before tapping %s: %s", target, open_msg)

        # Strategy 0: App-specific resource-id / description hints
        hinted_ids = app_hints.get("ids", [])
        el = self.ui.find_any_by_ids(hinted_ids)
        if el is not None:
            clicked, via_parent = self._click_element_with_parent(el)
            if clicked:
                if via_parent:
                    return True, f"tap → 「{target}」（resource-id+父容器）"
                return True, f"tap → 「{target}」（通过resource-id定位）"

        hinted_descs = [d for d in app_hints.get("descs", []) if d]
        if hinted_descs:
            el = self.ui.find_any_description(hinted_descs, exact=False)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"tap → 「{target}」（App描述别名匹配）"

        # Strategy 0.5: App-specific XPath hints (useful for drawer items with shared ids)
        hinted_xpaths = [x for x in app_hints.get("xpaths", []) if x]
        for xp in hinted_xpaths:
            el = self.ui.find_by_xpath(xp)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"tap → 「{target}」（App XPath提示匹配）"

        # Strategy 1: XPath — clickable element with exact text
        for candidate in target_candidates:
            xpath_clickable = f'//*[@text="{candidate}" and @clickable="true"]'
            el = self.ui.find_by_xpath(xpath_clickable)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"tap → 「{target}」（XPath可点击精确匹配: {candidate}）"

        # Strategy 2: XPath — exact text match, click it or its parent
        for candidate in target_candidates:
            xpath_exact = f'//*[@text="{candidate}"]'
            el = self.ui.find_by_xpath(xpath_exact)
            if el is not None:
                clicked, via_parent = self._click_element_with_parent(el)
                if clicked:
                    if via_parent:
                        return True, f"tap → 「{target}」（点击父容器成功: {candidate}）"
                    return True, f"tap → 「{target}」（XPath精确文本: {candidate}）"

        # Strategy 2b: XPath — content-description match (icons, image buttons)
        for candidate in target_candidates:
            xpath_desc = f'//*[@content-desc="{candidate}"]'
            el = self.ui.find_by_xpath(xpath_desc)
            if el is not None:
                clicked, via_parent = self._click_element_with_parent(el)
                if clicked:
                    if via_parent:
                        return True, f"tap → 「{target}」（content-desc+父容器: {candidate}）"
                    return True, f"tap → 「{target}」（通过content-desc XPath找到: {candidate}）"

        # Strategy 3: UiAutomator exact text
        for candidate in target_candidates:
            el = self.ui.find_by_text(candidate, exact=True)
            if el is not None:
                clicked, via_parent = self._click_element_with_parent(el)
                if clicked:
                    if via_parent:
                        return True, f"tap → 「{target}」（点击父容器成功: {candidate}）"
                    return True, f"tap → 「{target}」（精确匹配: {candidate}）"

        # Strategy 4: UiAutomator partial text
        for candidate in target_candidates:
            el = self.ui.find_by_text(candidate, exact=False)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"tap → 「{target}」（模糊匹配: {candidate}）"

        # Strategy 5: Scroll to text
        scroll_terms = self._extract_scroll_keywords(target_candidates)
        for term in scroll_terms:
            el = self.ui.scroll_to_text_uiautomator(term)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"tap → 「{target}」（滚动后找到: {term}）"

        # Strategy 6: Content description
        for candidate in target_candidates:
            el = self.ui.find_by_description(candidate, exact=False)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"tap → 「{target}」（通过accessibility标签: {candidate}）"

        # Strategy 7: icon-only fallback — tap bottom tab by index
        fallback_index = app_hints.get("bottom_index")
        if fallback_index is None and target in {"我", "我的", "个人", "Me", "Profile"}:
            fallback_index = 4
        if isinstance(fallback_index, int):
            if self.ui.tap_bottom_tab(fallback_index):
                return True, f"tap → 「{target}」（图标底栏兜底: index={fallback_index}）"

        # Strategy 8: Cross-screen fallback
        # 当前页未命中时，尝试切到个人页；若仍失败，再打开侧边栏后重试。
        if self._should_try_cross_screen_fallback(target):
            switched = self._jump_to_personal_view()
            if switched:
                found, how = self._probe_target_on_current_screen(target, target_candidates, app_hints)
                if found:
                    return True, f"tap → 「{target}」（跨页兜底：个人页后命中，{how}）"

            opened, _ = self._open_app_menu_if_needed()
            if opened:
                found, how = self._probe_target_on_current_screen(target, target_candidates, app_hints)
                if found:
                    return True, f"tap → 「{target}」（跨页兜底：侧边栏后命中，{how}）"

        return False, f"tap → 「{target}」：未找到该元素"

    def _click_element_with_parent(self, el) -> tuple[bool, bool]:
        """Try clicking an element, then its clickable parent.

        Returns:
            (clicked, via_parent)
        """
        try:
            el.click()
            return True, False
        except Exception:
            pass

        try:
            parent = self.ui.find_clickable_parent(el)
            if parent is not None:
                parent.click()
                return True, True
        except Exception:
            pass
        return False, False

    def _probe_target_on_current_screen(
        self,
        target: str,
        target_candidates: list[str],
        app_hints: dict,
    ) -> tuple[bool, str]:
        """Probe target quickly on current screen without recursive fallback."""
        hinted_ids = [rid for rid in app_hints.get("ids", []) if rid]
        el = self.ui.find_any_by_ids(hinted_ids)
        if el is not None:
            clicked, _ = self._click_element_with_parent(el)
            if clicked:
                return True, "resource-id"

        hinted_xpaths = [xp for xp in app_hints.get("xpaths", []) if xp]
        for xp in hinted_xpaths:
            el = self.ui.find_by_xpath(xp)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, "app-xpath"

        for candidate in target_candidates:
            el = self.ui.find_by_xpath(f'//*[@text="{candidate}" or @content-desc="{candidate}"]')
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"text/desc={candidate}"

        for candidate in target_candidates:
            el = self.ui.find_by_text(candidate, exact=True)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"exact={candidate}"

        for candidate in target_candidates:
            el = self.ui.find_by_description(candidate, exact=False)
            if el is not None:
                clicked, _ = self._click_element_with_parent(el)
                if clicked:
                    return True, f"desc={candidate}"

        return False, "not-found"

    def _jump_to_personal_view(self) -> bool:
        """Try switching to personal/profile page as fallback."""
        me_candidates = self._expand_target_candidates("我")
        me_hints = self._get_app_hint("我")

        found, _ = self._probe_target_on_current_screen("我", me_candidates, me_hints)
        if found:
            time.sleep(0.3)
            return True

        fallback_index = me_hints.get("bottom_index", 4)
        if isinstance(fallback_index, int) and self.ui.tap_bottom_tab(fallback_index):
            time.sleep(0.3)
            return True

        return False

    def _should_try_cross_screen_fallback(self, target: str) -> bool:
        """Avoid cross-screen fallback on obvious top-level targets."""
        t = (target or "").strip()
        if not t:
            return False
        blocked = {
            "我", "我的", "菜单",
            "首页", "推荐", "热门", "发现", "消息", "购物",
        }
        return t not in blocked

    def _execute_scroll_to(self, step: NavStep) -> tuple[bool, str]:
        """Scroll to find the target."""
        target = step.target
        terms = self._extract_scroll_keywords([target])
        for term in terms:
            el = self.ui.scroll_to_text(term, max_scrolls=8)
            if el is not None:
                return True, f"scroll → 「{target}」：已定位（关键词: {term}）"
        return False, f"scroll → 「{target}」：滚动后仍未找到"

    def _expand_target_candidates(self, target: str) -> list[str]:
        """Expand one target label into aliases to improve recall."""
        candidates: list[str] = []
        base = (target or "").strip()
        if not base:
            return candidates
        candidates.append(base)

        for canonical, aliases in UI_TEXT_ALIASES.items():
            if base == canonical or base in aliases:
                candidates.extend(aliases)
                if canonical not in candidates:
                    candidates.append(canonical)
                break

        # Add app-specific extra text candidates
        app_hints = self._get_app_hint(base)
        candidates.extend(app_hints.get("texts", []))

        uniq: list[str] = []
        for c in candidates:
            c = (c or "").strip()
            if c and c not in uniq:
                uniq.append(c)
        return uniq

    def _get_app_hint(self, target: str) -> dict:
        """Get app-specific locator hints for a target label."""
        app_cfg = APP_LOCATOR_HINTS.get(self.app_name, {})
        if target in app_cfg:
            return app_cfg[target]
        # Match by canonical alias group
        for canonical, aliases in UI_TEXT_ALIASES.items():
            if target == canonical or target in aliases:
                if canonical in app_cfg:
                    return app_cfg[canonical]
        return {}

    def _extract_scroll_keywords(self, texts: list[str]) -> list[str]:
        """Extract shorter keywords for scroll searches from long policy phrases."""
        kws: list[str] = []
        for text in texts:
            raw = (text or "").strip()
            if not raw:
                continue
            kws.append(raw)
            if len(raw) > 10:
                pieces = re.split(r"[，。、；：/\\\-\s]+", raw)
                for piece in pieces:
                    piece = piece.strip()
                    if 2 <= len(piece) <= 10 and piece not in kws:
                        kws.append(piece)
        return kws

    def _open_app_menu_if_needed(self) -> tuple[bool, str]:
        """Try opening an app side menu (e.g., Keep top-left hamburger icon)."""
        menu_hint = self._get_app_hint("菜单")
        menu_ids = [rid for rid in menu_hint.get("ids", []) if rid]
        desc_candidates = menu_hint.get(
            "descs",
            ["更多", "菜单", "更多选项", "侧边栏", "导航菜单"],
        )
        open_check_ids = [rid for rid in menu_hint.get("open_check_ids", []) if rid]
        open_check_texts = [txt for txt in menu_hint.get("open_check_texts", []) if txt]

        # Already opened
        if self._is_menu_open(open_check_ids, open_check_texts):
            return True, "菜单已展开"

        # 0) app-specific id
        el = self.ui.find_any_by_ids(menu_ids)
        if el is not None:
            try:
                el.click()
                time.sleep(0.3)
                if self._is_menu_open(open_check_ids, open_check_texts):
                    return True, "通过resource-id打开菜单"
                return True, "通过resource-id尝试打开菜单"
            except Exception:
                pass

        # 1) accessibility / content-desc
        el = self.ui.find_any_description(desc_candidates, exact=False)
        if el is not None:
            try:
                el.click()
                time.sleep(0.3)
                if self._is_menu_open(open_check_ids, open_check_texts):
                    return True, "通过accessibility描述打开菜单"
                return True, "通过accessibility尝试打开菜单"
            except Exception:
                pass

        # 2) explicit top-left icon tap fallback
        if menu_hint.get("top_left_icon"):
            try:
                size = self.ui.driver.get_window_size()
                x = int(size["width"] * 0.08)
                y = int(size["height"] * 0.08)
                self.ui.tap(x, y)
                time.sleep(0.3)
                if self._is_menu_open(open_check_ids, open_check_texts):
                    return True, "通过左上角图标坐标兜底打开菜单"
                return True, "通过坐标尝试打开菜单"
            except Exception:
                pass

        return False, "未识别到菜单图标"

    def _is_menu_open(self, open_ids: list[str], open_texts: list[str]) -> bool:
        """Check whether side menu has been expanded."""
        for rid in open_ids:
            if self.ui.find_by_id(rid) is not None:
                return True
        for txt in open_texts:
            if self.ui.find_by_text(txt, exact=True) is not None:
                return True
        return False

    def _merge_parallel_tap_steps(self, steps: list[NavStep]) -> list[NavStep]:
        """Collapse sequential peer taps into one tap_any(OR) step for target apps."""
        if not steps or not self.app_name:
            return steps
        rules = APP_PARALLEL_PATH_GROUPS.get(self.app_name, [])
        if not rules:
            return steps

        merged: list[NavStep] = []
        i = 0
        while i < len(steps):
            current = steps[i]
            if current.action != "tap":
                merged.append(current)
                i += 1
                continue

            collapsed = False
            for rule in rules:
                options = set(rule.get("options", []))
                anchor = (rule.get("anchor") or "").strip()
                prev_target = merged[-1].target if merged else ""
                if anchor and prev_target != anchor:
                    continue
                if current.target not in options:
                    continue

                run = [current.target]
                j = i + 1
                while j < len(steps):
                    nxt = steps[j]
                    if nxt.action == "tap" and nxt.target in options and nxt.target not in run:
                        run.append(nxt.target)
                        j += 1
                        continue
                    break
                if len(run) >= 2:
                    merged.append(
                        NavStep(
                            action="tap_any",
                            target=" / ".join(run),
                            alternatives=run,
                        )
                    )
                    i = j
                    collapsed = True
                break

            if not collapsed:
                merged.append(current)
                i += 1

        return merged

    def _get_page_signature(self) -> str:
        """Build a lightweight signature to detect whether page really changed."""
        try:
            src = self.ui.get_page_source() or ""
            texts = re.findall(r'text="([^"]+)"', src)
            descs = re.findall(r'content-desc="([^"]+)"', src)
            ids = re.findall(r'resource-id="([^"]+)"', src)
            items = [t.strip() for t in texts + descs + ids if t and t.strip()]
            if not items:
                return src[:2000]
            uniq = sorted(set(items))
            return "|".join(uniq[:200])
        except Exception:
            return ""

    def _page_changed(self, before_sig: str) -> bool:
        """Compare page signatures after a short UI settle."""
        time.sleep(0.2)
        after_sig = self._get_page_signature()
        if not before_sig and not after_sig:
            return False
        return after_sig != before_sig
