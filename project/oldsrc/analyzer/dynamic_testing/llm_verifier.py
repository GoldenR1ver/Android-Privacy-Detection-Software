"""LLM-based intelligent test verification using DeepSeek.

Instead of hardcoded keyword matching, sends the dynamic_test_goal +
current screen text to DeepSeek and asks it to determine:
  - Did the test pass?
  - What's the findability/functionality/efficiency/consistency score?
  - Why?

Falls back to keyword-based scoring when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# ── Prompt design ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Android app usability evaluator. Your job is to compare a privacy policy's stated execution method against what is actually visible on the app screen, and output a structured evaluation.

You will receive:
1. dynamic_test_goal: what the privacy policy claims should be verifiable
2. path_text: the navigation path described in the privacy policy
3. right_type: consent (同意权), access (访问权), or deletion (删除权)
4. screen_text: all visible text elements on the current app screen

Output ONLY valid JSON in this exact format:
{
  "pass": true/false,
  "findability": 1-5,
  "functionality": 1-5,
  "efficiency": 1-5,
  "consistency": 1-5,
  "summary": "one-line Chinese explanation"
}

Scoring guide:
- findability (可寻性): 5=entry on main screen, 3=2-3 taps in, 1=not found at all
- functionality (功能性): 5=works exactly as policy describes, 3=partial, 1=broken/missing
- efficiency (效率): 5=1-2 steps, 3=5-7 steps, 1=impossible via app
- consistency (一致性): 5=UI labels match policy text exactly, 3=similar meaning, 1=no match

Return JSON only, no other text."""

USER_PROMPT_TEMPLATE = """Evaluate whether the app screen matches the privacy policy claim.

dynamic_test_goal: {dynamic_test_goal}
path_text: {path_text}
right_type: {right_type}

=== Visible text on current screen ===
{screen_text}
===

Return JSON only:"""


# ── LLM Verifier ─────────────────────────────────────────────────────────────

class LLMVerifier:
    """Calls DeepSeek API to intelligently verify test results."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout_sec: int = 30,
    ):
        # Priority: param → env var → config
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            try:
                from .config import DEEPSEEK_API_KEY
                api_key = DEEPSEEK_API_KEY
            except ImportError:
                pass
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if LLM verification is usable."""
        if self._available is None:
            if not self.api_key:
                logger.warning("DEEPSEEK_API_KEY not set — LLM verification disabled")
                self._available = False
            else:
                self._available = True
        return self._available

    def verify(
        self,
        dynamic_test_goal: str,
        path_text: str,
        right_type: str,
        screen_text: str,
    ) -> dict:
        """Send screen state to LLM for intelligent verification.

        Returns a dict with: pass, findability, functionality, efficiency,
        consistency, summary. Falls back to None values on error.
        """
        if not self.available:
            return self._fallback()

        # Truncate screen text to avoid token limits
        screen_text = screen_text[:3000]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            dynamic_test_goal=dynamic_test_goal,
            path_text=path_text,
            right_type=right_type,
            screen_text=screen_text,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }

        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self.base_url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            content = data["choices"][0]["message"]["content"]
            # Extract JSON from response (may have ```json wrapper)
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                result = json.loads(json_match.group(0))
                logger.info("LLM verification: %s", result.get("summary", ""))
                return {
                    "pass": result.get("pass", False),
                    "findability": result.get("findability", 1),
                    "functionality": result.get("functionality", 1),
                    "efficiency": result.get("efficiency", 1),
                    "consistency": result.get("consistency", 1),
                    "summary": result.get("summary", ""),
                }
            else:
                logger.warning("LLM response had no JSON: %s", content[:200])
                return self._fallback()

        except Exception as e:
            logger.warning("LLM verification failed: %s", e)
            return self._fallback()

    def _fallback(self) -> dict:
        return {
            "pass": None,
            "findability": 0,
            "functionality": 0,
            "efficiency": 0,
            "consistency": 0,
            "summary": "LLM不可用",
        }
