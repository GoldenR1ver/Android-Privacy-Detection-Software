"""Helpers to load Frida hook scripts when ART starts late (common on MuMu)."""

from __future__ import annotations

import json
from pathlib import Path

# RPC-driven data-driven monitor (loaded raw — it manages its own Java wait).
PAPI_AGENT = "hook_papi.js"
HOOK_COMMON = "hook_common.js"
_STACK_HELPER_CACHE: str | None = None

# Hooks loaded for every real run regardless of candidate_pii (broad signals).
# The OkHttp network hook backs consent (still uploading?) and access
# (access / taint: did collected PII leave the device?).
GLOBAL_HOOKS: list[str] = ["hook_network_okhttp.js"]

# Excluded from frida runtime_event_count / runtime hit extraction.
FRIDA_NON_RUNTIME_API_NAMES: frozenset[str] = frozenset(
    {
        "",
        "hook_loaded",
        "hook_error",
        "java_active",
        "java_ready",
        "java_timeout",
        "papi_ready",
    }
)


def is_frida_runtime_api(api_name: str) -> bool:
    return str(api_name or "") not in FRIDA_NON_RUNTIME_API_NAMES

JAVA_READY_PROBE = r"""
(function () {
  var attempts = 0;
  function ready() {
    attempts += 1;
    if (typeof Java === "undefined" || !Java.available) {
      if (attempts < 80) {
        setTimeout(ready, 250);
      } else {
        send({ api_name: "java_timeout", pii_type: "frida", hook_file: "probe" });
      }
      return;
    }
    Java.perform(function () {
      send({ api_name: "java_ready", pii_type: "frida", hook_file: "probe" });
    });
  }
  ready();
})();
"""

_JAVA_CALL = "globalThis.__frida_java_install"

_JAVA_INSTALLER = r"""
(function () {
  var g = typeof globalThis !== "undefined" ? globalThis : this;
  if (g.__frida_java_install) {
    return;
  }
  g.__frida_java_install = function (fn, hookFile) {
    hookFile = hookFile || "hook";
    function wait() {
      if (typeof Java === "undefined" || !Java.available) {
        setTimeout(wait, 200);
        return;
      }
      Java.perform(function () {
        send({
          api_name: "java_active",
          pii_type: "frida",
          hook_file: hookFile,
          args_summary: "Java.perform entered",
          stack_summary: ""
        });
        fn();
      });
    }
    wait();
  };
})();
"""


def _stack_helper_source() -> str:
    global _STACK_HELPER_CACHE
    if _STACK_HELPER_CACHE is not None:
        return _STACK_HELPER_CACHE
    hooks_dir = Path(__file__).resolve().parent / "hooks"
    common_path = hooks_dir / HOOK_COMMON
    if common_path.exists():
        _STACK_HELPER_CACHE = common_path.read_text(encoding="utf-8") + "\n"
    else:
        _STACK_HELPER_CACHE = ""
    return _STACK_HELPER_CACHE


def wrap_hook_script(source: str, hook_file: str = "") -> str:
    """Wait for Java.available before running hook bodies that call Java.perform."""
    if "__frida_java_install" in source:
        return _stack_helper_source() + source
    label = json.dumps(
        hook_file.replace("\\", "/").split("/")[-1] if hook_file else "hook"
    )
    close_suffix = "}, " + label + ");"
    marker = "Java.perform(function () {"
    if marker in source:
        start = source.index(marker)
        close = source.rfind("});")
        if close > start:
            inner_start = start + len(marker)
            body = (
                source[:start]
                + _JAVA_CALL
                + "(function () {"
                + source[inner_start:close]
                + close_suffix
                + source[close + 3 :]
            )
            return _stack_helper_source() + _JAVA_INSTALLER + body
    body = source.replace("Java.perform(", _JAVA_CALL + "(", 1)
    return _stack_helper_source() + _JAVA_INSTALLER + body


def papi_agent_path(hooks_dir: Path) -> Path | None:
    """Path to the data-driven RPC agent (loaded raw, not wrapped)."""
    path = hooks_dir / PAPI_AGENT
    return path if path.exists() else None


def resolve_global_hooks(hooks_dir: Path) -> list[Path]:
    """Always-on hook scripts (e.g. network) that exist on disk."""
    paths: list[Path] = []
    for name in GLOBAL_HOOKS:
        path = hooks_dir / name
        if path.exists():
            paths.append(path)
    return paths
