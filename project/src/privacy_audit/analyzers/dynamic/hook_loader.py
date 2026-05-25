"""Helpers to load Frida hook scripts when ART starts late (common on MuMu)."""

from __future__ import annotations

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

_JAVA_INSTALLER = r"""
if (typeof __frida_java_install === "undefined") {
  __frida_java_install = function (fn) {
    function wait() {
      if (typeof Java === "undefined" || !Java.available) {
        setTimeout(wait, 200);
        return;
      }
      Java.perform(fn);
    }
    wait();
  };
}
"""


def wrap_hook_script(source: str) -> str:
    """Wait for Java.available before running hook bodies that call Java.perform."""
    if "__frida_java_install" in source:
        return source
    body = source.replace("Java.perform(", "__frida_java_install(", 1)
    return _JAVA_INSTALLER + body
