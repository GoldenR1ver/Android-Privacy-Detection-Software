/*
 * Baseline: confirm Java bridge + enumerate loaded classes under target package.
 * Lifecycle events use api_name "hook_loaded" (excluded from runtime counts).
 */
"use strict";

function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "frida",
    hook_file: "hook_java_bridge_probe.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary:
      typeof __frida_stack_summary === "function" ? __frida_stack_summary() : ""
  });
}

Java.perform(function () {
  sendEvent("hook_loaded", "java_bridge_probe ready");

  try {
    var hits = [];
    Java.enumerateLoadedClasses({
      onMatch: function (name) {
        if (
          name.indexOf("com.jiucaigongshe") === 0 ||
          name.indexOf("umeng") >= 0 ||
          name.indexOf("jpush") >= 0 ||
          name.indexOf("jiguang") >= 0 ||
          name.indexOf("getui") >= 0 ||
          name.indexOf("tencent.smtt") >= 0
        ) {
          hits.push(name);
        }
      },
      onComplete: function () {
        sendEvent(
          "ClassLoader.enumerateLoadedClasses",
          "matched=" + hits.length + " sample=" + hits.slice(0, 12).join("; ")
        );
      }
    });
  } catch (err) {
    sendEvent("hook_error", String(err));
  }
});
