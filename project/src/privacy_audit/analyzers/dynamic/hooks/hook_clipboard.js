function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "clipboard",
    hook_file: "hook_clipboard.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary: (typeof __frida_stack_summary === "function" ? __frida_stack_summary() : "")
  });
}

Java.perform(function () {
  try {
    var ClipboardManager = Java.use("android.content.ClipboardManager");
    if (ClipboardManager.getPrimaryClip) {
      var getClip = ClipboardManager.getPrimaryClip.overload();
      getClip.implementation = function () {
        var ret = getClip.call(this);
        sendEvent("ClipboardManager.getPrimaryClip", "");
        return ret;
      };
    }
    if (ClipboardManager.hasPrimaryClip) {
      var hasClip = ClipboardManager.hasPrimaryClip.overload();
      hasClip.implementation = function () {
        var ret = hasClip.call(this);
        sendEvent("ClipboardManager.hasPrimaryClip", "result=" + ret);
        return ret;
      };
    }
    sendEvent("hook_loaded", "ClipboardManager");
  } catch (err) {
    sendEvent("hook_error", String(err));
  }
});
