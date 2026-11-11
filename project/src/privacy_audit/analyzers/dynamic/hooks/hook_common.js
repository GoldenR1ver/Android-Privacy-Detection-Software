/*
 * Shared stack capture for hand-written hook_*.js scripts (R1 / PMA style).
 * Prepended by hook_loader.wrap_hook_script — do not load standalone.
 */
function __frida_stack_summary() {
  try {
    if (typeof Java === "undefined" || !Java.available) {
      return "";
    }
    var Exception = Java.use("java.lang.Exception");
    var frames = Exception.$new().getStackTrace();
    var out = [];
    var start = frames.length > 2 ? 2 : 0;
    for (var i = start; i < frames.length && out.length < 6; i++) {
      out.push(frames[i].toString());
    }
    return out.join("\n");
  } catch (e) {
    return "";
  }
}
