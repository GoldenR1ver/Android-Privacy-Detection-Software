function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "android_id",
    hook_file: "hook_android_id.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary: ""
  });
}

Java.perform(function () {
  try {
    var Secure = Java.use("android.provider.Settings$Secure");
    var origGetString = Secure.getString.overload(
      "android.content.ContentResolver",
      "java.lang.String"
    );
    origGetString.implementation = function (resolver, name) {
      var result = origGetString.call(this, resolver, name);
      try {
        if (name && name.toString() === "android_id") {
          sendEvent("Settings.Secure.getString", "name=android_id");
        }
      } catch (innerErr) {}
      return result;
    };
    sendEvent("hook_loaded", "Settings.Secure.getString");
  } catch (err) {
    sendEvent("hook_error", String(err));
  }
});
