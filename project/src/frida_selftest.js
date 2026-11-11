send({ api_name: "script_loaded", pii_type: "frida" });

function waitForJava(attempt) {
  attempt = attempt || 0;
  if (typeof Java !== "undefined" && Java.available) {
    Java.perform(function () {
      send({ api_name: "java_bridge_ok", pii_type: "frida" });
      try {
        var Secure = Java.use("android.provider.Settings$Secure");
        Secure.getString.overload(
          "android.content.ContentResolver",
          "java.lang.String"
        ).implementation = function (resolver, name) {
          var result = this.getString(resolver, name);
          if (name && name.toString() === "android_id") {
            send({ api_name: "Settings.Secure.getString", pii_type: "android_id" });
          }
          return result;
        };
        send({ api_name: "hook_loaded", pii_type: "android_id" });
      } catch (e) {
        send({ api_name: "hook_error", detail: String(e) });
      }
    });
    return;
  }
  if (attempt < 120) {
    setTimeout(function () { waitForJava(attempt + 1); }, 250);
  } else {
    send({ api_name: "java_never_ready", pii_type: "frida" });
  }
}
waitForJava();