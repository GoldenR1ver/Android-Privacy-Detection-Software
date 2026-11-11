function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "android_id",
    hook_file: "hook_oaid.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary: (typeof __frida_stack_summary === "function" ? __frida_stack_summary() : "")
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
        var key = name ? name.toString() : "";
        if (
          key === "android_id" ||
          key.indexOf("oaid") >= 0 ||
          key.indexOf("OAID") >= 0
        ) {
          sendEvent("Settings.Secure.getString", "name=" + key);
        }
      } catch (innerErr) {}
      return result;
    };
  } catch (err) {}

  try {
    var AdvertisingIdClient = Java.use("com.google.android.gms.ads.identifier.AdvertisingIdClient");
    if (AdvertisingIdClient.getAdvertisingIdInfo) {
      var getAdId = AdvertisingIdClient.getAdvertisingIdInfo.overload(
        "android.content.Context"
      );
      getAdId.implementation = function (ctx) {
        var ret = getAdId.call(this, ctx);
        sendEvent("AdvertisingIdClient.getAdvertisingIdInfo", "");
        return ret;
      };
    }
  } catch (err2) {}

  sendEvent("hook_loaded", "oaid/advertising_id");
});
