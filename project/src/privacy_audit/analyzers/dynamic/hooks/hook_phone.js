function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "phone_number",
    hook_file: "hook_phone.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary: ""
  });
}

Java.perform(function () {
  try {
    var TelephonyManager = Java.use("android.telephony.TelephonyManager");

    if (TelephonyManager.getLine1Number) {
      var getLine1 = TelephonyManager.getLine1Number.overload();
      getLine1.implementation = function () {
        var ret = getLine1.call(this);
        sendEvent("TelephonyManager.getLine1Number", "");
        return ret;
      };
    }

    if (TelephonyManager.getSubscriberId) {
      var getSub = TelephonyManager.getSubscriberId.overload();
      getSub.implementation = function () {
        var ret = getSub.call(this);
        sendEvent("TelephonyManager.getSubscriberId", "");
        return ret;
      };
    }

    if (TelephonyManager.getSimSerialNumber) {
      var getSim = TelephonyManager.getSimSerialNumber.overload();
      getSim.implementation = function () {
        var ret = getSim.call(this);
        sendEvent("TelephonyManager.getSimSerialNumber", "");
        return ret;
      };
    }

    sendEvent("hook_loaded", "TelephonyManager.phone");
  } catch (err) {
    sendEvent("hook_error", String(err));
  }
});
