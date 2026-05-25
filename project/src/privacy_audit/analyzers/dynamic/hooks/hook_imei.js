function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "imei",
    hook_file: "hook_imei.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary: ""
  });
}

Java.perform(function () {
  try {
    var TelephonyManager = Java.use("android.telephony.TelephonyManager");

    if (TelephonyManager.getImei) {
      var getImei0 = TelephonyManager.getImei.overload();
      getImei0.implementation = function () {
        var ret = getImei0.call(this);
        sendEvent("TelephonyManager.getImei", "");
        return ret;
      };
      if (TelephonyManager.getImei.overloads.length > 1) {
        var getImei1 = TelephonyManager.getImei.overload("int");
        getImei1.implementation = function (slot) {
          var ret = getImei1.call(this, slot);
          sendEvent("TelephonyManager.getImei(int)", "slot=" + slot);
          return ret;
        };
      }
    }

    if (TelephonyManager.getDeviceId) {
      var getDeviceId0 = TelephonyManager.getDeviceId.overload();
      getDeviceId0.implementation = function () {
        var ret = getDeviceId0.call(this);
        sendEvent("TelephonyManager.getDeviceId", "");
        return ret;
      };
      if (TelephonyManager.getDeviceId.overloads.length > 1) {
        var getDeviceId1 = TelephonyManager.getDeviceId.overload("int");
        getDeviceId1.implementation = function (slot) {
          var ret = getDeviceId1.call(this, slot);
          sendEvent("TelephonyManager.getDeviceId(int)", "slot=" + slot);
          return ret;
        };
      }
    }

    sendEvent("hook_loaded", "TelephonyManager");
  } catch (err) {
    sendEvent("hook_error", String(err));
  }
});
