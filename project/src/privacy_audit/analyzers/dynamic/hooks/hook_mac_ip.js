function sendEvent(piiType, apiName, argsSummary) {
  send({
    pii_type: piiType,
    hook_file: "hook_mac_ip.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary: (typeof __frida_stack_summary === "function" ? __frida_stack_summary() : "")
  });
}

Java.perform(function () {
  try {
    var WifiInfo = Java.use("android.net.wifi.WifiInfo");
    if (WifiInfo.getMacAddress) {
      var getMac = WifiInfo.getMacAddress.overload();
      getMac.implementation = function () {
        var ret = getMac.call(this);
        sendEvent("mac_address", "WifiInfo.getMacAddress", "");
        return ret;
      };
    }
    if (WifiInfo.getIpAddress) {
      var getIp = WifiInfo.getIpAddress.overload();
      getIp.implementation = function () {
        var ret = getIp.call(this);
        sendEvent("ip_address", "WifiInfo.getIpAddress", "");
        return ret;
      };
    }
  } catch (err) {
    sendEvent("mac_address", "hook_error", String(err));
  }

  try {
    var InetAddress = Java.use("java.net.InetAddress");
    if (InetAddress.getHostAddress) {
      var getHost = InetAddress.getHostAddress.overload();
      getHost.implementation = function () {
        var ret = getHost.call(this);
        sendEvent("ip_address", "InetAddress.getHostAddress", "");
        return ret;
      };
    }
  } catch (err2) {
    sendEvent("ip_address", "hook_error", String(err2));
  }

  try {
    var BluetoothAdapter = Java.use("android.bluetooth.BluetoothAdapter");
    if (BluetoothAdapter.getAddress) {
      var getBt = BluetoothAdapter.getAddress.overload();
      getBt.implementation = function () {
        var ret = getBt.call(this);
        sendEvent("mac_address", "BluetoothAdapter.getAddress", "");
        return ret;
      };
    }
  } catch (err3) {}

  sendEvent("mac_address", "hook_loaded", "network identifiers");
});
