function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "location",
    hook_file: "hook_location.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary: ""
  });
}

Java.perform(function () {
  try {
    var LocationManager = Java.use("android.location.LocationManager");

    if (LocationManager.getLastKnownLocation) {
      var getLastKnownLocation = LocationManager.getLastKnownLocation.overload("java.lang.String");
      getLastKnownLocation.implementation = function (provider) {
        var ret = getLastKnownLocation.call(this, provider);
        sendEvent("LocationManager.getLastKnownLocation", "provider=" + provider);
        return ret;
      };
    }

    if (LocationManager.requestLocationUpdates) {
      var overloads = LocationManager.requestLocationUpdates.overloads;
      for (var i = 0; i < overloads.length; i++) {
        (function (ov) {
          ov.implementation = function () {
            var args = Array.prototype.slice.call(arguments);
            var ret = ov.apply(this, args);
            sendEvent("LocationManager.requestLocationUpdates", "argc=" + args.length);
            return ret;
          };
        })(overloads[i]);
      }
    }

    if (LocationManager.getCurrentLocation) {
      var getCurrentLocation = LocationManager.getCurrentLocation.overload(
        "java.lang.String",
        "android.location.LocationRequest",
        "android.os.CancellationSignal",
        "java.util.concurrent.Executor",
        "java.util.function.Consumer"
      );
      getCurrentLocation.implementation = function (provider, request, cancelSignal, executor, consumer) {
        var ret = getCurrentLocation.call(this, provider, request, cancelSignal, executor, consumer);
        sendEvent("LocationManager.getCurrentLocation", "provider=" + provider);
        return ret;
      };
    }

    sendEvent("hook_loaded", "LocationManager");
  } catch (err) {
    sendEvent("hook_error", String(err));
  }
});
