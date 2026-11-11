function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "contact_list",
    hook_file: "hook_contacts.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary: (typeof __frida_stack_summary === "function" ? __frida_stack_summary() : "")
  });
}

Java.perform(function () {
  try {
    var ContentResolver = Java.use("android.content.ContentResolver");
    var queryOverloads = ContentResolver.query.overloads;
    for (var i = 0; i < queryOverloads.length; i++) {
      (function (ov) {
        ov.implementation = function () {
          var args = Array.prototype.slice.call(arguments);
          var ret = ov.apply(this, args);
          try {
            var uri = args[0];
            if (uri) {
              var uriStr = uri.toString();
              if (uriStr.indexOf("contacts") >= 0 || uriStr.indexOf("Contacts") >= 0) {
                sendEvent("ContentResolver.query", "uri=" + uriStr);
              }
            }
          } catch (innerErr) {}
          return ret;
        };
      })(queryOverloads[i]);
    }
    sendEvent("hook_loaded", "ContentResolver.contacts");
  } catch (err) {
    sendEvent("hook_error", String(err));
  }
});
