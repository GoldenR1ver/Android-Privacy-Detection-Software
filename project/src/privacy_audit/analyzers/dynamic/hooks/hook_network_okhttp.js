/*
 * hook_network_okhttp.js — OkHttp3 outbound-traffic hook (adapted from rekit
 * apktap/hooks/okhttp.js) re-shaped to the project's unified event schema, with
 * OBFUSCATION-AWARE class discovery.
 *
 * Why discovery: release apps (e.g. Keep) run R8/ProGuard which renames okhttp
 * CLASS names to single letters (okhttp3.e, okhttp3.internal.connection.<x>),
 * while METHOD names are usually kept (execute/enqueue/request/url/...). A hook
 * hardcoded to "okhttp3.internal.connection.RealCall" therefore matches nothing.
 * We instead enumerate okhttp3.* classes that expose execute()+enqueue()+request()
 * — the RealCall fingerprint — and hook whichever (obfuscated) class matches.
 *
 * Emits the unified schema:
 *   { pii_type:"network_transmission", hook_file:"hook_network_okhttp.js",
 *     api_name:"OkHttp.<event>", args_summary:<compact JSON>, stack_summary:"" }
 */

"use strict";

(function () {
  var MAX_BODY_CHARS = 2000;
  var correlationId = 0;
  var hookedClasses = {};

  function nextId() {
    return "okhttp-" + (++correlationId);
  }

  function sendEvent(apiName, summaryObj) {
    var argsSummary;
    try {
      argsSummary = JSON.stringify(summaryObj);
    } catch (e) {
      argsSummary = String(summaryObj);
    }
    if (argsSummary && argsSummary.length > 4000) {
      argsSummary = argsSummary.slice(0, 4000) + "...";
    }
    var stack = "";
    try {
      if (typeof __frida_stack_summary === "function") {
        stack = __frida_stack_summary();
      }
    } catch (e) { /* ignore */ }
    send({
      pii_type: "network_transmission",
      hook_file: "hook_network_okhttp.js",
      api_name: apiName,
      args_summary: argsSummary,
      stack_summary: stack
    });
  }

  function status(msg) {
    // "hook_loaded" => excluded from runtime_event_count
    send({
      pii_type: "frida",
      hook_file: "hook_network_okhttp.js",
      api_name: "hook_loaded",
      args_summary: msg || "",
      stack_summary: ""
    });
  }

  function readHeaderKeys(headers) {
    var keys = [];
    try {
      var size = headers.size();
      for (var i = 0; i < size && i < 40; i++) {
        keys.push(headers.name(i).toString());
      }
    } catch (e) {
      /* ignore */
    }
    return keys;
  }

  function readRequestBody(request) {
    try {
      var body = request.body();
      if (body === null || body.toString() === "null") {
        return null;
      }
      var contentType = body.contentType();
      var contentTypeStr = contentType ? contentType.toString() : "";
      var contentLength = -1;
      try {
        contentLength = body.contentLength();
      } catch (e) { /* unknown */ }
      if (contentLength > 1024 * 1024) {
        return { _truncated: true, size: contentLength, contentType: contentTypeStr };
      }
      if (contentTypeStr.indexOf("multipart") !== -1) {
        return { _multipart: true, contentType: contentTypeStr };
      }
      var Buffer = Java.use("okio.Buffer");
      var buffer = Buffer.$new();
      body.writeTo(buffer);
      var bytes = buffer.readByteArray();
      buffer.close();
      if (contentTypeStr.indexOf("json") !== -1 ||
        contentTypeStr.indexOf("text") !== -1 ||
        contentTypeStr.indexOf("xml") !== -1 ||
        contentTypeStr.indexOf("form-urlencoded") !== -1 ||
        contentTypeStr === "") {
        var JavaString = Java.use("java.lang.String");
        var s = JavaString.$new(bytes, "UTF-8").toString();
        if (s.length > MAX_BODY_CHARS) {
          s = s.slice(0, MAX_BODY_CHARS) + "...";
        }
        return s;
      }
      return { _binary: true, size: bytes.length, contentType: contentTypeStr };
    } catch (e) {
      return { _error: String(e) };
    }
  }

  function captureRequest(id, call, eventName) {
    // Best-effort: even if request()/url() are obfuscated away, still report
    // that an outbound call happened (the signal Lab4-2/4-3 need).
    var summary = { id: id };
    try {
      var request = call.request();
      summary.url = request.url().toString();
      summary.method = request.method().toString();
      summary.header_keys = readHeaderKeys(request.headers());
      summary.body = readRequestBody(request);
    } catch (e) {
      summary._note = "request introspection failed: " + e;
    }
    sendEvent(eventName, summary);
  }

  function captureResponse(id, response) {
    try {
      sendEvent("OkHttp.response", {
        id: id,
        code: response.code(),
        url: response.request().url().toString()
      });
    } catch (e) {
      /* response is secondary; ignore */
    }
  }

  // Fingerprint: okhttp3.Call impl (RealCall) exposes execute() + enqueue() +
  // request(), regardless of obfuscated class name.
  function looksLikeRealCall(Cls) {
    try {
      return !!(Cls.execute && Cls.enqueue && Cls.request);
    } catch (e) {
      return false;
    }
  }

  function hookCallClass(className) {
    if (hookedClasses[className]) {
      return;
    }
    var Cls;
    try {
      Cls = Java.use(className);
    } catch (e) {
      return;
    }
    if (!looksLikeRealCall(Cls)) {
      return;
    }
    var didHook = false;

    try {
      var execOverloads = Cls.execute.overloads;
      for (var i = 0; i < execOverloads.length; i++) {
        if (execOverloads[i].argumentTypes.length !== 0) {
          continue; // RealCall.execute() is no-arg
        }
        (function (ov) {
          ov.implementation = function () {
            var id = nextId();
            captureRequest(id, this, "OkHttp.execute");
            var resp = ov.call(this);
            captureResponse(id, resp);
            return resp;
          };
        })(execOverloads[i]);
        didHook = true;
      }
    } catch (e) {
      status("hook execute failed on " + className + ": " + e);
    }

    try {
      var enqOverloads = Cls.enqueue.overloads;
      for (var j = 0; j < enqOverloads.length; j++) {
        if (enqOverloads[j].argumentTypes.length !== 1) {
          continue; // enqueue(Callback)
        }
        (function (ov) {
          ov.implementation = function (cb) {
            var id = nextId();
            captureRequest(id, this, "OkHttp.enqueue");
            return ov.call(this, cb);
          };
        })(enqOverloads[j]);
        didHook = true;
      }
    } catch (e) {
      status("hook enqueue failed on " + className + ": " + e);
    }

    if (didHook) {
      hookedClasses[className] = true;
      status("hooked okhttp call class: " + className);
    }
  }

  Java.perform(function () {
    // 1) Unobfuscated apps: direct class names.
    hookCallClass("okhttp3.internal.connection.RealCall");
    hookCallClass("okhttp3.RealCall");

    // 2) Obfuscated apps: enumerate okhttp3.* classes that have execute(),
    //    then fingerprint-filter to the RealCall-like one.
    try {
      var groups = Java.enumerateMethods("okhttp3.*!execute");
      for (var g = 0; g < groups.length; g++) {
        var classes = groups[g].classes || [];
        for (var c = 0; c < classes.length; c++) {
          hookCallClass(classes[c].name);
        }
      }
    } catch (e) {
      status("enumerate okhttp3.*!execute failed: " + e);
    }

    var count = Object.keys(hookedClasses).length;
    if (count === 0) {
      status("no OkHttp RealCall class found (app may not use OkHttp, or uses HttpURLConnection)");
    } else {
      status("OkHttp network hooks loaded (classes=" + count + ")");
    }
  });
})();
