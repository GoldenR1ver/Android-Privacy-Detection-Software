/*
 * hook_papi.js — data-driven privacy-API monitor.
 *
 * RPC-driven agent loaded by frida_runner / FridaCaptureSession. The Python
 * host calls rpc.exports.installhooks(spec) AFTER the script loads (and, in
 * spawn mode, BEFORE device.resume), so framework identifier APIs called at
 * cold start are still intercepted.
 *
 * Engine adapted from PAPIMonitor's javadynamichook (overload-aware hooking)
 * plus Java.enumerateMethods wildcard support (frida-tools / frida-trace style).
 *
 * Every intercepted call emits the project's unified event schema:
 *   { pii_type, hook_file:"hook_papi.js", api_name:"Class.method",
 *     args_summary, stack_summary }
 * Lifecycle/status events use api_name "hook_loaded" so they are excluded from
 * runtime_event_count on the Python side.
 *
 * spec: Array of entries, each either
 *   { pii_type, category, clazz, method }                  // explicit target
 *   { pii_type, category, enumerate: "pkg.Class!method*" } // wildcard pattern
 */

"use strict";

var MAX_ARGS_CHARS = 600;
var MAX_STACK_CHARS = 700;
var __papi_installed = {}; // dedupe key "clazz#method"

function __papi_send(piiType, apiName, argsSummary, stackSummary) {
  send({
    pii_type: piiType || "unknown",
    hook_file: "hook_papi.js",
    api_name: apiName,
    args_summary: (argsSummary || "").slice(0, MAX_ARGS_CHARS),
    stack_summary: (stackSummary || "").slice(0, MAX_STACK_CHARS)
  });
}

function __papi_status(msg) {
  // api_name "hook_loaded" => not counted as a runtime event
  __papi_send("frida", "hook_loaded", msg, "");
}

function __papi_summarize_args(args) {
  var parts = [];
  try {
    for (var i = 0; i < args.length; i++) {
      var v;
      try {
        v = args[i] === null || args[i] === undefined ? String(args[i]) : args[i].toString();
      } catch (e) {
        v = "<unprintable>";
      }
      if (v && v.length > 120) {
        v = v.slice(0, 120) + "...";
      }
      parts.push(v);
    }
  } catch (e) {
    parts.push("<args_error:" + e + ">");
  }
  return parts.join(" | ");
}

function __papi_caller(exception) {
  // Caller chain (skip the framework method + our wrapper). Identifies which
  // SDK / app class triggered the read — key evidence for undeclared collection.
  try {
    var frames = exception.getStackTrace();
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

function __papi_hook_target(Exception, piiType, category, clazz, method) {
  var dedupeKey = clazz + "#" + method;
  if (__papi_installed[dedupeKey]) {
    return false;
  }
  var handle;
  try {
    handle = Java.use(clazz)[method];
    if (!handle) {
      __papi_status("missing " + clazz + "." + method);
      return false;
    }
  } catch (e) {
    __papi_status("cannot use " + clazz + "." + method);
    return false;
  }

  var apiName = clazz + "." + method;

  function makeImpl(overload) {
    return function () {
      var args = [].slice.call(arguments);
      var retval = overload.apply(this, arguments);
      try {
        var stack = __papi_caller(Exception.$new());
        __papi_send(piiType, apiName, __papi_summarize_args(args), stack);
      } catch (e) {
        // never break the app on logging failure
      }
      return retval;
    };
  }

  try {
    if (handle.overloads === undefined) {
      handle.implementation = makeImpl(handle);
    } else {
      for (var i = 0; i < handle.overloads.length; i++) {
        handle.overloads[i].implementation = makeImpl(handle.overloads[i]);
      }
    }
    __papi_installed[dedupeKey] = true;
    return true;
  } catch (e) {
    __papi_status("hook failed " + apiName + ": " + e);
    return false;
  }
}

function __papi_hook_enumerate(Exception, piiType, category, pattern) {
  var installed = 0;
  try {
    var groups = Java.enumerateMethods(pattern);
    for (var g = 0; g < groups.length; g++) {
      var classes = groups[g].classes || [];
      for (var c = 0; c < classes.length; c++) {
        var className = classes[c].name;
        var methods = classes[c].methods || [];
        for (var m = 0; m < methods.length; m++) {
          if (__papi_hook_target(Exception, piiType, category, className, methods[m])) {
            installed += 1;
          }
        }
      }
    }
  } catch (e) {
    __papi_status("enumerate failed " + pattern + ": " + e);
  }
  return installed;
}

function __papi_do_install(spec) {
  var installed = 0;
  var Exception = Java.use("java.lang.Exception");
  for (var i = 0; i < spec.length; i++) {
    var entry = spec[i];
    if (!entry) {
      continue;
    }
    var piiType = entry.pii_type || "unknown";
    var category = entry.category || "";
    if (entry.enumerate) {
      installed += __papi_hook_enumerate(Exception, piiType, category, entry.enumerate);
    } else if (entry.clazz && entry.method) {
      if (__papi_hook_target(Exception, piiType, category, entry.clazz, entry.method)) {
        installed += 1;
      }
    }
  }
  __papi_status("papi installed=" + installed + " targets=" + spec.length);
  __papi_send("frida", "papi_ready", "installed=" + installed + " targets=" + spec.length, "");
}

function __papi_wait_java(fn) {
  if (typeof Java === "undefined" || !Java.available) {
    setTimeout(function () {
      __papi_wait_java(fn);
    }, 200);
    return;
  }
  Java.perform(function () {
    __papi_send("frida", "java_active", "hook_papi.js Java.perform entered", "");
    fn();
  });
}

rpc.exports = {
  installhooks: function (spec) {
    if (!spec || !spec.length) {
      return 0;
    }
    // Run async once ART is ready; intercepts begin before resume in spawn mode.
    __papi_wait_java(function () {
      __papi_do_install(spec);
    });
    return spec.length;
  },
  ping: function () {
    return true;
  }
};
