/*
 * 腾讯 X5 / TBS WebView SDK — mitm 曾在韭研公社观测 log.tbs.qq.com。
 */
"use strict";

function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "sdk_info",
    hook_file: "hook_sdk_tencent_x5.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary:
      typeof __frida_stack_summary === "function" ? __frida_stack_summary() : ""
  });
}

function tryHook(clazz, methodName, sig, label) {
  try {
    var Clz = Java.use(clazz);
    var m = sig ? Clz[methodName].overload.apply(Clz[methodName], sig) : Clz[methodName];
    m.implementation = function () {
      var parts = [];
      for (var i = 0; i < arguments.length; i++) {
        parts.push(String(arguments[i]).slice(0, 160));
      }
      sendEvent(label, parts.join(" | "));
      return m.apply(this, arguments);
    };
    return true;
  } catch (e) {
    return false;
  }
}

Java.perform(function () {
  var n = 0;
  if (tryHook("com.tencent.smtt.sdk.QbSdk", "initX5Environment", ["android.content.Context", "com.tencent.smtt.sdk.QbSdk$PreInitCallback"], "X5.QbSdk.initX5Environment")) {
    n += 1;
  }
  if (tryHook("com.tencent.smtt.sdk.stat.MttLoader", "loadUrl", ["android.content.Context", "java.lang.String", "java.util.HashMap"], "X5.MttLoader.loadUrl")) {
    n += 1;
  }
  if (tryHook("com.tencent.smtt.sdk.WebView", "loadUrl", ["java.lang.String"], "X5.WebView.loadUrl")) {
    n += 1;
  }
  if (tryHook("com.tencent.smtt.sdk.TbsLogReport", "eventReport", ["int", "java.lang.String"], "X5.TbsLogReport.eventReport")) {
    n += 1;
  }
  sendEvent("hook_loaded", "tencent_x5 hooks installed=" + n);
});
