/*
 * 常见推送 SDK：极光 JPush / 个推 GeTui / 厂商通道。
 * 韭研公社 probe 会尝试 attach :push / pushcore 子进程。
 */
"use strict";

function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "sdk_info",
    hook_file: "hook_sdk_push.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary:
      typeof __frida_stack_summary === "function" ? __frida_stack_summary() : ""
  });
}

function hookStatic(clazz, methodName, sig, label) {
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
  var targets = [
    ["cn.jpush.android.api.JPushInterface", "init", ["android.content.Context"], "JPush.init"],
    ["cn.jpush.android.api.JPushInterface", "getRegistrationID", ["android.content.Context"], "JPush.getRegistrationID"],
    ["cn.jiguang.api.JCoreInterface", "init", ["android.content.Context"], "JCore.init"],
    ["com.igexin.sdk.PushManager", "initialize", ["android.content.Context"], "GeTui.initialize"],
    ["com.igexin.sdk.PushManager", "getClientid", ["android.content.Context"], "GeTui.getClientid"],
    ["com.xiaomi.mipush.sdk.MiPushClient", "registerPush", ["android.content.Context", "java.lang.String", "java.lang.String"], "MiPush.registerPush"],
    ["com.huawei.hms.push.HmsMessaging", "getToken", ["java.lang.String"], "HmsMessaging.getToken"]
  ];

  for (var i = 0; i < targets.length; i++) {
    var t = targets[i];
    if (hookStatic(t[0], t[1], t[2], t[3])) {
      n += 1;
    }
  }

  sendEvent("hook_loaded", "push_sdk hooks installed=" + n);
});
