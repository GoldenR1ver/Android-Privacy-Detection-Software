/*
 * 友盟 Umeng — AAID / 设备标识 / 统计上报常见入口。
 * mitm 曾在韭研公社观测 aaid.umeng.com。
 */
"use strict";

function sendEvent(apiName, argsSummary) {
  send({
    pii_type: "sdk_info",
    hook_file: "hook_sdk_umeng.js",
    api_name: apiName,
    args_summary: argsSummary || "",
    stack_summary:
      typeof __frida_stack_summary === "function" ? __frida_stack_summary() : ""
  });
}

function hookMethod(clazz, methodName, overloadSig, label) {
  try {
    var Clz = Java.use(clazz);
    var overload = overloadSig
      ? Clz[methodName].overload.apply(Clz[methodName], overloadSig)
      : Clz[methodName];
    overload.implementation = function () {
      var args = [];
      for (var i = 0; i < arguments.length; i++) {
        args.push(String(arguments[i]));
      }
      sendEvent(label || clazz + "." + methodName, args.join(" | "));
      return overload.apply(this, arguments);
    };
    return true;
  } catch (e) {
    return false;
  }
}

Java.perform(function () {
  var installed = 0;

  if (hookMethod("com.umeng.commonsdk.statistics.common.DeviceConfig", "getDeviceId", ["android.content.Context"], "Umeng.DeviceConfig.getDeviceId")) {
    installed += 1;
  }
  if (hookMethod("com.umeng.commonsdk.statistics.common.DeviceConfig", "getAndroidId", ["android.content.Context"], "Umeng.DeviceConfig.getAndroidId")) {
    installed += 1;
  }
  if (hookMethod("com.umeng.commonsdk.statistics.common.DeviceConfig", "getMac", ["android.content.Context"], "Umeng.DeviceConfig.getMac")) {
    installed += 1;
  }
  if (hookMethod("com.umeng.analytics.MobclickAgent", "onEvent", ["android.content.Context", "java.lang.String"], "Umeng.MobclickAgent.onEvent")) {
    installed += 1;
  }
  if (hookMethod("com.umeng.commonsdk.UMConfigure", "init", ["android.content.Context", "java.lang.String", "java.lang.String", "int", "java.lang.String"], "Umeng.UMConfigure.init")) {
    installed += 1;
  }

  try {
    Java.enumerateLoadedClasses({
      onMatch: function (name) {
        if (name.indexOf("com.umeng") !== 0) {
          return;
        }
        if (name.indexOf("DeviceConfig") >= 0 || name.indexOf("Id") >= 0) {
          sendEvent("Umeng.class.present", name);
        }
      },
      onComplete: function () {}
    });
  } catch (e) {}

  sendEvent("hook_loaded", "umeng hooks installed=" + installed);
});
