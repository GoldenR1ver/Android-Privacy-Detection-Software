"""Frida hook wrapper and diagnostic event helpers."""

from privacy_audit.analyzers.dynamic.hook_loader import (
    is_frida_runtime_api,
    wrap_hook_script,
)


def test_wrap_hook_script_emits_java_active_with_label() -> None:
    source = """Java.perform(function () {
  send({ api_name: "hook_loaded" });
});"""
    wrapped = wrap_hook_script(source, hook_file="hooks/hook_imei.js")
    assert "__frida_stack_summary" in wrapped
    assert "globalThis.__frida_java_install" in wrapped
    assert "java_active" in wrapped
    assert "hook_imei.js" in wrapped
    assert "globalThis.__frida_java_install(function () {" in wrapped
    assert wrapped.count("Java.perform(function () {") == 1  # installer only
    assert '}, "hook_imei.js");' in wrapped
    assert wrapped.count("{") == wrapped.count("}")


def test_is_frida_runtime_api_excludes_diagnostics() -> None:
    assert not is_frida_runtime_api("java_active")
    assert not is_frida_runtime_api("hook_loaded")
    assert is_frida_runtime_api("TelephonyManager.getImei")
