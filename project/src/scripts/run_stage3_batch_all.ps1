# 批量重跑全部 App 的 Stage3（真机 Appium + Frida）
# 前置：MuMu/adb、Appium(4723)、frida-server；每个 App 执行前请登录并保持前台
#
#   cd project\src
#   .\scripts\run_stage3_batch_all.ps1

$ErrorActionPreference = "Stop"
$SrcRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $SrcRoot

$logDir = Join-Path $SrcRoot "output\stage3_dynamic_real"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("batch_stage3_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Write-Host "Log: $logFile" -ForegroundColor Cyan
Write-Host "Apps: input/apps_full_batch_15.txt (15 apps)" -ForegroundColor Cyan

python scripts/run_full_experiment_batch.py `
  --only-step stage3 `
  --apps-file input/apps_full_batch_15.txt `
  --real-run-stage3 `
  -v 2>&1 | Tee-Object -FilePath $logFile

exit $LASTEXITCODE
