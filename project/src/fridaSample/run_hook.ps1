# 需在 project\src 目录下执行
param(
    [ValidateSet("attach", "spawn", "both")]
    [string]$Mode = "attach",
    [int]$WaitSec = 25,
    [switch]$WithProjectPii,
    [switch]$ListHooks
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$argsList = @("fridaSample/run_hook.py", "--mode", $Mode, "--wait-sec", "$WaitSec")
if ($WithProjectPii) { $argsList += "--with-project-pii" }
if ($ListHooks) { $argsList += "--list-hooks" }

python @argsList
