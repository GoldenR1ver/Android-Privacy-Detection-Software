# 全量实验批处理（PowerShell 入口）
# 用法见: python scripts/run_full_experiment_batch.py --help
#
# 示例：
#   cd project\src
#   .\scripts\run_full_experiment_batch.ps1
#   .\scripts\run_full_experiment_batch.ps1 -RealRunStage3 -Verbose
#   .\scripts\run_full_experiment_batch.ps1 -FromStep build-evidence
#   .\scripts\run_full_experiment_batch.ps1 -OnlyStep stage3 -Stage3App Keep -RealRunStage3 -Verbose

param(
    [string]$AppsFile = "input/apps_full_batch_10.txt",
    [string]$JobId = "",
    [switch]$Verbose,
    [switch]$RealRunStage3,
    [switch]$SkipStage3,
    [switch]$SkipStage2,
    [switch]$NoFastStage2,
    [string]$FromStep = "",
    [string]$OnlyStep = "",
    [string]$Apps = "",
    [string]$Stage3App = "",
    [switch]$NoKnowledgeBase,
    [int]$RetryCount = 2
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcRoot = Split-Path -Parent $ScriptDir
Set-Location $SrcRoot

$pyArgs = @("scripts/run_full_experiment_batch.py", "--apps-file", $AppsFile)
if ($JobId) { $pyArgs += @("--job-id", $JobId) }
if ($Verbose) { $pyArgs += "-v" }
if ($RealRunStage3) { $pyArgs += "--real-run-stage3" }
if ($SkipStage3) { $pyArgs += "--skip-stage3" }
if ($SkipStage2) { $pyArgs += "--skip-stage2" }
if ($NoFastStage2) { $pyArgs += "--no-fast-stage2" }
if ($FromStep) { $pyArgs += @("--from-step", $FromStep) }
if ($OnlyStep) { $pyArgs += @("--only-step", $OnlyStep) }
if ($Apps) { $pyArgs += @("--apps", $Apps) }
if ($Stage3App) { $pyArgs += @("--stage3-app", $Stage3App) }
if ($NoKnowledgeBase) { $pyArgs += "--no-knowledge-base" }
$pyArgs += @("--retry-count", "$RetryCount")

Write-Host "cwd: $SrcRoot" -ForegroundColor Cyan
Write-Host "python $($pyArgs -join ' ')" -ForegroundColor Cyan
python @pyArgs
exit $LASTEXITCODE
