<#!
  对 WEEK_5\src\output\run_* 目录批量执行 aggregate_pipeline_report.py（仅汇总，不重跑模型）。
#>
param(
    [string]$OutputParent = "",

    [string]$NameGlob = "run_*",

    [int]$MaxRuns = 50
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SrcRoot = $PSScriptRoot
if (-not $OutputParent) {
    $OutputParent = Join-Path $SrcRoot "output"
}
$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$agg = Join-Path $SrcRoot "aggregate_pipeline_report.py"

if (-not (Test-Path -LiteralPath $OutputParent)) {
    throw "Output parent not found: $OutputParent"
}

$dirs = @(Get-ChildItem -LiteralPath $OutputParent -Directory -Filter $NameGlob |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First $MaxRuns)

if ($dirs.Count -eq 0) {
    throw "No directories matching $NameGlob under $OutputParent"
}

foreach ($d in $dirs) {
    $summary = Join-Path $d.FullName "pipeline_summary.json"
    if (-not (Test-Path -LiteralPath $summary)) {
        Write-Host "[skip] no pipeline_summary.json: $($d.FullName)"
        continue
    }
    Write-Host "==> aggregate: $($d.Name)"
    & $Py $agg "--pipeline-root" $d.FullName
    if ($LASTEXITCODE -ne 0) { throw "aggregate failed for $($d.FullName)" }
}

Write-Host "Batch aggregate done. Processed up to $($dirs.Count) run folder(s)."
