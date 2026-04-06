<#!
  WEEK_4 一键流水线：data/*.txt -> 切句/分类 -> WEEK_3 风格 CSV -> 可选 audit -> 图表与汇总表

  用法（在 WEEK_4/src 下执行）:
    .\run_full_pipeline.ps1
    .\run_full_pipeline.ps1 -PrepareMode classify -Provider mock
    .\run_full_pipeline.ps1 -RunAudit -AuditLimit 50 -Provider mock
    $env:DEEPSEEK_API_KEY="sk-..."; .\run_full_pipeline.ps1 -PrepareMode classify -Provider deepseek -RunAudit -AuditLimit 20

  环境变量（可选）:
    $env:PYTHON = "python3"   # 覆盖默认 python 可执行文件
    $env:DEEPSEEK_API_KEY     # Provider=deepseek 时必填（勿提交到 Git）
#>
param(
    [ValidateSet("split-only", "classify")]
    [string]$PrepareMode = "split-only",

    [ValidateSet("mock", "ollama", "deepseek")]
    [string]$Provider = "mock",

    [string]$DeepSeekModel = "deepseek-chat",

    [int]$ClassifyLimit = 0,

    [switch]$RunAudit,

    [int]$AuditLimit = 0,

    [switch]$SkipPlot,

    [string]$DataDir = "",

    [string]$OutDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$SrcRoot = $PSScriptRoot
$WEEK4Root = Split-Path -Parent $SrcRoot
$RepoRoot = Split-Path -Parent $WEEK4Root
$Week32Dir = Join-Path $RepoRoot "WEEK_3\src\2-2"

if (-not $DataDir) {
    $DataDir = Join-Path $SrcRoot "data"
}
if (-not (Test-Path -LiteralPath $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    Write-Host "Created empty data directory: $DataDir"
}
$txtFiles = @(Get-ChildItem -LiteralPath $DataDir -Filter "*.txt" -File | Sort-Object Name)
if ($txtFiles.Count -eq 0) {
    throw "No .txt files found in $DataDir. Add UTF-8 .txt privacy policies there."
}

if (-not $OutDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutDir = Join-Path $WEEK4Root "out\pipeline_$stamp"
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

if ($Provider -eq "deepseek" -and -not $env:DEEPSEEK_API_KEY) {
    if ($PrepareMode -eq "classify" -or $RunAudit) {
        throw "Provider=deepseek requires DEEPSEEK_API_KEY when using classify or -RunAudit"
    }
}

$perDoc = [System.Collections.ArrayList]@()

Push-Location -LiteralPath $SrcRoot
try {
    foreach ($f in $txtFiles) {
        $stem = $f.BaseName
        $subOut = Join-Path $OutDir $stem
        Write-Host "==> [$stem] prepare -> $subOut"

        $prep = @(
            "run_prepare.py",
            "--input", $f.FullName,
            "--output-dir", $subOut,
            "--write-week3-csv",
            "--mode", $PrepareMode,
            "--provider", $Provider
        )
        if ($ClassifyLimit -gt 0) {
            $prep += @("--limit", "$ClassifyLimit")
        }
        if ($Provider -eq "deepseek") {
            $prep += @("--deepseek-model", $DeepSeekModel)
        }
        & $Py @prep
        if ($LASTEXITCODE -ne 0) { throw "run_prepare failed for $($f.Name)" }

        $week3csv = Join-Path $subOut "sentences_week3_2_2.csv"
        $auditRaw = Join-Path $subOut "audit_raw.csv"
        $auditProc = Join-Path $subOut "audit_processed.csv"

        if ($RunAudit) {
            if (-not (Test-Path -LiteralPath $Week32Dir)) {
                throw "WEEK_3 2-2 directory not found: $Week32Dir"
            }
            Write-Host "==> [$stem] WEEK_3 run_audit audit"
            Push-Location -LiteralPath $Week32Dir
            try {
                $auditArgs = @(
                    "run_audit.py", "audit",
                    "--input-csv", (Resolve-Path -LiteralPath $week3csv).Path,
                    "--output-csv", $auditRaw,
                    "--provider", $Provider,
                    "--log-file", (Join-Path $subOut "audit.log")
                )
                if ($AuditLimit -gt 0) {
                    $auditArgs += @("--limit", "$AuditLimit")
                }
                if ($Provider -eq "deepseek") {
                    $auditArgs += @("--deepseek-model", $DeepSeekModel)
                }
                & $Py @auditArgs
                if ($LASTEXITCODE -ne 0) { throw "run_audit audit failed for $($f.Name)" }

                Write-Host "==> [$stem] WEEK_3 postprocess"
                & $Py @(
                    "run_audit.py", "postprocess",
                    "--input-csv", $auditRaw,
                    "--output-csv", $auditProc,
                    "--log-file", (Join-Path $subOut "postprocess.log")
                )
                if ($LASTEXITCODE -ne 0) { throw "run_audit postprocess failed for $($f.Name)" }
            }
            finally {
                Pop-Location
            }
        }

        if (-not $SkipPlot) {
            Write-Host "==> [$stem] plot_experiment"
            $plotArgs = @(
                "plot_experiment.py",
                "--experiment-dir", $subOut
            )
            if ($RunAudit -and (Test-Path -LiteralPath $auditProc)) {
                $plotArgs += @("--audit-processed-csv", $auditProc)
            }
            & $Py @plotArgs
            if ($LASTEXITCODE -ne 0) { throw "plot_experiment failed for $($f.Name)" }
        }

        [void]$perDoc.Add(@{
                doc_id       = $stem
                output_dir   = $subOut
                week3_csv    = $week3csv
                audit_raw    = if (Test-Path -LiteralPath $auditRaw) { $auditRaw } else { $null }
                audit_processed = if (Test-Path -LiteralPath $auditProc) { $auditProc } else { $null }
            })
    }
}
finally {
    Pop-Location
}

$summary = [ordered]@{
    created_at_local = (Get-Date -Format "o")
    data_dir         = (Resolve-Path -LiteralPath $DataDir).Path
    output_root      = (Resolve-Path -LiteralPath $OutDir).Path
    prepare_mode     = $PrepareMode
    provider         = $Provider
    classify_limit   = $ClassifyLimit
    run_audit        = [bool]$RunAudit
    audit_limit      = $AuditLimit
    documents        = @($perDoc)
}
$summaryPath = Join-Path $OutDir "pipeline_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "Pipeline finished. Output root: $OutDir"
Write-Host "Summary: $summaryPath"
