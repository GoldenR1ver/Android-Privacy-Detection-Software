<#!
  WEEK_4 一键流水线：data/*.txt -> 切句/分类 -> WEEK_3 风格 CSV -> 可选 audit -> 图表
  -> 可选 送标导出 / 可选 强化聚类分析（cluster_analysis.py）

  用法（在 WEEK_4/src 下执行）:
    .\run_full_pipeline.ps1
    .\run_full_pipeline.ps1 -PrepareMode classify -Provider mock
    .\run_full_pipeline.ps1 -RunAudit -AuditLimit 50 -Provider mock
    $env:DEEPSEEK_API_KEY="sk-..."; .\run_full_pipeline.ps1 -PrepareMode classify -Provider deepseek -RunAudit -AuditLimit 20
    .\run_full_pipeline.ps1 -PrepareMode classify -Provider deepseek -RunAudit -RunClusterAnalysis -ExportLabelingQueue -LabelingTopN 200

  环境变量（可选）:
    $env:PYTHON = "python3"
    $env:DEEPSEEK_API_KEY     # Provider=deepseek 时必填（勿提交到 Git）
    $env:HF_ENDPOINT          # 例: https://hf-mirror.com（句向量下载超时）
#>
param(
    [ValidateSet("split-only", "classify")]
    [string]$PrepareMode = "split-only",

    [ValidateSet("mock", "ollama", "deepseek")]
    [string]$Provider = "deepseek",

    [string]$DeepSeekModel = "deepseek-chat",

    [int]$ClassifyLimit = 0,

    [switch]$RunAudit,

    [int]$AuditLimit = 0,

    [switch]$SkipPlot,

    [switch]$ExportLabelingQueue,

    [switch]$ExportReviewJson,

    [int]$LabelingTopN = 0,

    [switch]$RunClusterAnalysis,

    [ValidateSet("local", "deepseek-api")]
    [string]$ClusterEmbedBackend = "local",

    [string]$ClusterModel = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

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
if ($ClusterEmbedBackend -eq "deepseek-api" -and -not $env:DEEPSEEK_API_KEY) {
    throw "ClusterEmbedBackend=deepseek-api requires DEEPSEEK_API_KEY"
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
        if ($ExportLabelingQueue) {
            $prep += @("--export-labeling-queue")
        }
        if ($ExportReviewJson) {
            $prep += @("--export-review-json")
        }
        if ($LabelingTopN -gt 0) {
            $prep += @("--labeling-top-n", "$LabelingTopN")
        }
        & $Py @prep
        if ($LASTEXITCODE -ne 0) { throw "run_prepare failed for $($f.Name)" }

        $week3csv = Join-Path $subOut "sentences_week3_2_2.csv"
        $auditRaw = Join-Path $subOut "audit_raw.csv"
        $auditProc = Join-Path $subOut "audit_processed.csv"
        $sentJsonl = Join-Path $subOut "sentences.jsonl"

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

        $clusterDir = $null
        if ($RunClusterAnalysis) {
            if (-not (Test-Path -LiteralPath $sentJsonl)) {
                throw "RunClusterAnalysis requires sentences.jsonl at $sentJsonl"
            }
            $clusterDir = Join-Path $subOut "cluster_analysis"
            Write-Host "==> [$stem] cluster_analysis -> $clusterDir"
            $clusterArgs = @(
                "cluster_analysis.py",
                "--sentences-jsonl", (Resolve-Path -LiteralPath $sentJsonl).Path,
                "--output-dir", $clusterDir,
                "--embed-backend", $ClusterEmbedBackend,
                "--model", $ClusterModel
            )
            if ($RunAudit -and (Test-Path -LiteralPath $auditProc)) {
                $clusterArgs += @("--audit-processed", (Resolve-Path -LiteralPath $auditProc).Path)
            }
            & $Py @clusterArgs
            if ($LASTEXITCODE -ne 0) { throw "cluster_analysis failed for $($f.Name)" }
        }

        $forLabeling = Join-Path $subOut "for_labeling.jsonl"
        $reviewBundle = Join-Path $subOut "review_bundle.json"
        [void]$perDoc.Add(@{
                doc_id            = $stem
                output_dir        = $subOut
                week3_csv         = $week3csv
                audit_raw         = if (Test-Path -LiteralPath $auditRaw) { $auditRaw } else { $null }
                audit_processed   = if (Test-Path -LiteralPath $auditProc) { $auditProc } else { $null }
                for_labeling_jsonl = if (Test-Path -LiteralPath $forLabeling) { $forLabeling } else { $null }
                review_bundle_json = if (Test-Path -LiteralPath $reviewBundle) { $reviewBundle } else { $null }
                cluster_analysis_dir = $clusterDir
            })
    }
}
finally {
    Pop-Location
}

$summary = [ordered]@{
    created_at_local         = (Get-Date -Format "o")
    data_dir                 = (Resolve-Path -LiteralPath $DataDir).Path
    output_root              = (Resolve-Path -LiteralPath $OutDir).Path
    prepare_mode             = $PrepareMode
    provider                 = $Provider
    classify_limit           = $ClassifyLimit
    run_audit                = [bool]$RunAudit
    audit_limit              = $AuditLimit
    export_labeling_queue    = [bool]$ExportLabelingQueue
    export_review_json       = [bool]$ExportReviewJson
    labeling_top_n           = $LabelingTopN
    run_cluster_analysis     = [bool]$RunClusterAnalysis
    cluster_embed_backend    = $ClusterEmbedBackend
    cluster_model            = $ClusterModel
    documents                = @($perDoc)
}
$summaryPath = Join-Path $OutDir "pipeline_summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "Pipeline finished. Output root: $OutDir"
Write-Host "Summary: $summaryPath"
