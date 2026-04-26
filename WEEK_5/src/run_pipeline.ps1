<#!
  WEEK_5 一体化流水线（本目录已包含原 WEEK_4 预处理 + 原 WEEK_3 run_audit 逻辑）

  输入：默认读取与本脚本同级的 data\*.txt
  输出：默认每次运行自动创建 output\run_yyyyMMdd_HHmmss\（可用 -OutDir 覆盖为任意绝对路径）

  在 WEEK_5\src 下执行:
    一键全流程（分类 + 审计 + 聚类 + 送标队列 + review_bundle；默认 LabelingTopN=200；若存在 ref\shots.json 则带 few-shot；
    写 WEEK3 CSV 前固定做句向量+HDBSCAN（≥2 句时），DS=同簇句拼接、PP=当前句）:
    $env:DEEPSEEK_API_KEY="sk-..."; .\run_pipeline.ps1 -FullPipeline
    分步或 mock:
    .\run_pipeline.ps1 -PrepareMode classify -Provider mock
    .\run_pipeline.ps1 -PrepareMode classify -Provider deepseek -RunAudit -RunClusterAnalysis
    .\run_pipeline.ps1 -PrepareMode classify -Provider deepseek -UsePiiShots

  环境变量:
    $env:PYTHON  $env:DEEPSEEK_API_KEY
    $env:HF_ENDPOINT   # 审计前聚类与 RunClusterAnalysis 拉 BGE；未设置时本脚本会设为 hf-mirror
#>
param(
    [switch]$FullPipeline,

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

    [switch]$UsePiiShots,

    [string]$PiiShotsJson = "",

    [int]$PiiShotsMax = 12,

    [switch]$RunClusterAnalysis,

    [ValidateSet("flag", "sentence_transformers")]
    [string]$ClusterEmbedBackend = "flag",

    [string]$ClusterModel = "BAAI/bge-small-zh-v1.5",

    [string]$DataDir = "",

    [string]$OutDir = "",

    [switch]$SkipAggregate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($FullPipeline) {
    Write-Host "==> FullPipeline: classify + RunAudit + RunClusterAnalysis + ExportLabelingQueue + ExportReviewJson; LabelingTopN=200 if unset; PII shots if ref\shots.json exists."
    $PrepareMode = "classify"
    $RunAudit = $true
    $RunClusterAnalysis = $true
    $ExportLabelingQueue = $true
    $ExportReviewJson = $true
    if ($LabelingTopN -le 0) {
        $LabelingTopN = 200
    }
}

$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$SrcRoot = $PSScriptRoot
$Week5Root = Split-Path -Parent $SrcRoot

if (-not $DataDir) {
    $DataDir = Join-Path $SrcRoot "data"
}
if (-not (Test-Path -LiteralPath $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    Write-Host "Created empty data directory: $DataDir"
}
$txtFiles = @(Get-ChildItem -LiteralPath $DataDir -Filter "*.txt" -File | Sort-Object Name)
if ($txtFiles.Count -eq 0) {
    throw "No .txt files found in $DataDir. Add UTF-8 .txt privacy policies here."
}

if (-not $OutDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outParent = Join-Path $SrcRoot "output"
    if (-not (Test-Path -LiteralPath $outParent)) {
        New-Item -ItemType Directory -Path $outParent -Force | Out-Null
    }
    $OutDir = Join-Path $outParent "run_$stamp"
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

if ($Provider -eq "deepseek" -and -not $env:DEEPSEEK_API_KEY) {
    if ($PrepareMode -eq "classify" -or $RunAudit) {
        throw "Provider=deepseek requires DEEPSEEK_API_KEY when using classify or -RunAudit"
    }
}
if (-not $env:HF_ENDPOINT) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host "HF_ENDPOINT was unset: set to https://hf-mirror.com (pre-audit clustering + optional cluster_analysis)."
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
        $shotsPathResolved = $null
        if ($PiiShotsJson) {
            $shotsPathResolved = $PiiShotsJson
        }
        elseif ($UsePiiShots -or $FullPipeline) {
            $candShots = Join-Path $SrcRoot "ref\shots.json"
            if (Test-Path -LiteralPath $candShots) { $shotsPathResolved = $candShots }
        }
        if ($shotsPathResolved) {
            $prep += @("--pii-shots-json", (Resolve-Path -LiteralPath $shotsPathResolved).Path)
            $prep += @("--pii-shots-max", "$PiiShotsMax")
        }
        $prep += @("--cluster-embed-backend", $ClusterEmbedBackend)
        $prep += @("--cluster-model", $ClusterModel)
        & $Py @prep
        if ($LASTEXITCODE -ne 0) { throw "run_prepare failed for $($f.Name)" }

        $week3csv = Join-Path $subOut "sentences_week3_2_2.csv"
        $auditRaw = Join-Path $subOut "audit_raw.csv"
        $auditProc = Join-Path $subOut "audit_processed.csv"
        $sentJsonl = Join-Path $subOut "sentences.jsonl"

        if ($RunAudit) {
            Write-Host "==> [$stem] run_audit.py audit (WEEK_5 bundled)"
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

            Write-Host "==> [$stem] run_audit.py postprocess"
            & $Py @(
                "run_audit.py", "postprocess",
                "--input-csv", $auditRaw,
                "--output-csv", $auditProc,
                "--log-file", (Join-Path $subOut "postprocess.log")
            )
            if ($LASTEXITCODE -ne 0) { throw "run_audit postprocess failed for $($f.Name)" }
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
                doc_id             = $stem
                output_dir         = $subOut
                week3_csv          = $week3csv
                audit_raw          = if (Test-Path -LiteralPath $auditRaw) { $auditRaw } else { $null }
                audit_processed    = if (Test-Path -LiteralPath $auditProc) { $auditProc } else { $null }
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
    bundle                   = "WEEK_5"
    week5_root               = $Week5Root
    created_at_local         = (Get-Date -Format "o")
    full_pipeline            = [bool]$FullPipeline
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
    use_pii_shots            = [bool]$UsePiiShots
    pii_shots_json           = if ($PiiShotsJson) { $PiiShotsJson } else { $null }
    pii_shots_max            = $PiiShotsMax
    run_cluster_analysis     = [bool]$RunClusterAnalysis
    pre_audit_clustering_in_prepare = $true
    cluster_embed_backend    = $ClusterEmbedBackend
    cluster_model            = $ClusterModel
    documents                = @($perDoc)
}
$summaryPath = Join-Path $OutDir "pipeline_summary.json"
$summaryJson = $summary | ConvertTo-Json -Depth 8
# PowerShell 5 UTF8 默认带 BOM，会导致 Python json.loads 报错；写入无 BOM UTF-8
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($summaryPath, $summaryJson, $utf8NoBom)

if (-not $SkipAggregate) {
    Write-Host "==> aggregate_pipeline_report.py"
    $agg = Join-Path $SrcRoot "aggregate_pipeline_report.py"
    & $Py $agg "--pipeline-root" (Resolve-Path -LiteralPath $OutDir).Path
    if ($LASTEXITCODE -ne 0) { throw "aggregate_pipeline_report.py failed" }
}

Write-Host ""
Write-Host "Pipeline finished. Output root: $OutDir"
Write-Host "Summary: $summaryPath"
