<#!
  兼容入口：等价于在同级目录执行 run_pipeline.ps1（参数名 -PipelineOutDir 映射为 -OutDir）。
  一键全流程：.\run_week5_full_audit.ps1 -FullPipeline -PipelineOutDir "D:\work\my_run"
#>
param(
    [switch]$FullPipeline,

    [Parameter(Mandatory = $true)]
    [string]$PipelineOutDir,

    [string]$DataDir = "",

    [ValidateSet("split-only", "classify")]
    [string]$PrepareMode = "classify",

    [ValidateSet("mock", "ollama", "deepseek")]
    [string]$Provider = "deepseek",

    [string]$DeepSeekModel = "deepseek-chat",

    [int]$ClassifyLimit = 0,

    [bool]$RunAudit = $true,

    [int]$AuditLimit = 0,

    [switch]$SkipPlot,

    [bool]$ExportLabelingQueue = $true,

    [bool]$ExportReviewJson = $false,

    [int]$LabelingTopN = 0,

    [bool]$RunClusterAnalysis = $true,

    [ValidateSet("flag", "sentence_transformers")]
    [string]$ClusterEmbedBackend = "flag",

    [string]$ClusterModel = "BAAI/bge-small-zh-v1.5",

    [switch]$SkipAggregate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$here = $PSScriptRoot
$main = Join-Path $here "run_pipeline.ps1"
if (-not (Test-Path -LiteralPath $main)) {
    throw "Not found: $main"
}

if ($FullPipeline) {
    $forward = @{
        FullPipeline        = $true
        OutDir              = $PipelineOutDir
        Provider            = $Provider
        DeepSeekModel       = $DeepSeekModel
        ClassifyLimit       = $ClassifyLimit
        AuditLimit          = $AuditLimit
        ClusterEmbedBackend = $ClusterEmbedBackend
        ClusterModel        = $ClusterModel
    }
    if ($DataDir) { $forward.DataDir = $DataDir }
    if ($LabelingTopN -gt 0) { $forward.LabelingTopN = $LabelingTopN }
    if ($SkipPlot) { $forward.SkipPlot = $true }
    if ($SkipAggregate) { $forward.SkipAggregate = $true }
    & $main @forward
}
else {
    $forward = @{
        OutDir                   = $PipelineOutDir
        PrepareMode              = $PrepareMode
        Provider                 = $Provider
        DeepSeekModel            = $DeepSeekModel
        ClassifyLimit            = $ClassifyLimit
        AuditLimit               = $AuditLimit
        ClusterEmbedBackend      = $ClusterEmbedBackend
        ClusterModel             = $ClusterModel
    }
    if ($DataDir) { $forward.DataDir = $DataDir }
    if ($RunAudit) { $forward.RunAudit = $true }
    if ($SkipPlot) { $forward.SkipPlot = $true }
    if ($ExportLabelingQueue) { $forward.ExportLabelingQueue = $true }
    if ($ExportReviewJson) { $forward.ExportReviewJson = $true }
    if ($LabelingTopN -gt 0) { $forward.LabelingTopN = $LabelingTopN }
    if ($RunClusterAnalysis) { $forward.RunClusterAnalysis = $true }
    if ($SkipAggregate) { $forward.SkipAggregate = $true }

    & $main @forward
}
if ($LASTEXITCODE -ne 0) { throw "run_pipeline.ps1 failed" }
