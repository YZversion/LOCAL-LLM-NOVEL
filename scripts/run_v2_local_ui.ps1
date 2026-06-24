param(
    [string]$ContextFile = "",
    [int]$Seed = 1101,
    [int]$MaxSeqLength = 4096,
    [int]$TargetChars = 2300,
    [string]$Adapter = "outputs\qlora_run_v2"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv-train\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv-train Python: $Python"
}
if (-not (Test-Path $Adapter)) {
    throw "Missing adapter directory: $Adapter"
}

# Unsloth checks whether HF cache is writable during import.  In Codex sandbox
# the user HF cache can be read-only, so route writes to a local gitignored proxy.
$Proxy = Join-Path $RepoRoot "outputs\hf_stage0_proxy"
$Hub = Join-Path $Proxy "hub"
$Xet = Join-Path $Proxy "xet"
New-Item -ItemType Directory -Force $Hub, $Xet | Out-Null

$ModelCacheName = "models--huihui-ai--Huihui-Qwen3-8B-abliterated-v2"
$ProxyModel = Join-Path $Hub $ModelCacheName
$UserModel = Join-Path $env:USERPROFILE ".cache\huggingface\hub\$ModelCacheName"
if (-not (Test-Path $ProxyModel) -and (Test-Path $UserModel)) {
    New-Item -ItemType Junction -Path $ProxyModel -Target $UserModel | Out-Null
}

$env:HF_HOME = $Proxy
$env:HF_HUB_CACHE = $Hub
$env:HF_XET_CACHE = $Xet
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$ArgsList = @(
    "pipeline\adapter_cli.py",
    "--adapter", $Adapter,
    "--max-seq-length", "$MaxSeqLength",
    "--target-chars", "$TargetChars",
    "--seed", "$Seed"
)

if ($ContextFile) {
    if (-not (Test-Path $ContextFile)) {
        throw "Context file not found: $ContextFile"
    }
    $ArgsList += @("--context-file", $ContextFile)
}

Write-Host "LOCAL NOVEL UI (v2 adapter, no web)" -ForegroundColor Cyan
Write-Host "Adapter      : $Adapter"
Write-Host "Seed         : $Seed"
Write-Host "Max seq      : $MaxSeqLength"
Write-Host "Target chars : $TargetChars"
if ($ContextFile) { Write-Host "Context file : $ContextFile" }
else { Write-Host "Context file : (paste interactively)" }
Write-Host ""

& $Python @ArgsList
