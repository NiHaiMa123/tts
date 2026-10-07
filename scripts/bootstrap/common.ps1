# Common bootstrap helpers — mirror-first, proxy-fallback, provenance log.
# Dot-source this file from the setup_*.ps1 scripts.

$ErrorActionPreference = "Stop"

$Script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Script:DownloadLog = Join-Path $Script:RepoRoot "logs\downloads.jsonl"
$Script:ProxyUrl = "http://127.0.0.1:7897"
$Script:MirrorEndpoint = "https://hf-mirror.com"

function Write-DownloadRecord {
    param(
        [string]$Asset,
        [string]$SourceType,   # mirror | proxy | local_cache | failed
        [string]$ResolvedRevision,
        [string]$LocalPath,
        [bool]$Success,
        [string]$Detail = ""
    )
    $dir = Split-Path $Script:DownloadLog -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $record = @{
        asset = $Asset
        source_type = $SourceType
        resolved_revision = $ResolvedRevision
        local_path = $LocalPath
        success = $Success
        detail = $Detail
        at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Compress
    Add-Content -Path $Script:DownloadLog -Value $record -Encoding utf8
}

function Test-HfMirror {
    try {
        $r = Invoke-WebRequest -Uri "$($Script:MirrorEndpoint)/api/models/openbmb/VoxCPM2" `
            -Method Head -TimeoutSec 10 -UseBasicParsing
        return $r.StatusCode -lt 500
    } catch { return $false }
}

function Invoke-HfDownload {
    <#
    Download an HF model with resume, mirror-first then proxy fallback.
    Returns $true on success. Every attempt is logged — no silent switching.
    #>
    param(
        [Parameter(Mandatory=$true)][string]$RepoId,
        [Parameter(Mandatory=$true)][string]$PythonExe,
        [string]$CacheDir = ""
    )
    $envPairs = @(
        @{ Name = "mirror"; Vars = @{ HF_ENDPOINT = $Script:MirrorEndpoint } },
        @{ Name = "proxy";  Vars = @{
            HTTP_PROXY = $Script:ProxyUrl
            HTTPS_PROXY = $Script:ProxyUrl
            ALL_PROXY  = $Script:ProxyUrl } }
    )
    foreach ($attempt in $envPairs) {
        Write-Host "== download $RepoId via $($attempt.Name) =="
        $saved = @{}
        foreach ($k in $attempt.Vars.Keys) {
            $saved[$k] = [Environment]::GetEnvironmentVariable($k, "Process")
            [Environment]::SetEnvironmentVariable($k, $attempt.Vars[$k], "Process")
        }
        try {
            $args = @("-m", "huggingface_hub.commands.huggingface_cli",
                      "download", $RepoId)
            if ($CacheDir) { $args += @("--cache-dir", $CacheDir) }
            & $PythonExe @args
            $ok = ($LASTEXITCODE -eq 0)
        } catch { $ok = $false }
        finally {
            foreach ($k in $attempt.Vars.Keys) {
                [Environment]::SetEnvironmentVariable($k, $saved[$k], "Process")
            }
        }
        if ($ok) {
            Write-DownloadRecord -Asset $RepoId -SourceType $attempt.Name `
                -ResolvedRevision "" -LocalPath $CacheDir -Success $true
            return $true
        }
        Write-Host "   $($attempt.Name) attempt failed; trying next source"
        Write-DownloadRecord -Asset $RepoId -SourceType $attempt.Name `
            -ResolvedRevision "" -LocalPath $CacheDir -Success $false `
            -Detail "exit code $LASTEXITCODE"
    }
    return $false
}

function New-BackendVenv {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [string]$Python = "3.11"
    )
    $venvDir = Join-Path $Script:RepoRoot "backend_envs\$Name"
    if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
        Write-Host "== creating venv $venvDir (python $Python) =="
        uv venv $venvDir --python $Python
        if ($LASTEXITCODE -ne 0) { throw "uv venv failed for $Name" }
    }
    return Join-Path $venvDir "Scripts\python.exe"
}

function Install-Packages {
    param(
        [Parameter(Mandatory=$true)][string]$PythonExe,
        [Parameter(Mandatory=$true)][string[]]$Packages,
        [string]$IndexUrl = ""
    )
    $args = @("pip", "install", "--python", $PythonExe)
    if ($IndexUrl) { $args += @("--index-url", $IndexUrl) }
    $args += $Packages
    Write-Host "== uv $($args -join ' ') =="
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw "package install failed: $($Packages -join ', ')" }
}
