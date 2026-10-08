# Bootstrap the VoxCPM2 backend env + model download.
# Usage: powershell -File scripts/bootstrap/setup_voxcpm.ps1 [-SkipDownload]
param([switch]$SkipDownload)

. (Join-Path $PSScriptRoot "common.ps1")

$py = New-BackendVenv -Name "voxcpm2" -Python "3.11"
Write-Host "venv python: $py"

# torch cu128 first (RTX 5080 / sm_120 needs CUDA >= 12.8 builds).
Install-Packages -PythonExe $py -Packages @(
    "torch", "torchaudio"
) -IndexUrl "https://download.pytorch.org/whl/cu128"

Install-Packages -PythonExe $py -Packages @(
    "voxcpm", "soundfile", "huggingface_hub[cli]"
)

if (-not $SkipDownload) {
    $cacheDir = Join-Path $Script:RepoRoot "hf_cache\voxcpm2"
    # Revision pinned to match configs/backends/voxcpm2.yaml.
    $info = Invoke-HfDownload -RepoId "openbmb/VoxCPM2" -PythonExe $py `
        -CacheDir $cacheDir `
        -Revision "32279effe8c19989596f05d353d1447f51d9e915"
    if (-not $info) {
        Write-DownloadRecord -Asset "openbmb/VoxCPM2" -SourceType "failed" `
            -ResolvedRevision "" -LocalPath $cacheDir -Success $false
        throw "model download failed on all sources (see logs/downloads.jsonl)"
    }
}

Write-Host "VoxCPM2 backend ready: $py"
