# Bootstrap the Qwen3-TTS backend env + model downloads.
# Usage: powershell -File scripts/bootstrap/setup_qwen3_tts.ps1 [-SkipDownload]
param([switch]$SkipDownload)

. (Join-Path $PSScriptRoot "common.ps1")

$py = New-BackendVenv -Name "qwen3_tts" -Python "3.11"
Write-Host "venv python: $py"

# torch cu128 first (RTX 5080 / sm_120 needs CUDA >= 12.8 builds).
# flash-attn is intentionally NOT installed — sdpa is the supported
# fallback on sm_120 (see backend config notes).
Install-Packages -PythonExe $py -Packages @(
    "torch", "torchaudio"
) -IndexUrl "https://download.pytorch.org/whl/cu128"

Install-Packages -PythonExe $py -Packages @(
    "qwen-tts", "soundfile", "huggingface_hub[cli]", "transformers", "accelerate"
)

if (-not $SkipDownload) {
    $cacheDir = Join-Path $Script:RepoRoot "hf_cache\qwen3_tts"
    foreach ($repo in @(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    )) {
        $ok = Invoke-HfDownload -RepoId $repo -PythonExe $py -CacheDir $cacheDir
        if (-not $ok) {
            Write-DownloadRecord -Asset $repo -SourceType "failed" `
                -ResolvedRevision "" -LocalPath $cacheDir -Success $false
            throw "model download failed for $repo (see logs/downloads.jsonl)"
        }
    }
}

Write-Host "Qwen3-TTS backend ready: $py"
