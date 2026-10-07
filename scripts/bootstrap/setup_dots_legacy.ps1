# Verify-only bootstrap for the Dots legacy backend.
# The legacy backend reuses the old dotstts repo + its own venv — nothing
# is installed here. This script checks that every required piece exists.
# Usage: powershell -File scripts/bootstrap/setup_dots_legacy.ps1

. (Join-Path $PSScriptRoot "common.ps1")

$root = $env:DOTSTTS_ROOT
if (-not $root) { $root = "E:\project\dotstts" }

$checks = @(
    @{ Path = "$root\src\dots_tts";                 Label = "dots_tts source" },
    @{ Path = "$root\.venv\Scripts\python.exe";     Label = "legacy venv" },
    @{ Path = "$root\pretrained_models\dots.tts-soar"; Label = "base model" },
    @{ Path = "$root\data\work\suoming\suoming_lora_v1\checkpoint-00000500\model";
       Label = "suoming Step-500 adapter" },
    @{ Path = "$root\datasets\suoming\v1\validation.jsonl";
       Label = "suoming dataset manifests" }
)

$failed = 0
foreach ($c in $checks) {
    if (Test-Path $c.Path) {
        Write-Host "  ok   $($c.Label): $($c.Path)"
    } else {
        Write-Host "  MISS $($c.Label): $($c.Path)"
        $failed++
    }
}

if ($failed -gt 0) {
    Write-Host "BLOCKED: $failed required legacy asset(s) missing. " +
        "Set DOTSTTS_ROOT or restore the files."
    exit 2
}
Write-Host "Dots legacy backend verified (reuse only, nothing installed)."
