# Commit + push portable CrowdNav run artifacts under results/ for cross-machine resume.
# Usage:
#   .\sync_crowdnav_progress.ps1
#   .\sync_crowdnav_progress.ps1 -Message "card step 2 done" -LogPath results\runs\crowdnav\...
param(
    [string]$Message = "",
    [string]$LogPath = "",
    [switch]$NoPush
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here "..\..")).Path
# Do not leak Set-Location into the caller (run_crowdnav_card.ps1 needs self_reflection cwd).
Push-Location $repo
try {
    if (-not (Test-Path -LiteralPath (Join-Path $repo ".git"))) {
        throw "Not a git repo: $repo"
    }

    # Pointer for the other machine
    $runsRoot = Join-Path $repo "results\runs\crowdnav"
    New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null
    if ($LogPath -ne "") {
        $resolved = (Resolve-Path -LiteralPath $LogPath).Path
        Set-Content -LiteralPath (Join-Path $runsRoot "LAST_LOGPATH.txt") -Value $resolved
        # Relative pointer is more portable across machines/clones
        $rel = $resolved
        if ($rel.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
            $rel = $rel.Substring($repo.Length).TrimStart("\", "/")
        }
        Set-Content -LiteralPath (Join-Path $runsRoot "LAST_LOGPATH_REL.txt") -Value ($rel -replace "\\", "/")
    }

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($Message -eq "") {
        $Message = "Sync CrowdNav CARD progress ($stamp)"
    } else {
        $Message = "$Message ($stamp)"
    }

    git add -A -- "results" ".gitattributes"
    # Never stage secrets even if present under results by mistake
    git reset HEAD -- "**/xai_key" "**/xai_keys.json" "xai_key" "xai_keys.json" 2>$null

    function Invoke-GitPushRetry {
        # Git LFS verify often fails once with EOF on flaky networks; retry.
        $maxAttempts = 4
        for ($i = 1; $i -le $maxAttempts; $i++) {
            git push origin HEAD
            if ($LASTEXITCODE -eq 0) { return }
            if ($i -eq $maxAttempts) {
                throw "git push failed after $maxAttempts attempts — progress is committed locally; push when online"
            }
            Write-Host "git push failed (attempt $i/$maxAttempts); retrying in $($i * 3)s..."
            Start-Sleep -Seconds ($i * 3)
        }
    }

    $status = git status --porcelain -- "results" ".gitattributes"
    if (-not $status) {
        Write-Host "Nothing new under results/ to commit."
        if (-not $NoPush) {
            Invoke-GitPushRetry
        }
        return
    }

    git commit -m "$Message"
    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed"
    }

    if (-not $NoPush) {
        Invoke-GitPushRetry
        Write-Host "Pushed CrowdNav progress to origin."
    } else {
        Write-Host "Committed locally (NoPush)."
    }
} finally {
    Pop-Location
}
