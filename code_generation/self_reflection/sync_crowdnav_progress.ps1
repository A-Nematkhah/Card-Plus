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
Set-Location $repo

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

$status = git status --porcelain -- "results" ".gitattributes"
if (-not $status) {
    Write-Host "Nothing new under results/ to commit."
    if (-not $NoPush) {
        git push origin HEAD
    }
    return
}

git commit -m "$Message"
if ($LASTEXITCODE -ne 0) {
    throw "git commit failed"
}

if (-not $NoPush) {
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed — progress is committed locally; push when online"
    }
    Write-Host "Pushed CrowdNav progress to origin."
} else {
    Write-Host "Committed locally (NoPush)."
}
