# Push this project to GitHub (first-time setup)
# Run from repo root:  powershell -ExecutionPolicy Bypass -File scripts\push-to-github.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "Project root: $Root" -ForegroundColor Cyan

# 1) Init git if needed
if (-not (Test-Path ".git")) {
    git init -b main
    Write-Host "Git initialized (branch: main)" -ForegroundColor Green
} else {
    Write-Host "Git repo already exists." -ForegroundColor Yellow
}

# 2) Stage (respects .gitignore — no .venv, checkpoints, full datasets)
git add -A
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing new to commit (already clean?)." -ForegroundColor Yellow
} else {
    Write-Host "Staged files ($($staged.Count) items):" -ForegroundColor Cyan
    $staged | Select-Object -First 30
    if ($staged.Count -gt 30) { Write-Host "  ... and $($staged.Count - 30) more" }
    git commit -m "Portfolio: UDIS panorama pipeline, docs, and examples"
    Write-Host "Commit created." -ForegroundColor Green
}

# 3) Remote
$remote = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "=== GitHub remote not set ===" -ForegroundColor Yellow
    Write-Host "1) Open https://github.com/new"
    Write-Host "2) Repository name e.g. deep-image-stitching-portfolio"
    Write-Host "3) Public or Private — do NOT add README/license (you already have files)"
    Write-Host "4) Create repository, then paste your URL below."
    Write-Host ""
    $url = Read-Host "Paste GitHub repo URL (https://github.com/USER/REPO.git)"
    if ($url) {
        git remote add origin $url.Trim()
        Write-Host "Remote 'origin' added." -ForegroundColor Green
    } else {
        Write-Host "Skipped remote. Run later:" -ForegroundColor Yellow
        Write-Host '  git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git'
        exit 0
    }
} else {
    Write-Host "Remote origin: $remote" -ForegroundColor Cyan
}

# 4) Push
Write-Host "Pushing to origin main ..." -ForegroundColor Cyan
git push -u origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done! Open your repo on GitHub." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Push failed. Common fixes:" -ForegroundColor Red
    Write-Host "  - Log in: Git Credential Manager should open in browser"
    Write-Host "  - Or use Personal Access Token as password when prompted"
    Write-Host "  - Or install GitHub CLI: winget install GitHub.cli"
    Write-Host "    then: gh auth login && gh repo create YOUR_REPO --public --source=. --push"
}
