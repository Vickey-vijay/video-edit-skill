# One-time setup: Python environment, dependency check, and the Claude Code skill.
#   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
$Home2 = Join-Path $env:USERPROFILE ".video-edit-skill"
$Venv = Join-Path $Home2 "venv"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg is not on PATH. Install a full build (with libass), e.g. winget install Gyan.FFmpeg"
}
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { throw "Python 3.10+ is not on PATH." }

New-Item -ItemType Directory -Force (Join-Path $Home2 "work"), (Join-Path $Home2 "models") | Out-Null
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Write-Host "Creating Python environment in $Venv"
    & $py -m venv $Venv
}
$vpy = Join-Path $Venv "Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install -r (Join-Path $Repo "requirements.txt")

# Claude Code skill (personal skills folder), pointing at this repo
$SkillDir = Join-Path $env:USERPROFILE ".claude\skills\nagarajan-video-edit"
New-Item -ItemType Directory -Force $SkillDir | Out-Null
$text = (Get-Content (Join-Path $Repo "skill\SKILL.md") -Raw -Encoding UTF8).Replace("{{REPO}}", $Repo)
# no byte-order mark, or the skill's front-matter isn't recognised
[System.IO.File]::WriteAllText((Join-Path $SkillDir "SKILL.md"), $text, (New-Object System.Text.UTF8Encoding $false))
Write-Host "Skill installed: $SkillDir"

& $vpy -m vedit cards | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Repo "inbox") | Out-Null
Write-Host "Ready. Try:  .\vedit.cmd run `"C:\path\class.mp4`" --title `"Harmonic Patterns`""
