# Kreira virtualno okruzenje .venv i instalira sve pakete za UDIS
# Pokretanje: u PowerShellu -> .\setup_venv.ps1
# Za projekat je potreban Python 3.6 (TensorFlow 1.13). Ako imas vise verzija: py -3.6 -m venv .venv

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".venv") {
    Write-Host "Mapa .venv vec postoji. Za cisti pocetak obrisi ju pa pokreni skriptu ponovo." -ForegroundColor Yellow
    Write-Host "Aktivacija: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    exit 0
}

# Projekt zahtijeva Python 3.6 (TensorFlow 1.13). Probaj prvo py -3.6.
$py36 = $null
try { $null = & py -3.6 -c "exit(0)" 2>$null; $py36 = $true } catch {}
$venvPython = "python"
if ($py36) {
    Write-Host "Koristim Python 3.6 (py -3.6) za .venv ..." -ForegroundColor Green
    $venvPython = "py -3.6"
} else {
    Write-Host "Upozorenje: Python 3.6 nije pronaden. TensorFlow 1.13 zahtijeva Python 3.6." -ForegroundColor Yellow
    Write-Host "Instaliraj Python 3.6: https://www.python.org/downloads/release/python-3615/" -ForegroundColor Yellow
    Write-Host "Kreiram .venv s trenutnim Pythonom (paketi mogu ne raditi) ..." -ForegroundColor Yellow
}

Write-Host "Kreiram virtualno okruzenje .venv ..." -ForegroundColor Green
if ($py36) {
    & py -3.6 -m venv .venv
} else {
    & python -m venv .venv
}
if ($LASTEXITCODE -ne 0) { Write-Host "Kreiranje venv nije uspjelo." -ForegroundColor Red; exit 1 }

Write-Host "Aktiviram .venv i instaliran pakete ..." -ForegroundColor Green
& .\.venv\Scripts\pip.exe install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host ""
Write-Host "Gotovo. Aktivacija okruzenja:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Zatim za evaluaciju:" -ForegroundColor Green
Write-Host "  cd ImageReconstruction\Codes; python inference.py" -ForegroundColor Cyan
