@echo off
REM Kreira virtualno okruzenje .venv i instalira sve pakete za UDIS
REM Pokretanje: dvostruki klik ili: setup_venv.bat

cd /d "%~dp0"

if exist ".venv" (
    echo Mapa .venv vec postoji. Za cisti pocetak obrisi ju pa pokreni skriptu ponovo.
    echo Aktivacija: .venv\Scripts\activate.bat
    pause
    exit /b 0
)

echo Kreiram virtualno okruzenje .venv ...
python -m venv .venv
if errorlevel 1 (
    echo Nije uspjelo. Probaj: py -3.6 -m venv .venv
    pause
    exit /b 1
)

echo Aktiviram i instaliran pakete ...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Gotovo. Kad god zelis raditi na projektu:
echo   1. Otvori CMD u ovoj mapi
echo   2. .venv\Scripts\activate.bat
echo   3. cd ImageReconstruction\Codes
echo   4. python inference.py
pause
