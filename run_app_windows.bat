@echo off
REM FitSync - one-click launcher for Windows. Double-click this file.
cd /d "%~dp0"

REM Pick a supported Python (3.12 or 3.11 preferred; very new versions may lack packages)
set PY=
py -3.12 -c "import sys" >nul 2>&1 && set PY=py -3.12
if not defined PY py -3.11 -c "import sys" >nul 2>&1 && set PY=py -3.11
if not defined PY py -3.13 -c "import sys" >nul 2>&1 && set PY=py -3.13
if not defined PY set PY=python

if not exist venv (
    echo [1/3] Creating virtual environment with: %PY%
    %PY% -m venv venv || (echo Python not found. Install Python 3.12 from python.org and tick "Add to PATH". & pause & exit /b)
)
call venv\Scripts\activate
python -c "import sys; v=sys.version_info; print('Using Python', v.major, v.minor)"
echo [2/3] Installing packages (first time takes 5-10 min)...
pip install -q -r requirements.txt || (echo. & echo Install failed. If you see Python 3.14 or 3.15 above, install Python 3.12, delete the "venv" folder and run this again. & pause & exit /b)
echo [3/3] Starting FitSync... your browser will open at http://localhost:8501
streamlit run app.py
pause
