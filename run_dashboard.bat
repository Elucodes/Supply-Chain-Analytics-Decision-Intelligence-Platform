@echo off
setlocal
cd /d "%~dp0"
if exist ".\john\Scripts\streamlit.exe" (
    .\john\Scripts\streamlit.exe run ".\src\analytics dashboard.py"
) else (
    echo Virtual environment not found. Run: py -m venv john
    echo Then install requirements with: .\john\Scripts\python.exe -m pip install -r requirements.txt
    pause
)
