@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Please install Python 3.11 or newer and add it to PATH.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating local virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -c "import streamlit,chromadb,cryptography,pandas,pymupdf,docx,openpyxl,reportlab" >nul 2>nul
if errorlevel 1 (
  echo Installing or updating local dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)
echo Starting Supplier Quality AI and background worker at http://127.0.0.1:8501 ...
".venv\Scripts\python.exe" launcher.py
exit /b %errorlevel%
:failed
echo.
echo Setup failed. Check the messages above and README.md.
pause
exit /b 1
