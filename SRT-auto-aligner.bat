@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" main.py
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    start "" ".venv\Scripts\python.exe" main.py
    exit /b 0
)

echo [ERROR] Virtual environment (.venv) not found or incomplete.
echo         Please run setup.bat first to install the environment.
echo.
pause
exit /b 1
