@echo off
cd /d "%~dp0"

echo.
echo ========================================
echo  SRT Auto Aligner - Environment Setup
echo ========================================
echo.

where winget >nul 2>&1
if errorlevel 1 goto :err_winget

echo [1/5] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 goto :install_python
echo   Python 3.11 OK
goto :check_ffmpeg

:install_python
echo   Python 3.11 not found - installing via winget...
winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
if errorlevel 1 goto :err_python_install
echo.
echo Python installed. Open a NEW command prompt and run setup.bat again
echo so the updated PATH takes effect.
pause
exit /b 0

:check_ffmpeg
echo [2/5] Checking ffmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 goto :install_ffmpeg
echo   ffmpeg OK
goto :make_venv

:install_ffmpeg
echo   ffmpeg not found - installing via winget...
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
if errorlevel 1 echo   [WARN] ffmpeg install failed - manual install: https://www.gyan.dev/ffmpeg/builds/
goto :make_venv

:make_venv
echo [3/5] Preparing virtual environment (.venv)...
if exist ".venv\Scripts\python.exe" (
    echo   .venv already exists - reusing
    goto :install_pkgs
)
py -3.11 -m venv .venv
if errorlevel 1 goto :err_venv
echo   .venv created

:install_pkgs
echo [4/5] Installing pip packages (PyTorch ~2GB+, may take 5-15 min)...
".venv\Scripts\python.exe" -m pip install --upgrade pip

nvidia-smi >nul 2>&1
if errorlevel 1 goto :cpu_torch
echo   NVIDIA GPU detected - installing CUDA 12.6 PyTorch (pinned to 2.8.0 for whisperx compat)
".venv\Scripts\python.exe" -m pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 echo   [WARN] CUDA PyTorch install failed - falling back to CPU build
goto :install_rest

:cpu_torch
echo   No NVIDIA GPU detected - installing CPU PyTorch

:install_rest
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :err_pip

".venv\Scripts\python.exe" -c "import torch; print('  PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo [5/5] Downloading spaCy language models...
".venv\Scripts\python.exe" -m spacy download en_core_web_sm
".venv\Scripts\python.exe" -m spacy download ko_core_news_sm
".venv\Scripts\python.exe" -m spacy download ja_core_news_sm

echo.
echo ========================================
echo  Setup complete!
echo  Run SRT-auto-aligner.bat to launch GUI.
echo ========================================
echo.
pause
exit /b 0

:err_winget
echo [ERROR] winget not found. Requires Windows 10 1809+ with "App Installer" from Microsoft Store.
pause
exit /b 1

:err_python_install
echo [ERROR] Python install via winget failed.
echo         Manual install: https://www.python.org/downloads/release/python-3119/
pause
exit /b 1

:err_venv
echo [ERROR] venv creation failed.
pause
exit /b 1

:err_pip
echo [ERROR] pip package install failed.
pause
exit /b 1
