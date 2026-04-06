@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==============================================
echo ImageConvertPro setup / dependency updater
echo ==============================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    set "PYTHON_CMD=python"
)

echo Using Python command: %PYTHON_CMD%
call %PYTHON_CMD% --version
if errorlevel 1 (
    echo.
    echo Python was not found. Install Python 3.11+ and run this file again.
    pause
    exit /b 1
)

echo.
echo Upgrading pip tooling...
call %PYTHON_CMD% -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo.
echo Installing / updating base dependencies...
call %PYTHON_CMD% -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Verifying desktop + NiceGUI dependencies...
call %PYTHON_CMD% -c "import PIL, PySide6, numpy, cv2, rawpy, pillow_heif, nicegui; print('Base dependencies verified')"
if errorlevel 1 goto :fail

echo.
choice /M "Install or update NVIDIA GPU support (CuPy + CUDA runtime + NVRTC)"
if errorlevel 2 goto :postgpu

echo.
echo Installing / updating GPU dependencies...
call %PYTHON_CMD% -m pip install --upgrade -r requirements-gpu.txt
if errorlevel 1 goto :fail

echo.
echo Verifying GPU Python packages...
call %PYTHON_CMD% -c "import cupy; import cuda; print('GPU Python packages verified')"
if errorlevel 1 goto :fail

echo.
echo GPU packages installed.
echo If GPU acceleration is still unavailable, restart the terminal and app once.
echo.

:postgpu
echo Running quick syntax verification...
call %PYTHON_CMD% -m py_compile ImageConvertPro.py ImageConvertPro_nicegui.py
if errorlevel 1 goto :fail

echo.
echo Setup complete.
echo You can now launch ImageConvertPro.py or ImageConvertPro_nicegui.py
pause
exit /b 0

:fail
echo.
echo Setup stopped because one of the steps failed.
pause
exit /b 1
