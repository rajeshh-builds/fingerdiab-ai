@echo off
title FingerDiab AI - Non-Invasive Diabetes Screening
color 0A

:: Ensure working directory is set properly
cd /d "%~dp0"

:: If running from the outer folder, step into Code
if exist "Code\app.py" (
    cd /d "%~dp0Code"
)

cls
echo ================================================================
echo               FINGERDIAB AI - SCREENING SYSTEM
echo ================================================================
echo.

:: Detect Python executable
:: System Python in AppData is trusted by Windows Smart App Control
set "PYTHON_EXEC=python"

:: Check if system python works
%PYTHON_EXEC% --version >nul 2>&1
if errorlevel 1 (
    if exist "..\venv\Scripts\python.exe" (
        set "PYTHON_EXEC=..\venv\Scripts\python.exe"
    ) else if exist "venv\Scripts\python.exe" (
        set "PYTHON_EXEC=venv\Scripts\python.exe"
    )
)

echo [1/3] Detecting Python environment...
%PYTHON_EXEC% --version
if errorlevel 1 (
    echo.
    echo [ERROR] Python was not detected on this system.
    echo Please make sure Python 3.10+ is installed and added to PATH.
    echo.
    pause
    exit /b 1
)
echo [OK] Python detected successfully.

echo.
echo [2/3] Checking essential libraries...
%PYTHON_EXEC% -c "import flask, pandas, numpy, joblib, cv2, PIL" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required packages... This may take a minute.
    %PYTHON_EXEC% -m pip install Flask scikit-learn pandas numpy joblib opencv-python Pillow
) else (
    echo [OK] Core libraries verified.
)

echo.
echo [3/3] Starting server on http://127.0.0.1:5000...
echo.
echo ================================================================
echo   The web application is starting!
echo   Your default web browser will open automatically in 2 seconds.
echo.
echo   Press Ctrl + C in this terminal window to stop the server.
echo ================================================================
echo.

:: Launch the browser in the background after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

:: Run the Flask application
%PYTHON_EXEC% -u app.py

if errorlevel 1 (
    echo.
    echo [ERROR] The application stopped unexpectedly.
    echo.
    pause
)
