@echo off
setlocal EnableDelayedExpansion
title GTAE-IDS: Review-II Demo and Web Server

:: Ensure execution from the project root directory
set "PROJECT_DIR=C:\Users\Aniket Patil\Desktop\GTAE-IDS"
cd /d "%PROJECT_DIR%"

echo ===============================================================================
echo                      GTAE-IDS: LOCAL DEMO LAUNCHER
echo ===============================================================================
echo Project Location : %PROJECT_DIR%
echo Date and Time    : %DATE% %TIME%
echo ===============================================================================
echo.

:: 1. Validate existing virtual environment Python
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment Python executable not found at:
    echo   "%PYTHON_EXE%"
    echo Please make sure the .venv directory is intact.
    echo.
    pause
    exit /b 1
)

:: 2. Activate virtual environment in this shell if activate.bat exists
if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
)

:: 3. Execute Review-II Demonstration Pipeline
echo [STEP 1/2] Executing Review-II Demonstration Script...
echo Running: "%PYTHON_EXE%" scripts\run_review2_demo.py
echo -------------------------------------------------------------------------------
"%PYTHON_EXE%" scripts\run_review2_demo.py
set "DEMO_ERR=%ERRORLEVEL%"
echo -------------------------------------------------------------------------------

:: 4. Check Demo Result
if %DEMO_ERR% neq 0 (
    echo.
    echo ===============================================================================
    echo [FAILURE] Review-II Demo execution failed with exit code %DEMO_ERR%.
    echo Flask frontend will NOT be started due to pipeline errors.
    echo ===============================================================================
    echo.
    pause
    exit /b %DEMO_ERR%
)

echo.
echo ===============================================================================
echo [SUCCESS] Review-II Demo execution finished with all checks passing!
echo ===============================================================================
echo.

:: 5. Launch Background Browser Opener (waits until Flask listens on port 5000)
echo [STEP 2/2] Starting GTAE-IDS Flask Application (app.py)...
echo Target URL: http://127.0.0.1:5000
echo The web dashboard will automatically open in your default browser once ready.
echo Server logs will stream below. Press Ctrl+C in this window to stop the server.
echo ===============================================================================
echo.

start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 35; $i++) { Start-Sleep -Seconds 1; try { $client = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 5000); if ($client.Connected) { $client.Close(); Start-Process 'http://127.0.0.1:5000'; break; } } catch {} }"

:: 6. Run Flask application in the foreground of this terminal
"%PYTHON_EXE%" app.py

:: 7. If Flask exits or is stopped, keep the terminal window open
echo.
echo ===============================================================================
echo GTAE-IDS server process has terminated.
echo ===============================================================================
pause
