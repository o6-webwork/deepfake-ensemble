@echo off
REM NexInspect Setup Script for Windows
REM Automates installation and configuration

echo ============================================
echo   NexInspect - Deepfake Ensemble Setup
echo   Windows Installation Script
echo ============================================
echo.

REM Check Python installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Install dependencies
echo [4/6] Installing Python dependencies...
echo This may take several minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Check SPAI weights
echo [5/6] Checking SPAI model weights...
if exist spai\weights\spai.pth (
    echo [OK] SPAI weights found
) else (
    echo [!] SPAI weights not found
    echo.
    echo Please download SPAI weights manually:
    echo 1. Visit: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
    echo 2. Download spai.pth (~2GB^)
    echo 3. Create directory: mkdir spai\weights
    echo 4. Move file to: spai\weights\spai.pth
    echo.
)

REM Create .env file
echo [6/6] Setting up environment configuration...
if exist .env (
    echo .env file already exists, skipping...
) else (
    if exist .env.example (
        copy .env.example .env >nul
        echo [OK] Created .env file from template
        echo Please edit .env to add your API keys
    ) else (
        echo .env.example not found, skipping...
    )
)
echo.

REM Final instructions
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next steps:
echo.
echo 1. Download SPAI weights (if not already done^):
echo    https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
echo.
echo 2. (Optional^) Edit .env file with your API keys:
echo    notepad .env
echo.
echo 3. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 4. Launch application:
echo    streamlit run app.py
echo.
echo 5. Open browser to:
echo    http://localhost:8501
echo.
echo ============================================
pause
