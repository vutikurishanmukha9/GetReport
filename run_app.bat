@echo off
REM GetReport Startup Script
REM This script ensures you are in the correct directory before running components.

echo Starting GetReport Stack...
echo --------------------------------

if not exist "Backend" (
    echo [ERROR] Backend directory not found!
    echo Please run this script from the root of the repository (d:\Git_Repo\GetReport).
    pause
    exit /b
)

echo [1/3] Starting Redis (Verify it is running manually if this fails)...
REM Check if redis-cli works or just warn user
REM For now we assume Redis is running as service or user started it.

echo [2/3] Starting Celery Worker...
start "Celery Worker" cmd /k "cd Backend && venv\Scripts\celery -A app.core.celery_app worker --loglevel=info -P solo"

echo [3/3] Starting FastAPI Backend...
start "FastAPI Server" cmd /k "cd Backend && venv\Scripts\uvicorn app.main:app --reload"

echo --------------------------------
echo Stack started!
echo API: http://127.0.0.1:8000
echo Check the new terminal windows for logs.
pause
