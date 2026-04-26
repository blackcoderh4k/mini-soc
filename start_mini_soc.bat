@echo off
echo ==========================================
echo 🛡️ Mini SOC - Local AI Analyst Launcher 🛡️
echo ==========================================
echo.

echo [1/2] Starting Backend Server...
start "Mini SOC Server" /B python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

echo [WAIT] Waiting for server to initialize (10 seconds)...
timeout /t 10 /nobreak > nul

echo [2/2] Launching AI Analyst Inference...
python inference.py

echo.
echo ==========================================
echo ✅ Test Complete! Press any key to exit.
echo ==========================================
pause
