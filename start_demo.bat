@echo off
echo ==============================================================
echo AML Prototype - Full System Launch
echo ==============================================================
echo.

echo [1/4] Starting Bank A API (Port 8001)...
start "Bank A API" cmd /c "set BANK_ID=bank_a&& C:\Users\praka\Desktop\projekt\.venv\Scripts\python.exe C:\Users\praka\Desktop\projekt\aml_prototype\bank_node\api.py & pause"

timeout /t 1 /nobreak >nul

echo [2/4] Starting Bank B API (Port 8002)...
start "Bank B API" cmd /c "set BANK_ID=bank_b&& C:\Users\praka\Desktop\projekt\.venv\Scripts\python.exe C:\Users\praka\Desktop\projekt\aml_prototype\bank_node\api.py & pause"

timeout /t 1 /nobreak >nul

echo [3/4] Starting Bank C API (Port 8003)...
start "Bank C API" cmd /c "set BANK_ID=bank_c&& C:\Users\praka\Desktop\projekt\.venv\Scripts\python.exe C:\Users\praka\Desktop\projekt\aml_prototype\bank_node\api.py & pause"

timeout /t 1 /nobreak >nul

echo [4/4] Starting Compliance Dashboard (Port 8080)...
start "Dashboard" cmd /c "C:\Users\praka\Desktop\projekt\.venv\Scripts\python.exe C:\Users\praka\Desktop\projekt\aml_prototype\dashboard\app.py & pause"

echo.
echo ==============================================================
echo All services starting! Wait ~5 seconds, then run:
echo   .venv\Scripts\python.exe aml_prototype\aggregator\pipeline.py
echo.
echo Health checks:
echo   Bank A:    http://127.0.0.1:8001/health
echo   Bank B:    http://127.0.0.1:8002/health
echo   Bank C:    http://127.0.0.1:8003/health
echo   Dashboard: http://127.0.0.1:8080
echo ==============================================================
echo.
pause
