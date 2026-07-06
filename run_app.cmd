@echo off
cd /d D:\football-model
set PIP_CACHE_DIR=D:\football-model\.cache\pip
set TEMP=D:\football-model\.cache\tmp
set TMP=D:\football-model\.cache\tmp
"D:\football-model\.venv\Scripts\python.exe" "D:\football-model\launcher.py"
if errorlevel 1 pause
