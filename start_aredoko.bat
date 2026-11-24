@echo off
title AreDoko Local AI Memory Server
set "APP_DIR=%~dp0"
set "AREDOKO_PORT=1925"
cd /d "%APP_DIR%"
if not exist "app.py" goto ERR_NO_APP
echo Starting AreDoko AI Memory on http://127.0.0.1:%AREDOKO_PORT% ...
start "" "http://127.0.0.1:%AREDOKO_PORT%"
python "app.py"
goto END

:ERR_NO_APP
echo [ERROR] app.py not found in %APP_DIR%
pause

:END
