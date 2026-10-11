@echo off
rem Double-click to open interactive Plex Nano with the verified local P2-48 checkpoint.
rem The execution policy override applies only to this new PowerShell process.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0plex.ps1"
set "plex_exit=%ERRORLEVEL%"
echo.
if not "%plex_exit%"=="0" (
    echo Plex did not finish successfully. Check the message above.
)
pause
exit /b %plex_exit%
