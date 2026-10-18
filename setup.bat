@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo Установка прервана. Скопируйте сообщение об ошибке.
  pause
  exit /b 1
)
echo.
echo Готово. Для запуска откройте start.bat
pause
