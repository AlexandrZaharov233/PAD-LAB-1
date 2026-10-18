@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Сначала запустите setup.bat
  pause
  exit /b 1
)
echo Закройте start.bat перед экспериментами, чтобы не держать копии моделей в памяти.
echo 1 - Сравнение поиска: chunks, overlap, embeddings, Top-K, reranker, thresholds
echo 2 - Оценка базовой системы на всех вопросах с генерацией ответов
echo 3 - Сравнение двух prompts и двух режимов температуры на dev-наборе
echo 4 - Проверка базового поиска на отложенном test-наборе
set /p choice="Выберите 1, 2, 3 или 4: "
if "%choice%"=="1" ".venv\Scripts\python.exe" -m scada_rag experiments --suite retrieval --split dev
if "%choice%"=="2" ".venv\Scripts\python.exe" -m scada_rag evaluate --generate --split all
if "%choice%"=="3" ".venv\Scripts\python.exe" -m scada_rag experiments --suite generation --split dev
if "%choice%"=="4" ".venv\Scripts\python.exe" -m scada_rag evaluate --split test
pause
