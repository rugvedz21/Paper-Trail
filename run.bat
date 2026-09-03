@echo off
REM Paper Trail - run the project.
REM Uses the 3.12 venv: the interpreter the GPU stages need, since
REM PyTorch ships no CUDA wheels for the system Python 3.14.
setlocal
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
    echo ERROR: .venv not found. Create it with:
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\python -m pip install numpy pillow pytest
    exit /b 1
)

if "%1"==""     goto all
if "%1"=="web"  ( "%PY%" webapp.py & exit /b )
if "%1"=="demo" ( "%PY%" demo.py & exit /b )
if "%1"=="eval" ( "%PY%" evaluate.py & exit /b )
if "%1"=="full" ( "%PY%" evaluate.py --seeds 10 --sweep & exit /b )
if "%1"=="test" ( "%PY%" -m pytest tests\ -q & exit /b )
if "%1"=="docs" ( "%PY%" render_samples.py & exit /b )
if "%1"=="ocr"  ( "%PY%" evaluate_ocr.py & exit /b )
if "%1"=="tamper" ( "%PY%" evaluate_tamper.py & exit /b )
if "%1"=="trocr" ( "%PY%" train_trocr.py & exit /b )
if "%1"=="e2e"  ( "%PY%" evaluate_e2e.py & exit /b )
if "%1"=="stages" ( "%PY%" demo_pipeline.py & exit /b )
if "%1"=="try"  ( "%PY%" try_it.py & exit /b )
if "%1"=="samples" ( "%PY%" try_it.py --samples & exit /b )
echo Usage: run [web^|stages^|try^|demo^|eval^|full^|e2e^|ocr^|tamper^|trocr^|test^|docs^|samples]
exit /b 1

:all
echo ===== TESTS =====
"%PY%" -m pytest tests\ -q
echo.
echo ===== DEMO =====
"%PY%" demo.py
echo.
echo ===== EVALUATION =====
"%PY%" evaluate.py
