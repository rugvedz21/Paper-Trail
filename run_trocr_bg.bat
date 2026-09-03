@echo off
REM Detached TrOCR training. Survives the parent shell exiting, which a
REM foreground background-job does not.
cd /d "%~dp0"
.venv\Scripts\python.exe -u train_trocr.py --n 6000 --epochs 6 > trocr_train.log 2>&1
