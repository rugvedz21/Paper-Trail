#!/usr/bin/env sh
# Paper Trail - run the project. The POSIX twin of run.bat.
#
# Uses ./.venv if one exists, otherwise whatever `python3` is on PATH.
set -e

if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

case "${1:-all}" in
    web)     "$PY" webapp.py ;;
    demo)    "$PY" demo.py ;;
    eval)    "$PY" evaluate.py ;;
    full)    "$PY" evaluate.py --seeds 10 --sweep ;;
    test)    "$PY" -m pytest tests/ -q ;;
    docs)    "$PY" render_samples.py ;;
    ocr)     "$PY" evaluate_ocr.py ;;
    tamper)  "$PY" evaluate_tamper.py ;;
    trocr)   "$PY" train_trocr.py ;;
    e2e)     "$PY" evaluate_e2e.py ;;
    stages)  "$PY" demo_pipeline.py ;;
    try)     "$PY" try_it.py ;;
    samples) "$PY" try_it.py --samples ;;
    all)
        echo "===== TESTS ====="      && "$PY" -m pytest tests/ -q
        echo "" && echo "===== DEMO =====" && "$PY" demo.py
        echo "" && echo "===== EVALUATION =====" && "$PY" evaluate.py
        ;;
    *)
        echo "Usage: ./run.sh [web|stages|try|demo|eval|full|e2e|ocr|tamper|trocr|test|docs|samples]"
        exit 1
        ;;
esac
