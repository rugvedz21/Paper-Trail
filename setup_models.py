"""Fetch the models the GPU stages need.

    python setup_models.py

Downloads ~500 MB (one small GGUF for the explanation stage). The tamper CNN is
*trained*, not downloaded — run `python evaluate_tamper.py` to produce
`models/tamper_resnet18.pt`.

Nothing here is required for stages 2 and 3, `evaluate.py`, or the test suite:
those are pure Python and run without a GPU or a network.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Small enough to sit alongside OCR on a 4GB card, and quantised so it loads
# fast. Explanation quality is limited by this choice, which the README says.
REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def main() -> None:
    target = Path("models") / FILENAME
    if target.exists():
        size_mb = target.stat().st_size / 1024 ** 2
        print(f"already present: {target} ({size_mb:.0f} MB)")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed. Run:")
        print("    .venv/Scripts/python -m pip install huggingface_hub")
        raise SystemExit(1)

    print(f"downloading {FILENAME} (~500 MB)...")
    path = hf_hub_download(REPO, FILENAME, local_dir="models")
    print(f"saved to {path}")

    print()
    print("The tamper CNN is trained rather than downloaded:")
    print("    python evaluate_tamper.py     -> models/tamper_resnet18.pt")


if __name__ == "__main__":
    main()
