"""Field-level read accuracy as a function of image quality.

    python evaluate_ocr.py                 # 25 bundles x 6 levels
    python evaluate_ocr.py --n 50          # more samples, slower
    python evaluate_ocr.py --cpu           # no GPU

Reading a pristine PIL render tells you nothing: any OCR scores 100% on text it
drew itself. The number worth reporting is accuracy *against image quality*,
which is what this table gives.

On case folding
---------------
`render.py` prints names through `.upper()`, so the certificate genuinely reads
`NIMBUS TRADERS` where the Bundle says `Nimbus Traders`. Scoring that as a miss
would measure the renderer's styling, not the reader. Exact match is therefore
**case-insensitive and whitespace-normalised** for free-text fields.

Identifiers are compared **strictly**: a PAN is uppercase by construction, and
a single wrong character there invalidates the cross-document check that the
whole project rests on. There is no tolerance to give.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from datagen.degrade import LEVELS, degrade
from datagen.generate import consistent_bundle
from datagen.render import (
    GST_SIZE,
    PAN_SIZE,
    identifier_boxes,
    render_gst_certificate,
    render_pan_card,
)
from extract.ocr import extract_bundle, release

SEED = 42

# Reported in this order: identifiers first, because they are the fields the
# invariant layer actually consumes.
FIELDS: tuple[str, ...] = (
    "pan_number",
    "gstin",
    "pan_holder_name",
    "gst_legal_name",
    "gst_trade_name",
    "declared_business_type",
    "gst_address_state",
    "gst_registration_date",
)

STRICT_FIELDS = frozenset({"pan_number", "gstin"})


def matches(field: str, read: str, truth: str) -> bool:
    """Did we read this field correctly?

    Strict for identifiers, case- and whitespace-insensitive otherwise. See the
    module docstring for why the two differ.
    """
    if field in STRICT_FIELDS:
        return read == truth
    return " ".join(read.upper().split()) == " ".join(truth.upper().split())


def evaluate_level(level, bundles, gpu: bool, seed: int):
    """Read every bundle at one degradation level. Returns per-field hit counts."""
    hits = {field: 0 for field in FIELDS}
    identifier_hits = 0        # both PAN and GSTIN correct on the same bundle
    bundle_hits = 0            # every field correct
    confidences: list[float] = []

    for index, bundle in enumerate(bundles):
        pan_image, _ = render_pan_card(bundle)
        gst_image, _ = render_gst_certificate(bundle)

        # Seeded per (bundle, level) so a rerun degrades identically.
        pan_image = degrade(pan_image, level, seed=seed + index)
        gst_image = degrade(gst_image, level, seed=seed + index + 10_000)

        result = extract_bundle(pan_image, gst_image, identifier_boxes(bundle),
                                pan_size=PAN_SIZE, gst_size=GST_SIZE, gpu=gpu)
        confidences.append(result.mean_confidence)

        correct = {
            field: matches(field, getattr(result.bundle, field),
                           getattr(bundle, field))
            for field in FIELDS
        }
        for field, ok in correct.items():
            hits[field] += ok
        identifier_hits += all(correct[f] for f in STRICT_FIELDS)
        bundle_hits += all(correct.values())

    n = len(bundles)
    return {
        "per_field": {f: hits[f] / n for f in FIELDS},
        "identifiers": identifier_hits / n,
        "bundle": bundle_hits / n,
        "confidence": sum(confidences) / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=25,
                        help="bundles per level (default 25)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true", help="disable GPU")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bundles = [consistent_bundle(rng) for _ in range(args.n)]
    gpu = not args.cpu

    print("=" * 78)
    print(f"OCR FIELD ACCURACY vs IMAGE QUALITY — {args.n} bundles per level")
    print("=" * 78)
    print("Identifiers compared strictly; free-text fields case-insensitively")
    print("(the renderer prints names uppercase — see module docstring).")
    print()

    short = {
        "pan_number": "PAN", "gstin": "GSTIN",
        "pan_holder_name": "pan.name", "gst_legal_name": "legal",
        "gst_trade_name": "trade", "declared_business_type": "constit",
        "gst_address_state": "state", "gst_registration_date": "date",
    }
    header = (f"{'level':<11}" + "".join(f"{short[f]:>9}" for f in FIELDS)
              + f"{'BOTH ID':>9}{'all':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    started = time.time()
    for level in LEVELS:
        result = evaluate_level(level, bundles, gpu, args.seed)
        rows.append((level, result))
        line = f"{level.name:<11}"
        line += "".join(f"{result['per_field'][f]:>8.0%} " for f in FIELDS)
        line += f"{result['identifiers']:>8.0%} {result['bundle']:>6.0%}"
        print(line)

    print()
    print(f"  {len(LEVELS) * args.n * 2} images read in "
          f"{time.time() - started:.0f}s on {'GPU' if gpu else 'CPU'}")
    print()

    print("WHAT THIS MEANS FOR THE PIPELINE")
    print("-" * 78)
    pristine = rows[0][1]["identifiers"]
    worst = rows[-1][1]["identifiers"]
    print(f"  Both identifiers correct: {pristine:.0%} pristine -> "
          f"{worst:.0%} at '{LEVELS[-1].name}'.")
    print()
    print("  The PAN<->GSTIN equality check needs BOTH identifiers exactly")
    print("  right. A misread on either produces a mismatch that is")
    print("  indistinguishable from a forgery, so this column is the ceiling")
    print("  on what stage 2 can be asked to do from images.")
    print()
    print("  This is also the missing ingredient for the cost matrix: misreads")
    print("  push legitimate cases off a zero risk score, creating the overlap")
    print("  region that gives the cost-weighted threshold something to decide.")
    print("=" * 78)

    release()


if __name__ == "__main__":
    main()
