"""End-to-end evaluation: real images through all five stages.

    python evaluate_e2e.py                 # 150 applications
    python evaluate_e2e.py --n 400
    python evaluate_e2e.py --level typical

Why this exists separately from `evaluate.py`
---------------------------------------------
`evaluate.py` feeds `Bundle` objects straight to the checks. That measures the
*detection logic* in isolation, which is the right way to measure it — but it
produces a corpus where every legitimate case scores exactly zero, no class
overlap exists, and the cost matrix therefore has nothing to decide. The README
says so plainly.

This script closes that gap. Documents are rendered, degraded to a chosen
quality, read back by OCR, and only then checked. Extraction errors put
legitimate merchants at non-zero risk, which is what creates the overlap region
where a false positive and a false negative genuinely trade off — and where the
cost-weighted threshold finally earns its place.

It is slower by three orders of magnitude (a GPU inference per document, not a
string comparison), which is exactly why both scripts exist.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from datagen.degrade import degrade
from datagen.generate import (
    attack_identity_theft,
    attack_impossible_date,
    attack_naive_edit,
    attack_pixel_perfect,
    consistent_bundle,
)
from datagen.render import (
    identifier_boxes,
    render_gst_certificate,
    render_pan_card,
)
from evaluation.metrics import AttackRecall, CostMatrix, choose_threshold, pr_auc
import pipeline
from registry.gstn import MockGSTNRegistry

SEED = 42
AS_OF = date(2026, 8, 26)
FRAUD_RATE = 0.30          # higher than production: this corpus is small
N_REGISTERED = 200


def build_corpus(n: int, rng: random.Random):
    population = [consistent_bundle(rng) for _ in range(N_REGISTERED)]
    registry = MockGSTNRegistry.from_bundles(population)

    cases = []
    for _ in range(n):
        if rng.random() >= FRAUD_RATE:
            cases.append((rng.choice(population), 0, "legitimate"))
            continue

        attack = rng.choice([1, 2, 3, 4, 5])
        if attack == 1:
            cases.append((attack_naive_edit(rng.choice(population), rng), 1,
                          "attack_1_naive_edit"))
        elif attack == 2:
            cases.append((consistent_bundle(rng), 1, "attack_2_fabrication"))
        elif attack == 3:
            cases.append((attack_identity_theft(rng.choice(population), rng), 1,
                          "attack_3_identity_theft"))
        elif attack == 4:
            cases.append((attack_pixel_perfect(rng.choice(population), rng), 1,
                          "attack_4_pixel_perfect"))
        else:
            cases.append((attack_impossible_date(rng.choice(population), rng), 1,
                          "attack_5_impossible_date"))
    return cases, registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--level", default="good_photo",
                        help="degradation level (see datagen/degrade.py)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from extract.ocr import release

    rng = random.Random(args.seed)
    cases, registry = build_corpus(args.n, rng)
    costs = CostMatrix()

    print("=" * 76)
    print(f"END-TO-END — {args.n} applications, images at '{args.level}'")
    print("=" * 76)
    print("  Documents are rendered, degraded, read by OCR, then checked.")
    print("  Extraction errors are part of the measurement, not excluded.")
    print()

    scores, labels, kinds = [], [], []
    ocr_perfect = 0
    gated = 0
    started = time.time()

    for index, (bundle, label, kind) in enumerate(cases):
        pan_image, _ = render_pan_card(bundle)
        gst_image, _ = render_gst_certificate(bundle)
        pan_image = degrade(pan_image, args.level, seed=args.seed + index)
        gst_image = degrade(gst_image, args.level, seed=args.seed + index + 9973)

        # Through the real pipeline, not the checks directly. Measuring a
        # different code path than the one that ships is how a mitigation
        # gets credited without ever running - the OCR confidence gate lives
        # in `pipeline.run`, and calling `check_bundle` here would silently
        # skip it.
        result = pipeline.run(pan_image=pan_image, gst_image=gst_image,
                              boxes=identifier_boxes(bundle),
                              registry=registry, today=AS_OF,
                              gpu=not args.cpu)
        read = result.bundle

        ocr_perfect += (read.pan_number == bundle.pan_number
                        and read.gstin == bundle.gstin)
        # Count actual downgrades, not merely cases with a shaky field.
        # Reporting the latter overstates what the gate did by 10x.
        gated += (result.decision == "escalate"
                  and bool(result.low_confidence_fields))

        # An escalation is not a rejection: the merchant is reviewed, not
        # declined. Scoring it as a flag would erase the whole point of the
        # gate, so escalations are scored below the rejection threshold.
        scores.append(result.risk if result.decision == "reject"
                      else min(result.risk, 0.2))
        labels.append(label)
        kinds.append(kind)

        if (index + 1) % 25 == 0:
            print(f"    {index + 1}/{len(cases)} "
                  f"({time.time() - started:.0f}s)")

    release()

    scores = np.array(scores)
    labels = np.array(labels)
    kinds = np.array(kinds)

    threshold = choose_threshold(labels, scores, costs)
    flagged = scores >= threshold.threshold

    print()
    print(f"  OCR read both identifiers exactly: "
          f"{ocr_perfect}/{len(cases)} ({ocr_perfect / len(cases):.0%})")
    print(f"  rejections downgraded to human review by the "
          f"OCR-confidence gate: {gated}")
    print()

    print("DETECTION")
    print("-" * 76)
    print(f"  PR-AUC             : {pr_auc(labels, scores):.4f}   "
          f"(base rate {labels.mean():.4f})")
    print(f"  threshold          : {threshold.threshold:.4f}")
    print(f"  precision / recall : {threshold.precision:.4f} / "
          f"{threshold.recall:.4f}")
    print()

    print("PER-ATTACK RECALL")
    print("-" * 76)
    for kind in ("attack_1_naive_edit", "attack_2_fabrication",
                 "attack_3_identity_theft", "attack_4_pixel_perfect",
                 "attack_5_impossible_date"):
        mask = kinds == kind
        if not mask.any():
            continue
        row = AttackRecall(kind, int(flagged[mask].sum()), int(mask.sum()),
                           kind != "attack_4_pixel_perfect")
        note = "" if row.expected_to_catch else "   <- UNCAUGHT BY DESIGN"
        print(f"  {kind:<26} {row.caught:>3}/{row.total:<3} = "
              f"{row.recall:6.1%}{note}")
    print()

    # The point of the whole exercise.
    legit = labels == 0
    legit_scores = scores[legit]
    above_zero = int((legit_scores > 0).sum())

    print("DOES THE COST MATRIX NOW HAVE SOMETHING TO DECIDE?")
    print("-" * 76)
    print(f"  legitimate applications scoring above zero: "
          f"{above_zero}/{int(legit.sum())} "
          f"({above_zero / max(1, int(legit.sum())):.0%})")

    if above_zero:
        print("  YES. OCR errors put legitimate merchants at non-zero risk,")
        print("  so the classes overlap and the FN:FP ratio now determines")
        print("  where to cut. Sweeping it:")
        print()
        print(f"    {'FN':>9} {'FP':>7} {'threshold':>10} {'prec':>7} "
              f"{'recall':>7} {'flagged':>8}")
        seen = set()
        for fn, fp in [(5_000, 800), (50_000, 800), (200_000, 800),
                       (50_000, 20_000), (50_000, 50_000)]:
            choice = choose_threshold(
                labels, scores,
                CostMatrix(false_negative=fn, false_positive=fp))
            seen.add(round(choice.threshold, 4))
            print(f"    {fn:9,} {fp:7,} {choice.threshold:10.4f} "
                  f"{choice.precision:7.4f} {choice.recall:7.4f} "
                  f"{choice.flagged:8}")
        print()
        if len(seen) > 1:
            print(f"  {len(seen)} distinct operating points across the sweep —")
            print("  the cost weighting is now doing real work.")
        else:
            print("  The threshold still does not move: the overlap exists but")
            print("  is not yet dense enough to shift the cost minimum.")
    else:
        print("  NO. Every legitimate case still scores zero at this")
        print("  degradation level. Try --level typical or poor.")
    print("=" * 76)


if __name__ == "__main__":
    main()
