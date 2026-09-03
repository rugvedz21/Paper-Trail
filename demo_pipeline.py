"""Walk one merchant application through all five stages.

    python demo_pipeline.py              # from a Bundle, no GPU
    python demo_pipeline.py --images     # render, degrade, and read with OCR
    python demo_pipeline.py --images --llm

Shows what each stage contributes and, more importantly, what each one cannot
do. The attack-4 case at the end passes everything — that is correct, and it is
the limitation the whole submission is built around stating clearly.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pipeline
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
from registry.gstn import MockGSTNRegistry

SEED = 42
TODAY = date(2026, 8, 26)
RULE = "=" * 74


def show(title: str, note: str, bundle, registry, args, detector=None,
         model_path=None) -> None:
    print(RULE)
    print(title)
    print(RULE)
    print(f"  {note}")
    print()
    print(f"  PAN card   : {bundle.pan_number}   ({bundle.pan_holder_name})")
    print(f"  GSTIN      : {bundle.gstin}")
    print(f"  registered : {bundle.gst_registration_date}   "
          f"state: {bundle.gst_address_state}")

    kwargs = {}
    if args.images:
        pan_image, _ = render_pan_card(bundle)
        gst_image, _ = render_gst_certificate(bundle)
        pan_image = degrade(pan_image, args.level, seed=SEED)
        gst_image = degrade(gst_image, args.level, seed=SEED + 1)
        kwargs = dict(pan_image=pan_image, gst_image=gst_image,
                      boxes=identifier_boxes(bundle))
        print(f"  (rendered, degraded to '{args.level}', read back by OCR)")
        result = pipeline.run(registry=registry, tamper_detector=detector,
                              model_path=model_path, today=TODAY, **kwargs)
        print(f"  OCR read   : {result.bundle.pan_number} / "
              f"{result.bundle.gstin}")
    else:
        result = pipeline.run(bundle=bundle, registry=registry,
                              model_path=model_path, today=TODAY)

    print()
    print(f"  stages     : {' -> '.join(result.stages_run)}")
    print(f"  DECISION   : {result.decision.upper()}   "
          f"(risk {result.risk:.3f})")

    for finding in result.findings:
        if not finding.passed:
            print(f"     [{finding.severity}] {finding.message}")

    if result.tamper is not None:
        print(f"     tamper score {result.tamper.score:.3f}")

    if result.explanation is not None:
        print()
        print(f"  REVIEWER NOTE ({result.explanation.source}):")
        for line in result.explanation.text.splitlines():
            print(f"     {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", action="store_true",
                        help="render and read documents rather than using a Bundle")
    parser.add_argument("--level", default="good_photo")
    parser.add_argument("--llm", action="store_true",
                        help="use a local GGUF model for escalation notes")
    parser.add_argument("--tamper", action="store_true",
                        help="run the tamper CNN (needs --images)")
    args = parser.parse_args()

    model_path = None
    if args.llm:
        found = glob.glob("models/*.gguf")
        model_path = found[0] if found else None
        if model_path is None:
            print("  (no GGUF model in models/ — using templates)\n")

    detector = None
    if args.tamper and args.images:
        from tamper.detect import TamperDetector
        weights = "models/tamper_resnet18.pt"
        detector = TamperDetector(weights if os.path.exists(weights) else None)

    rng = random.Random(SEED)
    population = [consistent_bundle(rng) for _ in range(60)]
    registry = MockGSTNRegistry.from_bundles(population)
    victim = population[0]

    show("CLEAN APPLICATION",
         "A real registered business applying honestly.",
         victim, registry, args, detector, model_path)

    show("ATTACK 1 — naive edit",
         "Forger edits the PAN card and forgets the GST certificate. "
         "Caught by arithmetic.",
         attack_naive_edit(victim, rng), registry, args, detector, model_path)

    show("ATTACK 2 — coherent fabrication",
         "An invented identity that satisfies every invariant. "
         "Only the registry can see it.",
         consistent_bundle(rng), registry, args, detector, model_path)

    show("ATTACK 3 — identity theft",
         "A genuine PAN/GSTIN pair with someone else's name attached.",
         attack_identity_theft(population[1], rng), registry, args, detector,
         model_path)

    show("ATTACK 5 — impossible registration date",
         "Found by attacking our own checks, not designed in advance.",
         attack_impossible_date(population[2], rng), registry, args, detector,
         model_path)

    show("ATTACK 4 — pixel-perfect theft   [THE BLIND SPOT]",
         "Genuine documents belonging to someone else. Nothing was altered, "
         "so there is nothing to detect. This PASSES, and we report it.",
         attack_pixel_perfect(population[3], rng), registry, args, detector,
         model_path)

    print(RULE)
    print("Attack 4 passing is the honest result, not a bug. Detecting it")
    print("needs signals outside the documents: device fingerprinting,")
    print("behavioural analysis, liveness checks at onboarding.")
    print(RULE)


if __name__ == "__main__":
    main()
