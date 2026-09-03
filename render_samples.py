"""Render sample documents to `out/` — clean, and one per attack.

    python render_samples.py

Useful for the submission writeup: the attack-1 pair shows, in pixels, a PAN
card and a GST certificate that disagree about the same number.
"""

import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from datagen.generate import (
    attack_identity_theft,
    attack_impossible_date,
    attack_naive_edit,
    consistent_bundle,
)
from datagen.render import render_bundle
from invariants.checks import check_bundle, verdict

AS_OF_SEED = 42


def main() -> None:
    rng = random.Random(AS_OF_SEED)
    clean = consistent_bundle(rng)

    samples = {
        "clean": clean,
        "attack1_naive_edit": attack_naive_edit(clean, rng),
        "attack3_identity_theft": attack_identity_theft(clean, rng),
        "attack5_impossible_date": attack_impossible_date(clean, rng),
    }

    for name, bundle in samples.items():
        paths = render_bundle(bundle, "out", name)
        decision = verdict(check_bundle(bundle))["decision"]
        print(f"{name:<26} {decision:<9} "
              f"{bundle.pan_number} / {bundle.gstin}")
        for path in paths.values():
            print(f"    {path}")

    print("\nRendered to out/ — every identifier is synthetic.")


if __name__ == "__main__":
    main()
