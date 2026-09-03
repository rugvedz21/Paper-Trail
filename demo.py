"""Three worked examples on the invariant layer alone.

    python demo.py

No GPU, no models, no arguments — pure Python and a fixed seed. Shows a clean
bundle passing, attack 1 caught by arithmetic, attack 3 caught by the PAN's
name initial, and attack 2 passing every check, which is correct: a coherent
fabrication is the invariant layer's blind spot and only the registry stage
can see it.

`demo_pipeline.py` is the version that runs all five stages.
"""

import sys, os, random

# The findings use em-dashes; the default Windows console codepage mangles them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))
from invariants.checks import check_bundle, verdict
from datagen.generate import consistent_bundle, attack_identity_theft

rng = random.Random(42)
b = consistent_bundle(rng)
print("=" * 66)
print("CLEAN BUNDLE")
print("=" * 66)
print(f"  PAN card   : {b.pan_number}   ({b.pan_holder_name})")
print(f"  GSTIN      : {b.gstin}")
print(f"  declared   : {b.declared_business_type}   state: {b.gst_address_state}")
print(f"  -> {verdict(check_bundle(b))['decision']}")

print()
print("=" * 66)
print("ATTACK 1 — forger edits the PAN card, forgets the GST certificate")
print("=" * 66)
orig = b.pan_number
p = list(b.pan_number); p[1] = "X"
b.pan_number = "".join(p)
print(f"  PAN card   : {orig} -> {b.pan_number}")
print(f"  GSTIN      : {b.gstin}  (untouched)")
v = verdict(check_bundle(b))
print(f"  -> {v['decision']}")
for r in v["reasons"]:
    print(f"     {r}")

print()
print("=" * 66)
print("ATTACK 3 — real identifier pair, someone else's name")
print("=" * 66)
clean = consistent_bundle(rng)
stolen = attack_identity_theft(clean, rng)
print(f"  PAN card   : {stolen.pan_number}  (identifiers untouched)")
print(f"  GSTIN      : {stolen.gstin}")
print(f"  name       : {clean.gst_legal_name} -> {stolen.gst_legal_name}")
v = verdict(check_bundle(stolen))
print(f"  -> {v['decision']}")
for r in v["reasons"]:
    print(f"     {r}")

print()
print("=" * 66)
print("ATTACK 2 — coherent fabrication (the invariant layer's blind spot)")
print("=" * 66)
fake = consistent_bundle(rng)
print(f"  PAN card   : {fake.pan_number}   ({fake.pan_holder_name})")
print(f"  GSTIN      : {fake.gstin}")
print(f"  -> {verdict(check_bundle(fake))['decision']}")
print("     Every invariant holds — this identity is arithmetically perfect")
print("     and simply does not exist. Only the registry stage can catch it.")
