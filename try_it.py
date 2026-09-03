"""Type in two documents by hand and see what Paper Trail says.

    python try_it.py            interactive - it asks you for each field
    python try_it.py --samples  run four worked examples, no typing
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from invariants.checks import Bundle, check_bundle, verdict
from invariants.identifiers import gstin_check_char

BAR = "=" * 70


def show(title, b):
    print(f"\n{BAR}\n{title}\n{BAR}")
    print(f"  PAN card   : {b.pan_number}    name: {b.pan_holder_name}")
    print(f"  GSTIN      : {b.gstin}")
    print(f"  GST name   : {b.gst_legal_name}")
    print(f"  declared   : {b.declared_business_type}   state: {b.gst_address_state}")

    findings = check_bundle(b)
    v = verdict(findings)
    icon = {"pass": "PASS", "reject": "REJECT", "escalate": "ESCALATE"}[v["decision"]]
    print(f"\n  VERDICT -> {icon}")
    for f in findings:
        mark = "ok  " if f.passed else "FAIL"
        print(f"    [{mark}] {f.check}")
        if not f.passed:
            print(f"           {f.message}")
    return v


def ask():
    print("\nEnter what the two documents say. Press Enter to accept the default.\n")
    d = dict(
        pan_number="ABCPE1234F", pan_holder_name="Everest Traders",
        gstin="", gst_legal_name="", gst_trade_name="Everest",
        gst_address_state="Maharashtra", gst_registration_date="2021-07-15",
        declared_business_type="Individual",
    )
    for k in ["pan_number", "pan_holder_name", "gst_address_state",
              "declared_business_type"]:
        got = input(f"  {k:24s} [{d[k]}] : ").strip()
        if got:
            d[k] = got
    # offer a correctly-derived GSTIN so the default case is a clean bundle
    first14 = f"27{d['pan_number'].upper()}1Z"
    suggested = first14 + gstin_check_char(first14)
    got = input(f"  {'gstin':24s} [{suggested}] : ").strip()
    d["gstin"] = (got or suggested).upper()
    got = input(f"  {'gst_legal_name':24s} [{d['pan_holder_name']}] : ").strip()
    d["gst_legal_name"] = got or d["pan_holder_name"]
    d["pan_number"] = d["pan_number"].upper()
    return Bundle(**d)


def samples():
    from datagen.generate import consistent_bundle
    rng = random.Random(42)

    b = consistent_bundle(rng)
    show("1. CLEAN — everything agrees", b)

    b2 = consistent_bundle(rng)
    p = list(b2.pan_number); p[1] = "X"; b2.pan_number = "".join(p)
    show("2. ATTACK 1 — PAN card edited, GST certificate untouched", b2)

    b3 = consistent_bundle(rng)
    b3.pan_holder_name = b3.gst_legal_name = "Hemant Logistics"
    show("3. ATTACK 3 — real identifiers, someone else's name", b3)

    b4 = consistent_bundle(rng)
    b4.declared_business_type = "Private Limited Company"
    p = list(b4.pan_number); p[3] = "P"; b4.pan_number = "".join(p)
    show("4. ENTITY TYPE — claims a company, PAN says individual", b4)

    print(f"\n{BAR}")
    print("Sample 1 should PASS. Samples 2-4 should REJECT.")
    print(BAR)


if __name__ == "__main__":
    if "--samples" in sys.argv:
        samples()
    else:
        show("YOUR BUNDLE", ask())
