"""Paper Trail — web interface.

    python webapp.py            then open http://127.0.0.1:5000

A reviewer's console for the pipeline. Three ways in:

* **Generate** — build a synthetic application (clean, or any of the five
  attacks) and watch it go through. This is the demo path.
* **Type it in** — enter a PAN and GSTIN by hand. Useful for convincing
  yourself the cross-document check is real: change one character and watch it
  reject.
* **Upload** — supply two document images and run the full pipeline including
  OCR and tamper detection.

Why the stages are shown separately
-----------------------------------
The point of this project is that a decision is *explainable*: every rejection
traces to a named check comparing two specific values. A UI that showed only
"APPROVED / REJECTED" would hide exactly what makes the approach worth
anything. So each stage reports what it contributed, and every finding shows
the values it compared.

This is a local development server, not a production deployment. It binds to
localhost, holds no state between requests, and is meant for one person looking
at one application at a time.
"""

from __future__ import annotations

import base64
import io
import os
import random
import sys
from dataclasses import replace
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, render_template, request

import pipeline
from datagen.degrade import LEVELS, degrade
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
from invariants.checks import Bundle
from registry.gstn import MockGSTNRegistry

app = Flask(__name__)

AS_OF = date(2026, 8, 26)
SEED = 42

# One registry for the process, seeded once. Rebuilding it per request would
# make "this GSTIN is not registered" depend on when you clicked, which is not
# a property a reviewer should have to reason about.
_rng = random.Random(SEED)
POPULATION = [consistent_bundle(_rng) for _ in range(200)]
REGISTRY = MockGSTNRegistry.from_bundles(POPULATION)

# Each case carries three strings with different jobs:
#   short  — the sidebar label, scannable in one glance
#   caught — what stops it, so the list doubles as a summary of the defence
#   note   — the full sentence, shown only once a case is selected
ATTACKS = {
    "clean": {
        "short": "Clean",
        "caught": "—",
        "title": "Clean application",
        "note": "A real registered business applying honestly.",
    },
    "attack1": {
        "short": "Naive edit",
        "caught": "arithmetic",
        "title": "Attack 1 — naive edit",
        "note": "The forger edits the PAN card and forgets the GST "
                "certificate, so the two documents no longer agree.",
    },
    "attack2": {
        "short": "Fabricated",
        "caught": "registry",
        "title": "Attack 2 — coherent fabrication",
        "note": "An invented identity that satisfies every invariant. The "
                "forger did the maths correctly; only the registry can see "
                "that the number was never issued.",
    },
    "attack3": {
        "short": "Identity theft",
        "caught": "name mismatch",
        "title": "Attack 3 — identity theft",
        "note": "A genuine PAN/GSTIN pair with someone else's name attached. "
                "The identifiers agree with each other; the name does not "
                "match the registration.",
    },
    "attack5": {
        "short": "Impossible date",
        "caught": "date bounds",
        "title": "Attack 5 — impossible date",
        "note": "Registration dated before GST existed. Found by attacking "
                "our own checks after building them, not designed in advance.",
    },
    "attack4": {
        "short": "Stolen docs",
        "caught": "blind spot",
        "title": "Attack 4 — pixel-perfect theft",
        "note": "Genuine documents belonging to someone else, submitted "
                "unaltered. Every check passes because every check is "
                "correct. There is nothing in the documents to find.",
    },
}


def build_case(kind: str, rng: random.Random) -> Bundle:
    victim = rng.choice(POPULATION)
    if kind == "attack1":
        return attack_naive_edit(victim, rng)
    if kind == "attack2":
        return consistent_bundle(rng)
    if kind == "attack3":
        return attack_identity_theft(victim, rng)
    if kind == "attack5":
        return attack_impossible_date(victim, rng)
    if kind == "attack4":
        return attack_pixel_perfect(victim, rng)
    return victim


def encode_image(image) -> str:
    """PNG as a data URI, so the page needs no static file handling."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()).decode()


def serialise(result, truth: Bundle | None = None) -> dict:
    """Turn a PipelineResult into what the page needs.

    Findings carry their evidence through to the browser unchanged. A reviewer
    looking at a rejection must be able to see the two values that disagreed
    without going back to a terminal.
    """
    findings = [{
        "check": f.check,
        "passed": f.passed,
        "severity": f.severity,
        "message": f.message,
        "evidence": {k: str(v) for k, v in f.evidence.items()},
    } for f in result.findings]

    payload = {
        "decision": result.decision,
        "risk": round(result.risk, 4),
        "stages": result.stages_run,
        "findings": findings,
        "bundle": {
            "pan_number": result.bundle.pan_number,
            "pan_holder_name": result.bundle.pan_holder_name,
            "gstin": result.bundle.gstin,
            "gst_legal_name": result.bundle.gst_legal_name,
            "gst_trade_name": result.bundle.gst_trade_name,
            "gst_address_state": result.bundle.gst_address_state,
            "gst_registration_date": result.bundle.gst_registration_date,
            "declared_business_type": result.bundle.declared_business_type,
        },
    }

    if result.low_confidence_fields:
        payload["low_confidence"] = result.low_confidence_fields

    if result.tamper is not None:
        payload["tamper"] = {
            "score": round(result.tamper.score, 4),
            "suspicious": result.tamper.suspicious,
        }

    if result.explanation is not None:
        payload["explanation"] = {
            "text": result.explanation.text,
            "source": result.explanation.source,
        }

    # When the case was generated we know the truth, so OCR errors can be
    # shown as OCR errors rather than left looking like fraud.
    if truth is not None:
        payload["truth"] = {"pan_number": truth.pan_number,
                            "gstin": truth.gstin}
        payload["ocr_correct"] = (result.bundle.pan_number == truth.pan_number
                                  and result.bundle.gstin == truth.gstin)
    return payload


@app.route("/")
def index():
    return render_template("index.html",
                           attacks=ATTACKS,
                           levels=[level.name for level in LEVELS])


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate a case and run it, optionally through rendered images."""
    data = request.get_json(force=True)
    kind = data.get("kind", "clean")
    use_images = bool(data.get("use_images"))
    level = data.get("level", "good_photo")
    use_tamper = bool(data.get("use_tamper"))
    use_llm = bool(data.get("use_llm"))

    rng = random.Random(data.get("seed") or random.randrange(1 << 30))
    bundle = build_case(kind, rng)

    model_path = None
    if use_llm:
        import glob
        found = glob.glob("models/*.gguf")
        model_path = found[0] if found else None

    images = {}
    if use_images:
        pan_image, _ = render_pan_card(bundle)
        gst_image, _ = render_gst_certificate(bundle)
        pan_image = degrade(pan_image, level, seed=rng.randrange(1000))
        gst_image = degrade(gst_image, level, seed=rng.randrange(1000))
        images = {"pan": encode_image(pan_image), "gst": encode_image(gst_image)}

        detector = None
        if use_tamper and os.path.exists("models/tamper_resnet18.pt"):
            from tamper.detect import TamperDetector
            detector = TamperDetector("models/tamper_resnet18.pt")

        result = pipeline.run(pan_image=pan_image, gst_image=gst_image,
                              boxes=identifier_boxes(bundle),
                              registry=REGISTRY, tamper_detector=detector,
                              model_path=model_path, today=AS_OF)
        payload = serialise(result, truth=bundle)
    else:
        pan_image, _ = render_pan_card(bundle)
        gst_image, _ = render_gst_certificate(bundle)
        images = {"pan": encode_image(pan_image), "gst": encode_image(gst_image)}

        result = pipeline.run(bundle=bundle, registry=REGISTRY,
                              model_path=model_path, today=AS_OF)
        payload = serialise(result)

    payload["images"] = images
    meta = ATTACKS.get(kind, {})
    payload["title"] = meta.get("title", "Application")
    payload["note"] = meta.get("note", "")
    return jsonify(payload)


@app.route("/api/check", methods=["POST"])
def api_check():
    """Run the checks over hand-typed field values."""
    data = request.get_json(force=True)

    bundle = Bundle(
        pan_number=data.get("pan_number", "").strip().upper(),
        pan_holder_name=data.get("pan_holder_name", "").strip(),
        gstin=data.get("gstin", "").strip().upper(),
        gst_legal_name=data.get("gst_legal_name", "").strip(),
        gst_trade_name=data.get("gst_trade_name", "").strip(),
        gst_address_state=data.get("gst_address_state", "").strip(),
        gst_registration_date=data.get("gst_registration_date", "").strip(),
        declared_business_type=data.get("declared_business_type", "").strip(),
    )

    result = pipeline.run(bundle=bundle, registry=REGISTRY, today=AS_OF)
    return jsonify(serialise(result))


@app.route("/api/sample")
def api_sample():
    """A coherent identity to prefill the manual form."""
    bundle = random.choice(POPULATION)
    return jsonify({
        "pan_number": bundle.pan_number,
        "pan_holder_name": bundle.pan_holder_name,
        "gstin": bundle.gstin,
        "gst_legal_name": bundle.gst_legal_name,
        "gst_trade_name": bundle.gst_trade_name,
        "gst_address_state": bundle.gst_address_state,
        "gst_registration_date": bundle.gst_registration_date,
        "declared_business_type": bundle.declared_business_type,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Run the pipeline over two uploaded document images.

    Identifier boxes are the renderer's own coordinates: this reads documents
    produced by `run docs`, not arbitrary photographs. Locating fields on an
    unknown layout is a document-understanding problem this project does not
    claim to solve — see the README.
    """
    from PIL import Image

    if "pan" not in request.files or "gst" not in request.files:
        return jsonify({"error": "supply both a PAN and a GST image"}), 400

    pan_image = Image.open(request.files["pan"].stream).convert("RGB")
    gst_image = Image.open(request.files["gst"].stream).convert("RGB")

    reference = POPULATION[0]
    detector = None
    if os.path.exists("models/tamper_resnet18.pt"):
        from tamper.detect import TamperDetector
        detector = TamperDetector("models/tamper_resnet18.pt")

    result = pipeline.run(pan_image=pan_image, gst_image=gst_image,
                          boxes=identifier_boxes(reference),
                          registry=REGISTRY, tamper_detector=detector,
                          today=AS_OF)

    payload = serialise(result)
    payload["images"] = {"pan": encode_image(pan_image),
                         "gst": encode_image(gst_image)}
    payload["title"] = "Uploaded documents"
    payload["note"] = "Read with OCR, then checked."
    return jsonify(payload)


if __name__ == "__main__":
    print("=" * 66)
    print("  Paper Trail — http://127.0.0.1:5000")
    print("=" * 66)
    print(f"  registry seeded with {len(POPULATION)} businesses")
    print("  Ctrl-C to stop")
    print()
    app.run(host="127.0.0.1", port=5000, debug=False)
