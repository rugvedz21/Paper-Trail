"""Fine-tune TrOCR to read PAN and GSTIN crops.

    python train_trocr.py                  # ~25 min on a GTX 1650
    python train_trocr.py --n 10000 --epochs 4
    python train_trocr.py --eval-only      # score the saved checkpoint

Why fine-tune at all
--------------------
EasyOCR is generic: it reads any text in any font. Our problem is far narrower —
two fixed-format identifiers, one typeface, known lengths, a 36-character
alphabet. A model specialised on exactly that should beat a generalist, and OCR
is the ceiling on everything downstream: a misread PAN produces a mismatch the
invariant layer cannot distinguish from a forgery.

On training data
----------------
**Trained entirely on synthetic crops from our own renderer.** That is not a
compromise, it is the right choice twice over:

* CLAUDE.md forbids real identifiers in this repo, and every public dataset of
  Indian KYC documents contains real people's tax IDs, names and photographs.
* The model only ever needs to read *this* font in *this* layout. Generic OCR
  corpora (IAM, SROIE, MJSynth) are the wrong domain and would not help.

Every character the model trains on is generated, and the identifiers are
constructed to be structurally valid and mutually coherent — never looked up
against, or derived from, real registry data.

Model choice
------------
`trocr-small-printed` (62M), not `trocr-base` (334M). Measured on the target
4GB card:

    trocr-base   batch 2   2.40 s/sample   (spills to shared memory)
    trocr-small  batch 8   0.078 s/sample  3.1 GB peak, fits genuinely

31x faster, and the base model's "peak 6.4 GB" on a 4 GB card was Windows
silently paging rather than anything that fits. Speed here is not a nicety: it
is the difference between a 25-minute run and a 4-hour one.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image

from datagen.degrade import LEVELS, degrade
from datagen.generate import consistent_bundle
from datagen.render import (
    GST_SIZE,
    PAN_SIZE,
    identifier_boxes,
    render_gst_certificate,
    render_pan_card,
)

MODEL_NAME = "microsoft/trocr-small-printed"
OUTPUT_DIR = Path("models/trocr-identifiers")
SEED = 42

# Crop geometry is imported from the inference path rather than restated here.
# The model must train on exactly the framing the pipeline will feed it;
# a second copy of these would drift and the failure would be invisible —
# good evaluation numbers, bad reads in production.
from extract.ocr import PAD, _map_box, crop_field

# Levels sampled during training. `pristine` is included but rare: the model
# must handle clean scans, yet spending most of its capacity there would waste
# it on the easy case.
TRAIN_LEVELS = ("pristine", "clean_scan", "good_photo", "good_photo",
                "typical", "typical", "poor")


def build_dataset(n_bundles: int, rng: random.Random, levels=TRAIN_LEVELS):
    """Render bundles and cut identifier crops paired with their true text.

    Each bundle yields two samples — the PAN from the card and the GSTIN from
    the certificate — so the model sees both formats in proportion.
    """
    samples: list[tuple[Image.Image, str]] = []

    for index in range(n_bundles):
        bundle = consistent_bundle(rng)
        boxes = identifier_boxes(bundle)
        level = rng.choice(levels)

        pan_image, _ = render_pan_card(bundle)
        pan_image = degrade(pan_image, level, seed=index)
        samples.append((crop_field(pan_image, boxes["pan"]["pan_number"],
                                   PAN_SIZE), bundle.pan_number))

        gst_image, _ = render_gst_certificate(bundle)
        gst_image = degrade(gst_image, level, seed=index + 500_000)
        samples.append((crop_field(gst_image, boxes["gst"]["gstin"],
                                   GST_SIZE), bundle.gstin))

    return samples


def spaced(text: str) -> str:
    """Insert spaces so the BPE tokeniser emits one token per character.

    **This is the fix that made fine-tuning work at all.** TrOCR's tokeniser is
    BPE, trained on English prose, so it merges identifier characters into
    subword chunks — and those chunks are *unstable*:

        AAPFU0939F  -> ['AAP', 'FU', '09', '39', 'F']      5 tokens
        AAPFV0939F  -> ['AAP', 'F', 'V', '09', '39', 'F']  6 tokens

    One character changed, and the whole segmentation shifts. The model was
    being asked to predict a chunking that varies unpredictably with the
    content it is trying to read, which is far harder than reading characters
    and is close to unlearnable at this data scale. It scored 0-2%.

    Spacing the characters makes every identifier exactly `len(text)` tokens,
    each one a single character. See NOTES.md.
    """
    return " ".join(text)


def unspaced(text: str) -> str:
    """Recover an identifier from the model's spaced output."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def encode(processor, samples, batch: slice):
    """Turn a slice of samples into model inputs."""
    import torch

    chunk = samples[batch]
    pixel_values = processor(images=[img.convert("RGB") for img, _ in chunk],
                             return_tensors="pt").pixel_values
    labels = processor.tokenizer([spaced(text) for _, text in chunk],
                                 return_tensors="pt", padding=True).input_ids

    # -100 tells the loss to ignore padding. Without this the model is rewarded
    # for predicting pad tokens, which it will happily learn to do.
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return pixel_values, labels


def train(samples, epochs: int, batch_size: int, device: str, lr: float):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    order = list(range(0, len(samples), batch_size))

    model.train()
    for epoch in range(epochs):
        random.shuffle(order)
        total, seen = 0.0, 0
        started = time.time()

        for step, start in enumerate(order):
            pixel_values, labels = encode(
                processor, samples, slice(start, start + batch_size))
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            total += float(loss) * len(labels)
            seen += len(labels)

            if (step + 1) % 50 == 0:
                print(f"      step {step + 1}/{len(order)}  "
                      f"loss {total / seen:.4f}  "
                      f"({time.time() - started:.0f}s)")

        print(f"    epoch {epoch + 1}/{epochs}  loss {total / seen:.4f}  "
              f"({time.time() - started:.0f}s)")

    return processor, model


def evaluate(processor, model, samples, device: str, batch_size: int = 16):
    """Exact-match rate. A single wrong character is a wrong read."""
    import torch

    model.eval()
    correct, results = 0, []

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            pixel_values = processor(
                images=[img.convert("RGB") for img, _ in chunk],
                return_tensors="pt").pixel_values.to(device)

            # max_length must cover a spaced 15-character GSTIN plus the
            # start/end tokens; 24 truncated them and looked like a model
            # failure rather than a decoding limit.
            generated = model.generate(pixel_values, max_length=40)
            texts = processor.batch_decode(generated, skip_special_tokens=True)

            for (_, truth), predicted in zip(chunk, texts):
                cleaned = unspaced(predicted)
                correct += cleaned == truth
                results.append((truth, cleaned))

    return correct / len(samples), results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000,
                        help="bundles (each yields 2 crops)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8,
                        help="8 fits in 4GB; 16 spills to shared memory")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import torch
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 74)
    print("FINE-TUNING TrOCR ON IDENTIFIER CROPS")
    print("=" * 74)
    print(f"  model    : {MODEL_NAME} (62M)")
    print(f"  device   : {device}")
    print("  data     : synthetic, from our own renderer — no real identifiers")
    print()

    rng = random.Random(args.seed)

    if args.eval_only:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained(OUTPUT_DIR)
        model = VisionEncoderDecoderModel.from_pretrained(OUTPUT_DIR).to(device)
    else:
        print(f"  building {args.n} bundles -> {args.n * 2} crops...")
        train_samples = build_dataset(args.n, rng)
        print(f"  {len(train_samples)} training crops")
        print()

        started = time.time()
        processor, model = train(train_samples, args.epochs, args.batch,
                                 device, args.lr)
        print(f"  trained in {time.time() - started:.0f}s")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print(f"  saved to {OUTPUT_DIR}")
        print()

    # Held-out evaluation, per degradation level, against the EasyOCR numbers
    # the README already reports.
    print("EXACT-MATCH READ RATE BY IMAGE QUALITY")
    print("-" * 74)
    print(f"  {'level':<12}{'TrOCR':>10}{'EasyOCR':>10}   (EasyOCR from run ocr)")

    easyocr_reference = {
        "pristine": 0.96, "clean_scan": 0.96, "good_photo": 0.94,
        "typical": 0.94, "poor": 0.80, "terrible": 0.10,
    }

    for level in LEVELS:
        # A different seed stream from training, so these bundles are
        # genuinely unseen. The *same* seed for every level on purpose: the
        # identities are held constant across the table so image quality is
        # the only variable moving.
        test_rng = random.Random(args.seed + 7777)
        held_out = build_dataset(60, test_rng, levels=(level.name,))
        rate, results = evaluate(processor, model, held_out, device)

        reference = easyocr_reference.get(level.name)
        comparison = f"{reference:>9.0%}" if reference else " " * 10
        print(f"  {level.name:<12}{rate:>9.0%}{comparison}")

        if level.name == "typical":
            wrong = [(t, p) for t, p in results if t != p][:3]
            for truth, predicted in wrong:
                print(f"      miss: {truth} -> {predicted or '(empty)'}")

    print()
    print("  EasyOCR figures are the mean of its PAN and GSTIN columns, which")
    print("  is the closest like-for-like comparison: both are exact-match")
    print("  rates over the same identifier crops at the same quality levels.")
    print("=" * 74)


if __name__ == "__main__":
    main()
