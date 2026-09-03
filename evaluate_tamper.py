"""Train the tamper detector and measure held-out attack generalisation.

    python evaluate_tamper.py                # train + evaluate
    python evaluate_tamper.py --n 400        # bigger corpus
    python evaluate_tamper.py --epochs 6

The number that matters
-----------------------
Two figures are reported and they mean very different things:

* **Seen methods** — tampering the model trained on. This mostly measures
  whether the model can recognise our own generator, and is easy to make look
  good.
* **Held-out method** — a tampering technique the model has *never seen*
  (`datagen.tamper.HELD_OUT`). This is the honest answer to the sharpest
  objection a judge can raise: *"you designed both the attack and the defence."*

If the held-out number is much worse than the seen number, the model memorised
our editing operations rather than learning what tampering looks like. We report
both, in that order, and let the gap speak.

The `resave` control
--------------------
A portion of the clean set is re-encoded at a lower JPEG quality but not
otherwise edited. It is **not fraud**, and a detector that flags it has learned
"this region was recompressed" — true of every legitimately scanned document.
Its false-positive rate is reported separately for exactly that reason.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from datagen.degrade import degrade
from datagen.generate import consistent_bundle
from datagen.render import (
    identifier_boxes,
    render_gst_certificate,
    render_pan_card,
)
from datagen.tamper import BY_NAME, HELD_OUT, METHODS, apply_method
from evaluation.metrics import pr_auc
from tamper.detect import ela_features, prepare_crop

SEED = 42

# Degradation applied to every sample. Without it the model separates tampered
# from clean by JPEG history alone, which is not a skill that survives contact
# with a real upload.
DEGRADE_LEVEL = "good_photo"


def _random_identifier(rng: random.Random) -> str:
    import string
    return ("".join(rng.choice(string.ascii_uppercase) for _ in range(5))
            + "".join(rng.choice(string.digits) for _ in range(4))
            + rng.choice(string.ascii_uppercase))


def build_samples(n_bundles: int, rng: random.Random, methods: list[str]):
    """Render bundles and produce (crop, label, method) triples.

    Each bundle yields one clean crop and one tampered crop, so the classes are
    balanced by construction and accuracy on this set is not a meaningless
    number in the way it is for fraud overall.
    """
    samples = []
    donors = []

    for index in range(n_bundles):
        bundle = consistent_bundle(rng)
        use_pan = rng.random() < 0.5

        if use_pan:
            image, _ = render_pan_card(bundle)
            box = identifier_boxes(bundle)["pan"]["pan_number"]
        else:
            image, _ = render_gst_certificate(bundle)
            box = identifier_boxes(bundle)["gst"]["gstin"]

        donors.append(image)
        donor = donors[rng.randrange(len(donors))] if len(donors) > 1 else image

        # Clean. A quarter are `resave`d: recompressed but not edited, so the
        # model must learn that recompression alone is not evidence.
        clean_image = image
        clean_method = "clean"
        if rng.random() < 0.25:
            clean_image = apply_method("resave", image, box, rng)
            clean_method = "resave"

        clean_degraded = degrade(clean_image, DEGRADE_LEVEL, seed=index)
        samples.append((clean_degraded.crop(box), 0, clean_method))

        # Tampered.
        method = rng.choice(methods)
        tampered = apply_method(method, image, box, rng,
                                new_text=_random_identifier(rng), donor=donor)
        tampered_degraded = degrade(tampered, DEGRADE_LEVEL, seed=index)
        samples.append((tampered_degraded.crop(box), 1, method))

    return samples


def train(samples, epochs: int, device: str, seed: int):
    """Fine-tune ResNet18 on the training split."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)

    features = np.stack([prepare_crop(crop) for crop, _, _ in samples])
    labels = np.array([label for _, label, _ in samples], dtype=np.float32)

    dataset = TensorDataset(torch.from_numpy(features),
                            torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    from tamper.detect import build_model
    model = build_model(pretrained=True).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimiser.zero_grad()
            loss = criterion(model(batch_x).squeeze(1), batch_y)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch_y)

        print(f"    epoch {epoch + 1}/{epochs}  loss {total / len(dataset):.4f}")

    return model


def predict(model, samples, device: str) -> np.ndarray:
    import torch

    model.eval()
    features = np.stack([prepare_crop(crop) for crop, _, _ in samples])

    scores = []
    with torch.no_grad():
        for start in range(0, len(features), 32):
            batch = torch.from_numpy(features[start:start + 32]).to(device)
            scores.append(torch.sigmoid(model(batch).squeeze(1)).cpu().numpy())
    return np.concatenate(scores)


def ela_baseline(samples) -> np.ndarray:
    """ELA spatial-inconsistency alone, as the baseline the CNN must beat.

    If the CNN cannot beat six hand-computed statistics, the CNN is not
    earning its VRAM and we should say so rather than ship it.
    """
    return np.array([ela_features(crop)[4] for crop, _, _ in samples])


def report(name: str, scores: np.ndarray, labels: np.ndarray,
           methods: np.ndarray) -> None:
    flagged = scores >= 0.5
    positives = labels == 1

    recall = float(flagged[positives].mean()) if positives.any() else 0.0
    fpr = float(flagged[~positives].mean()) if (~positives).any() else 0.0

    print(f"  {name}")
    print(f"    PR-AUC          : {pr_auc(labels, scores):.4f}")
    print(f"    recall @0.5     : {recall:.1%}")
    print(f"    false positives : {fpr:.1%}")

    for method in sorted(set(methods.tolist())):
        mask = methods == method
        if not mask.any():
            continue
        rate = float(flagged[mask].mean())
        is_fraud = BY_NAME[method].is_fraud if method in BY_NAME else False
        label = "caught" if is_fraud else "FALSE ALARM"
        print(f"      {method:<12} {rate:6.1%} flagged   ({label})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300,
                        help="bundles per split (default 300)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save", default="models/tamper_resnet18.pt",
                        help="where to write the trained weights")
    args = parser.parse_args()

    import torch
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    seen_methods = [m.name for m in METHODS if m.is_fraud and m.name != HELD_OUT]

    print("=" * 74)
    print("TAMPER DETECTION — held-out attack generalisation")
    print("=" * 74)
    print(f"  trained on   : {', '.join(seen_methods)}")
    print(f"  HELD OUT     : {HELD_OUT}  (never seen during training)")
    print(f"  device       : {device}")
    print()

    rng = random.Random(args.seed)
    print("  building corpus...")
    train_samples = build_samples(args.n, rng, seen_methods)
    test_seen = build_samples(args.n // 3, rng, seen_methods)
    test_held_out = build_samples(args.n // 3, rng, [HELD_OUT])

    print(f"  train {len(train_samples)}  test-seen {len(test_seen)}  "
          f"test-held-out {len(test_held_out)}")
    print()

    started = time.time()
    model = train(train_samples, args.epochs, device, args.seed)
    print(f"  trained in {time.time() - started:.0f}s")

    if args.save:
        import torch
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"  weights saved to {args.save}")
    print()

    print("RESULTS")
    print("-" * 74)
    for name, samples in (("SEEN methods (trained on these)", test_seen),
                          (f"HELD-OUT method ({HELD_OUT}) — never trained on",
                           test_held_out)):
        scores = predict(model, samples, device)
        labels = np.array([label for _, label, _ in samples])
        methods = np.array([method for _, _, method in samples])
        report(name, scores, labels, methods)
        print()

    print("ELA-ONLY BASELINE (no learning — the bar the CNN must clear)")
    print("-" * 74)
    for name, samples in (("seen", test_seen), ("held-out", test_held_out)):
        labels = np.array([label for _, label, _ in samples])
        print(f"  {name:<10} PR-AUC {pr_auc(labels, ela_baseline(samples)):.4f}")
    print()

    print("HOW TO READ THIS")
    print("-" * 74)
    print("  The held-out figure is the honest one. A large gap between it and")
    print("  the seen figure means the model learned our editing operations")
    print("  rather than what tampering looks like.")
    print()
    print("  `resave` is a CONTROL: recompressed but not edited, so it is not")
    print("  fraud. Anything flagged there is a false alarm on a document that")
    print("  was merely scanned twice.")
    print()
    print("  Attack 4 is unaffected by any of this. A genuine document")
    print("  belonging to someone else has no pixel artefact to find.")
    print("=" * 74)


if __name__ == "__main__":
    main()
