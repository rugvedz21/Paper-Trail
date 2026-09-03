"""Reproduce every number in the README.

    python evaluate.py

One command, fixed seed, no GPU and no network. Anything the README claims that
this script does not print is a claim the repo cannot support.
"""

import argparse
import os
import random
import sys
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from datagen.generate import (
    attack_identity_theft,
    attack_impossible_date,
    attack_naive_edit,
    attack_pixel_perfect,
    consistent_bundle,
)
from evaluation.metrics import (
    AttackRecall,
    CostMatrix,
    EvaluationReport,
    brier_score,
    calibration_curve,
    choose_threshold,
    pr_auc,
)
from evaluation.score import risk_score
from invariants.checks import check_bundle
from registry.gstn import MockGSTNRegistry, check_registry

SEED = 42

# The date the evaluation is computed "as of". Fixed so the registration-date
# check cannot make the reported numbers drift as real time passes.
AS_OF = date(2026, 8, 26)

# Roughly the shape of real merchant onboarding: most applications are honest.
FRAUD_RATE = 0.08
N_CASES = 5_000
N_REGISTERED = 4_000


def build_corpus(rng: random.Random):
    """Build the evaluation corpus and the registry it is checked against.

    The registry is seeded *first*, from a population of genuine businesses.
    Legitimate applicants are drawn from that population; fraudulent ones are
    not. That ordering matters: if the registry were seeded from the corpus,
    every case would be found and the registry stage would look useless.
    """
    population = [consistent_bundle(rng) for _ in range(N_REGISTERED)]
    registry = MockGSTNRegistry.from_bundles(population)

    cases = []
    for _ in range(N_CASES):
        if rng.random() >= FRAUD_RATE:
            # Legitimate: a real registered business applying honestly.
            cases.append((rng.choice(population), 0, "legitimate"))
            continue

        attack = rng.choice([1, 2, 3, 4, 5])
        if attack == 1:
            victim = rng.choice(population)
            cases.append((attack_naive_edit(victim, rng), 1, "attack_1_naive_edit"))
        elif attack == 2:
            cases.append((consistent_bundle(rng), 1, "attack_2_fabrication"))
        elif attack == 3:
            victim = rng.choice(population)
            cases.append((attack_identity_theft(victim, rng), 1,
                          "attack_3_identity_theft"))
        elif attack == 4:
            # Attack 4: a genuine document set submitted by someone else.
            # Nothing about the *data* differs — that is the whole problem.
            victim = rng.choice(population)
            cases.append((attack_pixel_perfect(victim, rng), 1,
                          "attack_4_pixel_perfect"))
        else:
            # Attack 5: an impossible registration date. Found by attacking
            # the checks, not by design — see NOTES.md, 26 Aug.
            victim = rng.choice(population)
            cases.append((attack_impossible_date(victim, rng), 1,
                          "attack_5_impossible_date"))

    return cases, registry


def evaluate(cases, registry, costs: CostMatrix) -> EvaluationReport:
    scores, labels, kinds = [], [], []

    for bundle, label, kind in cases:
        # Clock pinned: an evaluation whose numbers drift with the wall clock
        # is not reproducible, and the README quotes these figures.
        findings = check_bundle(bundle, today=AS_OF)
        findings.append(check_registry(bundle, registry))
        scores.append(risk_score(findings))
        labels.append(label)
        kinds.append(kind)

    scores = np.array(scores)
    labels = np.array(labels)
    kinds = np.array(kinds)

    threshold = choose_threshold(labels, scores, costs)
    flagged = scores >= threshold.threshold

    # Baseline: approve everything. The cost the pipeline has to beat.
    baseline = costs.expected_cost(labels, np.zeros_like(labels, dtype=bool))

    per_attack = []
    for kind in ["attack_1_naive_edit", "attack_2_fabrication",
                 "attack_3_identity_theft", "attack_4_pixel_perfect",
                 "attack_5_impossible_date"]:
        mask = kinds == kind
        if not mask.any():
            continue
        per_attack.append(AttackRecall(
            attack=kind,
            caught=int(flagged[mask].sum()),
            total=int(mask.sum()),
            expected_to_catch=(kind != "attack_4_pixel_perfect"),
            note=("stated blind spot — genuine documents, nothing to detect"
                  if kind == "attack_4_pixel_perfect" else ""),
        ))

    return EvaluationReport(
        n_cases=len(cases),
        n_fraud=int(labels.sum()),
        pr_auc=pr_auc(labels, scores),
        brier=brier_score(labels, scores),
        threshold=threshold,
        baseline_cost=baseline,
        per_attack=per_attack,
        calibration=calibration_curve(labels, scores),
    )


def print_report(report: EvaluationReport, costs: CostMatrix) -> None:
    rule = "=" * 70

    print(rule)
    print("PAPER TRAIL — EVALUATION")
    print(rule)
    print(f"  cases              : {report.n_cases:,}")
    print(f"  fraudulent         : {report.n_fraud:,} "
          f"({report.fraud_rate:.1%})")
    print("  accuracy           : not reported — meaningless under this "
          "imbalance")
    print()

    print("RANKING QUALITY")
    print(f"  PR-AUC             : {report.pr_auc:.4f}   "
          f"(baseline = fraud rate = {report.fraud_rate:.4f})")
    print(f"  Brier score        : {report.brier:.4f}   (lower is better)")
    print("  ROC-AUC            : not reported — flatters under imbalance")
    print()

    print("COST-WEIGHTED OPERATING POINT")
    print(f"  cost matrix        : FN ₹{costs.false_negative:,.0f}  "
          f"FP ₹{costs.false_positive:,.0f}  review ₹{costs.review:,.0f}")
    print(f"  chosen threshold   : {report.threshold.threshold:.4f}  "
          "(minimises expected cost, not F1)")
    print(f"  precision / recall : {report.threshold.precision:.4f} / "
          f"{report.threshold.recall:.4f}")
    print(f"  flagged            : {report.threshold.flagged:,} of "
          f"{report.n_cases:,}")
    print(f"  expected cost      : ₹{report.threshold.expected_cost:,.0f}  "
          f"(₹{report.threshold.cost_per_case:,.2f}/case)")
    print(f"  approve-all cost   : ₹{report.baseline_cost:,.0f}")
    print(f"  avoided            : ₹{report.cost_saved:,.0f}")
    print()

    print("PER-ATTACK RECALL")
    for row in report.per_attack:
        status = "" if row.expected_to_catch else "  <- UNCAUGHT BY DESIGN"
        print(f"  {row.attack:<26} {row.caught:>4}/{row.total:<4} "
              f"= {row.recall:6.1%}{status}")
        if row.note:
            print(f"      {row.note}")
    print()

    print("CALIBRATION  (predicted -> observed fraud rate)")
    predicted, observed, counts = report.calibration
    for p, o, n in zip(predicted, observed, counts):
        bar = "#" * int(round(o * 40))
        print(f"  {p:.2f} -> {o:.2f}  n={n:<6} {bar}")
    print()
    print("  Scores are rule-derived, not a learned posterior — the weights")
    print("  are hand-set, so calibration is an artefact of those weights.")
    print("  Shown rather than hidden. See src/evaluation/score.py.")
    print(rule)


def print_seed_stability(n_seeds: int, costs: CostMatrix) -> None:
    """Is the headline number an artefact of one lucky seed?

    Reported because a single-seed result is not evidence, and a reader is
    entitled to check the spread rather than take one run on trust.
    """
    import statistics

    print("=" * 70)
    print(f"SEED STABILITY — {n_seeds} independent corpora")
    print("=" * 70)
    print(f"{'seed':>5} {'PR-AUC':>9} {'threshold':>10} {'precision':>10} "
          f"{'recall':>8} {'attack 4':>9}")

    pr_aucs, recalls = [], []
    for seed in range(n_seeds):
        report = evaluate(*build_corpus(random.Random(seed)), costs)
        blind = next(r for r in report.per_attack if "pixel" in r.attack)
        pr_aucs.append(report.pr_auc)
        recalls.append(report.threshold.recall)
        print(f"{seed:>5} {report.pr_auc:9.4f} {report.threshold.threshold:10.4f} "
              f"{report.threshold.precision:10.4f} "
              f"{report.threshold.recall:8.4f} {blind.recall:9.1%}")

    print()
    print(f"  PR-AUC : mean {statistics.mean(pr_aucs):.4f}  "
          f"sd {statistics.pstdev(pr_aucs):.4f}  "
          f"range [{min(pr_aucs):.4f}, {max(pr_aucs):.4f}]")
    print(f"  recall : mean {statistics.mean(recalls):.4f}  "
          f"sd {statistics.pstdev(recalls):.4f}")
    print()


def print_cost_sweep(seed: int) -> None:
    """How much does the operating point depend on the illustrative costs?

    The answer on this corpus is *not at all*, and that is a limitation rather
    than a strength — see the note printed below.
    """
    cases, registry = build_corpus(random.Random(seed))

    print("=" * 70)
    print("COST SENSITIVITY — does the threshold depend on the cost matrix?")
    print("=" * 70)
    print(f"{'FN':>10} {'FP':>8} {'ratio':>8} {'threshold':>10} "
          f"{'precision':>10} {'recall':>8}")

    pairs = [(5_000, 800), (20_000, 800), (50_000, 800), (200_000, 800),
             (1_000_000, 800), (50_000, 5_000), (50_000, 20_000),
             (50_000, 50_000)]

    thresholds = set()
    for fn, fp in pairs:
        report = evaluate(cases, registry,
                          CostMatrix(false_negative=fn, false_positive=fp))
        thresholds.add(round(report.threshold.threshold, 4))
        print(f"{fn:10,.0f} {fp:8,.0f} {fn / fp:8.1f} "
              f"{report.threshold.threshold:10.4f} "
              f"{report.threshold.precision:10.4f} "
              f"{report.threshold.recall:8.4f}")

    print()
    if len(thresholds) == 1:
        print("  The operating point is INVARIANT across a 200x range of")
        print("  FN:FP ratios. This is a limitation of the corpus, not a")
        print("  strength of the method: rule-derived scores are bimodal, no")
        print("  legitimate case scores above zero, and with no overlap region")
        print("  every threshold in the gap costs the same. Real OCR noise")
        print("  would push legitimate cases off zero and give the cost matrix")
        print("  something to decide. Until then the cost weighting is")
        print("  machinery that has not yet been exercised.")
    else:
        print(f"  {len(thresholds)} distinct operating points across the sweep.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fn-cost", type=float, default=50_000.0,
                        help="cost of approving a fraudulent merchant")
    parser.add_argument("--fp-cost", type=float, default=800.0,
                        help="cost of declining a legitimate merchant")
    parser.add_argument("--seeds", type=int, metavar="N",
                        help="also run N seeds and report the spread")
    parser.add_argument("--sweep", action="store_true",
                        help="also sweep the cost matrix")
    args = parser.parse_args()

    costs = CostMatrix(false_negative=args.fn_cost, false_positive=args.fp_cost)

    cases, registry = build_corpus(random.Random(args.seed))
    print_report(evaluate(cases, registry, costs), costs)

    if args.seeds:
        print()
        print_seed_stability(args.seeds, costs)
    if args.sweep:
        print()
        print_cost_sweep(args.seed)


if __name__ == "__main__":
    main()
