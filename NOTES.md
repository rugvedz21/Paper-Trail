# What broke, and how I got out

Running log. Razorpay's form asks this as its final question and they say it's
the first thing they read — so write entries as they happen, not at 3am on the 3rd.

Format: what I saw → what I measured → what I changed → what the number became.

---

## 27 Aug — PAN check digit turned out to be unimplementable

**Saw:** planned to validate the PAN's 10th character as a check digit, the way
the GSTIN's 15th is validated.

**Found:** the GSTIN check character is documented (mod-36 weighted scheme,
verifiable against the published example `27AAPFU0939F1Z` → `V`). The PAN check
digit algorithm has **never been published** by the Income Tax Department —
every "PAN validator" online only checks the regex.

**Changed:** dropped the check rather than shipping an invented one. Validate PAN
*structure* plus *cross-document consistency* instead, and state the limitation
explicitly in the README.

**Result:** six working checks instead of seven, and no false claim in the repo.
Attack-1 catch rate unaffected — 500/500 — because the PAN↔GSTIN equality check
was always doing that work.

---

## 26 Aug — renderer needed field coordinates the tamper stage could reuse

**Saw:** first cut of `datagen/render.py` drew each field with an inline
`draw.text(...)` call. Fine for producing images, but stage 4 has to crop the
identifier regions to feed the CNN, and those crop coordinates would have been
a second copy of the same numbers — guaranteed to drift the first time a label
moved.

**Changed:** fields are declared in a `Field` table (key, label, value accessor,
position, font) and drawn from it. `identifier_boxes()` re-renders and returns
the `textbbox` of every field flagged `identifier=True`, so crops come from the
same table that did the drawing. Rendering twice to get boxes is wasteful; at
datagen volumes it hasn't been worth caching.

**Result:** PAN and GST certificate both render; identifier boxes come back
tight around the ink (PAN `(72, 311, 384, 345)`, GSTIN `(120, 369, 481, 394)`).
The GSTIN renders as `27AAPFU0939F1ZV` against PAN `AAPFU0939F` — the
containment the whole project rests on is now visible in pixels.

---

## 26 Aug — the attack-1 test nearly measured the wrong check

**Saw:** first version of `attack_naive_edit` replaced a random PAN character
with a random character from `A-Z0-9`. Catch rate 500/500 immediately.

**Found:** too easy, and for the wrong reason. The PAN structure is 5 letters,
4 digits, 1 letter. An edit that dropped a digit into the letter block was
caught by `pan_structure` — a regex — before `pan_gstin_agreement` ever ran.
The headline number would have been partly measuring a regex rather than the
cross-document equality that is the point of the project.

**Changed:** the attack now draws its replacement from the pool valid at that
position, so every forged PAN stays structurally legal and the equality check
is the only thing that can catch it. Added
`test_attack_1_preserves_pan_structure` to hold that property, so the test
cannot silently regress into the easy version.

**Result:** still 500/500, but now the number means what the README will claim
it means. Confirmed by mutation: stubbing `pan_gstin_agreement` to always pass
drops it to 0/500 and fails four tests — before the fix, the regex would have
absorbed some of those.

---

## 26 Aug — asserting the blind spots instead of hoping for them

**Saw:** attacks 2 and 4 are uncatchable by the invariant layer by design —
attack 2 is a coherent fabrication, attack 4 is a genuine document set. Easy to
leave them as prose in the README.

**Changed:** wrote them as tests that assert the layer returns `pass`.

**Why it earns its place:** if an invariant ever becomes accidentally sensitive
to one of them, these fail loudly. Without them, the registry and tamper stages
could quietly get credit for work the invariants were doing by accident — which
would make the per-attack recall table wrong in the most embarrassing direction.

**Result:** 35 tests, all passing. Blind spots are now part of the suite rather
than a claim nobody re-checks.

**Check count, for the record:** the 27 Aug entry above says "six checks" — that
was the count at the time. `state_agreement` has since been added, so the
current total is **seven**: PAN structure, GSTIN structure, GSTIN checksum,
PAN↔GSTIN equality, entity type, name, state. Still no PAN check digit, and
`test_no_pan_check_digit_function_exists` fails the build if anyone adds one.

---

## 26 Aug — the evaluation reported 100% on the attack we cannot detect

**Saw:** first run of `evaluate.py` printed `attack_4_pixel_perfect 95/95 =
100.0%` on the same line as the label "UNCAUGHT BY DESIGN". The two claims
cannot both be true.

**Measured:** dumped the score distribution per case type. It was actually
correct — attack 4 scores exactly `0.000`, identical to legitimate cases, while
attacks 1–3 score 0.85–0.97. The detector was fine. The *threshold search* was
not: it had chosen `0.0000` and flagged all 5,000 cases.

**Why it happened:** with FN at ₹50,000 against FP at ₹800, a blanket decline
genuinely is the cost-minimising policy on this corpus, and `scores >= 0.0` is
true for everything. The arithmetic was right and the conclusion was garbage —
a blanket decline is not a detector, and reporting its recall credits us for
catching cases we cannot see.

**Changed:** candidate thresholds must be strictly positive. A zero score means
*no check fired*, i.e. no evidence, and "no evidence" must never be an operating
point. Where nothing scores above zero, the honest answer is to flag nothing and
report recall 0.0, which is now what happens.

**Result:** attack 4 reports **0/95 = 0.0%**, as it always should have.
Precision 1.0000, recall 0.7532 — and that recall ceiling is *correct*, because
attack 4 is roughly a quarter of the fraud in the corpus and is undetectable by
this layer. PR-AUC 0.7722 against a 0.0770 base rate.

**The lesson worth keeping:** this bug made the system look *better* than it is.
Nothing in the test suite would have caught it, because every individual
component was working. Regression tests now exist
(`test_zero_score_is_never_a_threshold`, `test_undetectable_fraud_is_not_
credited`) and I check any metric that improves unexpectedly before believing it.

---

## 26 Aug — GPU stages were silently blocked by the Python version

**Saw:** before writing any model code, checked whether the 1650 was actually
reachable. `torch.cuda.is_available()` returned `False` on a healthy card
(driver 592.82, 4096MiB free).

**Found:** the system interpreter is Python 3.14, and PyTorch ships no CUDA
wheels for it — `pip install torch --index-url .../cu124` gives *No matching
distribution found*. The installed `torch 2.11.0+cpu` would have run every model
on CPU without ever erroring. `llama_cpp_python` was likewise built without GPU
offload, and EasyOCR was not installed at all.

**Changed:** created a project venv on Python 3.12 (already present at
`C:\Python312`), installed `torch 2.6.0+cu124` into it. The invariant layer is
pure Python and was unaffected — 35 tests passed identically on both.

**Result:** `torch.cuda.is_available()` is `True`, device reports as GTX 1650
4.0GB SM 7.5, and a 2000×2000 matmul runs on device at 40.1MB peak. Found this
a week before submission rather than on the 1st, which is the whole argument for
checking the environment before writing code against it.

---

## 26 Aug — attacked my own checks and found an unvalidated field

**Saw:** rather than add stage 4, spent a session trying to break what exists.
Wrote seven attacks a competent forger would actually try, beyond the four in
the design.

**Found:** four got through. Three of them *should* — a forger who recomputes
the GSTIN after editing the PAN has produced a coherent identity, which is
attack 2 by another route and is caught by the registry (confirmed: reports
`not_found`). Trade-name substitution and a changed entity number have no ground
truth to check against.

The fourth was a real gap. **`gst_registration_date` was read by no check at
all.** It sat on every `Bundle`, was printed on every rendered certificate, and
nothing validated it. A forger backdating a registration to imply years of
trading history went straight through, as did `31/12/2099`.

**Changed:** added `check_registration_date`. Two closed-form bounds, which is
what makes it belong in this layer: GST commenced 1 July 2017 so nothing can
predate it, and nothing can be dated in the future. An *unparseable* date is a
WARN not a FAIL — OCR mangles dates constantly and declining a merchant over a
misread digit is precisely the false positive the cost matrix says to avoid.
`today` is injectable so the suite cannot rot as real time passes.

**Then it broke eight existing tests** — and the fault was the *generator*, not
the check. `consistent_bundle` drew year 2017–2025 with any month, so it happily
emitted registrations dated before GST existed. It had been producing impossible
"clean" identities from the start and nothing noticed, because nothing looked at
the field. Fixed the generator to constrain 2017 to July onward.

Also replaced a hardcoded `len(check_bundle(...)) == 7` with a comparison
against `len(ALL_CHECKS)`, so adding check nine cannot silently leave a stale
assertion behind.

**Result:** eight checks. Attack 5 added to the corpus and caught 66/66. Recall
rose 0.7532 → 0.7995 because a catchable attack joined the mix. 72 tests.

**Worth saying plainly in the writeup:** attack 5 was not designed in advance.
It was found by attacking the defence after building it, which is the only
honest answer to "you designed both the attack and the defence."

---

## 26 Aug — the cost matrix turns out not to be doing anything

**Saw:** wanted to know whether the reported threshold was fragile to the
illustrative cost figures, since those are placeholders for Razorpay's real
numbers.

**Measured:** swept the FN:FP ratio from 6:1 to 1250:1 — a 200× range. The
chosen threshold did not move. Not approximately: identically 0.3500 in all
eight configurations, same precision, same recall, same cases flagged.

**Why:** dumped the score distribution. Only **seven distinct scores** occur in
the whole corpus, and **zero legitimate cases score above 0.0**. The classes are
perfectly separated with a clean gap, so every threshold inside that gap has
identical cost and the argmin is arbitrary within it. The cost matrix has
nothing to decide.

**Not "fixed", because it is not a bug.** The code is correct; the *corpus* is
too easy. Scores are bimodal because there is no OCR noise — in reality one
misread character pushes a legitimate application off zero, creating the overlap
region where the FN:FP ratio genuinely determines where to cut.

**Changed:** added `--sweep` and `--seeds` to `evaluate.py` so both analyses are
reproducible rather than something I ran once in a terminal, and wrote the
limitation into the README in the cost section itself — where a reader
evaluating that claim will actually be looking, not buried at the bottom.

**Result:** the README now says the cost weighting is correct machinery that has
not yet been exercised. Presenting an invariant threshold as evidence of
robustness would have been the easy read and the dishonest one.

**Also from the same run:** seed stability over 10 corpora — PR-AUC 0.8097 ±
0.0255, precision 1.0000 and attack-4 recall 0.0% on every seed. Seed 42 is
slightly below the mean, so the headline number is not cherry-picked.

---

## 26 Aug — "fragment-tolerant fuzzy matching" was never specified

**Saw:** the extraction task named fragment-tolerant fuzzy matching as a
requirement, described in CLAUDE.md. Grepped for it: `fuzzy`, `fragment` and
`tolerant` appear nowhere in CLAUDE.md or anywhere else in the repo. The
extraction section says only "EasyOCR line detection + fine-tuned TrOCR. A
front end onto `Bundle`, never a hard dependency."

**Changed:** built it anyway, because the need is real — EasyOCR splits a
15-character GSTIN across boxes and confuses `0`/`O`, `1`/`I`, `5`/`S` — but
designed it myself and said so at the top of `src/extract/ocr.py` rather than
implying it came from a spec. The design: reassemble fragments into
fixed-length candidates (each alone, windowed if over-long, and consecutive
joins), filter by the published structural regex, then rank survivors by a
per-position character-class prior.

**Flagged for review:** the priors are the part worth checking. GSTIN position
13 (entity number) and 15 (check character) are `[0-9A-Z]` in the published
structure, so the regex cannot separate a genuine letter from a misread digit.
The prior *prefers* digits there — which fixes the observed `...N1ZK` read as
`...NIZK` — but never rejects a letter, so `27AAPFU0939F1ZV` keeps its real
`V`. If that preference is wrong for real GSTINs, it is one constant to change.

---

## 26 Aug — the OCR evaluation was measuring my own bug

**Saw:** first `evaluate_ocr.py` run showed GSTIN at **67% on pristine
renders**. A perfect PIL-drawn image should read at ~100%; something was wrong
with the reader, not the images.

**Measured:** dumped the raw fragments per crop. Two separate faults:

1. `PAD = 26` around the identifier box pulls in the label printed above it,
   which OCR reads as garbage (`TE9SMNUMDEQJOTINA` for "Registration Number
   (GSTIN)").
2. On one bundle the real fragment carried **three** simultaneous misreads
   (`5`→`S`, `5`→`S`, `1`→`I`), exceeding `_class_repairs(max_edits=2)`. No
   repair validated, and `best_match` then fell back to `candidates[0]` — which
   was a 15-character *window of the label garbage*. It returned noise in
   preference to the actual identifier.

**Worse:** `_misread_penalty` returned `0.0` on a length mismatch, so any
wrong-length garbage scored as a *perfect* match to the expected shape.

**Changed:** the fallback now ranks candidates by how well they fit the
character-class prior instead of taking the first; `_misread_penalty` returns
`inf` for a length mismatch; `max_edits` raised to 3 to cover the misread count
real crops actually produce.

**Result:** pristine went 67% → **100%** on every field. The degradation curve
below it is unchanged in shape, which is the tell that this was a
scoring/selection bug rather than a reading one.

**The lesson, again:** the first number was *too low* rather than too high, so
it looked like an honest bad result and was nearly accepted as one. Both
directions of surprising number are worth a debugger.

---

## 26 Aug — case folding, and what counts as a misread

**Saw:** three name fields read as `NIMBUS TRADERS` against a truth of
`Nimbus Traders`, and were being scored as failures.

**Found:** not an OCR error at all. `render.py` prints names through `.upper()`,
so the certificate genuinely says `NIMBUS TRADERS`. Scoring that as a miss
measures the renderer's styling, not the reader.

**Changed:** `evaluate_ocr.py` compares free-text fields case-insensitively and
whitespace-normalised, and identifiers **strictly**. The asymmetry is
deliberate and documented: a PAN is uppercase by construction and one wrong
character breaks the cross-document check the project rests on, so there is no
tolerance to give there.

---

## 26 Aug — I fixed three OCR cases and generalised from a sample of three

**Saw:** the 3-bundle smoke test read every field at 100% on pristine renders.
Declared the extraction fixed. The 25-bundle run then reported GSTIN at **56%**.

**What I had actually done:** debugged the three failures in front of me,
inferred a rule from them, and shipped it. One of those three had a misread `1`
at GSTIN position 13, so I marked positions 13 *and 15* as digit-preferring in
the character-class prior.

**What that broke:** position 15 is the mod-36 check character, which is a
**letter roughly 26 times in 36**. My prior was rewriting *correctly read*
letters into digits — `...S1ZB` → `...S1Z8`, `...A1ZU` → `...A1Z0`. Across 25
bundles the prior was *causing* more failures than it fixed. A rule induced
from one observation, applied to a position where it was simply false.

**Changed, and this is the better design:** stopped guessing character classes
at unconstrained positions and used the arithmetic instead. `gstin_check_char`
already exists and is verified against the published example, so `best_match`
now takes an optional `validator`. When OCR produces nothing that validates, it
searches every known confusion at every position (up to 4 edits) and keeps only
readings whose check character agrees with their own first fourteen.

That is a much stronger filter than any class heuristic — 1 in 36 by chance —
and it is *documented arithmetic* rather than my inference. The wide search is
gated on the validator existing, precisely so it can never manufacture a
plausible-looking PAN, which has no published checksum.

**Result:** GSTIN **56% → 92%** on pristine; both-identifiers **56% → 88%**.

**Two smaller bugs found on the way down:**

1. `GSTIN_PRIOR` position 14 held the literal `Z`, but `_class_repairs` only
   understood the class codes `D`/`A`/`?` and silently ignored anything else —
   so a misread `Z` was never repaired. Literals are now handled explicitly.

2. `_DIGIT_TO_LETTER` was a `dict[str, str]`. `0` is confusable with `O`, `D`
   *and* `Q`, so `("0", "Q")` overwrote `("0", "O")` and the **most likely**
   repair was never generated: the real PAN `GROCT2624O` came back as
   `GROCT2624Q`. It needed a multimap. Now every alternative is generated in
   likelihood order.

**The lesson:** three examples is not a sample, it is an anecdote. The smoke
test existed to check the code *ran*, and I let it answer a question about
whether the code was *right*. Every claim in the README now comes from the
25-bundle run.

**Known ceiling, not worth forcing:** a PAN whose final character is a genuine
`O` read as `0` is unrecoverable — `O`, `D` and `Q` are all valid letters there
and there is no PAN checksum to adjudicate. Inventing one is forbidden and
would be worse than the miss.

---

## 30 Aug — the held-out tamper attack was trivially detectable, for the wrong reason

**Saw:** `patch_paste` — the method deliberately held out of training — produced
an **all-zero** ELA feature vector. Every other method gave non-zero values.

**Found:** `patch_paste` sampled its donor region with uniform random
coordinates. A PAN card is mostly blank card stock, so the "region from another
document" was almost always a **flat rectangle of background**. That is not a
composite forgery. It has one unique colour, no compression history worth
speaking of, and a detector would flag it instantly — for a reason that has
nothing to do with tampering.

The held-out generalisation number, the single figure that answers *"you
designed both the attack and the defence"*, would have been measuring a bug.

**Changed:** the donor region is now chosen by variance — sample up to 24
candidate positions and take the most textured, stopping early once clearly
inked. `test_patch_paste_pastes_real_content` asserts the pasted region has more
than 20 distinct colours so this cannot regress into pasting blank paper again.

**Result:** held-out PR-AUC **0.9936**, recall 95.0%.

**And an honest reading of that number.** It is *higher* than the seen-methods
figure (0.8989), which is the opposite of the usual failure. Not because the
model generalises brilliantly — because `splice`, one of the methods it *was*
trained on, is caught only **10.5%** of the time. Copy-move within a single
document leaves noise, lighting and JPEG history all matching. There is almost
nothing to find. One attack is easy and another is nearly invisible, and which
is which does not depend on what we trained on. The README says exactly that.

**The control earned its place too.** A quarter of the clean set is `resave` —
recompressed but not edited. It is flagged 3.6% of the time. Had we omitted it,
a detector that had merely learned "this region was re-encoded" would have
looked identical to one that learned forgery.

---

## 30 Aug — grammar-constrained sampling segfaults, so the guarantee moved

**Saw:** the explanation stage constrains output with a GBNF grammar, per
CLAUDE.md. On the first real run it fell back to templates every time.

**Found:** my `except Exception` was swallowing the cause — a flaw in my own
code before anything else. Removing it surfaced
`OSError: exception: access violation reading 0x0000000000000000` from inside
`llama_sampler_sample`. The grammar itself parses fine; grammar-constrained
sampling is broken in the prebuilt `llama-cpp-python` 0.3.35 Windows wheel.

**Changed:** try grammar sampling first, fall back to free generation, then
enforce the same shape in Python (`conform`): exactly one SUMMARY line, at most
four bullets, and a next step from a closed set. Anything else the model
produced is discarded rather than shown to a reviewer — a reviewer cannot tell
which parts of loose model prose came from the evidence, and traceability is the
entire point of the stage.

**Then the model misbehaved in three more ways, all handled in `conform`:**

1. It looped, repeating one sentence six times. Fixed with `repeat_penalty=1.25`
   and a 200-character cap on the summary.
2. It emitted markdown section headers as bullets (`- **Disagreements:**`).
   Headers are dropped; content glued to a header is kept.
3. It repeated identical bullets. De-duplicated.

**Result:** clean, on-shape reviewer notes from a 0.5B model. The guard
`_assert_explains` still raises if any output contains decision language, so the
"explains, never decides" contract holds regardless of which path produced the
text.

**Worth noting:** the wheel is also CPU-only. That is fine here and arguably
helps — the explanation model never competes with OCR for the 4GB, and only
escalated cases reach it.

---

## 30 Aug — all five stages, and what the VRAM actually does

**Built:** `src/pipeline.py` wires extraction → invariants → registry → tamper →
decision/explanation, releasing each stage before the next loads.

**Measured, not assumed:** peak **1,373 MB** on the 4GB GTX 1650 for a full run
including OCR and the tamper CNN, dropping to **14 MB reserved** afterwards.

**One design decision worth defending:** a tamper hit on documents that are
otherwise internally consistent **escalates rather than rejects**. The CNN is
evidence, not proof. Letting a model's opinion about pixels decline a merchant
whose documents provably agree with each other would invert the whole argument
of the project. `test_tamper_hit_on_clean_documents_escalates` holds that line.

---

## 30 Aug — a clean application escalated because of a truncated word

**Saw:** running the five-stage demo on rendered-and-degraded images, a
perfectly legitimate merchant came back as ESCALATE with
`constitution 'Family' is not one we map to a PAN entity type`.

**Measured:** dumped the raw detections. OCR had read the certificate
correctly — `Constitution of Business` at y=676, then `Hindu Undivided` at
x=137 and `Family` at x=349. Two boxes, one value. My label matcher took a
single line and got the tail.

**Three bugs, each revealed by fixing the one before it:**

1. **Split values were not rejoined.** Merging fragments on the same row fixed
   the constitution.

2. **...which then merged across columns.** `State` and `Date of Liability`
   sit on one row ~640px apart, so `Punjab` became `Punjab 08/06/2023`. Fixed
   by splitting a row into runs wherever the horizontal gap exceeds 300px —
   wider than a word break, narrower than the column separation.

3. **...and my column filter was one-sided.** I kept fragments with
   `x < anchor.x + tolerance`, which works for the left column and fails for
   the right: the date's anchor is at x=756, so the state at x=137 passed the
   test and the *date* picked up the *state*. Fixed by choosing the run whose
   start is nearest the label's own x, rather than filtering by direction.

**Also worth recording:** detections within a row are not ordered by y.
`Family` came back at y=700.1 and `Hindu Undivided` at y=703.0 — the tail
before the head. Anchoring on "first line below the label" was therefore wrong
even before the column issues.

**Result:** 0 mismatches across 40 free-text fields on degraded images, where
before it was 8. Six regression tests cover the cases, including the
out-of-order one, since none of them are things I would have thought to write
without seeing the geometry.

**Why this one matters more than it looks:** it produced a **false positive on
a legitimate merchant** — the exact error the cost matrix prices at ₹800 and
which the whole design is meant to avoid. It came not from the fraud logic but
from a wrapped word on a form.

---

## 30 Aug — the cost matrix finally has something to decide (partly)

**The standing limitation:** `evaluate.py` feeds `Bundle` objects straight to
the checks. Every legitimate case scored exactly zero, the classes had no
overlap, and so every threshold in the gap cost the same. The README has said
since the start that the cost weighting was "correct machinery that has not yet
been exercised".

**Changed:** wrote `evaluate_e2e.py`, which renders each document, degrades it
to a phone-photo quality, and reads it back with OCR before checking anything.
Extraction errors are part of the measurement rather than excluded.

**First run, at `typical` quality — a warning, not a success:** 4 distinct
operating points across the cost sweep, so the matrix *was* deciding. But **99%
of legitimate applications scored above zero** and attack 4 read as 100% caught
because effectively everything was flagged. The cost matrix had something to
decide only because the detector was drowning in OCR noise. That is not the
result I wanted to report.

**After the label-matching fixes, at `good_photo`:**

| | before fixes (`typical`) | after fixes (`good_photo`) |
|---|---|---|
| legitimate scoring > 0 | 99% | **18%** |
| attack 4 | 100% (wrong) | **0%** (correct) |
| PR-AUC | 0.4967 | 0.5062 |
| precision / recall | 0.31 / 1.00 | 0.67 / 0.83 |
| distinct operating points | 4 | 1 |

**Result, stated precisely:** the overlap region is now *real* — 18% of honest
merchants get a non-zero risk score purely from extraction error, which is
exactly the thing that was missing. It is still not dense enough to move the
cost minimum at this quality. At `typical` it does move, but only because the
signal has degraded past usefulness.

So: **partly exercised.** The README says that rather than claiming the
limitation is closed. Real merchant uploads, with their real error
distribution, are what would settle it.

**The number I would put in front of a reviewer:** every detectable attack is
still caught 100% of the time when read from photographs; the precision drop to
0.67 is entirely OCR misreads producing mismatches indistinguishable from
forgery. That is the true cost of running this on images, and it is the figure a
deployment would have to improve.

---

## 30 Aug — retrained the tamper CNN on 2.7x the data

**Changed:** 800 bundles / 12 epochs, up from 300 / 5. Same architecture, same
held-out method, same controls. Old weights backed up before overwriting.

| | 300 bundles, 5 epochs | 800 bundles, 12 epochs |
|---|---|---|
| held-out PR-AUC | 0.9937 | **0.9986** |
| seen PR-AUC | 0.9040 | **0.9498** |
| seen recall | 67.0% | **79.3%** |
| `splice` recall | 13.2% | **38.9%** |
| `resave` false alarms | 0.0% / 4.2% | **0.0% / 0.0%** |
| false positives on clean | 8.3% / 0.0% | **2.0% / 0.5%** |
| CNN advantage over ELA (seen) | +0.044 | **+0.068** |

**Everything improved, which is exactly when to be suspicious.** The `splice`
jump from 13.2% to 38.9% was the one I had flagged in advance as implausible —
copy-move within a single document leaves matching noise, lighting and JPEG
history, so there should be very little to learn.

**Checked it rather than banking it:** re-scored 12 fresh documents at a seed
the model had never seen, outside the evaluation split entirely. `splice` 9/12
flagged (mean 0.752), clean 0/12, `resave` control 0/12. The improvement is
real, not an artefact of the split.

**The number that matters is still the held-out one**, and it barely moved
(0.9937 -> 0.9986) because it was already near ceiling. More data on a synthetic
corpus mostly teaches a model our generator better; the held-out figure is the
one that resists that, which is why it is the one reported first.

**What did NOT change:** attack 4 is still undetectable, because nothing about
it is a pixel artefact. No amount of training touches it.

---

## 30 Aug — TrOCR scored 0-2%, and the cause was the tokeniser

**Saw:** fine-tuned `trocr-small-printed` on 12,000 identifier crops for 5
epochs. Loss fell cleanly from 3.96 to 0.53, so it was learning *something*.
Exact-match read rate: **0-2%**, against EasyOCR's 80-96%.

**Measured:** dumped raw predictions instead of just the score. They were not
random — mostly empty, and where non-empty, revealing:

    truth 27FIEFV4270S1ZS   ->  '27F4770S1S'
    truth 05ZBHPK8154O1ZH   ->  '05ZR8154H'
    truth AQPTI3187C        ->  'AQTI31C'

First characters right, last characters right, **middle dropped**. A model
emitting EOS far too early, not one that had failed to see the image.

**Found:** TrOCR's tokeniser is BPE trained on English prose, and it merges
identifier characters into subword chunks — chunks that are *unstable*:

    AAPFU0939F  ->  ['AAP', 'FU', '09', '39', 'F']       5 tokens
    AAPFV0939F  ->  ['AAP', 'F', 'V', '09', '39', 'F']   6 tokens
    ZQXWM1234K  ->  ['Z','Q','X','WM','123','4','K']     7 tokens

One character different and the whole segmentation shifts. The model was being
asked to predict a chunking that varies unpredictably with the very content it
is trying to read. That is a far harder problem than reading characters, and at
12,000 samples it is close to unlearnable.

**Changed:** space the characters before tokenising (`"A A P F U 0 9 3 9 F"`),
so every identifier is exactly `len(text)` tokens, one per character, stable
across every string. Also raised `max_length` from 24 to 40 — a spaced GSTIN
needs 15 tokens plus specials, and 24 was truncating output in a way that
looked like model failure rather than a decoding limit.

**The lesson:** the loss curve looked healthy the whole time. Loss going down
means the model is fitting *the objective you gave it*, which is not the same
as the objective you meant. Only dumping raw predictions showed the shape of
the error, and the shape is what identified the cause.

**Also worth recording:** a `tail -30` on the training job captured only stderr
warnings and lost the entire stdout log. The run had actually completed and
saved a 246MB checkpoint. I nearly reported a training failure that had not
happened — check the artefacts, not just the log.

---

## 1 Sep — OCR confidence gating, and knowing when to stop tuning it

**The problem:** end-to-end, 18-21% of *legitimate* merchants get rejected
because a misread PAN produces a mismatch the invariant layer cannot
distinguish from a forgery. That is the worst number in the project.

**The idea:** `FieldRead` already carries a confidence that nothing read. Gate
on it — where extraction is shaky, downgrade a rejection to human review rather
than declining the merchant. At Rs 50,000 per missed fraud against Rs 150 per
review, that trade is obviously right.

**Measured the threshold rather than picking one.** Over 30 bundles at
`good_photo`: correct reads had median confidence 0.568, and every misread fell
below 0.26. Floor set at 0.30. But checking harder levels first was worth it —
at `poor`, some *wrong* reads come back at 0.967, so the separation is much
weaker than the first sample suggested. This is a mitigation, not a fix, and the
code says so.

**First attempt did nothing.** Seven cases downgraded, PR-AUC unchanged. Cause:
`evaluate_e2e.py` called `check_bundle` and `risk_score` directly, bypassing
`pipeline.run` — so the gate never executed in the measurement. **Measuring a
different code path than the one that ships is how a mitigation gets credited
without ever running.** Routed the evaluation through the real pipeline.

**Second attempt over-corrected.** Diagnosed what was actually failing on
legitimate cases at `typical`: not the identifiers (which still read at 80%) but
`entity_type_agreement` and `registration_date` — the *free-text* fields. Widened
the gate to cover them, and it downgraded **87 of 120** applications. PR-AUC got
*worse*: 0.507 -> 0.483.

Three-quarters of applications sent to human review is not mitigation, it is
abdication — and the metric said so.

**Where it landed:** the gate covers every extracted field, but only fires when
**at most two** fields read poorly. One bad field on an otherwise-clean read is
what it was measured for. If more than two are bad, the image was too poor to
process and excusing every failure is the wrong answer.

**Result, measured properly:** at `good_photo`, **3 rejections of 120**
downgraded to review, attack 4 correctly 0%, precision 0.617. At `typical`,
**0** — the cap correctly refuses to excuse anything when the whole extraction
is unreliable.

**Reported as modest, because it is.** Three cases in 120 is a real improvement
on a real failure mode and nothing more. The 21% false-positive rate needs
*better extraction*, not better handling of bad extraction. An earlier draft of
the README implied more; it now states the number.

**The lesson:** I had a plausible mitigation, a measured threshold, and a
metric that got worse. Trusting the metric over the story is the whole job.

---

## 3 Sep - the renderer only worked on my machine, and said nothing about it

**Saw:** ran the test suite on Linux for the first time, preparing the repo to
be pushed publicly. Two failures, both in `test_tamper.py`:

    pasted region has only 12 colours - it is blank card stock,
    not content from another document

**Found:** `datagen/render.py` asks for `arial.ttf`, `arialbd.ttf` and
`times.ttf`, and `datagen/tamper.py` for `courbd.ttf`. Windows filenames,
because Windows is where this was written. Both wrapped the load in
`except OSError: return ImageFont.load_default()`, so off Windows every face
resolved to PIL's 11px bitmap default: a 52pt PAN number rendered a few pixels
tall, and the identifier box came out mostly blank card. `patch_paste` then
sampled that blank card - which is exactly the regression the 30 Aug entry
added a test for. The test was right. It simply could not fire on the machine
that wrote it.

The two failures are the least of it. The renderer is the input to **every**
measurement in this repo: OCR accuracy by quality level, the tamper corpus, the
end-to-end run. Anyone cloning this on Linux or macOS would have reproduced
different numbers from the README, with nothing anywhere to tell them why.

**Changed:** added `datagen/fonts.py`. Each requested face resolves against a
candidate list - Windows names first, so rendering on Windows is byte-for-byte
what it was, then macOS, then the Liberation, DejaVu and Noto families that
ship on Linux. The fallback survives, because a wrong-looking document still
beats a crashed data generator, but it is no longer silent: it warns once per
face on stderr and through `warnings`, records the face in `DEGRADED`, and
`tests/test_fonts.py` fails outright if any face is missing rather than letting
the suite pass against documents nobody else can reproduce. Added a GitHub
Actions workflow so the suite runs on Linux on every push.

**Result:** **189 passed, 0 failed** on Linux. The four faces resolve to
Liberation equivalents; on Windows the first candidate still wins and nothing
about the rendered documents changes.

**The lesson:** every other entry in this log is about a number I could see was
wrong. This one I could not see at all, because the code was correct on the
only machine that ever ran it. "Works on my machine" is the oldest joke in
software and I still shipped it into the layer that every measurement depends
on. A CI runner is not process for its own sake - it is a second machine.

---

## What this log is for

Razorpay's form asks what broke and how it was fixed, and says it is the first
thing they read. Twenty-one entries above, written as they happened. The pattern
worth noticing across them:

**Four of the bugs made the system look BETTER than it was.**

- The threshold search flagged everything, reporting 100% on the attack we
  cannot detect.
- The attack-1 generator produced structurally invalid PANs, so a regex was
  quietly doing work the headline check claimed credit for.
- The held-out tamper method pasted blank card stock, which would have made the
  generalisation number meaningless.
- The clean-bundle generator emitted registrations dated before GST existed, and
  nothing noticed because no check read the field.

None of these would have been caught by a failing test, because in every case
each individual component was working correctly. They were caught by looking at
a number that was *surprising* and refusing to accept it — in one case a number
that was surprisingly **low** (OCR at 56%) and therefore looked like an honest
bad result.

**Two made it look worse than it was**, which is the safer direction but still
worth finding: the OCR fallback returning label garbage over a real identifier,
and a character-confusion dict silently dropping the most likely repair.

**One produced a false positive on a legitimate merchant** from a wrapped word
on a form — not from the fraud logic at all.

**And one was invisible from where I was standing.** The renderer's fonts were
correct on Windows and silently wrong everywhere else, so the code feeding every
measurement in the repo was broken for every reader except me. No amount of
staring at the numbers would have found it. Running the suite on a second
machine found it in eighteen seconds.

The habit that found all of them: when a metric moves in a direction you like,
go and check why before believing it.
