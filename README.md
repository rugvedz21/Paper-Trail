# Paper Trail

Merchant-onboarding fraud detection built on a single structural fact:

```
GSTIN  2 7 [A A P F U 0 9 3 9 F] 1 Z V
            └────────┬────────┘
PAN        [A A P F U 0 9 3 9 F]
```

**GSTIN characters 3–12 are the PAN, character for character.**

Most document fraud detection asks *"does this look genuine?"* — a losing game
against anyone with a decent image editor. Paper Trail asks whether the
documents **agree with each other**, which is much harder to fake: a forger who
edits the PAN card must also recompute the GSTIN, which requires knowing both
the embedding and the mod-36 check character. Most don't. String comparison
catches them.

---

## Reproduce every number

```
run web          # ← the web interface at http://127.0.0.1:5000
run              # tests, demo and evaluation in one go
run test         # 189 tests
run stages       # all five stages on one application of each attack type
run demo         # three worked examples (invariants only)
run try          # type in your own documents and see the verdict
run samples      # four worked examples, no typing
run eval         # every figure below
run full         # + seed stability and cost sweep
run e2e          # images -> OCR -> checks, end to end   (GPU, ~10 min)
run ocr          # OCR accuracy vs image quality         (GPU, ~8 min)
run tamper       # tamper CNN + held-out generalisation  (GPU, ~5 min)
run trocr        # fine-tune TrOCR on identifier crops    (GPU, ~50 min)
run docs         # render sample documents to out/
```

`run.bat` uses the project's Python 3.12 venv; `run.sh` is the POSIX twin. To
call the scripts directly:

```bash
.venv/Scripts/python evaluate.py            # Windows
.venv/bin/python evaluate.py                # macOS / Linux
.venv/Scripts/python evaluate.py --seeds 10 --sweep
.venv/Scripts/python -m pytest tests/ -q
```

Every number below is printed by one of those commands. The first six need
**no GPU and no network** — fixed seed, pinned clock, pure Python. The three
marked `(GPU)` load models and take minutes rather than seconds.

Any claim in this README that no command prints is a claim the repo cannot
support.

---

## Results

5,000 applications, 7.7% fraudulent, spread across the five attacks.

| Metric | Value | |
|---|---|---|
| **PR-AUC** | **0.8149** | against a 0.0768 base rate — 10× chance |
| Brier score | 0.0215 | lower is better |
| Precision | 1.0000 | at the chosen threshold |
| Recall | 0.7995 | ceiling explained below |
| Accuracy | *not reported* | meaningless at 7.7% positives |
| ROC-AUC | *not reported* | flatters under class imbalance |

**Per-attack recall** — the number that actually matters:

| # | Attack | Recall | Caught by |
|---|---|---|---|
| 1 | Naive edit — one document changed | **100%** (85/85) | Arithmetic: PAN ≠ GSTIN[3:12] |
| 2 | Consistent fabrication — coherent invented identity | **100%** (82/82) | Registry: never issued |
| 3 | Identity theft — real pair, name replaced | **100%** (74/74) | Name mismatch vs. registry |
| 4 | Pixel-perfect theft — genuine documents, wrong holder | **0%** (0/77) | **Nothing. Stated blind spot.** |
| 5 | Impossible registration date | **100%** (66/66) | Date before GST existed, or in the future |

That 79.95% overall recall is not a tuning failure. **Attack 4 is a fifth of the
fraud in this corpus and is undetectable by these methods** — the documents are
real, unaltered, and internally consistent. There is nothing in the data to
find. Detecting it needs signals outside the documents entirely: device
fingerprinting, behavioural analysis, liveness checks at onboarding.

We report it as 0% rather than dropping it from the evaluation.

> Attack 5 was **not designed in advance**. It was found by attacking our own
> checks after they were written: `gst_registration_date` was carried on every
> bundle, printed on every rendered certificate, and read by no check at all.
> A forger backdating a registration to imply trading history went straight
> through. See `NOTES.md`.

### Is this one lucky seed?

`python evaluate.py --seeds 10` runs ten independent corpora:

| | PR-AUC | Recall |
|---|---|---|
| mean | 0.8097 | 0.7933 |
| sd | 0.0255 | 0.0274 |
| range | 0.7661 – 0.8521 | |

Precision is 1.0000 and attack 4 recall is 0.0% on **all ten**. The headline
seed (42) sits slightly *below* the mean, so it is not cherry-picked.

---

## Cost, not F1

F1 asserts that a false positive and a false negative hurt equally. In merchant
onboarding they do not, by roughly two orders of magnitude.

| Outcome | Cost | |
|---|---|---|
| False negative | ₹50,000 | chargebacks, penalties, remediation |
| False positive | ₹800 | a legitimate merchant lost |
| Human review | ₹150 | reviewer time on an escalation |

> These figures are **illustrative** and must be replaced with Razorpay's real
> numbers before informing any live threshold. What matters here is the shape:
> a miss costs far more than a false alarm, so the threshold belongs well below
> the F1-optimal point.

The threshold minimises expected cost over the corpus. At the chosen point
(0.3500) the pipeline costs **₹3,850,000** against **₹19,200,000** for approving
everything — **₹15.35M avoided**, ₹770/case.

### Does the cost matrix do any work? Partly — and we measured it

On the `Bundle`-fed corpus (`python evaluate.py --sweep`), **no**. Varying the
FN:FP ratio over a 200× range does not move the threshold at all: it is 0.3500
every time. Rule-derived scores are bimodal, **no legitimate case scores above
zero**, and with a clean gap between the classes every threshold inside it costs
the same.

That is a property of the corpus, not a strength of the method — so we built the
corpus that breaks it. `python evaluate_e2e.py` renders each document, degrades
it to a phone-photo quality, and reads it back with OCR before checking
anything:

| | Bundle-fed | End-to-end (`good_photo`) |
|---|---|---|
| legitimate cases scoring > 0 | 0% | **18%** |
| distinct operating points across the sweep | 1 | 1 |

So the overlap region is now real — extraction errors put legitimate merchants
at non-zero risk, which is exactly what was missing. It is still **not dense
enough to shift the cost minimum** at this quality level. At `typical` quality
it does move (4 distinct operating points), but by then 99% of legitimate cases
score above zero and the detector is drowning in OCR noise rather than doing
useful work.

The honest summary: **the cost weighting is correct machinery that this corpus
exercises only partially.** Real merchant uploads, with their real error
distribution, are what would settle it. We would rather say that than present a
threshold that never moves as evidence of robustness.

---

## End-to-end, from pixels

`python evaluate_e2e.py` — 150 applications rendered, degraded, read by OCR,
then checked. Extraction errors are part of the measurement, not excluded.

| | |
|---|---|
| OCR read both identifiers exactly | **83%** (124/150) |
| PR-AUC | 0.5062 (base rate 0.3067) |
| precision / recall | 0.6667 / 0.8261 |
| legitimate cases false-flagged | 18% |

| # | Attack | Recall |
|---|---|---|
| 1 | Naive edit | **100%** (9/9) |
| 2 | Coherent fabrication | **100%** (13/13) |
| 3 | Identity theft | **100%** (4/4) |
| 4 | Pixel-perfect theft | **0%** (0/8) — blind spot, as always |
| 5 | Impossible date | **100%** (12/12) |

Every detectable attack still lands at 100% when read from images. The precision
drop to 0.6667 is entirely OCR: a misread PAN produces a mismatch that the
invariant layer cannot distinguish from a forgery, and 18% of honest merchants
pay for it. **That is the real cost of running this on photographs**, and it is
the number a deployment would have to improve.

### Mitigating it: OCR confidence gating

`FieldRead` carries a confidence score, and `pipeline.run` uses it: **a
rejection resting on a low-confidence read is downgraded to human review rather
than declining the merchant.** The invariant layer cannot tell a misread PAN
from a forged one — both look like two documents disagreeing — so where
extraction is shaky, the honest answer is "a person should look".

The threshold was measured, not chosen. Over 30 bundles at `good_photo`,
correct reads had median confidence 0.568 and every misread fell below 0.26; the
floor sits at 0.30.

Two guards keep it from becoming a loophole:

- **A registry miss is never excused.** A GSTIN that does not exist is not an
  OCR artefact. Only failures a bad read could actually explain are eligible.
- **At most two fields may be low-confidence.** Widening the gate to cover every
  field downgraded **87 of 120** applications at `typical` quality and made
  PR-AUC *worse* (0.507 → 0.483). Three-quarters of applications sent to review
  is abdication, not mitigation. With the cap, the gate helps at `good_photo`
  and correctly declines to help at `typical`, where extraction is not
  trustworthy enough for any downstream repair to be honest.

**How much does it actually help? Modestly.** Over 120 applications at
`good_photo`, it downgrades **3 rejections** to human review; at `typical`, **0**.

| | `good_photo` | `typical` |
|---|---|---|
| rejections downgraded | 3 | 0 |
| precision / recall | 0.6170 / 0.8056 | 0.3025 / 1.0000 |
| attack 4 | 0% (correct) | 100% (drowned in OCR noise) |

Three cases in 120 is a real improvement on a real failure mode, and it is not
a solution to the 21% false-positive rate. **The honest fix for that is better
extraction, not better handling of bad extraction.** We would rather report a
mitigation that helps a little than dress it up as one that helps a lot.

---

## The five stages

```
  PAN card + GST certificate  (or a Bundle directly)
             │
             ▼
   ┌──────────────────┐
   │ 1  EXTRACTION    │  EasyOCR -> Bundle          ~1.3 GB, then released
   │    crops for the identifiers, labels for the rest
   └──────────────────┘
             │  Bundle
             ▼
   ┌──────────────────┐
   │ 2  INVARIANTS    │  8 checks, NO MODEL                     0 GB
   │    PAN == GSTIN[3:12]  ·  mod-36 checksum  ·  name  ·  date
   └──────────────────┘
             │  Findings (each carrying its evidence)
             ▼
   ┌──────────────────┐
   │ 3  REGISTRY      │  mocked GSTN lookup                     0 GB
   │    not_found / name_mismatch / unavailable
   └──────────────────┘
             │
             ▼
   ┌──────────────────┐
   │ 4  TAMPER        │  ELA + ResNet18 on crops     ~50 MB, then released
   │    a score, never a verdict
   └──────────────────┘
             │
             ▼
   ┌──────────────────┐
   │    DECISION      │  cost-weighted threshold over rule-derived evidence
   └──────────────────┘
             │  escalate?
             ▼
   ┌──────────────────┐
   │ 5  EXPLANATION   │  Qwen2.5-0.5B, CPU        escalations only
   │    explains; never decides
   └──────────────────┘
```


| | Stage | Model | Peak VRAM | Why |
|---|---|---|---|---|
| 1 | Extraction | EasyOCR | ~1.3 GB | Reads the documents. A *front end* onto `Bundle`, never a hard dependency — the pipeline accepts a `Bundle` directly, so OCR problems can't block downstream work. |
| 2 | **Invariants** | **none — deliberately** | 0 | The centrepiece. Ground truth is closed-form; a classifier could only add error. |
| 3 | Registry | mocked GSTN lookup | 0 | Catches identities that are internally valid but don't exist. |
| 4 | Tamper detection | ResNet18 + Error Level Analysis | ~50 MB | Pixel-level evidence on identifier crops. |
| 5 | Explanation | Qwen2.5-0.5B (GGUF, CPU) | 0 | An explanation **for escalated cases only**. The model explains; it never decides. |

All five are implemented. `src/pipeline.py` wires them together and releases each
stage's memory before the next loads — measured peak **1,373 MB** on a 4 GB card,
dropping to 14 MB reserved after the run. OCR and the explanation model are
never resident together.

### Stage 2 uses no AI, on purpose

The relationship between a PAN and a GSTIN is exact. `27AAPFU0939F1ZV` either
contains `AAPFU0939F` or it doesn't. Training a classifier to approximate string
equality would introduce error where none needs to exist, and would produce a
confidence score in place of a proof.

The eight checks:

| Check | Catches |
|---|---|
| `pan_gstin_agreement` | **The load-bearing one.** PAN ≠ GSTIN[3:12] |
| `gstin_checksum` | Invalid GSTIN (documented mod-36 algorithm) |
| `name_agreement` | Name differs across documents, or contradicts PAN char 5 |
| `entity_type_agreement` | PAN entity letter vs. GST constitution |
| `state_agreement` | GSTIN state code vs. certificate address |
| `registration_date` | Dated before GST commenced (1 Jul 2017), or in the future |
| `pan_structure` | Malformed PAN |
| `gstin_structure` | Malformed GSTIN |

Every check returns a `Finding` carrying the values it compared — never a bare
boolean. "Name check failed" is unactionable; the reviewer needs both strings.

---

## Reading real documents

`python evaluate_ocr.py` — 25 bundles rendered, degraded to each quality level,
and read back. Identifiers are compared **strictly**; free-text fields
case-insensitively, because the renderer prints names uppercase.

| quality | PAN | GSTIN | **both** | names/dates | every field |
|---|---|---|---|---|---|
| pristine | 100% | 92% | **92%** | 100% | 92% |
| clean_scan | 100% | 92% | **92%** | 100% | 92% |
| good_photo | 96% | 92% | **88%** | 100% | 88% |
| typical | 96% | 92% | **88%** | 4–88% | 0% |
| poor | 84% | 76% | **68%** | 0% | 0% |
| terrible | 20% | 0% | **0%** | 0% | 0% |

The **both** column is the ceiling on everything downstream: the PAN↔GSTIN
equality check needs both identifiers exactly right, and a misread produces a
mismatch indistinguishable from a forgery.

### Fine-tuning TrOCR — what happened

CLAUDE.md's architecture specifies "EasyOCR line detection + fine-tuned TrOCR".
We built the fine-tuning (`run trocr`) and it is worth reading the result
honestly.

**First attempt: 0–2% exact match**, against EasyOCR's 80–96%. The loss curve
looked healthy the whole time (3.96 → 0.53), which is exactly why the score
alone was not enough to diagnose it. Dumping raw predictions showed the shape:

```
truth 27FIEFV4270S1ZS  ->  '27F4770S1S'
truth AQPTI3187C       ->  'AQTI31C'
```

First and last characters right, middle dropped. **The cause was the
tokeniser.** TrOCR's BPE was trained on English prose and merges identifier
characters into subword chunks that are *unstable*:

```
AAPFU0939F  ->  ['AAP', 'FU', '09', '39', 'F']       5 tokens
AAPFV0939F  ->  ['AAP', 'F', 'V', '09', '39', 'F']   6 tokens
```

One character different, and the whole segmentation shifts. The model was being
asked to predict a chunking that varies with the content it is reading — much
harder than reading characters, and close to unlearnable at this data scale.

Fixed by spacing characters before tokenising, so every identifier is exactly
one token per character. The next checkpoint emitted **full-length** strings
(`01VTLPO1862C1ZG -> '27FPPCC0222Z1Z'`) — right length, wrong glyphs. That is
the tokeniser problem solved and a pure data-scale problem remaining.

**Status: EasyOCR is what the pipeline uses.** TrOCR is not wired in, and will
not be unless it demonstrably wins. Shipping a second OCR path that loses to
the one already there would be complexity with no payoff. `run trocr`
reproduces the experiment; `NOTES.md` has the full trail.

### How a misread identifier is recovered

Fragments are reassembled into fixed-length candidates and filtered by the
published structural regex. Where that leaves a choice, **the documented GSTIN
mod-36 check character decides** — a reading whose 15th character agrees with
its own first fourteen is arithmetically self-consistent, which is far stronger
than guessing whether a glyph "should" be a digit.

That search is **gated on having a validator**. The PAN has no published
checksum, so a PAN is repaired only where the structure is unambiguous.
Searching without an independent test would manufacture plausible-looking
identifiers — precisely what this project exists to detect.

> An earlier version guessed character classes instead, marking GSTIN positions
> 13 and 15 as digit-preferring. Position 15 is a **letter** roughly 26 times in
> 36, so that prior was rewriting correctly-read letters into digits and made
> things *worse*: GSTIN accuracy was 56%. Using the checksum took it to 92%.
> See `NOTES.md`.

---

## Held-out attack generalisation

The sharpest objection to any result here is *"you designed both the attack and
the defence."* The tamper detector answers it directly: it is trained on some
tampering methods and evaluated on one it has **never seen**.

`python evaluate_tamper.py` — 300 bundles, ResNet18 over 6-channel input
(RGB + Error Level Analysis):

| | PR-AUC | recall @0.5 | false positives |
|---|---|---|---|
| Seen methods (`retype`, `splice`, `inpaint`) | 0.9498 | 79.3% | 1.5% |
| **Held out (`patch_paste`) — never trained on** | **0.9986** | **95.9%** | 0.4% |

Scores from the trained detector on one document, which shows the split
plainly:

Mean scores over 12 documents at an **unseen seed**, to check the numbers above
are not an artefact of the evaluation split:

| | mean score | flagged |
|---|---|---|
| clean | 0.062 | 0/12 |
| `resave` (control — not fraud) | 0.056 | 0/12 |
| `retype` | 1.000 | 12/12 |
| `inpaint` | 1.000 | 12/12 |
| `patch_paste` (held out) | 0.832 | 10/12 |
| `splice` | 0.752 | 9/12 |

The held-out number is *higher* than the seen number, which is the opposite of
the usual failure. Two things explain it, and both are worth stating:

- `patch_paste` introduces pixels from a different document — foreign lighting
  and compression history, a strong and genuinely generalisable signal.
- `splice` (a **seen** method) is caught only **38.9%** of the time. Copy-move
  within one document leaves noise, lighting and JPEG history all matching, so
  there is much less to find. It drags the seen average down.

So the honest reading is not "the detector generalises brilliantly" but "one
attack is easy and another is nearly invisible, and which is which does not
depend on what we trained on."

### The control that stops us fooling ourselves

A quarter of the clean set is **re-encoded at lower JPEG quality but not
otherwise edited** (`resave`). It is not fraud — every legitimately scanned
document has been recompressed. A detector that flags it has learned
compression history rather than forgery.

It is flagged **0.0% of the time on both splits**. Reported separately, always.

### Against a baseline with no learning at all

| | CNN | ELA statistics only |
|---|---|---|
| seen | 0.9498 | 0.8814 |
| held-out | 0.9986 | 0.9050 |

The CNN beats six hand-computed numbers, but not by a wide margin. That is the
honest framing: it earns its ~50 MB, and a reviewer entitled to ask "is the
neural network doing anything?" gets a real answer.

---

## What this does not do

**We do not verify the PAN check digit.** The Income Tax Department has never
published the algorithm for PAN character 10. Every "PAN checksum validator" in
circulation is invented. We validate PAN *structure* and *cross-document
consistency* only. A test (`test_no_pan_check_digit_function_exists`) fails the
build if anyone adds one.

The GSTIN check character **is** documented, **is** implemented, and is verified
against the published worked example `27AAPFU0939F1Z` → `V`.

**Risk scores are rule-derived, not learned.** The weights in
`src/evaluation/score.py` are hand-set from how diagnostic each check is. Scores
are *ordinally* meaningful — a higher score is a stronger case — but calibration
is an artefact of those weights until a model is fitted on real outcomes. The
calibration curve is printed to show this rather than hide it.

**The registry is mocked.** The real GSTN API needs credentials that cannot ship
in a public repo. What is modelled is the *interface* and the *failure modes* —
including `UNAVAILABLE`, which warns rather than rejects, because an API outage
is not evidence against a merchant.

**OCR is not a solved problem here.** At `good_photo` quality both identifiers
are read exactly 88% of the time; at `terrible` it is 0%. A misread PAN produces
a mismatch **indistinguishable from a forgery**, so extraction quality is a hard
ceiling on what the invariant layer can be asked to do from images. `run ocr`
prints the full table.

**The tamper CNN is not proof.** It contributes a score, and a tamper hit on
documents that are otherwise internally consistent **escalates to a human rather
than rejecting**. A model's opinion about pixels should not decline a merchant
whose documents provably agree with each other.

**`splice` tampering is caught less than half the time** — 38.9%. Copy-move within a
single document leaves matching noise, lighting and compression, so there is
very little to find.

**The explanation model runs on CPU.** The prebuilt `llama-cpp-python` wheel for
Windows has no CUDA support, and grammar-constrained sampling segfaults in that
build (0.3.35). Output shape is therefore enforced in Python after generation
rather than during it — the same contract, checked differently. A ~300-token
note on an escalation takes a few seconds, which is acceptable for the small
fraction of applications that reach it.

**This is not a claim about Razorpay's KYC stack.** Assume they already do these
checks. This demonstrates engineering judgment, not IP.

---

## The explanation stage

The decision is already made by the time the model runs. This stage turns
`Finding` objects into prose a reviewer can act on — **for escalated cases
only**, because a clear reject is already fully explained by its findings and
each invocation costs a model load.

A real escalation, from `run stages --llm`:

```
DECISION   : ESCALATE   (risk 0.150)
   [warn] constitution 'Nidhi Company' is not one we map to a PAN entity
          type; PAN says 'Firm/LLP'

REVIEWER NOTE (llm):
   SUMMARY: The merchant's PAN entity type is 'Firm/LLP' and the
            constitution of Nidhi Company does not match that.
   - The merchant's PAN entity type is 'Firm/LLP'
   - Nidhi Company does not match that
   SUGGESTED NEXT STEP: escalate to a senior reviewer
```

**Three guards keep the model from deciding anything:**

1. The prompt **states** the decision rather than asking for one.
2. Output shape is constrained — a GBNF grammar where the build supports it,
   otherwise enforced in Python after generation. The next step comes from a
   closed set of five; the model cannot invent an instruction.
3. `_assert_explains` **raises** if the output contains decision language
   ("I reject", "should be approved"). Shipping a model's verdict to a reviewer
   is the one failure this stage must never have.

**Quality is limited by the model.** Qwen2.5-0.5B is what fits comfortably
alongside everything else; its notes range from genuinely useful to merely
restating the finding. Everything it says is traceable to a `Finding`, and when
no model is present the same evidence is rendered from a template — so the
reviewer is never worse off than the deterministic path.

---

## Every identifier here is synthetic

All PANs, GSTINs, names and addresses are generated. Nothing is derived from or
looked up against real registry data. Real certificates were used as *layout
references* only. Any collision with a real registration is coincidence.

### Why we did not use a public dataset

The obvious way to improve OCR is to train on real scanned PAN cards. We
deliberately did not, and the reason is not squeamishness:

**Every real PAN card image carries a living person's tax identifier, legal
name and photograph.** Putting that in a public repository would breach the
first hard rule of this project and Indian data-protection law along with it.

We checked the legitimate public OCR corpora too, and none of them fit:

| Dataset | Why not |
|---|---|
| [NVIDIA OCR-Synthetic-Multilingual](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1) | Multilingual documents — wrong domain |
| Synth90K (9M) | English *words*, not alphanumeric identifiers |
| SynthText (800K) | Scene text on photographs |
| SynthAdd (1.6M) | Special characters and text lines |
| ICDAR 2003 / 2013 | Scene text, and far too small |

Those corpora teach a model to read English words in natural scenes.
`trocr-small-printed` already has that pretraining. Our task is a 36-character
alphabet, two fixed formats, one bold monospace typeface, one layout.

**Training on our own renderer is the correct answer, not a fallback.** The
model needs to read exactly the documents this system will see, our generator
produces them by the thousand with perfect labels, and every character is
synthetic.

---

## Layout

```
src/pipeline.py                 all five stages, wired together

src/extract/ocr.py              1. EasyOCR -> Bundle, fragment-tolerant matching
src/invariants/identifiers.py   2. checksum, structure, decoding
src/invariants/checks.py        2. cross-document checks, Finding, Bundle, verdict
src/registry/gstn.py            3. mocked GSTN lookup
src/tamper/detect.py            4. ELA + ResNet18 over identifier crops
src/explain/reviewer.py         5. reviewer notes, escalations only
src/evaluation/metrics.py       PR-AUC, cost matrix, calibration
src/evaluation/score.py         rule-derived risk score

datagen/generate.py             synthetic identities + the five attacks
datagen/render.py               PAN card and GST certificate images
datagen/degrade.py              phone-photo degradation, six levels
datagen/tamper.py               five tampering methods + the resave control
datagen/fonts.py                per-platform font resolution, loud on fallback

evaluate.py                     the headline numbers
evaluate_e2e.py                 images -> OCR -> checks, end to end
evaluate_ocr.py                 read accuracy vs image quality
evaluate_tamper.py              held-out attack generalisation
train_trocr.py                  fine-tune TrOCR on identifier crops
setup_models.py                 fetch the GGUF the explanation stage needs
webapp.py                       web interface (Flask)
templates/index.html            the page it serves
demo_pipeline.py                all five stages, one case per attack
demo.py                         three worked examples (invariants only)
try_it.py                       type in your own documents
render_samples.py               sample document images -> out/
run.bat                         convenience wrapper (Windows)
run.sh                          the same, for macOS and Linux
requirements.txt                stages 2-3, the evaluation and the tests
requirements-gpu.txt            stages 1, 4 and 5
tests/                          189 tests
```

`NOTES.md` is a running log of what broke and how it was fixed — including an
evaluation bug that made the system look *better* than it was.

---

## Setup

Requires **Python 3.12** — PyTorch ships no CUDA wheels for 3.13 or 3.14, and a
CPU-only build silently runs every model on the CPU without erroring.

**Stages 2 and 3, `evaluate.py`, `demo.py`, `try_it.py`, the web interface and
the whole test suite need only this** — pure Python, no GPU, no network, no
model downloads:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests/ -q       # 189 tests
.venv/Scripts/python evaluate.py               # the headline numbers
```

Stages 1, 4 and 5 — OCR, the tamper CNN, the explanation model — need the rest.
torch must come from the CUDA index; the PyPI wheel is CPU-only and will
silently run every model on the CPU rather than erroring:

```bash
.venv/Scripts/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements-gpu.txt
.venv/Scripts/python setup_models.py      # ~500MB GGUF for stage 5
.venv/Scripts/python evaluate_tamper.py   # trains + saves the stage-4 CNN
```

### Fonts

`datagen/render.py` names the faces it wants by their Windows filenames.
`datagen/fonts.py` resolves those against macOS and Linux equivalents, and
**says so loudly** if it cannot — a silent bitmap fallback renders near-empty
documents and quietly invalidates every OCR, tamper and end-to-end number
measured from them. On a bare Linux box:

```bash
sudo apt-get install fonts-liberation
```

`tests/test_fonts.py` fails if any face is missing, rather than letting the
suite pass against documents nobody can reproduce. See NOTES.md, 3 Sep.

Verified on a **GTX 1650 (4 GB, SM 7.5)**, driver 592.82, torch 2.6.0+cu124.
