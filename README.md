# FakeReasoning — LoRA fine-tuning on a single GPU

Fine-tunes the released **FakeReasoning** checkpoint — a vision-language model that
detects AI-generated images *and explains why* — on one GPU, with the instrumentation
needed to tell whether the fine-tune actually helped.

The last part turns out to be the hard part, and most of this repository is about it.

**Current result: separation between real and fake nearly doubled (+0.274 → +0.520) and
accuracy went 64% → 72%.** Details and caveats in [Results](#results).

---

## 1. The model

FakeReasoning does not output a probability. It outputs an argument, and the verdict is a
sentence inside that argument. That single design choice drives everything else: the
model has to *see* forgery artifacts and *say* what it saw.

Those two requirements pull in opposite directions, and the architecture resolves it by
using two vision encoders instead of one.

```
                    336×336 RGB (padded square, CLIP normalisation)
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        CLIP ViT-L/14-336                     DINOv2 ViT-L/14
        24 blocks, d=1024                     24 blocks, d=1024
        576 patch tokens                      576 tokens + attention map
        303.5M params  [FROZEN]               607.9M params  [FROZEN]
                    │                                   │
                    └──────────►  crossattention  ◄─────┘
                                    157.4M  [FROZEN]
                    ┌───────────────────┴───────────────┐
                    ▼                                   ▼
              mm_projector                      dino_mm_projector
              15.7M  1024→5120                  15.7M  1024→5120
                    └───────────────────┬───────────────┘
                                        ▼
                     visual tokens spliced in at the <image> sentinel
                                        │
                                        ▼
                            Vicuna-13B decoder
                            40 blocks, d=5120, ffn=13824
                            ~13B params  [LoRA r=16 → 62.6M trainable]
                                        │
                                        ▼
                      <SUMMARY> <CAPTION> <REASONING> <CONCLUSION>
```

### Why two towers

**CLIP** is trained contrastively on image–text pairs. Its features are *semantic* —
they encode what the picture is of, in a space already half-aligned with language, which
is why LLaVA-family models use it. What CLIP is bad at is precisely what forgery
detection needs: it was trained to be **invariant** to the low-level detail that
separates a diffusion sample from a photograph. Two images that "mean" the same thing sit
near each other in CLIP space whether or not one has six-fingered hands.

**DINOv2** is self-supervised — no text, no captions, trained to make different crops of
one image agree. Nothing in that objective rewards discarding low-level detail, so its
patch features keep texture statistics, local structure, and the periodic artifacts that
generative upsamplers leave behind. It can tell that a region was *rendered* rather than
photographed without knowing what the region depicts.

Neither is sufficient alone. CLIP can say "this is a portrait" but not "the ear is
structurally wrong". DINOv2 can say "this texture is synthetic" but cannot phrase it.
`crossattention` — 157M parameters, the largest non-decoder component — is where the two
streams combine before either reaches the language model. It also consumes DINOv2's
**attention map**, which is spatial evidence about where the tower is looking.

### The output schema

| block | content | why it matters |
|---|---|---|
| `<SUMMARY>` | a fixed sentence stating the method | identical on every image; carries no information but *is* the format |
| `<CAPTION>` | description of this specific image | the only block that forces per-image grounding |
| `<REASONING>` | evidence, then the verdict | the part worth training and worth auditing |
| `<CONCLUSION>` | "This image is real/fake." | the parseable answer |

Any training target that omits a block teaches the model to stop producing it. Dropping
`<CAPTION>` is the expensive one — it's what forces the model to look.

---

## 2. What gets fine-tuned

Three mechanisms, two of them hand-rolled because peft cannot reach the vision towers.

| component | targets/block | blocks | linears | trainable |
|---|---|---|---|---|
| Vicuna-13B decoder | `q,k,v,o_proj` + `gate,up,down_proj` | 40 | 280 | **62,586,880** |
| CLIP tower | `q,k,v,out_proj` + `fc1,fc2` | 24 | 144 | 3,538,944 |
| DINOv2 tower | `qkv, proj, fc1, fc2` | 24 | 96 | 3,145,728 |
| projectors + fusion | — | — | **0** | **0** |

DINOv2 needs only four targets because its attention is a single **fused** `qkv`
(1024→3072). One adapter on the fused layer is also cheaper than three on split ones:
`8×(1024+3072) = 32,768` against `3 × 8×(1024+1024) = 49,152`.

The defaults freeze both towers. They are the entire generalisation budget — every image
the model will ever see that is unlike the training set is handled by CLIP and DINOv2
features. The full reasoning, with code, is in **`explain_withcode.md`**.

### Two traps worth knowing

**peft matches target names by suffix.** Passing `["q_proj", ...]` as a list also matches
every `q_proj` inside the CLIP tower, so a config saying `"frozen"` would silently be
adapting it. The notebook builds an **anchored regex** instead, finds the decoder by
identity rather than by guessing its name, and re-checks the property *after*
`get_peft_model()` has run:

```python
strays = [n for n, m in model.named_modules()
          if isinstance(m, LoraLayer) and any(k in n for k in MULTIMODAL_KEYS)]
assert not strays, "the vision-tower freeze is not real"
```

**peft claims any parameter whose name contains `lora_`.** It decides what to write into
`adapter_model.safetensors` with a bare substring test, with no check that peft created
the parameter — so hand-injected factors named `lora_A` get swept into peft's file,
renamed on load, and silently dropped. The vision adapters are therefore called
`adapter_A` / `adapter_B`, leaving `vision_tower_weights.pt` as their sole owner.

---

## 3. Results

Measured on held-out FakeClue images, base checkpoint versus fine-tuned, both loaded
4-bit and scored identically.

```
                        base      fine-tuned      change
mean P(real | real)     0.619        1.000
mean P(real | fake)     0.345        0.480
separation             +0.274       +0.520        +90%
accuracy at 0.50        64.0%        72.0%        +8.0 pts
accuracy at best t      64.0%        76.0%        (t = 0.98)

per category      base        fine-tuned
  animal          17/23         19/23      +2
  deepfake        11/20         11/20       0
  human            4/7           6/7       +2
```

**The separation is the result, not the accuracy.** Accuracy is a thresholded count — at
n=50 the +8 points is four images, about 1.2σ, which on its own would be noise. The
separation between the two class distributions moved from +0.274 to +0.520, and that is a
distributional measure over every point rather than a count of crossings. It is the more
trustworthy number and it nearly doubled.

Read together they say something specific: **the model now assigns real images
P(real) ≈ 1.0 while fakes sit at 0.48**, just below the boundary. The classes are pushed
much further apart than before. That is a genuine gain in discriminative signal.

**There are 4 more points sitting unclaimed.** The optimal threshold is 0.98, not 0.50 —
fakes cluster right at the decision boundary, so a small shift catches more of them.
Scoring with `p_real > 0.98` rather than reading the generated word gives 76%. Fit that
threshold on a split you did not measure on before trusting it.

Where the gains came from: `human` and `animal` improved, `deepfake` was flat. Deepfake
is the hardest category and the one FakeClue supplies most of — worth its own experiment.

### Caveats, stated plainly

- **n = 50.** Enough to see the separation move, not enough to pin the accuracy. Raise
  `COMPARE_N` before quoting the 72% anywhere.
- Earlier configurations of this same pipeline made the model *worse*. The difference was
  not luck; it was the fixes in §4 below. But it does mean the recipe is sensitive.
- `P(real|real) = 1.000` is saturated. The model is maximally confident on reals, which
  is why the useful threshold is so high.

---

## 4. What actually made the difference

Five findings, in rough order of how much they mattered. Each was discovered by
measurement, and several contradicted what we believed at the time.

### The verdict was leaked into the reasoning

Every training target — **500/500** MMFR records checked, and FakeClue's conversion too —
ended its reasoning with *"Therefore, this image is real."* before restating it in
`<CONCLUSION>`. The model had a free shortcut: copy the earlier verdict. That needs no
image at all.

It also pinned every teacher-forced metric at exactly 100%, which is how it was found:

```
Step  Validation Loss  Verdict Acc  Says Real
  50         0.786        1.000        0.515
 100         0.677        1.000        0.515
 ...
1200         0.354        1.000        0.515
```

`says_real = 0.515` × 200 = 103 — simply how many validation examples were labelled real.
Both columns were constants of the dataset, not measurements of the model. A number that
is *exactly* stable to six decimals is never a model output.

`strip_verdict_leak()` now removes verdict assertions from `<REASONING>` only, leaving
`<CONCLUSION>` intact and preserving descriptive uses ("realistic depth of field", "a
real-world scenario", "the image is clear and sharp").

### The verdict was 4% of the gradient

Cross-entropy averages over ~280 supervised tokens, of which the verdict is ~12. The
model had almost no reason to get it right and every reason to make the surrounding prose
likely — which it can do without reading the image. `conclusion_weight = 10` lifts the
closing tokens to **31%** of the loss.

### `mmfr_weight` controlled examples, not gradient

The most counter-intuitive finding. The released checkpoint already fits MMFR-style
targets (loss ≈ 0.08) far better than FakeClue's (loss ≈ 0.79). Gradient magnitude scales
with loss, so:

| `mmfr_weight` | share of examples | share of the gradient |
|---|---|---|
| 0.6 | 60% | **14%** |
| 0.8 | 80% | **29%** |
| 0.95 | 95% | 66% |

Setting 0.8 and believing you were rehearsing four-fifths of the time meant rehearsing
under a third of the time. Every "raise mmfr_weight" decision had been calibrated against
the wrong quantity.

### Validation was measuring the wrong distribution

Training was 80% MMFR; validation was 100% FakeClue. The beautiful 0.786 → 0.251 descent
was the model adapting to the 20% minority it started furthest from — and checkpoints
were being selected on it. §7f now builds a **mixed** validation set in the training
proportion, carved from MMFR records the mixture did not take and checked against
training images so nothing leaks.

### Real targets were 12% longer than fake ones

Confirmed by the length report: real 176 words mean, fake 156. Under token-level
averaging that class quietly takes more gradient even with labels balanced 50/50. The
loss now normalises **per example**, so each image counts once regardless of how much
text its answer contains.

---

## 5. The instrumentation

Because the ordinary signals all failed, the notebook grew its own.

**`p_real`** — the model's probability at the verdict token, read from the generation
scores and renormalised over `{real, fake}`. This is what makes the results section
possible. Greedy decoding takes the argmax, so a model at P(real)=0.55 and one at 0.99
emit the *same string*; only the probabilities distinguish "needs a threshold" from "has
stopped looking".

**The calibration report** in §13 — separation, accuracy at 0.50 versus at the best
threshold, and a diagnosis. It checks separation *first*, because a threshold fitted over
99 candidates beats 0.50 by a few points on pure noise, and testing the threshold gain
first labels every collapse as a calibration problem.

**`GenerationDriftStopper`** — actually generates on a balanced sample of held-out
images every N evaluations and halts if the model answers one label outside 25–75% on two
consecutive probes. Free generation is the only measurement that sees the real behaviour;
it caught a drift and stopped a run at step 800 of 1249.

**Pre-flight checks that fire before expensive steps:**

```python
probe_loss = trainer.training_step(model, data_collator([train_dataset[i] for i in range(2)]))
assert torch.isfinite(probe_loss)
assert with_grad, "no trainable parameter received a gradient"
```

Ten seconds, and it covers the collator keys matching the forward signature, a finite
loss, and gradient actually reaching the adapters. Expect roughly **half** the trainable
tensors to report gradient — `adapter_B` starts at zero, so `dL/dA = 0` on step one.

Also: a dataset audit that fails on templated targets, a train/eval overlap assertion, a
label-masking survey that refuses to continue above 5% fully-masked examples, and a
freeze check that verifies **both** directions — frozen components must have zero
trainable parameters, and trained ones must have some.

---

## 6. Getting the data without the disk

FakeClue is ~28 GB compressed and MMFR ~42 GB, and the training config asks for a few
thousand images. Both are handled without downloading everything, which is what made this
runnable on a 20 GB Kaggle instance before it moved to a local A100.

**ZIP is random-access.** A central directory at the end means one member can be read
with an HTTP range request. §7b plans the selection from the annotation JSON first, then
pulls exactly those images — with one range request per member, sorted by offset.

The detail that decides whether this works: fsspec caches by block, so `block_size=4MB`
turns a 300 KB image into a 4 MB transfer. Extracting 6,250 members from a 28 GB archive
costs **26 GB at 4 MB blocks and 1.8 GB with exact per-member ranges** — a 15× difference
that is invisible unless you count transferred bytes.

**TAR is sequential.** MMFR's is split across 86 parts with no index, so `PartStream`
presents them as one continuous stream, fetching each part when the reader reaches it and
deleting it once consumed. `tarfile` in `"r|"` mode never seeks. All 42 GB crosses the
network; peak disk is one 500 MB part plus what you keep.

Budgets are in **bytes, not files** — MMFR's images span tens of KB to several MB, so a
file count cannot bound a disk. And there is a *per-folder* budget as well as a global
one: real and fake images live in different folders and a tar is read in archive order,
so a single global budget lets whichever folder comes first take all of it. Measured on a
fixture: 104 fake / 8 real under one budget, 26 / 250 with per-folder budgets — same
disk, 3× the usable data, and balanced.

---

## 7. The notebook itself

Rebuilt from the original, which had accumulated a defensive branch and a paragraph of
post-mortem comment for every bug it had ever hit.

| | before | after |
|---|---|---|
| cells | 51 | 44 |
| `try` / `except` blocks | 8 | **0** |
| `raise` statements | 21 | **1** |
| `assert` statements | 7 | **20** |

The one surviving `raise` rejects a record with no `<CONCLUSION>` tag — a data error, not
a bug. Everything else that mattered became a one-line assertion. Auto-detection was
replaced by plain constants with comments saying what they should be:

```python
LOAD_4BIT = True   # change this to False if you have a good gpu
```

Comment lines went *up* 21% while code went down 18%. That is the intended trade: the
code says what to set, the comments say why, and the narrative moved into
`fakereasoning_explained.md`.

The cleanup workflow is packaged as a reusable skill (`notebook-cleanup`), with scripts
for extracting a notebook to an editable cells module, rebuilding it with parse and
cross-cell name-resolution checks, and producing the before/after table above.

---

## 8. Files

| file | what |
|---|---|
| `fakereasoning_finetune.ipynb` | the pipeline: setup → data → LoRA → train → evaluate |
| `fakereasoning_compare.ipynb` | base vs fine-tuned side by side on your own images |
| `fakereasoning_explained.md` | architecture and fine-tuning, in depth |
| `explain_withcode.md` | what gets adapted, with the code that does it |
| `code_review.md` | findings from a full review of the pipeline |

## 9. Running it

```bash
conda create -n fakereasoning python=3.11 -y && conda activate fakereasoning
pip install "torch==2.7.*" torchvision --index-url https://download.pytorch.org/whl/cu126
pip install bitsandbytes einops einops-exts==0.0.4 sentencepiece protobuf safetensors \
  regex filelock pyyaml psutil tqdm packaging typing-extensions fsspec requests pillow \
  numpy scipy scikit-learn seaborn nltk rouge-score markdown2 shortuuid pyarrow \
  jupyterlab ipywidgets
pip install --no-deps transformers==4.37.2 tokenizers==0.15.2 \
  huggingface-hub==0.36.0 accelerate==0.21.0 peft==0.7.1
git clone https://github.com/PRIS-CV/FakeReasoning.git $FR_ROOT/scratch/FakeReasoning
pip install --no-deps --no-build-isolation -e $FR_ROOT/scratch/FakeReasoning/LLaVA
```

The four pinned versions are load-bearing: FakeReasoning's LLaVA fork calls
`transformers` internals that moved in 4.38. `--no-deps` stops pip resolving them back
up, which is also why the long list above exists — those are the transitive dependencies
the pinned installs will not pull in for themselves.

Then set `INSTALL = False` in §1, point `PROJECT_ROOT` at a volume with ~30 GB, and
**smoke-test before committing to a real run** — §6 has a four-value recipe that
exercises every code path in about five minutes.

### What to watch

| output | healthy |
|---|---|
| §8 `wrapped:` on a `"lora"` tower | 144 for CLIP, 96 for DINOv2 |
| §8 dtype breakdown | `float32` no larger than the adapters |
| §10 probe | ~half the trainable tensors receive gradient |
| §10 `[generation probe]` | `says_real` near 50% |
| §13 separation | larger than the base's |

The generation probe is the one that describes the model. Validation loss falls straight
through a prior collapse; it did so for 1,150 steps on an earlier run while the model
quietly settled on answering "real" for everything.
