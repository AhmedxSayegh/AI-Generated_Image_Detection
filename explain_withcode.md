# What actually gets fine-tuned, and the code that does it

FakeReasoning has four trainable-in-principle components: a Vicuna-13B decoder, a CLIP
ViT-L/14-336 tower, a DINOv2 ViT-L/14 tower, and a fusion stack (two projectors plus a
cross-attention module). This notebook adapts three of them with LoRA and leaves the
fourth completely untouched.

Three separate mechanisms are at work. Two are hand-rolled; one is peft. They are
separate because peft can only reach modules it recognises, and the vision towers are
not standard `transformers` modules.

---

## 1. The LLM decoder — peft does this one

**Targets: 7 linears per block × 40 blocks = 280 linears.**

| where | linears |
|---|---|
| self-attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| MLP (SwiGLU) | `gate_proj`, `up_proj`, `down_proj` |

### Building the target pattern

peft accepts either a list of names or a regex string, and the difference matters more
than it looks. Given a **list**, peft matches each entry by *suffix* —
`key.endswith(f".{target}")`. In a plain language model that is harmless. In a VLM,
`"q_proj"` also matches every `q_proj` inside the CLIP tower, so a config that says the
vision towers are frozen would silently be adapting them anyway.

So the notebook builds an **anchored regex** instead, and finds the decoder by identity
rather than guessing its name:

```python
decoder = model.get_model()
prefix = next(name for name, module in model.named_modules() if module is decoder)
pattern = (rf"{re.escape(prefix)}\.layers\.\d+\."
           rf"(?:self_attn|mlp)\.(?:{'|'.join(leaf_names)})")

matched = [n for n, m in model.named_modules()
           if isinstance(m, nn.Linear) and re.fullmatch(pattern, n)]
leaked = [n for n in matched if any(k in n for k in MULTIMODAL_KEYS)]
assert matched, f"LoRA target regex {pattern!r} matched nothing (prefix {prefix!r})."
assert not leaked, f"LoRA target regex reaches multimodal modules: {leaked[:3]}"
```

peft applies `re.fullmatch()` to a regex target, so an anchored pattern cannot reach a
vision module the way a bare leaf name can.

### Applying it

```python
model = get_peft_model(model, LoraConfig(
    r=FINETUNE_CONFIG.llm_lora_r,
    lora_alpha=FINETUNE_CONFIG.llm_lora_alpha,
    lora_dropout=FINETUNE_CONFIG.llm_lora_dropout,
    target_modules=llm_target_modules,
    modules_to_save=FINETUNE_CONFIG.modules_to_save or None,
    bias="none",
    task_type="CAUSAL_LM",
))
```

And the post-condition, checked *after* peft has run rather than trusted beforehand:

```python
def assert_no_lora_outside_llm(model):
    from peft.tuners.lora import LoraLayer
    strays = [name for name, module in model.named_modules()
              if isinstance(module, LoraLayer) and any(k in name for k in MULTIMODAL_KEYS)]
    assert not strays, (
        f"{len(strays)} peft LoRA layer(s) were injected OUTSIDE the language model, "
        f"e.g. {strays[:3]}. The vision-tower freeze is not real."
    )
```

This produces the `280 Linear layers, all inside the decoder` line in §8's output.

---

## 2 and 3. The vision towers — hand-rolled

peft cannot target these, so the notebook injects its own `LoRALinear` wrapper. The two
towers need different target sets because the architectures name things differently:

```python
DINO_LORA_TARGETS = {"qkv", "proj", "fc1", "fc2"}
CLIP_LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
```

| tower | targets/block | blocks | linears |
|---|---|---|---|
| CLIP ViT-L/14-336 | 6 | 24 | **144** |
| DINOv2 ViT-L/14 | 4 | 24 | **96** |

DINOv2 needs only four because its attention is a single **fused** `qkv` projection
(1024 → 3072) rather than three separate ones. One adapter on the fused layer is also
cheaper than three on the split ones: `8×(1024+3072) = 32,768` against
`3 × 8×(1024+1024) = 49,152`.

### The switch that reads your config

```python
def configure_vision_tower_finetuning(tower, which, mode, r, alpha, dropout):
    if mode == "frozen":
        return {"which": which, "mode": mode, "trainable_params": 0, "wrapped": 0}

    forward_fn, target_names = TOWER_SETUP[which]
    tower.forward = types.MethodType(forward_fn, tower)   # drop the @torch.no_grad()

    replaced = inject_lora_linear(tower.vision_tower, target_names,
                                  r=r, alpha=alpha, dropout=dropout)
```

The `tower.forward = ...` line matters as much as the injection. The released
`CLIPVisionTower.forward` and `DINOVisionTower.forward` are decorated with
`@torch.no_grad()` — correct for zero-shot inference, fatal for training. Without
replacing them, no gradient reaches the adapters no matter what `requires_grad` says.

`"frozen"` returns before any of this, so the released code is left exactly as
published. The frozen path is *no path at all*.

### The injection itself

```python
for name, module in list(root_module.named_modules()):
    if not isinstance(module, nn.Linear) or name.split(".")[-1] not in target_leaf_names:
        continue
    parent_name, _, child_name = name.rpartition(".")
    parent = root_module.get_submodule(parent_name) if parent_name else root_module
    setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
    replaced.append(name)
```

The `isinstance(module, nn.Linear)` check earns its keep here: DINOv2's
`patch_embed.proj` is a `Conv2d`, and `proj` is in the target set. It is skipped because
it is not a `Linear`.

### Called once per tower

```python
dino_info = configure_vision_tower_finetuning(
    model.get_dino_vision_tower(), "dino", FINETUNE_CONFIG.dino_finetune_mode,
    r=FINETUNE_CONFIG.dino_lora_r, alpha=FINETUNE_CONFIG.dino_lora_alpha,
    dropout=FINETUNE_CONFIG.dino_lora_dropout,
)
clip_info = configure_vision_tower_finetuning(
    model.get_vision_tower(), "clip", FINETUNE_CONFIG.clip_finetune_mode,
    r=FINETUNE_CONFIG.clip_lora_r, alpha=FINETUNE_CONFIG.clip_lora_alpha,
    dropout=FINETUNE_CONFIG.clip_lora_dropout,
)
```

### Why the adapters are not called `lora_A` / `lora_B`

`LoRALinear` names its factors `adapter_A` and `adapter_B` on purpose. peft decides what
to write into `adapter_model.safetensors` with a bare substring test —
`{k: v for k, v in state_dict.items() if "lora_" in k}` — with no check that peft
created the parameter. Hand-injected factors named `lora_A` are therefore swept into
peft's adapter file, renamed on load (`lora_A` → `lora_A.default`), and silently
dropped, because peft's own adapters are `ModuleDict`s keyed by adapter name and these
are bare `Parameter`s. Keeping `lora_` out of the name leaves the vision adapters
entirely to `vision_tower_weights.pt`, which is the file that actually round-trips them.

### The check that catches a silent no-op

A mode of `"lora"` that wrapped nothing is the worst outcome available: the config reads
as if the tower is training, the run completes, and the tower never moved.

```python
assert mode == "full" or replaced, (
    f"{which}: mode is {mode!r} but nothing under tower.vision_tower matched "
    f"{sorted(target_names)}. No adapter was created, so this tower would have "
    f"trained nothing while the config claimed otherwise."
)
```

---

## 4. mm_projector, dino_mm_projector, crossattention — **no LoRA, not trained at all**

These are excluded twice over.

**The regex cannot reach them.** That is what `MULTIMODAL_KEYS` asserts against in §1
above — if the pattern ever matched a name containing `mm_projector`, `crossattention`,
`vision_tower` or `dino_tower`, the build fails.

**And the only mechanism that could train them is empty:**

```python
# Anything named here is FULLY fine-tuned, not adapted. The three candidates are
# "mm_projector", "dino_mm_projector", "crossattention" -- 377M released params.
# Leave this empty unless §14 proves that training them helps.
modules_to_save: List[str] = field(default_factory=list)
```

Note what that field does if you fill it: **full fine-tuning, not LoRA.** peft replaces
the module with a trainable copy and keeps a frozen fp32 copy besides — no low-rank
constraint, no initialise-at-identity. Listing all three trains 377M released parameters
directly, which is how an earlier run of this notebook overwrote the checkpoint's own
detector.

The freeze is verified rather than assumed, in both directions:

```python
violations = [f"{c}: frozen by the config but has "
              f"{TRAINABLE_REPORT['breakdown'][c]:,} trainable params"
              for c, should_be_frozen in frozen_expected.items()
              if should_be_frozen and TRAINABLE_REPORT["breakdown"].get(c, 0) > 0]

violations += [f"{c}: the config trains it but it has 0 trainable params"
               for c, should_be_frozen in frozen_expected.items()
               if not should_be_frozen and TRAINABLE_REPORT["breakdown"].get(c, 0) == 0]

assert not violations, "Config and model disagree:\n  " + "\n  ".join(violations)
```

---

## The totals

With all three components on `"lora"`:

| | targets/block | blocks | linears | trainable params |
|---|---|---|---|---|
| LLM decoder (r=16) | 7 | 40 | 280 | 62,586,880 |
| CLIP tower (r=8) | 6 | 24 | 144 | 3,538,944 |
| DINOv2 tower (r=8) | 4 | 24 | 96 | 3,145,728 |
| projectors + fusion | — | — | **0** | **0** |
| | | **520** | **69,271,552** |

Every count comes from the same formula — `r × (d_in + d_out)` per adapted linear,
summed over the layers.

Two lines in §8's output confirm the whole thing:

```
Optimizer: AdamW8bit | {'llm': 560, 'vision': 480}
```

560 = 280 linears × 2 tensors (`lora_A`, `lora_B`). 480 = (144 + 96) × 2
(`adapter_A`, `adapter_B`). **No `projector` group at all** — that is the fusion stack
being genuinely untouched.

```
probe loss 0.1199 | 376/1040 trainable tensors received gradient
```

Roughly half is correct on the first step: `B` is initialised to zero, so
`dL/dA = Bᵀ(∂L/∂y)xᵀ = 0` for every adapter until `B` moves. A number far from half
means one of the three groups is not receiving gradient — 376 decomposes as 280 (LLM) +
96 (DINO) with nothing from CLIP's 144, which is exactly how a silently inert CLIP
injection shows up.

---

## Where the gradient goes

```
image ─┬─► CLIP tower ──┐
       │                ├─► crossattention ─► projectors ─► 576 visual tokens
       └─► DINO tower ──┘                                          │
                                                                   ▼
                             text tokens ──────────► [visual tokens spliced in]
                                                                   │
                                                                   ▼
                                                  Vicuna-13B decoder (LoRA)
                                                                   │
                                                                   ▼
                                             loss, on the answer tokens only
```

**With both towers frozen**, the backward pass reaches the decoder's adapters and stops
there. It never enters the projectors, the fusion module or either tower, because
nothing in that path requires grad. CLIP and DINOv2 run as fixed feature extractors:
they decide *what the model sees*, and the LoRA learns *what to say about it*. Their
weights are bit-identical before and after.

**With a tower on `"lora"`**, the gradient continues past the decoder, back through the
projectors and `crossattention`, into that tower's adapters. This is why unfreezing a
tower is not a local change: the fusion stack suddenly has to carry gradients it never
carried before. On a 4-bit load that surfaces as

```
RuntimeError: Output 0 of MatMul4BitBackward is a view and is being modified inplace
```

because `crossattention`'s FFN applies `nn.ReLU(inplace=True)` to the output of a 4-bit
linear — harmless until that output becomes part of an autograd graph.
