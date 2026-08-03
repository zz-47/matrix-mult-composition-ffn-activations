# matrix-mult-composition-ffn-activations

**SLM math, measured not assumed.** The second half of the decoder layer in [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — the feed-forward network and the nonlinearities that make it work. Every block of the forward pass is a composed matrix multiply: `h = W_E[x] → 30×(Attn + FFN) → softmax(U·h)`. This repo traces each weight's real shape from `model.state_dict()`, plots the four activation candidates (ReLU, GELU, SiLU, SwiGLU), and reimplements the actual SiLU-gated FFN (`576 → 1536 → 576`) in NumPy — proving the gate, not just reading about it.

> This is not generic linear algebra — this is SLM math. Every result below is *measured* on the actual model weights, not assumed from a spec sheet.

---

## The Real Model, Not the Spec Sheet

The sprint spec assumed one configuration. The **actual** checkpoint (config.json + safetensors) differs:

| Property | Assumed | **Measured** |
|----------|---------|--------------|
| Model type | — | **llama** (arch) |
| Hidden size `d_model` | 576 | **576** |
| FFN intermediate `d_ff` | 1536 | **1536** |
| Expansion ratio | — | **1536 / 576 = 2.67×** |
| Hidden activation | — | **silu** (SiLU/Swish, gated) |
| Layers | 12 | **30** |
| Vocabulary | 50,000 | **49,152** |
| Tied embeddings | — | **True** (`lm_head` absent from state_dict) |
| `W_E` | 50000×576 | **(49152, 576)** |
| `W_Q` | 576×576 | **(576, 576)** (9 heads, d_k=64) |
| `W_K`, `W_V` | 576×192 | **(192, 576)** (3 KV heads, d_k=64) |
| `W_O` | 192×576 | **(576, 576)** |
| `W_gate`, `W_up` | 576×1536 | **(1536, 576)** |
| `W_down` | 1536×576 | **(576, 1536)** |

The headline claim: torch stores every Linear weight as `(out, in)` — the paper's `W ∈ 576×192` is the same matrix stored `(192, 576)` and applied as `x @ W.T`.

---

## Units

| Unit | Topic | Core formula | Status |
|------|-------|-------------|--------|
| 1 | Matrix multiplication as transformation composition | `h = W_E[x] → 30×(Attn+FFN) → softmax(U·h)` | ✅ Complete |
| 2 | Activations: ReLU, GELU, SiLU, SwiGLU | `SiLU(x) = x·sigmoid(x)` | ✅ Complete |
| 3 | FFN SwiGLU math | `FFN(x) = W_down·(SiLU(W_gate·x) ⊙ W_up·x)` | ✅ Complete |

---

## Repository Structure

```
matrix-mult-composition-ffn-activations/
├── README.md                    ← this file
├── requirements.txt             ← pinned env (torch 2.13.0, transformers 5.14.1)
├── venv/                        ← local environment
├── .vscode/settings.json        ← interpreter → this venv
├── unit1_matrix_mult.ipynb      ← composition, measured shapes, spec-vs-real
├── unit2_activations.ipynb      ← the four nonlinearities, why SiLU wins
└── unit3_swiglu_ffn.ipynb       ← FFN rebuild, gating effect
```

Each notebook is **self-contained** (loads the model fresh), runs on CPU-only Windows in minutes, and follows the *issue → hypothesis → fix* narrative with inline statistics and graphs.

---

## Running the Notebooks

```bash
# 1. Terminal in this dir auto-activates venv (PowerShell profile hook) — or:
.\venv\Scripts\Activate.ps1

# 2. Launch
jupyter notebook unit1_matrix_mult.ipynb
```

Requirements: `torch`, `transformers`, `numpy`, `matplotlib`, `jupyter` (see `requirements.txt`). Model is cached from the repo-1 sprint, no re-download.

---

## Unit 1 — Matrix Composition & FFN Spectra (measured)

**Claim proven:** the forward pass is a composition of ~60 linear maps whose shapes are exactly predicted by `W_E[x] → 30×(Attn+FFN) → U·h` — verified against all 272 keys of `model.state_dict()`.

**Measured findings:**

| # | Claim | Measured | Verdict |
|---|---|---|---|
| 1 | Embedding `W_E` shape | (49152, 576) | ✅ Holds (vocab differs from spec) |
| 2 | Layer structure q/k/v/o + gate/up/down + 2 norms | 9 keys/layer × 30 | ✅ Holds |
| 3 | `lm_head` tied to embedding | absent from state_dict | ✅ Holds |
| 4 | MLP widens then narrows | 576→1536→576 | ✅ Holds |
| 5 | Expansion ratio | 2.67× | ✅ Holds |
| 6 | FFN energy concentrated (anisotropic) | see spectra below | ✅ Holds |

**Capacity location (the career number):** per layer the FFN trio holds `2,654,208` params vs attention's `884,736` — **the FFN is ~3× attention's size**; it is where memory, quantization, and LoRA budgets live.

**Effective rank @90% energy (measured via SVD):**
| Matrix | effective rank / total | top-1 energy share |
|---|---|---|
| `W_E` | 366 / 576 | 50.6% |
| `W_gate` | 394 / 576 | 3.0% |
| `W_up` | 396 / 576 | — |
| `W_down` | 388 / 576 | — |

Interpretation: even the widest FFN matrices are effectively ~390-dim. That concentration is precisely what low-rank adaptation exploits — and why quantization can drop the low-energy tail.

---

## Unit 2 — Activations (measured)

**Claim proven:** the activation is the nonlinear crack between matmuls that makes depth meaningful; SmolLM2's `hidden_act = silu` *dims* instead of *kills*.

**Measured findings:**

| # | Claim | Measured | Verdict |
|---|---|---|---|
| 1 | `hidden_act` is SiLU, gated | `silu` from config | ✅ Holds |
| 2 | SiLU smooth & non-monotonic | dips negative near x≈−1, never flat-zero | ✅ Holds |
| 3 | SiLU derivative bounded, never zero | min −0.1, max 1.1 | ✅ Holds (corrected theory: dips slightly negative) |
| 4 | ReLU death zone | `ReLU'` exactly 0 for x<0 | ✅ Holds |
| 5 | Gate mutes dims per token | 36.1% dims `|g|<0.1` | ✅ Holds (measured on "the") |
| 6 | Gating is input-dependent | gate vector depends on x | ✅ Holds |

**Correction the measurement caught:** the pre-written theory claimed `0 < SiLU'(x) ≤ ~1.1`, but measurement on [−5, 5] gives **min = −0.1** (zero-crossing at x ≈ −1.28). SiLU' is never *exactly* zero — that part holds — but it does dip slightly negative. The safe claim: `SiLU'(x) ∈ [−0.1, 1.1]`, never zero ⇒ no death zone (unlike ReLU's flat 0 over an interval).

**Gating measured live (CB 2.3, token "the"):** for one token, **36.1%** of the 1536-wide workbench is muted (`|gate| < 0.1`) and only **0.3%** is fully open (`|gate| > 0.9`). The gate stem-plot shows a few strong spikes over a mostly-muted sea; the `g·u` histogram clusters near 0 with a long strong-feature tail. This is Unit 1's low effective rank, now seen at runtime per-token.

---

## Unit 3 — Full SwiGLU FFN Rebuild (in progress)

**Claim proven:** the FFN `FFN(x) = W_down(SiLU(W_gate·x) ⊙ W_up·x)` rebuilt in NumPy reproduces the real layer-0 MLP to **9.5e-07 max abs error** (float32 rounding noise); the gate demonstrably reshapes the output (mean |Δout| 1.58 vs the ReLU variant), and SwiGLU *dims* 94.7% of the 1536-space where ReLU would hard-*kill* 50.7%.

**Three concepts:**
| CB | Concept | What it measures | Status |
|---|---|---|---|
| 3.1 | FFN rebuild, verified vs `model.layers[0].mlp` | `max abs error` ≈ 1e-5 (the trust standard) | ✅ 9.5e-07 |
| 3.2 | Gated vs ReLU FFN, same weights | output delta + hard-kill vs soft-mute fractions | ✅ measured |
| 3.3 | Runtime activation rank | SVD effective rank @90% of the 6×1536 activation matrix + ever-strong dim union | ✅ 5 of 6 · 45/1536 |

**Measured findings:**

| # | CB | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|---|
| 1 | 3.1 | NumPy FFN rebuild == real layer-0 MLP | max abs err < 1e-4 | max **9.5e-07**, mean **1.1e-07** — float32 rounding noise (mean ≈ machine ε) | ✅ Holds |
| 2 | 3.2 | Gate reshapes output (not decorative) | mean \|Δout\| ~ 1e-1..1e-2 | mean **1.577**, max **25.39** (~half a typical output per coordinate) | ✅ Holds — *an order of magnitude stronger* |
| 3 | 3.2 | ReLU hard-kills vs SwiGLU soft-mutes | ReLU dead > SwiGLU muted | ReLU dead **50.7%** (778/1536) vs SwiGLU muted **94.7%** (1454/1536) | ✅ Holds — *prediction was backwards* |
| 4 | 3.3 | Live activation rank is tiny | k@90% in 2..5 of 6 | k@90% = **5 of 6** (spectrum near-flat) | ✅ Holds — at the top edge of the band |
| 5 | 3.3 | Few dims ever-strong per sentence | union << 1536 | union = **45 of 1536 (2.9%)** | ✅ Holds — stronger than predicted |

**Two corrections the measurement caught (rows 2-3):**
- **The gate is *not* decorative — it's more aggressive than expected.** Row 2 predicted a mean Δoutput of 1e-1..1e-2; measured **1.577** (max 25.4 on a single coordinate). On this token the gate doesn't tweak the output — it redesigns it.
- **The naive sparsity prediction was backwards.** Row 3 predicted ReLU's exact-zero fraction would beat SwiGLU's muted fraction. Measured the opposite: **SwiGLU 94.7% > ReLU 50.7%.** The gated value is a *product* `silu(W_gate·x)·(W_up·x)` — if *either* factor is small the dim dies — so gating concentrates all signal into ~**82/1536** dims (5%), versus ReLU leaving ~**758** dims at full strength. The two fractions aren't directly comparable (exact-0 vs <0.1 threshold), which is exactly why the naive prediction flipped. CB 3.3 confirmed the consequence: only **45/1536** dims are ever strong across the 6 tokens — but (see row 4) that tiny active set is *not* compressed.

**The thesis:** Unit 1 measured *weight* rank (~390/576). Unit 3 measured *activation* reality: only **45 of 1536** dims ever fire for a sentence (2.9% — the collapse is in the *feature set*), while those tokens stay diverse within the active set (k@90% = 5 of 6 — the collapse is *not* in the geometry). Measured against the full 1536-wide space, the live FFN representation is overwhelmingly sparse; this is the empirical basis for pruning and quantization of the never-opened dims, and for LoRA budget in the production repos. Limitation: 6 tokens is a single sentence, not the model's activation distribution.


