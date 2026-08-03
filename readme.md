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
| 1 | Matrix multiplication as transformation composition | `h = W_E[x] → 30×(Attn+FFN) → softmax(U·h)` | ⏳ In progress |
| 2 | Activations: ReLU, GELU, SiLU, SwiGLU | `SiLU(x) = x·sigmoid(x)` | ⬜ Planned |
| 3 | FFN SwiGLU math | `FFN(x) = W_down·(SiLU(W_gate·x) ⊙ W_up·x)` | ⬜ Planned |

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

## unit 1

Concept it proves: X is the model's input to layer 0 — the concrete h₀ of the composition h₀ = W_E[x]. W_E is the first matrix of the pipeline. Everything downstream (attention mats, FFN mats) multiplies against vectors shaped exactly like X's rows (576).

