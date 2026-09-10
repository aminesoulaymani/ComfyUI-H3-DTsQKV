# ComfyUI-H3-DTsQKV

**Run DmitryDB's DT-sQKV MiniMax H3 checkpoints on a stock ComfyUI — no core patch, nothing to re-patch
after updates, LoRAs fully applied, your usual optimisation nodes untouched.**

## The problem, as users live it

You have a 12–16 GB card, Windows, and you want MiniMax H3 at its native canvas — 768 px short edge,
1344×768 at 16:9 — for ten or fifteen seconds. You press *Queue*.

```
0%|          | 0/8 [00:00<?, ?it/s,   Model Initializing ...  ]
```

Twenty minutes later it is still there. Or it dies with `CUDA error: illegal memory access`. That is
ComfyUI issue [#15628](https://github.com/Comfy-Org/ComfyUI/issues/15628) — 12 GB card, Windows, 1 MP ×
15 s, int8 and NVFP4 checkpoints alike — **open, with no maintainer diagnosis** at the time of writing. It
has company: GPU lost on machines with 64 GB of RAM ([#15488](https://github.com/Comfy-Org/ComfyUI/issues/15488)),
access violation at VAE decode ([#15337](https://github.com/Comfy-Org/ComfyUI/issues/15337)), crash after
*Clean / Unload* ([#15352](https://github.com/Comfy-Org/ComfyUI/issues/15352)), SageAttention producing
noise past ~160k tokens on RTX 50 ([#15263](https://github.com/Comfy-Org/ComfyUI/issues/15263)).

The "MiniMax H3 on 8 GB" workflows are real and honest about what they do: 480×864, block swapping,
4-step LoRAs, or generate small and upscale afterwards. They lower the canvas. None of them holds it.

We tried every H3 checkpoint we could find on a 16 GB RTX 5060 Ti — the official pruned int8 builds, the
12 GB w4a8 and NVFP4 ones. At the native canvas and 15 seconds, none produced a first step. **One family
did: [DmitryDB's DT-sQKV checkpoints](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV)**, which
start after a few minutes and sample at about 215 s per step. In our experience they are excellent.

They also came with three problems of their own:

1. **They need a patch to the ComfyUI core** — ten hunks across three files. Every ComfyUI update undoes it.
   You re-apply it by hand and hope the lines still match, or you stop updating.
2. **Standard LoRAs half-apply, silently.** Their attention layers do not find their target; ComfyUI logs
   `lora key not loaded` fifty-two times and carries on. The result looks *almost* right. Nothing tells
   you that every attention layer of your LoRA was thrown away.
3. **Your optimisation nodes stop recognising the model.** kjnodes' Sage Attention crashes with
   `'Attention' object has no attribute 'qkv_proj'`; others skip the model without a word, or take a code
   path that hangs at 0 %.

## What this module does for you

- **Install it like any node pack, and forget the patch.** Load the checkpoint with the stock *Load
  Diffusion Model* node. Update ComfyUI whenever you want; there is nothing to re-apply.
- **Use any MiniMax H3 LoRA with the stock LoraLoader.** Every layer is applied, attention included. The
  console tells you how many were converted.
- **Keep your nodes.** kjnodes Sage Attention, sol-attn, EasyCache, Spectrum, the core's Sparse Attention:
  they see a normal MiniMax H3 and behave as they do on one.
- **It tells you when something moved.** After a ComfyUI update the module checks the parts of the core it
  relies on. If they changed but it can still cope, it names them in the console and carries on. If it
  cannot, it refuses to load and says why — rather than loading a wrong model. It went through the
  0.34.0 → 0.35.0 update this way, unattended.
- **It does not make anything faster and does not change the picture.** It only lets an unmodified
  ComfyUI build the model. Measured on our machine: same time per step as the source patch (213–217 s
  at 115 000 tokens), 641 of 641 tensors matched against the checkpoint file.

| | source patch | this module |
|---|---|---|
| ComfyUI update | re-apply 10 hunks in 3 files by hand | nothing to do; the module reports what moved |
| standard LoRA | 52 attention layers silently dropped | fully applied |
| kjnodes Sage, sol-attn, Sparse Attention… | crash, silent skip, or hang | work as on a stock model |
| something incompatible upstream | a patch that half-applies | a refusal with the reason |

## What a working setup looks like (our machine)

RTX 5060 Ti 16 GB, 32 GB RAM, Windows 11, native canvas, 15 seconds, 8 steps. Nothing exotic in the
workflow: the stock MiniMax H3 template, DmitryDB's HQ checkpoint in *Load Diffusion Model*, and this pack
installed.

| what you add | time for 15 s at the native canvas |
|---|---|
| nothing but a turbo LoRA — [lightx2v `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16`](https://huggingface.co/lightx2v/Minimax-h3-Turbo), loaded as-is | **about 1 hour** (515 s per step) |
| + kjnodes **MiniMax H3 Chunk FeedForward** and **MiniMax H3 Mem Eff Sage Attention Patch** | **about 30 minutes** (213–217 s per step) |
| + **Spectrum Apply MiniMax H3**, out-of-the-box parameters | **about 22 minutes** — with, to our eye, a slightly lower quality; very good for prototyping |

Neither kjnodes node is required to *start*; a plain workflow with this checkpoint starts without
trouble. But on 16 GB they halve the time: they keep the activations from spilling into shared system
memory, which is where the other half of the hour goes. Turbo LoRAs are fully functional through the
stock LoraLoader — the one above is the one we use.

We also wired a whole set of attention and acceleration nodes at once — sol-attn's Sol sparse attention,
Spectrum, chunked FFN, memory-efficient Sage, EasyCache — to see what happens. Everything runs, no bugs.
Take that as a **technical** proof that the model is usable with the ecosystem, not as a quality
statement: we did not evaluate what each node does to the picture, and whatever trade-off an accelerator
makes, we expect it to make the same one on a stock checkpoint. Which nodes to use is your call. Ours, day
to day, is the two kjnodes nodes and nothing else.

> **Status: experimental.** Verified on one machine (see [Measurements](#measurements)), against ComfyUI
> 0.34.0 and 0.35.0, Windows only. Written in pair sessions with an AI model — see
> [How this was made](#how-this-was-made) before trusting it with your workflow. If you know a checkpoint
> that runs at the native canvas on 16 GB without a core patch, open an issue: we will list it.

---

## Table of contents

Everything below is the technical part.

- [What a working setup looks like](#what-a-working-setup-looks-like-our-machine)
- [The checkpoint](#the-checkpoint)
- [Context: MiniMax H3 on 12–16 GB, September 2026](#context-minimax-h3-on-1216-gb-september-2026)
- [Why a vanilla core cannot load it](#why-a-vanilla-core-cannot-load-it)
- [What the module does](#what-the-module-does)
- [The int32 wall — why separate Q/K/V matters beyond memory](#the-int32-wall--why-separate-qkv-matters-beyond-memory)
- [Install](#install)
- [Verify](#verify)
- [Updating ComfyUI](#updating-comfyui)
- [LoRAs](#loras)
- [Third-party packs](#third-party-packs)
- [Measurements](#measurements)
- [Limitations](#limitations)
- [Converting other checkpoints to DT-sQKV](#converting-other-checkpoints-to-dt-sqkv)
- [How this was made](#how-this-was-made)
- [Maintainer notes](#maintainer-notes)
- [Credits and license](#credits-and-license)

---

## The checkpoint

**[DmitryDB/MiniMax-H3-DynTime-sQKV](https://huggingface.co/DmitryDB/MiniMax-H3-DynTime-sQKV)** — two
quantisation profiles, each in FL2VA and Ref2VA variants. Weights are under the MiniMax-H3 community
license (not included here).

| file | size | int8 tensors | what is quantised |
|---|---|---|---|
| `MiniMax-H3_Ref2VA-DT-sQKV-INT8-ConvRot-HQ.safetensors` | 28.0 GiB | 122 / 641 | `mlp.fc1` on 50/50 blocks, `q/k/v_proj` on 24/50 blocks; everything else bf16 |
| `MiniMax-H3_Ref2VA-DT-sQKV-INT8-ConvRot.safetensors` | 21.0 GiB | 270 / 641 | `q/k/v_proj` 50/50, `fc1` 50/50, `fc2` 47/50, `out_proj` 23/50 |

(tensor census read from the safetensors headers; older downloads of the 21 GiB profile were named
`…-int8-lean-convrot-dynamic-k16-separate-qkv-quality21…`)

The author's model card recommends 24 GB-class GPUs for the first profile and 32 GB+ for HQ, and points
8–16 GB users to stock W4/NVFP4 checkpoints. Our experience on a **16 GB card with dynamic VRAM
offloading** is the opposite: the HQ profile is the only H3 checkpoint that ran at native 1 MP
resolution for us, and it did so at the same per-step cost as the source patch (see
[Measurements](#measurements)). The model card also states that no perceptual A/B against the stock
checkpoints is claimed; we do not claim one either.

### What is non-standard, and by how much

**sQKV — separate projections.** Instead of `blocks.N.attn.qkv_proj.weight [21504, 5376]`, the file
holds `q_proj`, `k_proj`, `v_proj`, each `[7168, 5376]`. Bit-exact slices of the fused matrix; three
GEMMs instead of one. Same for the two token-refiner blocks.

**DT — "DynTime", rank-16 adaLN.** In a full-width H3, every block's adaLN projection is
`adaln_proj.linear.weight [96768, 2688]` (the core passes `time_embed_dim = 2688` as the adaLN input
width; a vanilla LoRA trying to merge into this layer reports a delta of exactly 96768 × 2688 =
260 112 384 elements). Fifty blocks of that is **13.0 G parameters — 26 GB in bf16, more than the
entire attention + MLP backbone (19.3 G)**. Every reduced H3 checkpoint gets rid of them; the question is
how. Comfy-Org's "pruned" builds replace these modulation weights (~40 % of the parameters) with a
precomputed lookup table `adaln_t_table` (61.7 GB bf16 → 31.7 GB int8 → 19.5 GB pruned int8, "no loss in
output quality" per the [day-0 post](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui) and
the [weights note](https://comfyui-wiki.com/en/news/2026-08-03-minimax-h3-open-weights-comfyui)).
DT-sQKV takes the other route and keeps the time MLP:

```
adaln_curve_basis   [2688, 16]  fp32   shared by all blocks, orthonormal to 1.7e-8 (an SVD basis)
adaln_curve_mean    [2688]      fp32
blocks.N.adaln_proj.linear.weight  [96768, 16]  fp32
time_embedder.*     fp32        the original MLP 256 → 5376 → 2688, kept
```

and computes `t_emb = (silu(time_embedder(t)) − mean) @ basis`. That works because the time embedding
traces a **one-dimensional curve** in a 2688-dimensional space (a single scalar `t` drives it); sixteen
principal directions capture it — the author claims a full-time relative reconstruction error of about
3e-7. The result is 82 M adaLN parameters instead of 13 G, a factor of 160.

So the size win is **shared** with the stock pruned checkpoints; what DT changes is the fidelity trade:
a continuous fp32 evaluation of the time curve instead of a table interpolated on a grid. Whether that is
visible in the output is not something we measured, and the author claims no perceptual A/B either.

Both routes are **irreversible**: the full-width weights are in neither file. No loader can turn a DT
checkpoint back into a vanilla one; the only option is to *emulate* the DT geometry, which is what this
pack does. The core loads the table route natively; it does not load the MLP + basis route — hence the
patch, and hence this pack.

---

## Context: MiniMax H3 on 12–16 GB, September 2026

What we found when we looked for prior art, so you do not have to tell us. If something here is outdated,
open an issue.

| | what it is | what it means for a 16 GB card at the native canvas |
|---|---|---|
| [Comfy-Org pruned int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3) (19.5 GB) | the official reduced checkpoint, `adaln_t_table` route | the one we and issue #15628 could not get past `Model Initializing` at 1 MP × 15 s on Windows. Another 16 GB user's [measured notes](https://github.com/Tomiigo/minimax-h3-16gb) (RTX 5070 Ti, Linux, ComfyUI 0.30.1) run it at **640×384**, 226 frames, 190 s total |
| [kijai w4a8](https://comfyui-wiki.com/en/news/2026-08-05-kijai-minimax-h3-w4a8) (12.5 GB), [coolthor](https://huggingface.co/coolthor/MiniMax-H3-pruned-NVFP4) / [lilcheaty](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4) NVFP4 | 4-bit weight builds, Blackwell-oriented | smaller download, same fused layout; NVFP4 is emulated (no speed) below sm_120. Not started at 1 MP × 15 s in our hands |
| [DmitryDB/MiniMax-H3-ComfyUI-Quants](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants) | the same author's **stock-compatible** quants | no patch, no pack needed — and no separate Q/K/V |
| [DmitryDB/MiniMax-H3-10Eros-Max-DT-sQKV](https://huggingface.co/DmitryDB/MiniMax-H3-10Eros-Max-DT-sQKV) (also [mirrored](https://huggingface.co/Kerochake/MiniMax-H3-10Eros-Max-DT-sQKV)) | a community fine-tune ("10Eros_Max") in the same DT-sQKV format, 21 GiB, FL2VA | loads with this pack like the base checkpoints — same layout, same patch requirement upstream |
| "H3 on 8 GB" workflows — [reventadirecta](https://github.com/reventadirecta/MiniMax-H3-ComfyUI-8GB-VRAM), [Civitai](https://civitai.com/models/2901854/easy-workflow-minimax-h3-ultimate-upscale-for-turbo-hybrid-8gb-vram-friendly), [UniblockSwap](https://civitai.com/models/2863388/minimax-h3-48-step-accelerating-workflow-collection8gb) | `--lowvram`, block swapping, 4/8-step LoRAs, **480×864 recommended**, or lower-res generation + Ultimate Upscale to 1280×736 | real and useful — but they lower the canvas or upscale afterwards. None of them claims the native canvas at 15 s on 8 GB, and neither do we |
| Open ComfyUI issues on Windows / Blackwell / H3 | [#15628](https://github.com/Comfy-Org/ComfyUI/issues/15628) hang at init (12 GB, 1 MP × 15 s) · [#15488](https://github.com/Comfy-Org/ComfyUI/issues/15488) GPU lost with 64 GB RAM · [#15337](https://github.com/Comfy-Org/ComfyUI/issues/15337) VAE decode access violation · [#15352](https://github.com/Comfy-Org/ComfyUI/issues/15352) crash after unload · [#15263](https://github.com/Comfy-Org/ComfyUI/issues/15263) Sage FP8 PV noise above ~160k tokens on sm_120 | the 16 GB Windows experience is rough for everyone. #15263 is worth knowing if you push beyond 15 s: our runs at 115k tokens are below that threshold |
| [scottmudge HybridLoader](https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader), [abakanai hybrid](https://huggingface.co/abakanai/Minimax_h3_hybrid) | mixing FL2VA / Ref2VA blocks | orthogonal to this pack; untested together |

Native canvas per the ComfyUI docs: 768 px short edge, 4–15 s, 24 fps, 32 kHz stereo audio.

---

## Why a vanilla core cannot load it

Three places, all in the core, all hard-coded to the fused / full-width layout:

1. **Detection** — `model_detection.detect_unet_config` reads `blocks.0.attn.qkv_proj.weight` to infer
   the head count. Missing key → exception before anything is built.
2. **Construction** — `Attention.__init__` creates `qkv_proj`; `MiniMaxH3Model.__init__` creates
   `adaln_proj` with input width `time_embed_dim` and either a `time_embedder` *or* an `adaln_t_table`.
   A DT checkpoint needs `q/k/v_proj`, a width-16 fp32 adaLN, *and* the `time_embedder`, *and* two
   basis buffers.
3. **Forward** — `Attention.forward` calls `self.qkv_proj(x).split(...)`; `_forward` casts `t_emb` to
   the compute dtype, which would degrade the fp32 basis coordinates to bf16.

And a fourth, outside the core: third-party packs test `hasattr(attn, "qkv_proj")`,
`__class__.__name__ == "MiniMaxH3Model"`, `use_adaln_curves`, or read `adaln_t_table`. A model that
loads but fails those tests is a model half the ecosystem silently ignores.

---

## What the module does

Everything happens at import time, in memory. **No file under `comfy/` is ever modified.**

| layer | mechanism | file |
|---|---|---|
| **Detection** | wraps `detect_unet_config`: hands the upstream function a *view* of the state dict enriched with a `meta` tensor of the fused shape (no memory), then adds `separate_qkv` / `adaln_curve_basis_dim` | `core_shim.py` |
| **Construction** | subclasses `Attention`, `TimeEmbedder`, `MiniMaxH3Model` and installs them into `comfy.ldm.minimax.model`. The `separate_qkv` flag travels through a `ContextVar` so `DiTBlock` / `RefinerBlock` / `TokenRefiner` signatures are untouched. The DT geometry is obtained by calling the upstream constructor with `time_embed_dim = 16` and `adaln_curve_grid = 1` (the core's "table" mode already builds fp32, silu-free adaLN of that width), then removing the dummy table and installing a `TimeEmbedder` subclass that returns basis coordinates. The upstream `_forward` needs to know nothing. | `core_shim.py` |
| **Forward** | recompiles `Attention.forward` and `MiniMaxH3Model._forward` from their source with **one anchor line each** replaced (separate projections; keep `t_emb` fp32). If an anchor is missing the pack warns and degrades gracefully instead of failing. | `core_shim.py` |
| **Vanilla appearance** | substitute classes take the upstream `__name__` / `__qualname__` / `__module__`; the model reports `use_adaln_curves = False`; its `time_embedder` returns the 16 coordinates directly, so a pack that recomputes `t_emb` through it just works | `core_shim.py` |
| **LoRA** | wraps `comfy.sd.load_lora_for_models` (LoraLoader path) and `comfy.lora.load_lora` (direct path): Q/K/V split, adaLN transported into the basis — see [LoRAs](#loras) | `lora_shim.py` |
| **Virtual fused view** | `attn.qkv_proj` exists (read-only weight, correct `in/out_features`) and its forward returns a **storage-less tensor subclass** of the fused shape: aligned `split` / `chunk` / slices hand back the separate projections without any fused buffer; anything else materialises, guarded — see [Third-party packs](#third-party-packs) | `qkv_shim.py` |
| **Guard rail** | at import: required constructor parameters checked (raise), a toy model built on `meta` and inspected (raise), fingerprints of the four upstream methods compared against every verified version (warn) | `core_shim.py` |
| **Patcher tolerance** | wraps `model_patcher.get_key_weight` to return `None` instead of raising when a quantised layer has no plain `.weight` | `core_shim.py` |

If the core is *already* patched with the source patch, the pack detects it (`separate_qkv` in the
`Attention` signature) and runs in **passive mode**: only the LoRA conversion and the wrappers are
installed. You can therefore install the pack first, validate, and remove the patch afterwards.

---

## The int32 wall — why separate Q/K/V matters beyond memory

At 1 MP × 15 s the packed sequence is **115 403 tokens** (117 464 with references). A fused projection
produces a `[S, 3 × 7168]` activation:

```
115 403 × 21 504  =  2 481 626 112  elements   >   2^31 − 1  =  2 147 483 647
```

Any kernel that indexes that buffer with 32-bit offsets — the `q`/`k` views handed to
`rms_rope_split_half_` do exactly that — reads past the end. Symptoms we observed: `CUDA error: illegal
memory access`, or a generation that sits at 0 % forever. With separate projections each activation is
`115 403 × 7168 = 827 M` elements, comfortably below the limit.

**Separate Q/K/V is therefore not just a memory layout: on a fused checkpoint this resolution is
unreachable on our hardware regardless of VRAM.** This is also why `attn.qkv_proj(x)` returns a *virtual*
tensor here rather than a real fused buffer: a pack that splits it in three gets the separate
projections and stays under the wall; a pack that needs the fused activation for real is refused above
2^31 elements with a message naming it, instead of re-creating the wall "for compatibility".

---

## Install

```bat
cd ComfyUI\custom_nodes
git clone https://github.com/aminesoulaymani/ComfyUI-H3-DTsQKV
```

No Python dependencies beyond ComfyUI's own. Requires Python ≥ 3.10 and a ComfyUI install with **source
files present** (the pack reads `comfy/ldm/minimax/model.py` with `inspect.getsource`; a `.pyc`-only
install cannot work).

If your core carries the source patch, remove it after validating — on a git checkout:

```bat
git checkout comfy/ldm/minimax/model.py comfy/model_detection.py comfy/model_patcher.py
```

On a portable build without git, restore the three files from an unmodified ComfyUI of the same version.

Restart ComfyUI and look for one line in the console:

```
[H3-DTsQKV] installed - core 0.35.0, mode vanilla, anchors {'Attention.forward': 'rewritten', 'MiniMaxH3Model._forward': 'rewritten'}, qkv_proj view disabled (default)
```

`mode vanilla` = active substitution on an unmodified core. `mode patched` = passive mode on a patched
core.

---

## Verify

```bat
cd ComfyUI_windows_portable
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-H3-DTsQKV\selftest.py
```

(or double-click `selftest.bat`). **No weights are read and no VRAM is used**: the model is built on the
`meta` device.

| test | what it proves |
|---|---|
| T1 | module installed, detection and LoRA wrappers in place |
| T2 | detection on the real checkpoint header → config; full model built on `meta`; **641 keys compared one by one, shape by shape** against the header; no `qkv_proj` in the state dict; view invisible to `named_modules()` |
| T3 | toy-size CPU: separate-QKV `Attention.forward` == fused `Attention.forward` (max gap measured 0.0); the virtual view: aligned `split`/`chunk`/slices return the projections without materialising, `detach` stays lazy, any other op materialises to exactly the fused tensor, the int32 guard refuses without allocating, `weight` is read-only |
| T4 | toy-size CPU: `time_embedder` output == `(silu(mlp(t)) − mean) @ basis` (gap 0.0); adaLN width 16, fp32, no silu; `_forward` rewritten |
| T5 | LoRA split is exact slice-for-slice on diffusers, kohya and prefix-less key formats; adaLN transport reproduces the vanilla delta (gap 4e-6) and is idempotent |
| T6 | `--lora PATH`: dry-run conversion report of a real LoRA file |

Then generate with a seed whose output you already have. Expected: identical to the source-patch
result, same time per step.

A **status node** (`MiniMax H3/DT-sQKV → module status`) prints the same report inside a workflow.

---

## Updating ComfyUI

This is the reason the pack exists, so here is exactly what happens.

| situation | behaviour |
|---|---|
| a required constructor parameter disappears, `TimeEmbedder` is gone, or the toy model built at import does not have the DT-sQKV structure | **exception at import** → ComfyUI shows `IMPORT FAILED` for this pack with the reason; the checkpoint does not load |
| the anchor line in `Attention.forward` moved | WARNING; the projection goes through the fused view (correct, transient ≈ 3× the QKV activation) |
| the anchor line in `_forward` moved | WARNING; adaLN coordinates are cast to bf16 instead of staying fp32 (slightly reduced fidelity) |
| the four upstream methods no longer match **any** verified version | WARNING per method: run `selftest.py`, then add the version's fingerprints (see [Maintainer notes](#maintainer-notes)) |

`H3_DTSQKV_STRICT=1` turns every WARNING into an error.

**What happened at 0.34.0 → 0.35.0 (2026-09-10).** All four fingerprints changed: PR #15958 added
`gate_compress` / `attn.to_gate_compress`, `v = v.clone()` was removed from `Attention.forward`, and
`_forward` started publishing `minimax_h3_layout` / `block_index`. The pack logged four warnings, found
both anchors intact, rewrote them, and generated. We diffed the four methods, confirmed the changes were
irrelevant to the emulation, and recorded 0.35.0 as verified. That is the intended cycle: **warn, verify,
record.** It will never silently guess.

---

## LoRAs

**Vanilla MiniMax H3 LoRAs are fully supported. Every layer they carry is applied; no attention layer is
silently dropped.**

Why this needs saying: a LoRA trained on a stock H3 addresses the attention projections as
`blocks.N.attn.qkv_proj`. On a DT-sQKV model that key does not exist, so a plain ComfyUI — patched core or
not — cannot map it: it logs `lora key not loaded` once per key and moves on. The LoRA then reaches
`out_proj`, `fc1` and `fc2` but **none of the 52 attention projections**, and the generation looks
"almost right" with no error anywhere. This pack removes that failure mode.

Load any MiniMax H3 LoRA with the stock **LoraLoader** / **LoraLoaderModelOnly**. The pack converts it in
memory before ComfyUI builds the patches and logs one line:

```
[H3-DTsQKV] LoRA adapted to the DT-sQKV model - qkv_proj -> q/k/v: 104 duplicated, 52 split
```

(for a 50-block + 2-refiner LoRA: 52 `lora_A` + 52 `alpha` duplicated, 52 `lora_B` split into three
row thirds; ComfyUI then reports `312 patches attached`.)

| LoRA key | conversion | exact? |
|---|---|---|
| `attn.qkv_proj.{lora_A, lora_down, alpha}` | duplicated verbatim onto `q_proj`, `k_proj`, `v_proj` | yes |
| `attn.qkv_proj.{lora_B, lora_up, diff, diff_b, dora_scale, w_norm, b_norm, set_weight}` | rows split in three | yes — `(B·A)[third] = B[third]·A` |
| `adaln_proj.linear.{lora_A / lora_down}` with input width 2688 | `A' = A @ pinv(basis)ᵀ` (→ width 16), plus a bias delta `scale · B · (A · mean)` added to `diff_b` | exact up to the same rank-16 approximation the checkpoint already makes |
| `adaln_proj.linear.dora_scale` | dropped with a notice | no equivalent in the compact basis |
| LoHa / LoKr / OFT on `qkv_proj` | left untouched, reported as unknown format | — |

Key prefixes handled: `diffusion_model.…`, kohya `lora_unet_…_qkv_proj`, and prefix-less
`blocks.N.…` (DiffSynth / ModelScope, mapped by the core since 0.35.0).

**Caveat that has nothing to do with this pack:** on the int8 layers (`fc1` everywhere, `q/k/v_proj` on
24 blocks of the HQ profile) ComfyUI dequantises, adds the delta, and re-quantises with stochastic
rounding. The LoRA is applied exactly; it is then quantised. The pack makes the LoRA *correct*, not
*stronger*. If a LoRA seems weak, compare strength 0.0 vs 3.0 on the same seed before blaming the
conversion.

---

## Third-party packs

What we observed on our machine. "Works" means the pack's own log confirmed activation; it does not mean
we benchmarked its benefit unless a number is given.

| pack | how it recognises the model | result |
|---|---|---|
| **ComfyUI-sol-attn** — Fused Modulation, Chunk FeedForward, Scheduled Sol Attention | `__class__.__name__ == "MiniMaxH3Model"`; checks `separate_qkv` and uses `q/k/v_proj` | all three active. Sol in `strict` mode ran 8/8 steps with zero dense fallback: **−15 % per step** ([Measurements](#measurements)) |
| **kjnodes ≥ 1.5.1** — MiniMax H3 Memory Efficient Sage Attention | `q, k, v = self.qkv_proj(x).split(inner, dim=-1)` (`ltxv_nodes.py:2088`) | served by the virtual view: the split returns the three separate projections, **no fused buffer**, same memory as the core path. **Verified 2026-09-10** on kjnodes 1.5.1 unmodified, 118 438 tokens, no materialisation. Without this pack (or with the view disabled): `AttributeError: 'Attention' object has no attribute 'qkv_proj'` |
| **kjnodes** — MiniMax LowVRAM Attention | requires `qkv_proj` (otherwise "returning model unchanged"); `qkv = self.qkv_proj(x); del x; q, k, v = qkv.split(...)` | same mechanism as above; its early `del x` works as intended since the projections are already computed |
| **ComfyUI core ≥ 0.35.0** — Sparse Attention (kijai, #16072) | `isinstance(…, MiniMaxH3Model)` (our subclass passes); calls `attn.qkv_proj` in **chunks of 4096 tokens** | virtual view; if it does more than split, each chunk materialises 4096 × 21504 = 88 M elements (176 MB) — far from the wall. **Untested by us** |
| **ComfyUI-Ref2VA-VSA** | `__class__.__name__` check; needs `attn.to_gate_compress` (core ≥ 0.35.0, PR #15958) | on a DT checkpoint the core creates `to_gate_compress = None`; the pack's guard now passes but its forward calls the attribute. **Untested since 0.35.0**; expect a `NoneType` error unless the pack injects its gate. Its `nodes.py` also needs the separate-QKV projection (we patched our local copy) |
| **ComfyUI-VDN-H3** | `blocks[].attn.qkv_proj` guard and `qkv_proj(x).split` in `hybrid.py` | needs a local patch to use `q/k/v_proj` (three small changes; we made them). Loads and runs; no speed benefit observed — it is eager PyTorch by design |
| **comfyui-spectrum-minimax-h3** (upstream) | guard: `adaln_t_table` if `use_adaln_curves` else `time_embedder` (`minimax_h3.py:60`); forecast path `inner.time_embedder(values).to(dtype)` (`:372`); `model_aware` samples `blocks.N.attn.qkv_proj.weight` **from the state dict by name** | **compatible unmodified**: the model reports `use_adaln_curves = False` and its `time_embedder` returns the basis coordinates, so the upstream vanilla path is the right one (Spectrum casts them to bf16 for its own forecast steps; the model keeps fp32). Runs out of the box: the 8-step run went from ~29 to **22 min**, output subjectively a little softer — good for prototyping. `Comfy model compiler graph breaks: 20` appears in the log with it; harmless. `model_aware`'s predictability profile stays degraded on any sQKV checkpoint (string lookup, not attribute) |
| EasyCache | — | `skipped 0/8 steps (1.00x)` on an 8-step turbo LoRA, three runs. Not a pack problem: with so few steps every residual differs |

### The virtual `qkv_proj` view

On by default; disable with `H3_DTSQKV_QKV_VIEW=0` in the environment before starting ComfyUI.

Each `attn` gets a read-only `qkv_proj` object: `hasattr` → `True`, `in_features` / `out_features`
correct, `weight` rebuilt on demand (dequantised for int8 layers). It is *not* a registered submodule
and carries no `comfy_cast_weights`, so the ModelPatcher never sees it.

Its forward computes the three separate projections and returns a **`VirtualFusedQKV`**: a
`torch.Tensor` subclass built with `_make_wrapper_subclass` — no storage, fused shape, correct dtype
and device — whose `__torch_dispatch__` handles:

| operation on the virtual tensor | result |
|---|---|
| `split` / `chunk` in three thirds on the last dim | the three separate, contiguous projections — **no fused buffer is ever built** |
| a slice aligned on a projection boundary (`[..., :inner]`, `[..., inner:2·inner]`, …) | that projection |
| `.shape`, `.dtype`, `.device`, `.numel()` | metadata, no compute |
| `detach`, `alias` | another virtual tensor over the same projections |
| anything else (`.view`, `cat`, arithmetic, `.contiguous()` …) | materialised **one third at a time**, WARNING naming the caller, **refused above 2^31 elements** |

So the "fused path" of kjnodes, LowVRAM Attention or sol-attn costs exactly what the core's separate
path costs. Only a pack that reshapes or indexes the fused activation in a non-aligned way pays for the
buffer — below the limit it works, above it the pack is named and refused (see
[The int32 wall](#the-int32-wall--why-separate-qkv-matters-beyond-memory)).

Known limit: a function compiled with `torch.compile` that manipulates the virtual tensor will
graph-break or fail; none of the packs above compile that path.

---

## Measurements

One machine, one workflow, one checkpoint. Treat these as a data point, not a benchmark.

**Setup.** RTX 5060 Ti 16 GB (PCIe 5.0 ×8), 32 GB RAM, Windows 11, Python 3.13.9, torch 2.9.1+cu130,
comfy-kitchen 0.2.31, comfy-aimdo 0.4.15, pytorch attention, dynamic VRAM (28 474 MB staged),
`MiniMax-H3_Ref2VA-DT-sQKV-INT8-ConvRot-HQ`, text encoder `qwen3vl_32b_minimax_h3_nvfp4_awq`,
`minimax_h3_ref2v_turbo_8step_v1.0_768p` LoRA at 1.0 (312 patches), 1 MP × 15 s → **115 403 tokens**,
8 steps, CFG 1. Chunk FeedForward (4 chunks) and a memory-efficient attention on, except where noted.

| configuration | s / step | 8 steps |
|---|---|---|
| turbo LoRA only — **no** Chunk FeedForward, **no** memory-efficient attention (this pack, core 0.35.0, kjnodes 1.5.1) | **515.7** (step 1) | ≈ 1 h |
| source-patched core 0.34.0 + kjnodes Sage + Chunk FeedForward | 213 – 217 | ≈ 28.7 min |
| **this pack** (vanilla core 0.34.0) + kjnodes Sage + Chunk FeedForward | 213 – 217 | — (identical) |
| this pack + kjnodes Sage + sol-attn **Sol** (`tau 1.30 → 0.80` linear, `strict`) | 176.7 early, **182.8 avg** | **24:22** (−15 %) |
| this pack + kjnodes Sage + **Spectrum** (defaults: blend 0.50, window 2, warm-up 1, max_history 8) | — (skips evaluations; per-step not comparable) | **22:06** wall time including decode; output subjectively a little softer |

Post-sampling (video + audio VAE decode, 192 preview frames): **4:34** — 16 % of wall time, and a
lever with zero quality risk.

**Where the time goes.** With `S = 115 403`, `d = 5 376`, 50 blocks:

```
attention   4·S²·d·L  ≈ 1.4 × 10^16 FLOP
linear      ≈ 0.45 × 10^16 FLOP
total       ≈ 1.9 × 10^16 FLOP per step   →  ≈ 200 s at ~95 TFLOPS bf16 peak
```

Once the activations fit, the card is compute-bound on quadratic attention, which is ~75 % of the work,
and 215 s/step is close to the ~200 s this estimate gives. Weight streaming (28 GB per step over PCIe
5.0 ×8) costs one to two seconds and is not the bottleneck at this token count.

Before the activations fit, the picture is different: without Chunk FeedForward and a memory-efficient
attention, the same run takes **515 s/step** — the `fc1` activation alone is 115 403 × 28 672 × 2 bytes
= 6.6 GB, and on a 16 GB card under Windows the overflow lands in shared system memory, demand-paged over
PCIe. On 16 GB, the two memory nodes are therefore the biggest speed lever there is (2.4×). Once they are
on, further memory-oriented nodes (fused modulation, more chunking) change nothing measurable, and the
remaining levers are fewer tokens, fewer steps, and a sparse-attention kernel that actually engages.

**Loading.** `28474MB Staged`, no `unet missing` / `unet unexpected`, checkpoint header 885 tensors
(641 weights + 122 `weight_scale` + 122 `comfy_quant`).

---

## Limitations

- **Not a converter.** It loads DT-sQKV files; it does not produce them (see next section).
- **adaLN as seen by packs.** The model presents as vanilla with a 16-wide `t_emb`. A pack that
  recomputes `t_emb` through `time_embedder` works unchanged; a pack that hard-codes 2688 fails loudly
  (e.g. merging a vanilla adaLN LoRA outside the LoraLoader path). Intentional.
- **Spectrum `model_aware`** reads `qkv_proj.weight` from the state dict by name; the view does not
  help it.
- **DoRA on adaLN** is dropped; **LoHa / LoKr / OFT** on `qkv_proj` are not split.
- **int8 floor** for LoRA deltas on quantised layers (a property of the checkpoint, not the pack).
- **Source required.** `inspect.getsource` on the core; `.pyc`-only installs cannot work.
- **Tested on Windows only.** Nothing in the code is OS-specific; Linux is expected to work and is
  untested.
- **Beyond 15 s on sm_120**, ComfyUI issue [#15263](https://github.com/Comfy-Org/ComfyUI/issues/15263)
  reports SageAttention FP8 PV kernels producing noise above ~160k tokens. Unrelated to this pack, but it
  is the next wall after the int32 one.
- **One machine, one workflow.** Everything above is n = 1.

---

## Converting other checkpoints to DT-sQKV

Not implemented here, but the recipe is short and needs no training. Both halves are exact or
measurably close:

**sQKV** is a byte-exact row split of `qkv_proj.weight` (and its quantisation scale) into three
tensors; the reverse of `cat`.

**DT** needs a checkpoint that still has its `time_embedder` and full-width `adaln_proj` weights:

```
E[g,:] = silu(time_embedder(t_g))      for G points t_g in [0, 1]      → [G, 2688] fp32
mean   = row mean of E
B      = first 16 right singular vectors of (E − mean)                → [2688, 16]
per block:  W' = W @ B                  [96768, 2688] @ [2688, 16]
            b' = b + W @ mean
```

Exactness: `W·e = W·mean + W·(e − mean) ≈ W·mean + W·B·Bᵀ·(e − mean) = b'-term + W'·c`. The residual
is the energy of `(e − mean)` outside the span of `B` — computable from the time embedder alone in
minutes, **before** downloading anything else. A checkpoint that only has an `adaln_t_table` has lost
its `time_embedder` and cannot be converted to this format.

---

## How this was made

This pack was **written in pair-programming sessions with Claude (Anthropic, model Fable 5.1)**. The
model wrote the code, read the ComfyUI sources, reconstructed the vanilla core by reverse-applying the
source patch, designed the tests and the guard rail, and did the analysis behind this README. The human
ran everything on the hardware, reported the logs, made the decisions, and pushed back when the model
was wrong.

It was wrong several times. Three of those mistakes reached the user's machine and are listed in
[CHANGELOG.md](CHANGELOG.md) under *Fixed during development*: the view was first registered as a
submodule (the patcher tried to load it: `KeyError`), it first returned a *real* fused buffer (kjnodes
Sage took the fused path: 5 GB of concatenation and the int32 wall, generation stuck at 0 % — which is
why it became a virtual tensor), and the substitute classes first had their own names (three packs
rejected the model). Each was found from a log, understood, and fixed the same day. Earlier
analysis errors — misreading the `adaln_t_table` shape, calling the workload bandwidth-bound — were
corrected in the analysis and are stated as such above.

What "verified" means here: `selftest.py` passes on a real checkpoint header; the module's five Python
files were shown, by normalised-AST comparison, to be logic-identical to the version that generated the
measured runs (only comments, strings and two defensive guards differ); and the generated videos matched
the source-patch reference at identical per-step timing. What it does **not** mean: a second human has
read this code. **Read it before you depend on it.** It is short — about 500 lines of logic across five
files — and every point of contact with the core is listed below.

---

## Maintainer notes

Points of contact with the ComfyUI core, in the order they would break:

1. `MiniMaxH3Model.__init__` — subclass. Depends on the parameters
   `time_embed_dim, timestep_input_dim, time_embed_hidden_size, adaln_curve_grid, dtype, device,
   operations` and on the "table" mode building fp32, silu-free adaLN of width `time_embed_dim`.
2. `Attention.__init__` — subclass. Depends on `hidden, heads, head_dim, eps, dtype, device,
   operations` and on `qkv_proj` being safe to delete right after construction.
3. `Attention.forward` and `MiniMaxH3Model._forward` — recompiled from source with one anchor line
   each replaced (`ATTN_ANCHOR`, `TEMB_ANCHOR` in `core_shim.py`). Each anchor must occur exactly once.
4. `model_detection.detect_unet_config` — wrapped; relies on the MiniMax branch reading
   `blocks.0.attn.qkv_proj.weight`'s shape and on `adaln_curve_basis` not being consumed upstream.
5. `model_patcher.get_key_weight` — wrapped, additive.
6. `comfy.sd.load_lora_for_models`, `comfy.lora.load_lora` — wrapped, additive.

**Recording a new verified version.** After a ComfyUI update logs fingerprint warnings, run
`selftest.py`; if it passes, run the status node (or read the warnings) to get the four new fingerprints
and add them to `REFERENCE_FINGERPRINTS` in `core_shim.py` under the new version key; bump
`REFERENCE_VERSION`. Fingerprints are SHA-256 of the method source with all whitespace stripped, first
16 hex digits.

**Testing active mode while the core is still patched.** `selftest.py --reference-dir DIR` loads
`model.py` and `model_detection.py` from `DIR` (an unmodified ComfyUI checkout of the same version) in
place of the installed ones. Upstream core files are GPL-3.0 and are not redistributed in this
repository.

**Before editing anything on a live install**, copy the file to `<name>.bak.<date>-<label>`; `.gitignore`
excludes those.

---

## Credits and license

- **[DmitryDB](https://huggingface.co/DmitryDB)** — the DT-sQKV checkpoints, the DynTime factorisation,
  and the original source patch this pack replaces. Every design decision here mirrors what that patch
  does; the contribution is *how* it is applied, not *what*.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** and **[kijai](https://github.com/kijai)** for
  the MiniMax H3 implementation, PDD LoRA support (#15908), `to_gate_compress` (#15958) and the core
  Sparse Attention node (#16072).
- The authors of **ComfyUI-sol-attn** — whose `separate_qkv` check is the model of how a pack should
  read this checkpoint — and of the other packs named above.
- **[MiniMaxAI](https://huggingface.co/MiniMaxAI/MiniMax-H3)** for the model.

This pack is released under the **GNU General Public License v3.0** (see [LICENSE](LICENSE)), the same
license as ComfyUI, whose methods it recompiles in memory. The checkpoint weights are **not** part of
this repository and are subject to the MiniMax-H3 community license.
