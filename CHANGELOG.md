# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-10

First public release. Verified against ComfyUI **0.34.0** and **0.35.0** on one machine
(RTX 5060 Ti 16 GB, Windows 11, Python 3.13.9, torch 2.9.1+cu130).

### Added
- Class substitution for `Attention` / `MiniMaxH3Model` / `TimeEmbedder` in `comfy.ldm.minimax.model`,
  building DT-sQKV checkpoints without any core source patch.
- Anchor rewrite of `Attention.forward` (separate projections) and `MiniMaxH3Model._forward`
  (fp32 adaLN coordinates), with graceful degradation when an anchor is missing.
- `detect_unet_config` wrapper: recognises separate Q/K/V and `adaln_curve_basis`.
- `get_key_weight` wrapper: tolerates quantised layers without a plain `.weight`.
- In-memory LoRA conversion at load time (`comfy.sd.load_lora_for_models` and `comfy.lora.load_lora`):
  Q/K/V split, adaLN transport into the compact basis including the bias term, DoRA on adaLN dropped
  with a notice. Handles diffusers, kohya and prefix-less (DiffSynth / ModelScope) key formats.
- Virtual fused `qkv_proj` view (on by default, `H3_DTSQKV_QKV_VIEW=0` to disable): `attn.qkv_proj(x)`
  returns a storage-less `torch.Tensor` subclass of the fused shape; aligned `split` / `chunk` / slices
  return the separate projections with no fused buffer; anything else materialises a third at a time,
  refusing above 2^31 elements and naming the calling pack. Makes kjnodes 1.5.1 (`qkv_proj(x).split`)
  work unmodified.
- Guard rail: structural self-check on the `meta` device at import (raises), per-version fingerprints of
  the four upstream methods (warns), `H3_DTSQKV_STRICT=1`.
- `selftest.py` (6 tests, no weights read, CPU toy-size numerics) and a status node.

### Fixed during development (kept for the record)
- The fused view was first registered as a submodule and carried `comfy_cast_weights`: the ModelPatcher
  tried to load `qkv_proj.weight` as a parameter (`KeyError: attribute 'weight' already exists`).
- The fused view first returned a *real* fused buffer: packs that select their path with
  `hasattr(attn, "qkv_proj")` (kjnodes Sage, sol-attn) switched to the fused path, paid a 5 GB
  concatenation and hit the int32 wall at 115k tokens (generation stuck at 0 %). It was made opt-in, then
  replaced by the virtual tensor, which makes the fused path cost the same as the separate one.
- Substitute classes now take the upstream `__name__` / `__qualname__` / `__module__`: three packs
  test `__class__.__name__ == "MiniMaxH3Model"`.
- The virtual tensor first returned the *same* projection objects on every aligned split; a second
  `split`/`chunk` on the same virtual tensor raised "already had autograd metadata" (caught by selftest
  T3, never reached at run time where each forward splits once). It now returns fresh aliases.
