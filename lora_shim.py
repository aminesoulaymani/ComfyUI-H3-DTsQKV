"""
Loading "vanilla" LoRAs (written for an H3 with fused qkv_proj and full-width adaLN) onto a DT-sQKV
model, in memory, at load time - so that every layer of the LoRA is applied and no attention module is
silently dropped as `lora key not loaded`.

1. sQKV: for every `...attn.qkv_proj` module:
     input side  (lora_A / lora_down / alpha)  -> duplicated verbatim onto q_proj, k_proj, v_proj
     output side (lora_B / lora_up / diff / diff_b / dora_scale / w_norm / b_norm / set_weight)
                                                -> split into three row thirds
   Exact: W_qkv = [Wq ; Wk ; Wv] stacked by rows, hence (B A)[third] = B[third] A.

2. DT (adaLN): the model computes y = W' c + b with c = (silu(e) - mean) @ basis, where the vanilla
   model computes y = W silu(e) + b. A vanilla delta dW carries over as
        dW silu(e) = dW mean + dW (silu(e) - mean) ~= dW mean + (dW pinv(basis)^T) c
   hence: lora_A' = lora_A @ pinv(basis)^T   (rank unchanged, alpha unchanged)
          diff_b += scale * B (A mean)       (constant term, scale = alpha/rank)
   The `~=` is the approximation the checkpoint already makes itself (orthonormal basis: pinv = transpose).
"""

import logging
import re

import torch

import comfy.lora
import comfy.sd

log = logging.getLogger("H3-DTsQKV")

_QKV_TOKEN = re.compile(r"(?<![A-Za-z0-9])qkv_proj(?=\.)")
_ADALN_TOKEN = re.compile(r"(?<![A-Za-z0-9])adaln_proj[._]linear(?=\.)")

# first component of the suffix after the module name
DUPLICATE_FIRST = {"lora_A", "lora_down", "alpha"}
SPLIT_FIRST = {"lora_B", "lora_up", "diff", "diff_b", "dora_scale", "w_norm", "b_norm", "set_weight"}


def _first(suffix):
    # ".lora_B.default.weight" -> "lora_B"
    return suffix[1:].split(".", 1)[0]


def is_dt_sqkv_model(diffusion_model):
    if diffusion_model is None:
        return False
    if getattr(diffusion_model, "separate_qkv", False):
        return True
    blocks = getattr(diffusion_model, "blocks", None)
    try:
        return all(hasattr(blocks[0].attn, n) for n in ("q_proj", "k_proj", "v_proj"))
    except Exception:
        return False


def split_qkv_lora(lora):
    """Returns (new dict, stats). Keys without `qkv_proj` are carried over unchanged."""
    out, stats = {}, {"dup": 0, "split": 0, "unknown": [], "not_divisible": []}
    for key, t in lora.items():
        m = _QKV_TOKEN.search(key)
        if m is None:
            out[key] = t
            continue
        pre, suf = key[:m.start()], key[m.end():]
        first = _first(suf)
        targets = [f"{pre}{n}_proj{suf}" for n in ("q", "k", "v")]
        if first in DUPLICATE_FIRST:
            for k2 in targets:
                out[k2] = t
            stats["dup"] += 1
        elif first in SPLIT_FIRST:
            if t.ndim == 0 or t.shape[0] % 3 != 0:
                stats["not_divisible"].append(key)
                out[key] = t
                continue
            third = t.shape[0] // 3
            for i, k2 in enumerate(targets):
                out[k2] = t[i * third:(i + 1) * third].clone().contiguous()
            stats["split"] += 1
        else:
            stats["unknown"].append(key)
            out[key] = t
    return out, stats


def convert_adaln_lora(lora, basis, mean):
    """Carries full-width adaLN modules over to the compact basis, in place. Returns stats."""
    stats = {"converted": 0, "already_compact": 0, "skipped": [], "dora_dropped": 0}
    basis = basis.detach().float().cpu()
    mean = mean.detach().float().cpu()
    full_dim, k = basis.shape
    pinv_t = torch.linalg.pinv(basis).T           # [full_dim, k]

    # group keys by adaLN module
    groups = {}
    for key in list(lora.keys()):
        m = _ADALN_TOKEN.search(key)
        if m is None:
            continue
        base, suf = key[:m.end()], key[m.end():]
        groups.setdefault(base, {})[_first(suf)] = key

    for base, parts in groups.items():
        a_key = parts.get("lora_A") or parts.get("lora_down")
        b_key = parts.get("lora_B") or parts.get("lora_up")
        d_key = parts.get("diff")
        if a_key is None and d_key is None:
            continue
        in_dim = lora[a_key].shape[1] if a_key is not None else lora[d_key].shape[1]
        if in_dim == k:
            stats["already_compact"] += 1
            continue
        if in_dim != full_dim:
            stats["skipped"].append(f"{base} (input {in_dim}, expected {full_dim} or {k})")
            continue

        bias_delta = torch.zeros(0)
        if a_key is not None and b_key is not None:
            A = lora[a_key]
            B = lora[b_key]
            rank = A.shape[0]
            alpha_key = parts.get("alpha")
            scale = (lora[alpha_key].item() / rank) if alpha_key is not None else 1.0
            A32 = A.detach().float().cpu()
            lora[a_key] = (A32 @ pinv_t).to(A.dtype)
            bias_delta = scale * (B.detach().float().cpu() @ (A32 @ mean))
        if d_key is not None:
            D = lora[d_key]
            D32 = D.detach().float().cpu()
            lora[d_key] = (D32 @ pinv_t).to(D.dtype)
            delta = D32 @ mean
            bias_delta = delta if bias_delta.numel() == 0 else bias_delta + delta
        if bias_delta.numel():
            db_key = parts.get("diff_b", base + ".diff_b")
            if db_key in lora:
                lora[db_key] = (lora[db_key].detach().float().cpu() + bias_delta).to(lora[db_key].dtype)
            else:
                lora[db_key] = bias_delta.to(lora[a_key if a_key is not None else d_key].dtype)
        if "dora_scale" in parts:
            # DoRA normalises the columns of the merged weight: no equivalent in the compact basis
            lora.pop(parts["dora_scale"], None)
            stats["dora_dropped"] += 1
        stats["converted"] += 1
    return stats


def convert_lora_for_model(lora, diffusion_model):
    """Full conversion (sQKV + adaLN when the model carries a basis). Returns a new dict."""
    out, qkv_stats = split_qkv_lora(lora)
    msg = [f"qkv_proj -> q/k/v: {qkv_stats['dup']} duplicated, {qkv_stats['split']} split"]
    if qkv_stats["unknown"]:
        msg.append(f"{len(qkv_stats['unknown'])} qkv_proj key(s) of unknown format left untouched "
                   f"(e.g. {qkv_stats['unknown'][0]})")
    if qkv_stats["not_divisible"]:
        msg.append(f"{len(qkv_stats['not_divisible'])} qkv_proj key(s) not divisible by 3 (e.g. {qkv_stats['not_divisible'][0]})")

    basis = getattr(diffusion_model, "adaln_curve_basis", None)
    mean = getattr(diffusion_model, "adaln_curve_mean", None)
    if basis is not None and mean is not None and not basis.is_meta:
        ad = convert_adaln_lora(out, basis, mean)
        if ad["converted"] or ad["already_compact"] or ad["skipped"]:
            msg.append(f"adaLN: {ad['converted']} module(s) carried into the basis, "
                       f"{ad['already_compact']} already compact"
                       + (f", {ad['dora_dropped']} dora_scale dropped" if ad["dora_dropped"] else "")
                       + (f", skipped: {ad['skipped'][:3]}" if ad["skipped"] else ""))
    log.info("[H3-DTsQKV] LoRA adapted to the DT-sQKV model - " + " ; ".join(msg))
    return out


_installed = False


def install():
    global _installed
    if _installed:
        return
    _installed = True

    # Standard path: LoraLoader / LoraLoaderModelOnly -> comfy.sd.load_lora_for_models(model, clip, lora, ...)
    orig_lfm = comfy.sd.load_lora_for_models

    def load_lora_for_models(model, clip, lora, *args, **kwargs):
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        if is_dt_sqkv_model(dm):
            lora = convert_lora_for_model(lora, dm)
        return orig_lfm(model, clip, lora, *args, **kwargs)

    load_lora_for_models.__h3_dtsqkv_wrapped__ = True
    comfy.sd.load_lora_for_models = load_lora_for_models

    # Safety net: packs that call comfy.lora.load_lora directly (no model at hand -> sQKV only).
    orig_ll = comfy.lora.load_lora

    def load_lora(lora, to_load, *args, **kwargs):
        targets = to_load.values() if hasattr(to_load, "values") else ()
        model_is_separate = any(str(v).endswith(".attn.q_proj.weight") for v in targets)
        if model_is_separate and any(_QKV_TOKEN.search(k) for k in lora.keys()):
            lora, stats = split_qkv_lora(lora)
            log.info("[H3-DTsQKV] LoRA (direct path): %d duplicated, %d split", stats["dup"], stats["split"])
        return orig_ll(lora, to_load, *args, **kwargs)

    load_lora.__h3_dtsqkv_wrapped__ = True
    comfy.lora.load_lora = load_lora
