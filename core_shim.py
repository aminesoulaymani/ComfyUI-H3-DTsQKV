"""
Native support for MiniMax H3 "DT-sQKV" checkpoints without patching the ComfyUI core.

Two non-standard properties are handled:

* sQKV  - separate attention projections (q_proj / k_proj / v_proj) instead of the fused qkv_proj.
          Byte-exact reversible, but we NEVER re-fuse: three separate GEMMs keep every product below
          the int32 indexing limit (2^31 elements) on long sequences.
* DT    - rank-k factorised adaLN (adaln_curve_basis [T, k] + adaln_curve_mean [T]); the fp32 time
          MLP is kept and its output is projected onto the basis. The full-width adaLN weights do NOT
          exist in the file: this is irreversible, it can only be emulated.

Strategy: class substitution (subclasses installed into the upstream module) plus a targeted rewrite of
two anchor lines inside upstream methods. No source file is ever edited.

Guard rail: anything that would break model construction raises at import time (the pack refuses to
load and says why); anything that only degrades performance or fidelity logs an explicit WARNING and
continues. H3_DTSQKV_STRICT=1 turns warnings into errors.
"""

import contextvars
import hashlib
import inspect
import logging
import os
import textwrap
import weakref

import torch
import torch.nn as nn

import comfy.ldm.minimax.model as mm
import comfy.model_detection as md
import comfy.model_management
import comfy.model_patcher as mp
import comfy.ops
import comfy.utils

from .qkv_shim import attach_qkv_shim, VIEW_ENABLED

log = logging.getLogger("H3-DTsQKV")

STRICT = os.environ.get("H3_DTSQKV_STRICT", "0") == "1"

# Latest core version verified with selftest.py.
REFERENCE_VERSION = "0.35.0"

# Whitespace-insensitive fingerprints of the upstream methods we depend on, per verified version.
# Purely informative: a mismatch against EVERY known version means "run selftest.py"; it never
# blocks by itself.
#   0.34.0 -> 0.35.0: PR #15958 adds `gate_compress` / `attn.to_gate_compress` (Attention.__init__
#   and MiniMaxH3Model.__init__), drops `v = v.clone()` (Attention.forward), and publishes
#   `minimax_h3_layout` / `block_index` in transformer_options (_forward). No impact on the
#   DT-sQKV emulation: both anchors stay unique and the required parameters are intact.
REFERENCE_FINGERPRINTS = {
    "0.34.0": {
        "Attention.__init__": "cbd50af0a7437b9f",
        "Attention.forward": "f5f03243dc85d715",
        "MiniMaxH3Model.__init__": "f05142133b0ead66",
        "MiniMaxH3Model._forward": "71c7e0061ff27078",
    },
    "0.35.0": {
        "Attention.__init__": "9e08064f37029f06",
        "Attention.forward": "4e8d9171a59f3ec0",
        "MiniMaxH3Model.__init__": "72ac658153413467",
        "MiniMaxH3Model._forward": "6d5340027f586d0d",
    },
}
METHODS = tuple(REFERENCE_FINGERPRINTS[REFERENCE_VERSION])

# Anchor lines rewritten inside upstream methods (compared after strip()).
ATTN_ANCHOR = "q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)"
ATTN_REPLACEMENT = (
    "q, k, v = (self.q_proj(x), self.k_proj(x), self.v_proj(x)) if getattr(self, 'separate_qkv', False) "
    "else self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)",
)
TEMB_ANCHOR = "t_emb = self.time_embedder(t_vals).to(dtype)"
TEMB_REPLACEMENT = (
    "t_emb = self.time_embedder(t_vals)",
    # the DT time MLP already returns fp32 basis coordinates: do not degrade them to bf16
    "t_emb = t_emb if getattr(self, 'h3_dt_basis', False) else t_emb.to(dtype)",
)

# Upstream constructor parameters without which the DT emulation is impossible.
REQUIRED_MODEL_PARAMS = ("time_embed_dim", "timestep_input_dim", "time_embed_hidden_size",
                         "adaln_curve_grid", "dtype", "device", "operations")
REQUIRED_ATTN_PARAMS = ("hidden", "heads", "head_dim", "eps", "dtype", "device", "operations")

STATUS = {
    "installed": False,
    "mode": None,             # "vanilla" (active substitution) or "patched" (core already patched: passive mode)
    "core_version": None,
    "matched_version": None,  # verified version whose fingerprints all match, or None
    "fingerprints": {},
    "anchors": {},
    "warnings": [],
}

# Flag carried from MiniMaxH3Model.__init__ down to the Attention modules built underneath it, without
# touching the intermediate core signatures (DiTBlock, RefinerBlock, TokenRefiner).
_SEPARATE_QKV = contextvars.ContextVar("h3_dtsqkv_separate_qkv", default=False)

# Original upstream classes, captured at install time (useful to tests and third-party packs).
ORIGINAL = {}


def _warn(msg):
    STATUS["warnings"].append(msg)
    if STRICT:
        raise RuntimeError("[H3-DTsQKV] (strict) " + msg)
    log.warning("[H3-DTsQKV] %s", msg)


def _fail(msg):
    raise RuntimeError("[H3-DTsQKV] " + msg)


def _normalized_source(fn):
    src = inspect.getsource(fn)
    return "\n".join(line.strip() for line in src.splitlines() if line.strip())


def fingerprint(fn):
    return hashlib.sha256(_normalized_source(fn).encode("utf-8")).hexdigest()[:16]


def _core_version():
    try:
        import comfyui_version
        return getattr(comfyui_version, "__version__", "?")
    except Exception:
        return "?"


def core_mode():
    """'patched' if the core already carries the DT-sQKV source patch (Attention signature), else 'vanilla'."""
    params = inspect.signature(mm.Attention.__init__).parameters
    return "patched" if "separate_qkv" in params else "vanilla"


def _rewrite_method(cls, name, anchor, replacement_lines, namespace):
    """Recompile `cls.name` with the `anchor` line replaced. Returns the function, or None when the
    anchor is missing or ambiguous (the upstream source changed at that spot)."""
    fn = getattr(cls, name)
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    lines = src.splitlines()
    hits = [i for i, line in enumerate(lines) if line.strip() == anchor]
    if len(hits) != 1:
        return None
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i:i + 1] = [indent + r for r in replacement_lines]
    code = compile("\n".join(lines) + "\n", f"<H3-DTsQKV rewrite {cls.__name__}.{name}>", "exec")
    local_ns = {}
    exec(code, namespace, local_ns)   # globals = the upstream module: every name it uses still resolves
    new_fn = local_ns[name]
    new_fn.__h3_dtsqkv_rewritten__ = True
    new_fn.__qualname__ = f"{cls.__name__}.{name}"
    return new_fn


def _check_signature(fn, required, label):
    params = inspect.signature(fn).parameters
    missing = [p for p in required if p not in params]
    if missing:
        _fail(f"{label} no longer has the parameters {missing} (core {STATUS['core_version']}). "
              f"The DT-sQKV emulation cannot be installed on this version: "
              f"resynchronise core_shim.py or fall back to the source patch.")


# --------------------------------------------------------------------------------------------
# Substitute classes (built at install time so they capture the upstream classes)
# --------------------------------------------------------------------------------------------

def _build_classes():
    OrigAttention = mm.Attention
    OrigModel = mm.MiniMaxH3Model
    OrigTimeEmbedder = mm.TimeEmbedder
    attn_sig = inspect.signature(OrigAttention.__init__)
    model_sig = inspect.signature(OrigModel.__init__)

    class DTAttention(OrigAttention):
        """Upstream Attention, with separate q_proj/k_proj/v_proj when the model asks for it."""

        def __init__(self, *args, separate_qkv=None, **kwargs):
            super().__init__(*args, **kwargs)
            sep = _SEPARATE_QKV.get() if separate_qkv is None else separate_qkv
            self.separate_qkv = bool(sep)
            if not self.separate_qkv:
                return
            bound = attn_sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            inner = a["heads"] * a["head_dim"]
            # The upstream qkv_proj is empty (aimdo lazy init) or uninitialised: removing it costs nothing.
            del self.qkv_proj
            lin = a["operations"].Linear
            self.q_proj = lin(a["hidden"], inner, bias=False, dtype=a["dtype"], device=a["device"])
            self.k_proj = lin(a["hidden"], inner, bias=False, dtype=a["dtype"], device=a["device"])
            self.v_proj = lin(a["hidden"], inner, bias=False, dtype=a["dtype"], device=a["device"])
            # Read-only fused view (opt-in): third-party packs find `attn.qkv_proj` again.
            attach_qkv_shim(self)

    class DTTimeEmbedder(OrigTimeEmbedder):
        """Upstream time MLP (fp32) whose output is projected onto the compact adaLN basis:
        t_emb = (silu(mlp(t)) - mean) @ basis. The buffers live on the model (checkpoint keys)."""

        def __init__(self, *args, owner=None, **kwargs):
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "_h3_owner", weakref.ref(owner))

        def forward(self, t):
            full = nn.functional.silu(super().forward(t))
            owner = self._h3_owner()
            if owner is None:
                raise RuntimeError("[H3-DTsQKV] owning model was freed before its time_embedder")
            basis = comfy.model_management.cast_to(owner.adaln_curve_basis, device=full.device)
            mean = comfy.model_management.cast_to(owner.adaln_curve_mean, device=full.device)
            return (full - mean) @ basis

    class DTMiniMaxH3Model(OrigModel):
        """Upstream model, accepted as-is; in DT-sQKV mode it is built so that the upstream _forward
        needs to know nothing: standard `time_embedder` path, width-k adaLN in fp32."""

        def __init__(self, *args, adaln_curve_basis_dim=None, separate_qkv=False, **kwargs):
            bound = model_sig.bind_partial(self, *args, **kwargs)
            bound.apply_defaults()
            kw = dict(bound.arguments)
            kw.pop("self")
            kw.update(kw.pop("kwargs", None) or {})

            basis_dim = adaln_curve_basis_dim
            if basis_dim is not None and kw.get("adaln_curve_grid") is not None:
                raise ValueError("[H3-DTsQKV] adaln_t_table and adaln_curve_basis are mutually exclusive")
            time_embed_dim = kw["time_embed_dim"]
            timestep_input_dim = kw["timestep_input_dim"]
            time_embed_hidden_size = kw["time_embed_hidden_size"]
            if basis_dim is not None:
                # Trick: the adaLN width of the blocks follows time_embed_dim, and the upstream "table"
                # mode already builds adaLN without silu and in fp32. This yields exactly the DT geometry
                # without touching DiTBlock/FinalLayer; the dummy table is removed afterwards.
                kw["time_embed_dim"] = basis_dim
                kw["adaln_curve_grid"] = 1

            token = _SEPARATE_QKV.set(bool(separate_qkv))
            try:
                super().__init__(**kw)
            finally:
                _SEPARATE_QKV.reset(token)

            self.separate_qkv = bool(separate_qkv)
            self.h3_dt_basis = basis_dim is not None
            # Explicit flags for packs that read them (Spectrum): the model presents itself as vanilla,
            # its time_embedder returns the basis coordinates directly.
            self.use_adaln_basis = False
            self.use_adaln_table = bool(kw.get("adaln_curve_grid")) and basis_dim is None
            if not self.h3_dt_basis:
                return

            del self.adaln_t_table
            self.use_adaln_curves = False
            self.time_embedder = DTTimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,
                                                dtype=torch.float32, device=kw["device"],
                                                operations=kw["operations"], owner=self)
            self.register_buffer("adaln_curve_basis", torch.empty(time_embed_dim, basis_dim, dtype=torch.float32))
            self.register_buffer("adaln_curve_mean", torch.empty(time_embed_dim, dtype=torch.float32))

    # Third-party packs often test `__class__.__name__ == "MiniMaxH3Model"` (sol-attn, Ref2VA-VSA,
    # Spectrum): the substitutes take the identity of the upstream classes so they stay
    # indistinguishable from a vanilla model.
    for sub, orig in ((DTAttention, OrigAttention), (DTTimeEmbedder, OrigTimeEmbedder), (DTMiniMaxH3Model, OrigModel)):
        sub.__name__ = orig.__name__
        sub.__qualname__ = orig.__qualname__
        sub.__module__ = orig.__module__
        sub.__h3_dtsqkv_substitute__ = True

    return DTAttention, DTTimeEmbedder, DTMiniMaxH3Model


# --------------------------------------------------------------------------------------------
# Checkpoint detection
# --------------------------------------------------------------------------------------------

def _wrap_detection():
    orig = md.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, metadata=None):
        keys = state_dict.keys()
        fused_key = f"{key_prefix}blocks.0.attn.qkv_proj.weight"
        sep_keys = [f"{key_prefix}blocks.0.attn.{n}_proj.weight" for n in ("q", "k", "v")]
        is_h3 = (f"{key_prefix}video_patch_proj.weight" in keys and f"{key_prefix}audio_patch_proj.weight" in keys)
        if not (is_h3 and fused_key not in keys and all(k in keys for k in sep_keys)):
            return orig(state_dict, key_prefix, metadata)

        q, k, v = (state_dict[s] for s in sep_keys)
        if not (q.shape == k.shape == v.shape):
            raise ValueError("[H3-DTsQKV] the separate Q/K/V projections do not share one shape")
        # Upstream detection reads the shape of qkv_proj: hand it a view of the state dict enriched
        # with a meta tensor of the fused shape (no memory), leaving the real state dict untouched.
        try:
            view = dict(state_dict)
        except TypeError:  # mapping-like object without the dict protocol
            view = {name: state_dict[name] for name in keys}
        view[fused_key] = torch.empty((q.shape[0] * 3, q.shape[1]), dtype=q.dtype, device="meta")
        cfg = orig(view, key_prefix, metadata)
        cfg["separate_qkv"] = True

        basis_key = f"{key_prefix}adaln_curve_basis"
        if basis_key in keys:
            shape = state_dict[basis_key].shape
            if shape[0] != cfg.get("time_embed_dim"):
                raise ValueError("[H3-DTsQKV] adaln_curve_basis does not match the time_embedder output width")
            cfg["adaln_curve_basis_dim"] = int(shape[1])
        return cfg

    detect_unet_config.__h3_dtsqkv_wrapped__ = True
    md.detect_unet_config = detect_unet_config


# --------------------------------------------------------------------------------------------
# model_patcher.get_key_weight: tolerate quantised layers without a plain .weight attribute
# --------------------------------------------------------------------------------------------

def _wrap_get_key_weight():
    orig = mp.get_key_weight
    if getattr(orig, "__h3_dtsqkv_wrapped__", False):
        return

    def get_key_weight(model, key):
        try:
            return orig(model, key)
        except AttributeError:
            op_keys = key.rsplit(".", 1)
            if len(op_keys) < 2:
                raise
            op = comfy.utils.get_attr(model, op_keys[0])
            return (None,
                    getattr(op, "set_{}".format(op_keys[1]), None),
                    getattr(op, "convert_{}".format(op_keys[1]), None))

    get_key_weight.__h3_dtsqkv_wrapped__ = True
    mp.get_key_weight = get_key_weight


# --------------------------------------------------------------------------------------------
# Structural self-check (meta device: zero memory)
# --------------------------------------------------------------------------------------------

TINY_CONFIG = dict(hidden_size=64, num_layers=1, token_refiner_num_layers=1, num_attention_heads=2,
                   attention_head_dim=32, ffn_hidden_size=64, latents_dim=24, audio_latents_dim=32,
                   text_dim=16, timestep_input_dim=8, time_embed_hidden_size=16, time_embed_dim=24,
                   rope_inv_freq_len=16, adaln_curve_basis_dim=4, separate_qkv=True)


def _self_check(DTTimeEmbedder):
    model = mm.MiniMaxH3Model(**TINY_CONFIG, dtype=torch.bfloat16, device=torch.device("meta"),
                              operations=comfy.ops.disable_weight_init)
    attn = model.blocks[0].attn
    problems = []
    if not all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
        problems.append("q_proj/k_proj/v_proj missing")
    has_view = getattr(getattr(attn, "qkv_proj", None), "is_h3_qkv_shim", False)
    if VIEW_ENABLED and not has_view:
        problems.append("qkv_proj virtual view missing although the view is enabled")
    if not VIEW_ENABLED and hasattr(attn, "qkv_proj"):
        problems.append("qkv_proj present although the view is disabled (H3_DTSQKV_QKV_VIEW=0)")
    if any(n.endswith("qkv_proj") for n, _ in model.named_modules()):
        problems.append("the qkv_proj view is registered as a submodule (the patcher would load it as a weight)")
    if hasattr(getattr(attn, "qkv_proj", None), "comfy_cast_weights"):
        problems.append("the qkv_proj view carries comfy_cast_weights (the patcher would pick it up)")
    if hasattr(model, "adaln_t_table"):
        problems.append("dummy adaln_t_table not removed")
    if tuple(model.adaln_curve_basis.shape) != (24, 4):
        problems.append(f"adaln_curve_basis {tuple(model.adaln_curve_basis.shape)} != (24, 4)")
    lin = model.blocks[0].adaln_proj.linear
    if getattr(lin, "in_features", None) != 4:
        problems.append(f"adaln_proj.linear.in_features = {getattr(lin, 'in_features', None)} != 4")
    if getattr(model.blocks[0].adaln_proj, "apply_silu", None) is not False:
        problems.append("adaln_proj.apply_silu should be False")
    if not isinstance(model.time_embedder, DTTimeEmbedder):
        problems.append("time_embedder is not the DTTimeEmbedder")
    if model.use_adaln_curves:
        problems.append("use_adaln_curves should be False")
    if type(model).__name__ != "MiniMaxH3Model" or type(attn).__name__ != "Attention":
        problems.append(f"non-vanilla class names: {type(model).__name__} / {type(attn).__name__}")
    # (not via state_dict(): under aimdo lazy init the weights are not Parameters yet)
    ref_attn = model.token_refiner.blocks[0].attn
    if not all(hasattr(ref_attn, n) for n in ("q_proj", "k_proj", "v_proj")):
        problems.append("token_refiner without q_proj/k_proj/v_proj")
    if getattr(ref_attn.q_proj, "out_features", None) != 64:
        problems.append(f"token_refiner q_proj.out_features = {getattr(ref_attn.q_proj, 'out_features', None)} != 64")
    if problems:
        _fail("structural self-check failed: " + " ; ".join(problems))


# --------------------------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------------------------

def _install_active():
    _check_signature(mm.MiniMaxH3Model.__init__, REQUIRED_MODEL_PARAMS, "MiniMaxH3Model.__init__")
    _check_signature(mm.Attention.__init__, REQUIRED_ATTN_PARAMS, "Attention.__init__")
    if not hasattr(mm, "TimeEmbedder"):
        _fail("comfy.ldm.minimax.model.TimeEmbedder not found")

    ORIGINAL["Attention"] = mm.Attention
    ORIGINAL["MiniMaxH3Model"] = mm.MiniMaxH3Model
    ORIGINAL["TimeEmbedder"] = mm.TimeEmbedder
    DTAttention, DTTimeEmbedder, DTMiniMaxH3Model = _build_classes()

    fn = _rewrite_method(ORIGINAL["Attention"], "forward", ATTN_ANCHOR, ATTN_REPLACEMENT, mm.__dict__)
    if fn is not None:
        DTAttention.forward = fn
        STATUS["anchors"]["Attention.forward"] = "rewritten"
    else:
        STATUS["anchors"]["Attention.forward"] = "anchor missing -> fused view"
        _warn("anchor missing in Attention.forward: the QKV projection will go through the fused view "
              "(transient concatenation, ~3x the QKV activation memory). The model stays correct.")

    fn = _rewrite_method(ORIGINAL["MiniMaxH3Model"], "_forward", TEMB_ANCHOR, TEMB_REPLACEMENT, mm.__dict__)
    if fn is not None:
        DTMiniMaxH3Model._forward = fn
        STATUS["anchors"]["MiniMaxH3Model._forward"] = "rewritten"
    else:
        STATUS["anchors"]["MiniMaxH3Model._forward"] = "anchor missing -> adaLN in bf16"
        _warn("anchor missing in MiniMaxH3Model._forward: the adaLN coordinates will be cast to bf16 "
              "instead of staying fp32 (slightly reduced fidelity). The model stays correct.")

    mm.Attention = DTAttention
    mm.MiniMaxH3Model = DTMiniMaxH3Model
    _wrap_detection()
    _self_check(DTTimeEmbedder)


def _install_passive():
    """Core already patched (modified sources): only add what the patch does not provide."""
    orig_init = mm.Attention.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if getattr(self, "separate_qkv", False) and not hasattr(self, "qkv_proj"):
            attach_qkv_shim(self)

    __init__.__h3_dtsqkv_wrapped__ = True
    mm.Attention.__init__ = __init__
    STATUS["anchors"]["Attention.forward"] = "core patched (not rewritten)"
    STATUS["anchors"]["MiniMaxH3Model._forward"] = "core patched (not rewritten)"


def install():
    if STATUS["installed"]:
        return STATUS
    STATUS["core_version"] = _core_version()
    STATUS["mode"] = core_mode()

    for name in METHODS:
        cls_name, meth = name.split(".")
        try:
            fp = fingerprint(getattr(getattr(mm, cls_name), meth))
        except Exception as e:  # source unavailable (pyc only, etc.)
            fp = f"unavailable ({type(e).__name__})"
        STATUS["fingerprints"][name] = fp

    matched = [v for v, fps in REFERENCE_FINGERPRINTS.items()
               if all(STATUS["fingerprints"].get(k) == f for k, f in fps.items())]
    STATUS["matched_version"] = matched[-1] if matched else None
    if STATUS["mode"] == "vanilla" and not matched:
        closest = REFERENCE_FINGERPRINTS.get(STATUS["core_version"], REFERENCE_FINGERPRINTS[REFERENCE_VERSION])
        for name, fp in STATUS["fingerprints"].items():
            if fp != closest.get(name):
                _warn(f"{name} changed (fingerprint {fp}): no verified version "
                      f"({', '.join(REFERENCE_FINGERPRINTS)}) matches. The module continues on its "
                      f"anchors; run selftest.py to confirm.")

    if STATUS["mode"] == "vanilla":
        _install_active()
    else:
        _install_passive()
    _wrap_get_key_weight()

    from . import lora_shim
    lora_shim.install()

    STATUS["installed"] = True
    STATUS["qkv_view"] = VIEW_ENABLED
    log.info("[H3-DTsQKV] installed - core %s, mode %s, anchors %s, qkv_proj view %s%s",
             STATUS["core_version"], STATUS["mode"], STATUS["anchors"],
             "virtual (default)" if VIEW_ENABLED else "disabled (H3_DTSQKV_QKV_VIEW=0)",
             f", {len(STATUS['warnings'])} warning(s)" if STATUS["warnings"] else "")
    return STATUS


def status_report():
    lines = [f"H3-DTsQKV - ComfyUI core {STATUS['core_version']} "
             f"(verified versions: {', '.join(REFERENCE_FINGERPRINTS)})",
             f"mode: {STATUS['mode']}",
             f"fingerprints: {'identical to ' + STATUS['matched_version'] if STATUS['matched_version'] else 'NO verified version matches'}",
             f"qkv_proj view: {'virtual (default) - aligned splits return the separate projections, no fused buffer' if VIEW_ENABLED else 'disabled (H3_DTSQKV_QKV_VIEW=0)'}"]
    for name, fp in STATUS["fingerprints"].items():
        known = [v for v, fps in REFERENCE_FINGERPRINTS.items() if fps.get(name) == fp]
        lines.append(f"  {name:28s} {fp}  {'= ' + ', '.join(known) if known else '!= every known version'}")
    for name, state in STATUS["anchors"].items():
        lines.append(f"  anchor {name:22s} {state}")
    if STATUS["warnings"]:
        lines.append("warnings:")
        lines += ["  - " + w for w in STATUS["warnings"]]
    else:
        lines.append("no warnings")
    return "\n".join(lines)
