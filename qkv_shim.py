"""
Fused `qkv_proj` view laid over an Attention module with separate projections.

Third-party packs do one of three things: `hasattr(attn, "qkv_proj")`, `attn.qkv_proj(x).split(...)`,
or read `attn.qkv_proj.in_features`. This module answers all three without a single extra weight byte
and - the important part - **without ever building the fused activation** on the common path:

`attn.qkv_proj(x)` returns a `VirtualFusedQKV`, a storage-less `torch.Tensor` subclass that announces the
fused shape `[.., 3*inner]` and holds the three separate projections. A `split` / `chunk` in three thirds
along the last dimension, or a slice aligned on a projection boundary, hands back the separate tensors
directly - exactly what the core's own separate-QKV path produces, same peak memory, same distance from
the int32 wall. Any other operation materialises the fused tensor, one third at a time, and refuses above
2^31 elements with a message naming the calling pack.

No parameter is registered (the model's state_dict and LoRA keys stay those of the separate
projections), and the view is not a submodule (invisible to named_modules() and to the ModelPatcher).
"""

import logging
import os
import sys
import weakref

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

try:
    from comfy.ops import QuantizedTensor
except Exception:  # core without integrated quantisation
    QuantizedTensor = ()

log = logging.getLogger("H3-DTsQKV")
aten = torch.ops.aten

INT32_LIMIT = 2 ** 31 - 1
_NOTED_CALLERS = set()
_WARNED_CALLERS = set()

# The view is ON by default and can be disabled with H3_DTSQKV_QKV_VIEW=0.
# It used to be opt-in when it returned a real fused buffer: packs that pick their path with
# `hasattr(attn, "qkv_proj")` (kjnodes, sol-attn) then paid a 5 GB concatenation and, above ~100k tokens,
# hit the int32 wall. With VirtualFusedQKV the fused path costs exactly what the separate path costs, so
# there is no longer a reason to hide `qkv_proj` from anyone.
VIEW_ENABLED = os.environ.get("H3_DTSQKV_QKV_VIEW", "1") != "0"


def _caller():
    """file:line of the first caller outside this module (cheap: no inspect.stack)."""
    getframe = getattr(sys, "_getframe", None)
    if getframe is None:
        return "?"
    torch_dir = os.sep + "torch" + os.sep
    f = getframe(1)
    # skip this module and torch's own frames (nn.Module.__call__ etc.) to name the calling pack
    while f is not None and (f.f_code.co_filename == __file__ or torch_dir in f.f_code.co_filename):
        f = f.f_back
    if f is None:
        return "?"
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"


def _dense_weight(linear, name):
    w = getattr(linear, "weight", None)
    if w is None:
        raise RuntimeError(f"[H3-DTsQKV] {name}.weight is not loaded yet (lazy initialisation)")
    if QuantizedTensor and isinstance(w, QuantizedTensor):
        w = w.dequantize()
    return w


def check_fused_limit(rows, total, caller):
    """Refuse a fused activation whose element count exceeds int32 indexing."""
    if rows * total > INT32_LIMIT:
        raise RuntimeError(
            f"[H3-DTsQKV] {caller} needs a materialised fused qkv activation of {rows} rows: "
            f"{rows * total:,} elements (> 2^31), beyond the int32 indexing of the attention kernels. "
            f"This pack must consume attn.qkv_proj(x) with .split(inner, dim=-1) (handled without any "
            f"fused buffer), or project through attn.q_proj/k_proj/v_proj, or be disabled for this model.")


class VirtualFusedQKV(torch.Tensor):
    """Storage-less tensor of shape [.., 3*inner] standing for cat(q, k, v) without building it.

    Aligned `split` / `chunk` / slices return the separate projections; everything else materialises
    (guarded). `detach` / `alias` keep the laziness.
    """

    @staticmethod
    def __new__(cls, q, k, v, caller="?"):
        shape = (*q.shape[:-1], q.shape[-1] + k.shape[-1] + v.shape[-1])
        return torch.Tensor._make_wrapper_subclass(cls, shape, dtype=q.dtype, device=q.device, requires_grad=False)

    def __init__(self, q, k, v, caller="?"):
        self._parts = (q, k, v)
        self._bounds = (0, q.shape[-1], q.shape[-1] + k.shape[-1], self.shape[-1])
        self._dense = None
        self._caller = caller

    __torch_function__ = torch._C._disabled_torch_function_impl

    # -- helpers -------------------------------------------------------------------------------
    def _alias(self):
        other = VirtualFusedQKV.__new__(VirtualFusedQKV, *self._parts, self._caller)
        other._parts, other._bounds, other._dense, other._caller = self._parts, self._bounds, self._dense, self._caller
        return other

    def _norm_dim(self, dim):
        return dim % self.ndim

    def _fresh(self, i):
        """A fresh alias of projection i. Tensors returned from __torch_dispatch__ get view metadata
        attached by autograd; returning the same object twice raises "already had autograd metadata"."""
        return self._parts[i].detach()

    def _part_for_range(self, start, end):
        """Index of the projection exactly covering [start, end) on the last dim, or None."""
        b = self._bounds
        for i in range(3):
            if start == b[i] and end == b[i + 1]:
                return i
        return None

    def materialize(self):
        if self._dense is None:
            rows = self.numel() // self.shape[-1] if self.numel() else 0
            check_fused_limit(rows, self.shape[-1], self._caller)
            if self._caller not in _WARNED_CALLERS:
                _WARNED_CALLERS.add(self._caller)
                log.warning("[H3-DTsQKV] %s uses attn.qkv_proj(x) beyond an aligned split: materialising a fused "
                            "buffer of %.2f GB per block. This pack would run leaner with .split(inner, dim=-1) "
                            "or q_proj/k_proj/v_proj.", self._caller, self.numel() * self.element_size() / 1e9)
            q, k, v = self._parts
            out = torch.empty(self.shape, dtype=q.dtype, device=q.device)
            b = self._bounds
            out[..., b[0]:b[1]].copy_(q)
            out[..., b[1]:b[2]].copy_(k)
            out[..., b[2]:b[3]].copy_(v)
            self._dense = out
        return self._dense

    def __repr__(self):
        return f"VirtualFusedQKV(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, materialised={self._dense is not None})"

    # -- dispatch ------------------------------------------------------------------------------
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self = next((a for a in args if isinstance(a, cls)), None)

        if self is not None and self._dense is None:
            if func in (aten.detach.default, aten.alias.default) and len(args) == 1:
                return self._alias()

            if func is aten.split.Tensor and args[0] is self:
                split_size = args[1] if len(args) > 1 else kwargs.get("split_size")
                dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
                b = self._bounds
                if self._norm_dim(dim) == self.ndim - 1 and [b[1] - b[0], b[2] - b[1], b[3] - b[2]] == [split_size] * 3:
                    return [self._fresh(0), self._fresh(1), self._fresh(2)]

            if func is aten.split_with_sizes.default and args[0] is self:
                sizes = list(args[1] if len(args) > 1 else kwargs.get("split_sizes"))
                dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
                b = self._bounds
                if self._norm_dim(dim) == self.ndim - 1 and sizes == [b[1] - b[0], b[2] - b[1], b[3] - b[2]]:
                    return [self._fresh(0), self._fresh(1), self._fresh(2)]

            if func is aten.slice.Tensor and args[0] is self:
                dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
                start = args[2] if len(args) > 2 else kwargs.get("start")
                end = args[3] if len(args) > 3 else kwargs.get("end")
                step = args[4] if len(args) > 4 else kwargs.get("step", 1)
                if self._norm_dim(dim) == self.ndim - 1 and step == 1:
                    last = self.shape[-1]
                    start = 0 if start is None else (start + last if start < 0 else min(start, last))
                    end = last if end is None else (end + last if end < 0 else min(end, last))
                    i = self._part_for_range(start, end)
                    if i is not None:
                        return self._fresh(i)

        # Everything else: materialise every virtual argument and run the real op.
        def unwrap(t):
            return t.materialize() if isinstance(t, cls) else t
        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


class FusedQKVShim(nn.Module):
    """Synthetic `attn.qkv_proj`. Its forward returns a VirtualFusedQKV; `.weight` is read-only.

    Deliberately WITHOUT a `comfy_cast_weights` attribute (nor weight_function/bias_function): the
    ModelPatcher picks up every module carrying one and would try to load `qkv_proj.weight` as a real
    parameter. The view is also not registered as a submodule (see attach_qkv_shim): it is invisible to
    named_modules(), state_dict() and the patcher.
    """

    is_h3_qkv_shim = True

    def __init__(self, attn):
        super().__init__()
        object.__setattr__(self, "_attn_ref", weakref.ref(attn))

    @property
    def _attn(self):
        attn = self._attn_ref()
        if attn is None:
            raise RuntimeError("[H3-DTsQKV] owning Attention was freed")
        return attn

    @property
    def in_features(self):
        return self._attn.q_proj.in_features

    @property
    def out_features(self):
        a = self._attn
        return a.q_proj.out_features + a.k_proj.out_features + a.v_proj.out_features

    @property
    def bias(self):
        return None

    @property
    def weight(self):
        a = self._attn
        return torch.cat([_dense_weight(a.q_proj, "q_proj"),
                          _dense_weight(a.k_proj, "k_proj"),
                          _dense_weight(a.v_proj, "v_proj")], dim=0)

    @weight.setter
    def weight(self, value):
        raise RuntimeError(
            "[H3-DTsQKV] attn.qkv_proj is a read-only view over q_proj/k_proj/v_proj: a delta written "
            "here would be lost. Go through the ModelPatcher (LoRA) or write directly to "
            "attn.q_proj / attn.k_proj / attn.v_proj.")

    @property
    def dtype(self):
        w = getattr(self._attn.q_proj, "weight", None)
        return None if w is None else w.dtype

    def forward(self, x, *args, **kwargs):
        """Three separate projections wrapped in a VirtualFusedQKV: no fused buffer unless the caller
        does something other than an aligned split."""
        a = self._attn
        caller = _caller()
        if caller not in _NOTED_CALLERS:
            _NOTED_CALLERS.add(caller)
            log.info("[H3-DTsQKV] %s uses attn.qkv_proj(x): served by the virtual fused view "
                     "(separate projections, no fused buffer on aligned splits)", caller)
        q = a.q_proj(x, *args, **kwargs)
        k = a.k_proj(x, *args, **kwargs)
        v = a.v_proj(x, *args, **kwargs)
        return VirtualFusedQKV(q, k, v, caller)

    def extra_repr(self):
        try:
            return f"virtual fused view over q/k/v_proj, in={self.in_features}, out={self.out_features}"
        except Exception:
            return "virtual fused view over q/k/v_proj"


def attach_qkv_shim(attn, force=False):
    """Lay the view over `attn` if it has separate projections and no qkv_proj yet.
    Does nothing while the view is disabled (H3_DTSQKV_QKV_VIEW=0), unless force=True."""
    if not (VIEW_ENABLED or force):
        return None
    if hasattr(attn, "qkv_proj"):
        return getattr(attn, "qkv_proj")
    if not all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
        return None
    shim = FusedQKVShim(attn)
    # Plain attribute, NOT through nn.Module.__setattr__: the view must not become a submodule
    # (it would be enumerated by named_modules() and loaded as a weight).
    object.__setattr__(attn, "qkv_proj", shim)
    return shim
