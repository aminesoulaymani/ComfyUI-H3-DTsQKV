"""
ComfyUI-H3-DTsQKV - native support for MiniMax H3 "DT-sQKV" checkpoints (separate Q/K/V projections
+ rank-k factorised adaLN) without patching the ComfyUI core; vanilla LoRAs loaded as-is; optional
fused `qkv_proj` view for third-party packs.

Installs at import. If the core has changed to the point where the emulation is impossible, the import
raises: ComfyUI shows "IMPORT FAILED" for this pack with the reason, and the DT-sQKV checkpoint does not
load - rather than a silently wrong model.
"""

from . import core_shim

core_shim.install()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
