"""Status node: text report on the module installation (mode, fingerprints, anchors, warnings)."""

from . import core_shim


class H3DTsQKVStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    CATEGORY = "MiniMax H3/DT-sQKV"
    OUTPUT_NODE = True

    def report(self):
        text = core_shim.status_report()
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"H3DTsQKVStatus": H3DTsQKVStatus}
NODE_DISPLAY_NAME_MAPPINGS = {"H3DTsQKVStatus": "H3 DT-sQKV - module status"}
