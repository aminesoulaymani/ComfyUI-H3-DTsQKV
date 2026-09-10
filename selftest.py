"""
Self-test for the H3-DTsQKV module. Run from the ComfyUI root with the embedded Python:

    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI-H3-DTsQKV\\selftest.py [options]

Options:
    --ckpt PATH           DT-sQKV checkpoint to compare against (default: first *sQKV*/*separate-qkv*
                          file in models/diffusion_models)
    --lora PATH           vanilla LoRA to dry-run convert (conversion report, nothing written)
    --reference-dir DIR   maintainer tool: load `model.py` and `model_detection.py` from DIR (a vanilla
                          ComfyUI checkout) in place of the installed core - lets you test the active
                          mode while the installed core still carries the DT-sQKV source patch

No weights are read: the model is built on the `meta` device (zero memory) and compared key by key and
shape by shape against the checkpoint header. Numerical tests run on CPU at toy size.
"""

import argparse
import glob
import importlib.util
import json
import os
import struct
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

RESULTS = []


def load_module_from(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[package_dir] if package_dir else None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def preload_reference(ref):
    for fname in ("model.py", "model_detection.py"):
        if not os.path.isfile(os.path.join(ref, fname)):
            sys.exit(f"--reference-dir: {fname} not found in {ref}")
    import comfy
    import comfy.ldm
    import comfy.ldm.minimax
    m = load_module_from("comfy.ldm.minimax.model", os.path.join(ref, "model.py"))
    comfy.ldm.minimax.model = m
    d = load_module_from("comfy.model_detection", os.path.join(ref, "model_detection.py"))
    comfy.model_detection = d
    print(f"[reference] {ref}: model.py and model_detection.py loaded in place of the installed core")


def run(name, fn):
    print(f"\n=== {name} ===")
    try:
        r = fn()
        status = "SKIP" if r == "skip" else "PASS"
    except Exception:
        traceback.print_exc()
        status = "FAIL"
    RESULTS.append((name, status))
    print(f"--> {status}")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--lora")
    ap.add_argument("--reference-dir")
    args = ap.parse_args()

    if args.reference_dir:
        preload_reference(os.path.abspath(args.reference_dir))

    import torch
    import comfy.ops
    import comfy.memory_management
    import comfy.model_management  # noqa: F401  (initialises the devices, like main.py)

    # real weights (on meta) rather than aimdo lazy init: required to compare shapes
    comfy.memory_management.aimdo_enabled = False

    pkg = load_module_from("h3_dtsqkv_selftest_pkg", os.path.join(HERE, "__init__.py"), HERE)
    core = pkg.core_shim
    lora_shim = pkg.lora_shim
    import comfy.ldm.minimax.model as mm
    import comfy.model_detection as md

    print(f"python {sys.version.split()[0]} - torch {torch.__version__}")
    print(core.status_report())

    DT = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I8": torch.int8, "U8": torch.uint8,
          "I64": torch.int64, "I32": torch.int32, "BOOL": torch.bool,
          "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8), "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8)}
    ops = comfy.ops.disable_weight_init
    cpu = torch.device("cpu")
    state = {}

    # ---------------------------------------------------------------- T1
    def t1_install():
        assert core.STATUS["installed"], "module not installed"
        print("mode:", core.STATUS["mode"], "| anchors:", core.STATUS["anchors"])
        if core.STATUS["warnings"]:
            print("WARNINGS:", *core.STATUS["warnings"], sep="\n  ")
        assert getattr(md.detect_unet_config, "__h3_dtsqkv_wrapped__", False) or core.STATUS["mode"] == "patched", \
            "detect_unet_config not wrapped"
        import comfy.sd
        assert getattr(comfy.sd.load_lora_for_models, "__h3_dtsqkv_wrapped__", False), "load_lora_for_models not wrapped"

    # ---------------------------------------------------------------- T2
    def t2_structure():
        ckpt = args.ckpt
        if ckpt is None:
            cands = [p for pat in ("*sQKV*.safetensors", "*separate-qkv*.safetensors")
                     for p in glob.glob(os.path.join(ROOT, "models", "diffusion_models", pat))]
            if not cands:
                print("no sQKV checkpoint found: pass --ckpt")
                return "skip"
            ckpt = sorted(cands)[0]
        print("checkpoint:", ckpt)
        header = read_header(ckpt)
        sd = {k: torch.empty(v["shape"], dtype=DT.get(v["dtype"], torch.uint8), device="meta") for k, v in header.items()}
        n_i8 = sum(1 for v in header.values() if v["dtype"] == "I8")
        print(f"{len(header)} tensors in the header, {n_i8} in int8")

        cfg = md.detect_unet_config(sd, "", None)
        print("detected config:", {k: v for k, v in cfg.items() if k != "image_model"})
        assert cfg.get("separate_qkv") is True, "separate_qkv not detected"
        assert cfg.get("image_model") == "minimax_h3"
        if "adaln_curve_basis" in sd:
            assert cfg.get("adaln_curve_basis_dim") == sd["adaln_curve_basis"].shape[1], "wrong adaln_curve_basis_dim"
            assert cfg.get("time_embed_dim") == sd["adaln_curve_basis"].shape[0], "wrong time_embed_dim"
        expected_heads = sd["blocks.0.attn.q_proj.weight"].shape[0] // sd["blocks.0.attn.q_norm.weight"].shape[0]
        assert cfg["num_attention_heads"] == expected_heads, f"num_attention_heads {cfg['num_attention_heads']} != {expected_heads}"

        model = mm.MiniMaxH3Model(**cfg, dtype=torch.bfloat16, device=torch.device("meta"), operations=ops)
        msd = model.state_dict()
        quant_suffixes = (".comfy_quant", ".weight_scale", ".weight_scale_2", ".input_scale")
        expected = {k: tuple(v["shape"]) for k, v in header.items() if not k.endswith(quant_suffixes)}
        got = {k: tuple(v.shape) for k, v in msd.items()}
        missing = sorted(set(expected) - set(got))
        unexpected = sorted(set(got) - set(expected))
        mismatched = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
        print(f"expected keys {len(expected)} / model {len(got)} - missing {len(missing)}, unexpected {len(unexpected)}, shape mismatches {len(mismatched)}")
        for label, lst in (("missing", missing), ("unexpected", unexpected), ("shape", mismatched)):
            for k in lst[:8]:
                print(f"   {label}: {k}  expected={expected.get(k)} model={got.get(k)}")
        assert not missing and not unexpected and not mismatched, "structure differs from the checkpoint"
        assert not any("qkv_proj" in k for k in got), "qkv_proj present in the state_dict"

        attn = model.blocks[0].attn
        assert not hasattr(model, "adaln_t_table")
        # the view must never be a submodule (regression 2026-09-09: KeyError 'weight' already exists)
        registered = [n for n, _ in model.named_modules() if n.endswith("qkv_proj")]
        assert not registered, f"qkv_proj registered as a submodule: {registered[:3]}"
        if pkg.qkv_shim.VIEW_ENABLED:
            shim = getattr(attn, "qkv_proj", None)
            assert getattr(shim, "is_h3_qkv_shim", False), "qkv_proj view missing although the view is enabled"
            assert shim.in_features == cfg["hidden_size"], shim.in_features
            assert shim.out_features == 3 * cfg["num_attention_heads"] * cfg["attention_head_dim"], shim.out_features
            assert not hasattr(shim, "comfy_cast_weights"), "the view must not look like a ComfyUI op"
            print(f"qkv_proj virtual view enabled (default): in={shim.in_features} out={shim.out_features}, outside named_modules()")
        else:
            assert not hasattr(attn, "qkv_proj"), "qkv_proj present although the view is disabled"
            print("qkv_proj view disabled (H3_DTSQKV_QKV_VIEW=0): attn has no qkv_proj")
        print(f"adaln_proj.linear.in_features={model.blocks[0].adaln_proj.linear.in_features}")
        state["model_meta"] = model

    # ---------------------------------------------------------------- T3
    def t3_attention_equivalence():
        torch.manual_seed(0)
        hidden, heads, hd = 64, 2, 32
        inner = heads * hd
        fused = mm.Attention(hidden, heads, hd, 1e-5, separate_qkv=False, dtype=torch.float32, device=cpu, operations=ops)
        sep = mm.Attention(hidden, heads, hd, 1e-5, separate_qkv=True, dtype=torch.float32, device=cpu, operations=ops)
        assert hasattr(fused, "qkv_proj") and not hasattr(fused, "q_proj")
        assert all(hasattr(sep, n) for n in ("q_proj", "k_proj", "v_proj"))
        with torch.no_grad():
            for p in fused.parameters():
                torch.nn.init.normal_(p, std=0.05)
            fused.q_norm.weight.fill_(1.0)
            fused.k_norm.weight.fill_(1.0)
            W = fused.qkv_proj.weight
            sep.q_proj.weight.copy_(W[:inner])
            sep.k_proj.weight.copy_(W[inner:2 * inner])
            sep.v_proj.weight.copy_(W[2 * inner:])
            sep.out_proj.weight.copy_(fused.out_proj.weight)
            sep.q_norm.weight.copy_(fused.q_norm.weight)
            sep.k_norm.weight.copy_(fused.k_norm.weight)
            x = torch.randn(7, hidden)
            # virtual fused view: tested on a forced copy when the view is disabled
            if not pkg.qkv_shim.VIEW_ENABLED:
                assert not hasattr(sep, "qkv_proj"), "qkv_proj present although the view is disabled"
                pkg.qkv_shim.attach_qkv_shim(sep, force=True)
            assert torch.equal(sep.qkv_proj.weight, W), "qkv_proj.weight (view) != fused weight"
            assert sep.qkv_proj.in_features == hidden and sep.qkv_proj.out_features == 3 * inner
            ref = fused.qkv_proj(x)
            vq = sep.qkv_proj(x)
            assert isinstance(vq, torch.Tensor) and type(vq).__name__ == "VirtualFusedQKV", type(vq)
            assert vq.shape == ref.shape and vq.dtype == ref.dtype and vq.device == ref.device
            # aligned split / chunk / slices -> the separate projections, nothing materialised
            qs, ks, vs = vq.split(inner, dim=-1)
            assert vq._dense is None, "an aligned split must not materialise the fused buffer"
            assert qs.is_contiguous() and torch.allclose(torch.cat([qs, ks, vs], -1), ref, atol=1e-6)
            c0, c1, c2 = vq.chunk(3, dim=-1)
            assert vq._dense is None and torch.equal(c1, ks), "chunk(3) must reuse the projections"
            assert torch.equal(vq[..., inner:2 * inner], ks) and torch.equal(vq[..., 2 * inner:], vs) and vq._dense is None
            d = vq.detach()
            assert type(d).__name__ == "VirtualFusedQKV" and d._dense is None, "detach must keep the view lazy"
            # anything else materialises (guarded) and matches the real fused tensor
            assert torch.allclose(vq + 0, ref, atol=1e-6) and vq._dense is not None
            assert torch.allclose(vq.view(7, 3, inner)[:, 1], ks, atol=1e-6)
            try:
                pkg.qkv_shim.check_fused_limit(120_000, 3 * 7168, "selftest")
                raise AssertionError("a > 2^31 fused activation should have been refused")
            except RuntimeError as e:
                print("int32 limit refused without allocating:", str(e)[:70], "...")
            try:
                sep.qkv_proj.weight = W.clone()
                raise AssertionError("writing to the view should have been refused")
            except RuntimeError as e:
                print("write refused as expected:", str(e)[:60], "...")
            print("virtual view: split/chunk/slices served from the separate projections, materialisation exact")
            if not pkg.qkv_shim.VIEW_ENABLED:
                del sep.__dict__["qkv_proj"]
                print("view disabled: tested on a forced copy, then removed")
            # full forward (no-rope path: no kitchen kernel)
            a = fused(x.clone(), rope_freqs=None)
            b = sep(x.clone(), rope_freqs=None)
        diff = (a - b).abs().max().item()
        print(f"max forward gap fused vs separate: {diff:.3e}")
        assert diff < 1e-5, "separate forward != fused forward"
        if core.STATUS["mode"] == "vanilla":
            assert getattr(type(sep).forward, "__h3_dtsqkv_rewritten__", False) == (
                core.STATUS["anchors"].get("Attention.forward") == "rewritten"), "Attention.forward anchor state inconsistent"

    # ---------------------------------------------------------------- T4
    def t4_time_embedder():
        torch.manual_seed(1)
        model = mm.MiniMaxH3Model(**core.TINY_CONFIG, dtype=torch.float32, device=cpu, operations=ops)
        assert hasattr(model, "adaln_curve_basis") and hasattr(model, "adaln_curve_mean")
        assert not hasattr(model, "adaln_t_table")
        assert model.blocks[0].adaln_proj.linear.in_features == core.TINY_CONFIG["adaln_curve_basis_dim"]
        assert model.blocks[0].adaln_proj.apply_silu is False
        assert model.blocks[0].adaln_proj.linear.weight.dtype == torch.float32, "adaLN must stay fp32"
        with torch.no_grad():
            Q, _ = torch.linalg.qr(torch.randn(24, 4))
            model.adaln_curve_basis.copy_(Q)
            model.adaln_curve_mean.normal_()
            for p in model.time_embedder.parameters():
                torch.nn.init.normal_(p, std=0.1)
            t = torch.rand(3)
            if core.STATUS["mode"] != "vanilla":
                print("patched core: the upstream time_embedder is raw, projection happens in _forward (not tested here)")
                return
            out = model.time_embedder(t)
            full = torch.nn.functional.silu(core.ORIGINAL["TimeEmbedder"].forward(model.time_embedder, t))
            ref = (full - model.adaln_curve_mean) @ model.adaln_curve_basis
        assert out.shape == (3, 4), out.shape
        assert out.dtype == torch.float32
        diff = (out - ref).abs().max().item()
        print(f"DT time_embedder vs reference formula: max gap {diff:.3e}")
        assert diff < 1e-6
        assert model.use_adaln_curves is False and model.use_adaln_basis is False, "the model must present itself as vanilla"
        rewritten = getattr(type(model)._forward, "__h3_dtsqkv_rewritten__", False)
        assert rewritten == (core.STATUS["anchors"].get("MiniMaxH3Model._forward") == "rewritten")
        print("_forward rewritten:", rewritten)

    # ---------------------------------------------------------------- T5
    def t5_lora_synthetic():
        torch.manual_seed(2)
        hidden, inner, r = 32, 64, 4
        A, B = torch.randn(r, hidden), torch.randn(3 * inner, r)
        db = torch.randn(3 * inner)
        lora = {
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": A,
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": B,
            "diffusion_model.blocks.0.attn.qkv_proj.alpha": torch.tensor(2.0),
            "diffusion_model.blocks.0.attn.qkv_proj.diff_b": db,
            "lora_unet_blocks_1_attn_qkv_proj.lora_down.weight": A,
            "lora_unet_blocks_1_attn_qkv_proj.lora_up.weight": B,
            "blocks.2.attn.qkv_proj.lora_A.weight": A,          # DiffSynth / ModelScope style (no prefix)
            "blocks.2.attn.qkv_proj.lora_B.weight": B,
            "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": A,
        }
        out, stats = lora_shim.split_qkv_lora(lora)
        assert not any("qkv_proj" in k for k in out), "qkv_proj remains"
        assert stats["dup"] == 4 and stats["split"] == 4 and not stats["unknown"], stats
        full = B @ A
        for i, n in enumerate("qkv"):
            Bn = out[f"diffusion_model.blocks.0.attn.{n}_proj.lora_B.weight"]
            An = out[f"diffusion_model.blocks.0.attn.{n}_proj.lora_A.weight"]
            assert torch.equal(Bn @ An, full[i * inner:(i + 1) * inner]), f"{n}_proj: delta != slice"
            assert torch.equal(out[f"diffusion_model.blocks.0.attn.{n}_proj.diff_b"], db[i * inner:(i + 1) * inner])
            assert f"lora_unet_blocks_1_attn_{n}_proj.lora_up.weight" in out
            assert f"blocks.2.attn.{n}_proj.lora_B.weight" in out
        assert "diffusion_model.blocks.0.attn.out_proj.lora_A.weight" in out
        print("sQKV: exact slices; diffusers, kohya and prefix-less key formats; diff_b split")

        # adaLN: vanilla delta == carried delta, for inputs inside the span of the basis
        full_dim, k, outd, r = 24, 4, 48, 3
        Q, _ = torch.linalg.qr(torch.randn(full_dim, k))
        mean = torch.randn(full_dim)
        A, B, alpha = torch.randn(r, full_dim), torch.randn(outd, r), 1.5
        base = "diffusion_model.blocks.0.adaln_proj.linear"
        lora = {f"{base}.lora_A.weight": A.clone(), f"{base}.lora_B.weight": B.clone(), f"{base}.alpha": torch.tensor(alpha),
                "lora_unet_final_layer_adaln_proj_linear.lora_down.weight": A.clone(),
                "lora_unet_final_layer_adaln_proj_linear.lora_up.weight": B.clone()}
        st = lora_shim.convert_adaln_lora(lora, Q, mean)
        assert st["converted"] == 2 and not st["skipped"], st
        A2, db2 = lora[f"{base}.lora_A.weight"], lora[f"{base}.diff_b"]
        assert A2.shape == (r, k) and db2.shape == (outd,)
        scale = alpha / r
        z = torch.randn(5, k)
        d = mean + z @ Q.T                       # silu(e) inside the span: the case where the model itself is exact
        vanilla = scale * d @ (B @ A).T
        c = (d - mean) @ Q
        ours = scale * c @ (B @ A2).T + db2
        diff = (vanilla - ours).abs().max().item()
        print(f"adaLN: vanilla vs carried delta gap {diff:.3e} (kohya converted too: "
              f"{'lora_unet_final_layer_adaln_proj_linear.diff_b' in lora})")
        assert diff < 1e-4
        # idempotence: a second pass touches nothing
        st2 = lora_shim.convert_adaln_lora(lora, Q, mean)
        assert st2["converted"] == 0 and st2["already_compact"] == 2, st2

    # ---------------------------------------------------------------- T6
    def t6_lora_file():
        if not args.lora:
            print("pass --lora PATH to dry-run convert a real LoRA")
            return "skip"
        from safetensors.torch import load_file
        lora = load_file(args.lora, device="cpu")
        n_qkv = sum(1 for k in lora if "qkv_proj" in k)
        n_adaln = sum(1 for k in lora if "adaln_proj" in k)
        print(f"{len(lora)} keys, {n_qkv} on qkv_proj, {n_adaln} on adaln_proj")
        model = state.get("model_meta")
        if model is None:
            out, st = lora_shim.split_qkv_lora(lora)
            print("split:", st["dup"], "dup,", st["split"], "split, unknown", len(st["unknown"]))
        else:
            out = lora_shim.convert_lora_for_model(lora, model)   # basis on meta -> adaLN skipped, reported in the log
        assert not any("qkv_proj" in k for k in out), "qkv_proj remains"
        print(f"{len(out)} keys after conversion (nothing written)")

    run("T1 installation and wrappers", t1_install)
    run("T2 detection + structure against the checkpoint header (meta)", t2_structure)
    run("T3 fused / separate Attention equivalence (CPU)", t3_attention_equivalence)
    run("T4 DT time_embedder and adaLN geometry (CPU)", t4_time_embedder)
    run("T5 synthetic LoRA conversion (sQKV + adaLN)", t5_lora_synthetic)
    run("T6 dry-run conversion of a real LoRA", t6_lora_file)

    print("\n================ SUMMARY ================")
    for name, status in RESULTS:
        print(f"{status:5s} {name}")
    if core.STATUS["warnings"]:
        print(f"{len(core.STATUS['warnings'])} module warning(s) - see above")
    failed = [n for n, s in RESULTS if s == "FAIL"]
    print("FAILED" if failed else "OK")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
