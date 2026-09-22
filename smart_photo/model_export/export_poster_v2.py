"""
Export POSTER V2 → Core ML embedding model (768-d).

Tap point: ViT.se_block output (input to ViT.head), L2-normalized.
Classification head is discarded.

Usage:
  1. pip install -r requirements.txt
  2. python export_poster_v2.py --download   # clones repo + downloads weights
  3. python export_poster_v2.py             # convert + validate

Output: output/PosterV2Embedding.mlpackage
Copy that into smart_photo/SmartPhoto/ then rebuild the iOS app.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
POSTER_DIR = ROOT / "vendor" / "POSTER_V2"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "output"

# Google Drive file IDs from https://github.com/Talented-Q/POSTER_V2
GDRIVE = {
    "ir50": "17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt",
    "mobilefacenet": "1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX",
    "rafdb": "1aVm_hmJyZ5E_0p25XTbm3X9ophsKqCxv",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def download_assets() -> None:
    """Clone POSTER_V2 and download pretrained + RAF-DB checkpoints via gdown."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not POSTER_DIR.exists():
        POSTER_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run([
            "git", "clone", "--depth", "1",
            "https://github.com/Talented-Q/POSTER_V2.git",
            str(POSTER_DIR),
        ])

    try:
        import gdown
    except ImportError as e:
        raise SystemExit("Install gdown: pip install gdown") from e

    targets = {
        "ir50.pth": GDRIVE["ir50"],
        "mobilefacenet_model_best.pth.tar": GDRIVE["mobilefacenet"],
        "rafdb_poster_v2.pth": GDRIVE["rafdb"],
    }
    for name, file_id in targets.items():
        dest = CKPT_DIR / name
        if dest.exists():
            print(f"already have {dest}")
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"downloading {name} ...")
        gdown.download(url, str(dest), quiet=False)
        if not dest.exists():
            raise SystemExit(f"Failed to download {name}. Download manually from POSTER_V2 README.")

    # Place pretrain weights where the patched model expects them
    pretrain = POSTER_DIR / "models" / "pretrain"
    pretrain.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CKPT_DIR / "ir50.pth", pretrain / "ir50.pth")
    shutil.copy2(
        CKPT_DIR / "mobilefacenet_model_best.pth.tar",
        pretrain / "mobilefacenet_model_best.pth.tar",
    )


def patch_poster_paths() -> None:
    """Fix Windows paths and make window ops Core-ML-traceable (no .item())."""
    target = POSTER_DIR / "models" / "PosterV2_7cls.py"
    text = target.read_text()

    if "import os" not in text.splitlines()[:20]:
        text = "import os\n" + text

    text = re.sub(
        r"torch\.load\(\s*r?['\"][^'\"]*mobilefacenet_model_best\.pth\.tar['\"]\s*,\s*"
        r"map_location\s*=\s*[^)]+\)",
        "torch.load(os.path.join(os.path.dirname(__file__), 'pretrain', "
        "'mobilefacenet_model_best.pth.tar'), map_location='cpu')",
        text,
    )
    text = re.sub(
        r"torch\.load\(\s*r?['\"][^'\"]*ir50\.pth['\"]\s*,\s*"
        r"map_location\s*=\s*[^)]+\)",
        "torch.load(os.path.join(os.path.dirname(__file__), 'pretrain', "
        "'ir50.pth'), map_location='cpu')",
        text,
    )

    # Replace .item()-based int casts — these become unsupported aten::Int in Core ML.
    # Safe for fixed 224×224 export (H/W divide evenly by window sizes 28/14/7).
    replacements = [
        (
            "h_w = int(torch.div(H, self.window_size).item())\n"
            "        w_w = int(torch.div(W, self.window_size).item())",
            "h_w = H // self.window_size\n"
            "        w_w = W // self.window_size",
        ),
        (
            "h_w = int(torch.div(H, self.window_size).item())\n"
            "        w_w = int(torch.div(W, self.window_size).item())",
            "h_w = H // self.window_size\n"
            "        w_w = W // self.window_size",
        ),
        (
            "head_dim = int(torch.div(C, self.num_heads).item())\n"
            "        B_dim = int(torch.div(B_, B).item())",
            "head_dim = C // self.num_heads\n"
            "        B_dim = B_ // B",
        ),
        (
            "B = int(windows.shape[0] / (H * W / window_size / window_size))",
            "B = windows.shape[0] // ((H // window_size) * (W // window_size))",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    # Also catch any remaining torch.div(...).item() patterns in this file
    text = re.sub(
        r"int\(torch\.div\(([^,]+),\s*([^)]+)\)\.item\(\)\)",
        r"(\1 // \2)",
        text,
    )

    target.write_text(text)
    print(f"patched {target}")


class PosterV2ExportWrapper(nn.Module):
    """
    Replays pyramid_trans_expr2.forward up through ViT.se_block,
    returning the 768-d gated vector (input to ClassificationHead).

    Spatial sizes are hardcoded for 224×224 export so Core ML never sees
    dynamic aten::Int casts from window / attention shape math.
    """

    # (spatial, channels, num_heads, head_dim) per pyramid level — fixed for 224 input
    _LEVELS = (
        (28, 64, 2, 32),
        (14, 128, 4, 32),
        (7, 256, 8, 32),
    )

    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.m = full_model.module if hasattr(full_model, "module") else full_model

    def _window_tokens(self, win_mod: nn.Module, x: torch.Tensor, spatial: int, channels: int):
        # x: (1, C, H, W) with H=W=spatial
        x = x.permute(0, 2, 3, 1)  # (1, H, W, C)
        x = win_mod.norm(x)
        shortcut = x
        tokens = x.reshape(1, spatial * spatial, channels)
        return tokens, shortcut

    def _attn(self, attn_mod: nn.Module, tokens: torch.Tensor, q_global: torch.Tensor,
              spatial: int, channels: int, num_heads: int, head_dim: int):
        n = spatial * spatial
        kv = attn_mod.qkv(tokens).reshape(1, n, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # q_global is (1, 1, heads, N, head_dim); single window ⇒ B_dim=1
        q = q_global.reshape(1, num_heads, n, head_dim) * attn_mod.scale
        attn = q @ k.transpose(-2, -1)
        bias = attn_mod.relative_position_bias_table[attn_mod.relative_position_index.view(-1)]
        bias = bias.view(n, n, num_heads).permute(2, 0, 1).unsqueeze(0)
        attn = attn_mod.softmax(attn + bias)
        attn = attn_mod.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(1, n, channels)
        return attn_mod.proj_drop(attn_mod.proj(out))

    def _ffn(self, ffn_mod: nn.Module, tokens: torch.Tensor, shortcut: torch.Tensor, spatial: int):
        # tokens: (1, N, C) → restore to (1, H, W, C)
        x = tokens.reshape(1, spatial, spatial, -1)
        x = shortcut + ffn_mod.drop_path(ffn_mod.gamma1 * x)
        x = x + ffn_mod.drop_path(ffn_mod.gamma2 * ffn_mod.mlp(ffn_mod.norm(x)))
        return x.permute(0, 3, 1, 2)  # (1, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from models.PosterV2_7cls import _to_channel_last  # type: ignore

        # Expect [0,1] RGB (Core ML ImageType scale=1/255). Apply ImageNet normalize here
        # so we don't need per-channel ImageType scale (which breaks milprogram).
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std

        m = self.m
        x_face = F.interpolate(x, size=112)
        x_face1, x_face2, x_face3 = m.face_landback(x_face)
        x_face3 = m.last_face_conv(x_face3)
        x_face1, x_face2, x_face3 = (
            _to_channel_last(x_face1),
            _to_channel_last(x_face2),
            _to_channel_last(x_face3),
        )

        qs = []
        faces = (x_face1, x_face2, x_face3)
        for i, (spatial, _ch, heads, head_dim) in enumerate(self._LEVELS):
            n = spatial * spatial
            # _to_query with B hardcoded to 1 — avoids aten::Int from x.shape[0]
            qs.append(faces[i].reshape(1, 1, n, heads, head_dim).permute(0, 1, 3, 2, 4))

        x_ir1, x_ir2, x_ir3 = m.ir_back(x)
        irs = (m.conv1(x_ir1), m.conv2(x_ir2), m.conv3(x_ir3))
        wins = (m.window1, m.window2, m.window3)
        attns = (m.attn1, m.attn2, m.attn3)
        ffns = (m.ffn1, m.ffn2, m.ffn3)

        outs = []
        for i, (spatial, channels, heads, head_dim) in enumerate(self._LEVELS):
            tokens, shortcut = self._window_tokens(wins[i], irs[i], spatial, channels)
            tokens = self._attn(attns[i], tokens, qs[i], spatial, channels, heads, head_dim)
            outs.append(self._ffn(ffns[i], tokens, shortcut, spatial))

        o1 = m.embed_q(outs[0]).flatten(2).transpose(1, 2)
        o2 = m.embed_k(outs[1]).flatten(2).transpose(1, 2)
        o3 = m.embed_v(outs[2])
        o = torch.cat([o1, o2, o3], dim=1)

        # Inline ViT.forward_features with B=1 hardcoded (avoids x.shape[0] → aten::Int)
        vit = m.VIT
        cls = vit.cls_token.expand(1, -1, -1)
        tokens = torch.cat((cls, o), dim=1)
        tokens = vit.pos_drop(tokens + vit.pos_embed)
        tokens = vit.blocks(tokens)
        tokens = vit.norm(tokens)
        emb = vit.pre_logits(tokens[:, 0])
        emb = vit.se_block(emb)
        return F.normalize(emb, dim=-1)


def _patch_vit_attention_static(model: nn.Module) -> None:
    """Replace ViT Attention.forward so reshape uses Python ints (B=1, N=148, C=768)."""
    import types

    def make_attn_forward(num_heads: int):
        def forward(self, x):
            # x: (1, 148, 768) — CLS + 147 tokens for POSTER V2 @ 224
            x_img = x[:, : self.img_chanel, :]
            b, n, c = 1, self.img_chanel, 768
            head_dim = c // num_heads
            qkv = self.qkv(x_img).reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(b, n, c)
            return self.proj_drop(self.proj(out))

        return forward

    vit = model.module.VIT if hasattr(model, "module") else model.VIT
    for block in vit.blocks:
        block.attn.forward = types.MethodType(
            make_attn_forward(block.attn.num_heads), block.attn
        )


def load_poster_model(rafdb_ckpt: Path) -> nn.Module:
    sys.path.insert(0, str(POSTER_DIR))
    # Heavy deps used by PosterV2_7cls import chain
    from models.PosterV2_7cls import pyramid_trans_expr2  # type: ignore

    model = pyramid_trans_expr2(img_size=224, num_classes=7)

    # RAF-DB checkpoint pickles training helpers (RecorderMeter*) into the file.
    # Provide stubs so unpickling succeeds; we only need state_dict.
    class RecorderMeter:  # noqa: N801
        pass

    class RecorderMeter1:  # noqa: N801
        pass

    import __main__
    __main__.RecorderMeter = RecorderMeter
    __main__.RecorderMeter1 = RecorderMeter1

    ckpt = torch.load(rafdb_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Strip DataParallel 'module.' prefix
    cleaned = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"loaded RAF-DB weights  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    _patch_vit_attention_static(model)
    return model


def convert(wrapper: nn.Module, out_path: Path) -> Path:
    import coremltools as ct

    # Trace with [0,1] floats — matches ImageType scale=1/255
    example = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name="input",
                shape=example.shape,
                scale=1.0 / 255.0,
                bias=[0.0, 0.0, 0.0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")
    return out_path


def validate(wrapper: nn.Module, mlpackage: Path, atol: float = 1e-3) -> None:
    """Compare PyTorch vs Core ML on a random 224×224 RGB image."""
    import coremltools as ct
    from PIL import Image

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")

    # PyTorch path: [0,1] then ImageNet inside wrapper
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        pt = wrapper(tensor).numpy().reshape(-1)

    mlmodel = ct.models.MLModel(str(mlpackage))
    cm = mlmodel.predict({"input": pil})
    key = "embedding" if "embedding" in cm else list(cm.keys())[0]
    cml = np.array(cm[key]).reshape(-1)

    cos = float(np.dot(pt, cml) / (np.linalg.norm(pt) * np.linalg.norm(cml) + 1e-8))
    max_abs = float(np.max(np.abs(pt - cml)))
    print(f"validate: cos={cos:.6f}  max_abs={max_abs:.6e}  dim={pt.shape[0]}")
    if cos < 0.995:
        print("WARNING: PyTorch vs Core ML diverge — check preprocess scale/bias")
    else:
        print("OK: Core ML matches PyTorch within tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Clone repo + download weights")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CKPT_DIR / "rafdb_poster_v2.pth",
        help="RAF-DB POSTER V2 .pth",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "PosterV2Embedding.mlpackage",
    )
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_assets()
        patch_poster_paths()
        print("Download done. Re-run without --download to convert.")
        return

    if not POSTER_DIR.exists():
        raise SystemExit("Run with --download first (or clone POSTER_V2 into vendor/).")
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint {args.checkpoint}. Run with --download.")

    patch_poster_paths()
    model = load_poster_model(args.checkpoint)
    wrapper = PosterV2ExportWrapper(model).eval()

    # Smoke-test embedding shape before conversion
    with torch.no_grad():
        emb = wrapper(torch.rand(1, 3, 224, 224))
    assert emb.shape == (1, 768), emb.shape
    print(f"pytorch embedding shape OK: {tuple(emb.shape)}")

    convert(wrapper, args.out)
    if not args.skip_validate:
        validate(wrapper, args.out)

    print(
        "\nNext: copy the .mlpackage into smart_photo/SmartPhoto/ and rebuild.\n"
        f"  cp -R {args.out} ../SmartPhoto/"
    )


if __name__ == "__main__":
    main()
