"""
Export FECNet-style model → Core ML embedding (16-d).

Architecture (from Approach C slide):
  InceptionResnetV1 (facenet-pytorch, VGGFace2, frozen)
    → tap block8 output (1792-d feature map, pre-pool)
    → DenseNet-BC head (growth_rate=64, 1 block × 5 layers)
    → 16-d L2-normalized embedding

IMPORTANT — licensing / weights:
  - Backbone (facenet-pytorch) is MIT and ships pretrained.
  - Original AmirSh15/FECNet head weights have no LICENSE — not used here.
  - The DenseNet-BC head is randomly initialized. Embeddings are NOT meaningful
    until you train the head on Google's FEC dataset with triplet loss.
  - This script still produces a valid .mlpackage so the iOS plumbing works;
    train later and re-export with --head-weights.

Preprocess: (x - 127.5) / 128.0   (FaceNet convention, 160×160 input)

Usage:
  pip install -r requirements.txt
  python export_fecnet.py
  python export_fecnet.py --head-weights path/to/trained_head.pth   # after training

Output: output/FECNetEmbedding.mlpackage
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, n_layers: int):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(n_layers):
            layers.append(_DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FECHead(nn.Module):
    """DenseNet-BC: growth_rate=64, 1 block of 5 layers → 16-d."""

    def __init__(
        self,
        in_channels: int = 1792,
        growth_rate: int = 64,
        n_layers: int = 5,
        out_dim: int = 16,
    ):
        super().__init__()
        self.dense = _DenseBlock(in_channels, growth_rate, n_layers)
        final_ch = in_channels + n_layers * growth_rate
        self.bn = nn.BatchNorm2d(final_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_ch, out_dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        x = self.dense(feat_map)
        x = F.relu(self.bn(x), inplace=True)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return F.normalize(x, dim=-1)


class FECNetExportWrapper(nn.Module):
    """
    Backbone through block8 (no pool / no last_linear), then DenseNet-BC head.
    forward() returns the 16-d embedding directly — no hooks.
    """

    def __init__(self, backbone: nn.Module, head: FECHead):
        super().__init__()
        self.backbone = backbone
        self.head = head
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward_block8(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv2d_1a(x)
        x = b.conv2d_2a(x)
        x = b.conv2d_2b(x)
        x = b.maxpool_3a(x)
        x = b.conv2d_3b(x)
        x = b.conv2d_4a(x)
        x = b.conv2d_4b(x)
        x = b.repeat_1(x)
        x = b.mixed_6a(x)
        x = b.repeat_2(x)
        x = b.mixed_7a(x)
        x = b.repeat_3(x)
        x = b.block8(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_block8(x))


def build_model(head_weights: Path | None) -> FECNetExportWrapper:
    from facenet_pytorch import InceptionResnetV1

    backbone = InceptionResnetV1(pretrained="vggface2").eval()
    head = FECHead()
    if head_weights is not None:
        state = torch.load(head_weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {
            (k[5:] if k.startswith("head.") else k): v for k, v in state.items()
        }
        head.load_state_dict(cleaned, strict=True)
        print(f"loaded head weights from {head_weights}")
    else:
        print(
            "WARNING: head is randomly initialized — "
            "embeddings are placeholders until you train on FEC."
        )

    return FECNetExportWrapper(backbone, head).eval()


def convert(wrapper: nn.Module, out_path: Path) -> Path:
    import coremltools as ct

    example = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    # FaceNet preprocess: (x - 127.5) / 128 with x in [0, 255]
    scale = 1.0 / 128.0
    bias = [-127.5 / 128.0] * 3

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name="input",
                shape=example.shape,
                scale=scale,
                bias=bias,
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


def validate(wrapper: nn.Module, mlpackage: Path) -> None:
    import coremltools as ct
    from PIL import Image

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")

    arr = (rgb.astype(np.float32) - 127.5) / 128.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        pt = wrapper(tensor).numpy().reshape(-1)

    mlmodel = ct.models.MLModel(str(mlpackage))
    cm = mlmodel.predict({"input": pil})
    key = "embedding" if "embedding" in cm else list(cm.keys())[0]
    cml = np.array(cm[key]).reshape(-1)

    cos = float(np.dot(pt, cml) / (np.linalg.norm(pt) * np.linalg.norm(cml) + 1e-8))
    max_abs = float(np.max(np.abs(pt - cml)))
    print(f"validate: cos={cos:.6f}  max_abs={max_abs:.6e}  dim={pt.shape[0]}")
    if cos < 0.999:
        print("WARNING: PyTorch vs Core ML diverge — check preprocess")
    else:
        print("OK: Core ML matches PyTorch within tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-weights", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "FECNetEmbedding.mlpackage")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    wrapper = build_model(args.head_weights)

    with torch.no_grad():
        emb = wrapper(torch.rand(1, 3, 160, 160))
    assert emb.shape == (1, 16), emb.shape
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
