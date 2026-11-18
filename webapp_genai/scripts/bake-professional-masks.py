"""Bake cleaned person masks for the Professional look so the browser only composites."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent / "public" / "scenes" / "professional"

PAIRS = [
    ("coat-navy.jpg", "person-navy.png", "person-navy-clean.png"),
    ("coat-charcoal.jpg", "person-charcoal.png", "person-charcoal-clean.png"),
    ("coat-camel.jpg", "person-camel.png", "person-camel-clean.png"),
]


def clean(photo_path: Path, mask_path: Path) -> Image.Image:
    photo = np.asarray(Image.open(photo_path).convert("RGB")).astype(np.int16)
    h, w = photo.shape[:2]
    mask = np.asarray(Image.open(mask_path).convert("L").resize((w, h), Image.BILINEAR)) >= 128
    r, g, b = photo[..., 0], photo[..., 1], photo[..., 2]
    luma = (r + g + b) / 3

    foliage = (g > r + 16) & (g > b + 10) & (g > 55) & (r < 170)
    mask &= ~foliage

    ys = np.where(mask.any(axis=1))[0]
    if ys.size:
        top, bottom = ys[0], ys[-1]
        cap = np.zeros_like(mask)
        cap[top : top + int((bottom - top) * 0.32)] = True
        chroma = photo.max(axis=2) - photo.min(axis=2)
        hair = (luma < 95) & (chroma < 22)
        skin = (r > g) & (g > b) & (r > 115) & (r - b > 18)
        mask &= ~(cap & ~(hair | skin))

    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)

    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=1)
    out = Image.fromarray((mask * 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(1.0))


def grabcut_original(photo_path: Path) -> Image.Image:
    """The old GrabCut mask kept only half the denim jacket; re-cut with seeds on both panels and the tee."""
    import cv2

    rgb = np.asarray(Image.open(photo_path).convert("RGB"))
    h, w = rgb.shape[:2]
    sx, sy = w / 660, h / 847
    box = lambda x0, y0, x1, y1: (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
    seed = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    x0, y0, x1, y1 = box(70, 95, 595, 847)
    seed[y0:y1, x0:x1] = cv2.GC_PR_FGD
    for x0, y0, x1, y1 in [box(0, 0, w / sx, 90), box(0, 0, 60, 847), box(605, 0, w / sx, 847), box(0, 0, 175, 335), box(440, 0, w / sx, 340), box(0, 430, 100, 560), box(0, 560, 84, 660), box(185, 335, 226, 386)]:
        seed[y0:y1, x0:x1] = cv2.GC_BGD
    for x0, y0, x1, y1 in [
        box(240, 150, 370, 190),
        box(250, 200, 360, 320),
        box(270, 320, 350, 380),
        box(235, 420, 355, 840),
        box(125, 470, 230, 800),
        box(395, 470, 540, 800),
        box(210, 380, 420, 440),
    ]:
        seed[y0:y1, x0:x1] = cv2.GC_FGD
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, seed, None, bg_model, fg_model, 8, cv2.GC_INIT_WITH_MASK)
    mask = (seed == cv2.GC_FGD) | (seed == cv2.GC_PR_FGD)

    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_erosion(mask, iterations=1)
    return Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))


def clip_body(mask: Image.Image, guides: list[Image.Image], neck_frac: float = 0.4) -> Image.Image:
    """Suit renders share one pose; clip stray background below the neck to the other suits' outline."""
    m = np.asarray(mask).astype(np.float32)
    guide = np.zeros(m.shape, bool)
    for g in guides:
        guide |= np.asarray(g.resize(mask.size)) >= 128
    guide = ndimage.binary_dilation(guide, iterations=10)
    neck = int(m.shape[0] * neck_frac)
    m[neck:][~guide[neck:]] = 0
    return Image.fromarray(m.astype(np.uint8))


def main() -> None:
    baked = {"person-original-clean.png": grabcut_original(ROOT / "original.jpg")}
    for photo, mask, target in PAIRS:
        baked[target] = clean(ROOT / photo, ROOT / mask)
    suits = ["person-navy-clean.png", "person-charcoal-clean.png", "person-camel-clean.png"]
    for target in suits:
        others = [baked[name] for name in suits if name != target]
        baked[target] = clip_body(baked[target], others)
    for target, image in baked.items():
        image.save(ROOT / target, optimize=True)
        print("wrote", target)


if __name__ == "__main__":
    main()
