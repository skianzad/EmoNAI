#!/usr/bin/env python3
"""Approach 5 — changed-pixel mask → island segmentation → purity metrics.

Perceptual change map (local SSIM), not raw RGB subtraction, then connected-
component labeling. Run against existing before/after pairs from run_edits:

    python island_pipeline.py --run latest
    python island_pipeline.py --run latest --conditions 8 --expected-islands 2

Writes island overlays + purity JSON under outputs/<run>/islands/.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
INPUT_DIR = ROOT / "inputs"

try:
    from skimage.metrics import structural_similarity
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_opening, binary_closing, disk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "island_pipeline requires scikit-image.\n"
        "  pip install scikit-image\n"
        f"Original error: {exc}"
    ) from exc

from layout import discover_tool_bundles, write_json
from run_edits import COMPARISON_TOOLS, CONDITIONS, DISPLAY_NAME, find_source


# ---------------------------------------------------------------------------
# Diff → mask → islands
# ---------------------------------------------------------------------------


@dataclass
class IslandStats:
    island_id: int
    area_px: int
    bbox: tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    centroid: tuple[float, float]
    mean_change: float
    fraction_of_mask: float


@dataclass
class DiffResult:
    width: int
    height: int
    change_fraction: float
    island_count: int
    islands: list[IslandStats]
    # arrays kept for optional overlay writes (not serialized)
    change_mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    ssim_map: np.ndarray | None = None


def _to_gray_float(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    rgb = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    # Rec. 709 luminance
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def perceptual_change_mask(
    original: Image.Image,
    edited: Image.Image,
    *,
    win_size: int = 11,
    ssim_threshold: float = 0.85,
    min_island_area: int = 64,
    morph_radius: int = 2,
) -> DiffResult:
    """Build a binary change mask from a local SSIM map, then label islands."""
    w = min(original.width, edited.width)
    h = min(original.height, edited.height)
    # Cap working resolution for speed; masks scale conceptually with pixels.
    max_side = 1024
    scale = min(1.0, max_side / max(w, h))
    size = (max(1, int(w * scale)), max(1, int(h * scale)))

    a = _to_gray_float(original, size)
    b = _to_gray_float(edited, size)

    # Odd window required; clamp to image size.
    ws = min(win_size, min(size) if min(size) % 2 == 1 else min(size) - 1)
    ws = max(3, ws if ws % 2 == 1 else ws - 1)

    score, ssim_map = structural_similarity(
        a, b, win_size=ws, data_range=1.0, full=True
    )
    del score

    # Low SSIM → changed. Invert to change intensity in [0, 1].
    change = np.clip(1.0 - ssim_map.astype(np.float32), 0.0, 1.0)
    binary = change >= (1.0 - ssim_threshold)

    if morph_radius > 0:
        selem = disk(morph_radius)
        binary = binary_opening(binary, selem)
        binary = binary_closing(binary, selem)

    labels = label(binary, connectivity=2)
    props = regionprops(labels, intensity_image=change)

    kept: list[IslandStats] = []
    remapped = np.zeros_like(labels)
    next_id = 1
    total_changed = int(binary.sum())
    for prop in props:
        if prop.area < min_island_area:
            continue
        min_r, min_c, max_r, max_c = prop.bbox
        remapped[labels == prop.label] = next_id
        kept.append(
            IslandStats(
                island_id=next_id,
                area_px=int(prop.area),
                bbox=(int(min_r), int(min_c), int(max_r), int(max_c)),
                centroid=(float(prop.centroid[0]), float(prop.centroid[1])),
                mean_change=float(prop.mean_intensity or 0.0),
                fraction_of_mask=(
                    float(prop.area) / total_changed if total_changed else 0.0
                ),
            )
        )
        next_id += 1

    # Drop tiny components from the binary mask for reporting.
    clean_mask = remapped > 0
    return DiffResult(
        width=size[0],
        height=size[1],
        change_fraction=float(clean_mask.mean()),
        island_count=len(kept),
        islands=kept,
        change_mask=clean_mask,
        labels=remapped,
        ssim_map=ssim_map.astype(np.float32),
    )


def composite_with_disabled(
    original: Image.Image,
    edited: Image.Image,
    labels: np.ndarray,
    disabled_ids: set[int],
) -> Image.Image:
    """Final = original, with edited pixels only where enabled islands remain."""
    size = (labels.shape[1], labels.shape[0])
    orig = np.asarray(original.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    edit = np.asarray(edited.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    out = edit.copy()
    for iid in disabled_ids:
        region = labels == iid
        out[region] = orig[region]
    return Image.fromarray(out)


def overlay_islands(
    base: Image.Image,
    labels: np.ndarray,
    *,
    alpha: float = 0.45,
) -> Image.Image:
    """Tint each island with a distinct hue over the edited image."""
    size = (labels.shape[1], labels.shape[0])
    rgb = np.asarray(base.convert("RGB").resize(size, Image.Resampling.LANCZOS)).astype(
        np.float32
    )
    n = int(labels.max())
    if n == 0:
        return Image.fromarray(rgb.astype(np.uint8))

    out = rgb.copy()
    for iid in range(1, n + 1):
        hue = (iid * 0.61803398875) % 1.0
        color = np.array(_hsv_to_rgb(hue, 0.85, 1.0), dtype=np.float32) * 255.0
        region = labels == iid
        out[region] = (1.0 - alpha) * out[region] + alpha * color
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s)
    i %= 6
    return [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][i]


# ---------------------------------------------------------------------------
# Purity scoring helpers
# ---------------------------------------------------------------------------


@dataclass
class PurityReport:
    tool: str
    condition: int
    condition_slug: str
    source: str
    edited: str
    island_count: int
    expected_islands: int | None
    over_segmentation: int | None  # max(0, actual - expected)
    under_segmentation: int | None  # max(0, expected - actual)
    change_fraction: float
    islands: list[dict[str, Any]]


def resolve_run_dir(run: str) -> Path:
    if run == "latest":
        runs = sorted(
            (p for p in OUTPUT_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not runs:
            raise SystemExit(f"No runs under {OUTPUT_DIR}")
        return runs[0]
    path = Path(run)
    if not path.is_absolute():
        path = OUTPUT_DIR / run
    if not path.is_dir():
        raise SystemExit(f"Run directory not found: {path}")
    return path


def find_edited(bundle_dir: Path) -> Path | None:
    for name in ("after.png", "after.jpg", "after.jpeg", "result.png"):
        p = bundle_dir / name
        if p.exists():
            return p
    # Legacy flat layouts
    pngs = sorted(bundle_dir.glob("*.png"))
    return pngs[0] if pngs else None


def evaluate_pair(
    tool: str,
    condition: int,
    slug: str,
    source_path: Path,
    edited_path: Path,
    *,
    expected_islands: int | None,
    ssim_threshold: float,
    min_island_area: int,
) -> tuple[PurityReport, DiffResult]:
    original = Image.open(source_path)
    edited = Image.open(edited_path)
    diff = perceptual_change_mask(
        original,
        edited,
        ssim_threshold=ssim_threshold,
        min_island_area=min_island_area,
    )
    over = under = None
    if expected_islands is not None:
        over = max(0, diff.island_count - expected_islands)
        under = max(0, expected_islands - diff.island_count)

    report = PurityReport(
        tool=tool,
        condition=condition,
        condition_slug=slug,
        source=str(source_path),
        edited=str(edited_path),
        island_count=diff.island_count,
        expected_islands=expected_islands,
        over_segmentation=over,
        under_segmentation=under,
        change_fraction=diff.change_fraction,
        islands=[asdict(i) for i in diff.islands],
    )
    return report, diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest", help="Run id under outputs/ or 'latest'")
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated condition numbers (default: all present in run)",
    )
    parser.add_argument(
        "--tools",
        default=",".join(COMPARISON_TOOLS),
        help="Comma-separated tool ids",
    )
    parser.add_argument(
        "--expected-islands",
        type=int,
        default=None,
        help="Ground-truth island count for purity (e.g. 5 for multi-object edit)",
    )
    parser.add_argument("--ssim-threshold", type=float, default=0.85)
    parser.add_argument("--min-island-area", type=int, default=64)
    parser.add_argument(
        "--write-overlays",
        action="store_true",
        default=True,
        help="Write tinted island overlays (default on)",
    )
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    out_dir = run_dir / "islands"
    out_dir.mkdir(parents=True, exist_ok=True)

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    cond_filter: set[int] | None = None
    if args.conditions.strip():
        cond_filter = {int(x) for x in args.conditions.split(",") if x.strip()}

    reports: list[PurityReport] = []
    write_overlays = args.write_overlays and not args.no_overlays

    all_bundles = discover_tool_bundles(run_dir)
    bundles_by_cond: dict[str, list] = {}
    for b in all_bundles:
        bundles_by_cond.setdefault(b.condition_dir.name, []).append(b)

    for condition in CONDITIONS:
        if cond_filter is not None and condition.number not in cond_filter:
            continue
        try:
            source = find_source(INPUT_DIR, condition)
        except Exception:
            source = None
        if source is None or not Path(source).exists():
            # Fall back to copied source inside the run folder
            cond_dir = run_dir / condition.folder
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = cond_dir / f"source{ext}"
                if candidate.exists():
                    source = candidate
                    break
        if source is None:
            continue

        cond_bundles = {
            b.variant: b
            for b in bundles_by_cond.get(condition.folder, [])
            if b.variant in tools
        }

        for tool in tools:
            bundle = cond_bundles.get(tool)
            if bundle is None:
                continue
            edited = bundle.after if bundle.after.exists() else find_edited(bundle.tool_dir)
            if edited is None or not Path(edited).exists():
                continue

            report, diff = evaluate_pair(
                tool,
                condition.number,
                condition.slug,
                Path(source),
                Path(edited),
                expected_islands=args.expected_islands,
                ssim_threshold=args.ssim_threshold,
                min_island_area=args.min_island_area,
            )
            reports.append(report)

            stem = f"{condition.number:02d}_{condition.slug}__{tool}"
            write_json(out_dir / f"{stem}.json", asdict(report))

            if write_overlays and diff.labels is not None:
                edited_img = Image.open(edited)
                overlay = overlay_islands(edited_img, diff.labels)
                overlay.save(out_dir / f"{stem}_overlay.png")
                if diff.change_mask is not None:
                    mask_img = Image.fromarray(
                        (diff.change_mask.astype(np.uint8) * 255)
                    )
                    mask_img.save(out_dir / f"{stem}_mask.png")

    # Cross-tool summary table
    summary_rows: list[dict[str, Any]] = []
    by_cond: dict[int, list[PurityReport]] = {}
    for r in reports:
        by_cond.setdefault(r.condition, []).append(r)

    for cond_num, group in sorted(by_cond.items()):
        for r in group:
            summary_rows.append(
                {
                    "condition": r.condition,
                    "slug": r.condition_slug,
                    "tool": r.tool,
                    "display": DISPLAY_NAME.get(r.tool, r.tool),
                    "island_count": r.island_count,
                    "expected": r.expected_islands,
                    "over_seg": r.over_segmentation,
                    "under_seg": r.under_segmentation,
                    "change_fraction": round(r.change_fraction, 5),
                }
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": str(run_dir),
        "ssim_threshold": args.ssim_threshold,
        "min_island_area": args.min_island_area,
        "expected_islands": args.expected_islands,
        "note": (
            "Island purity is the go/no-go for Approach 5. Compare island_count "
            "across tools before investing in boolean-ops UI."
        ),
        "rows": summary_rows,
    }
    write_json(out_dir / "purity_summary.json", summary)

    csv_path = out_dir / "purity_summary.csv"
    if summary_rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Console table
    print(f"Run: {run_dir}")
    print(f"Wrote {len(reports)} purity reports → {out_dir}")
    if args.expected_islands is not None:
        print(f"Expected islands: {args.expected_islands}")
    print()
    print(f"{'cond':>4}  {'tool':<16}  {'islands':>7}  {'over':>4}  {'under':>5}  change%")
    for row in summary_rows:
        over = row["over_seg"] if row["over_seg"] is not None else "-"
        under = row["under_seg"] if row["under_seg"] is not None else "-"
        print(
            f"{row['condition']:>4}  {row['tool']:<16}  {row['island_count']:>7}  "
            f"{over!s:>4}  {under!s:>5}  {100 * row['change_fraction']:6.2f}%"
        )


if __name__ == "__main__":
    main()
