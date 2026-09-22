"""Experiment folder layout shared by run_edits, evaluate_edits, and react_loop.

outputs/<run_id>/
  manifest.json
  01_waist_reshaping/
    source.jpg
    extras/
    nano_banana/
      prompt.txt
      after.png
      meta.json
      judgments/
        openai.json
        gemini.json
      react/
        round_01/
          prompt.txt
          after.png
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

TOOL_FAMILY = {
    "nano_banana": "google",
    "qwen_image_3": "alibaba",
    "seedream_5_pro": "bytedance",
    "flux_kontext": "bfl",
    "ideogram": "ideogram",
    "gpt_image_2": "openai",
}

JUDGE_FAMILY = {
    "openai": "openai",
    "gemini": "google",
    "qwen": "alibaba",
    "seedream": "bytedance",
}


@dataclass(frozen=True)
class ToolBundle:
    condition_dir: Path
    tool_dir: Path
    source: Path
    after: Path
    prompt_path: Path
    meta_path: Path
    extras_dir: Path
    judgments_dir: Path
    react_dir: Path
    variant: str

    @property
    def extras(self) -> list[Path]:
        if not self.extras_dir.exists():
            return []
        return sorted(
            p for p in self.extras_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )


def condition_dir(run_dir: Path, folder: str) -> Path:
    return run_dir / folder


def tool_bundle(run_dir: Path, folder: str, variant: str) -> ToolBundle:
    cond = condition_dir(run_dir, folder)
    tool_dir = cond / variant
    return ToolBundle(
        condition_dir=cond,
        tool_dir=tool_dir,
        source=_first_existing(cond, "source") or (cond / "source.jpg"),
        after=tool_dir / "after.png",
        prompt_path=tool_dir / "prompt.txt",
        meta_path=tool_dir / "meta.json",
        extras_dir=cond / "extras",
        judgments_dir=tool_dir / "judgments",
        react_dir=tool_dir / "react",
        variant=variant,
    )


def _first_existing(folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def copy_image(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def point_latest(output_root: Path, run_dir: Path) -> None:
    latest = output_root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        write_json(output_root / "latest.json", {"run": str(run_dir)})


def discover_tool_bundles(run_dir: Path) -> list[ToolBundle]:
    """Find per-tool experiment folders, including the older flat PNG layout."""
    bundles: list[ToolBundle] = []
    if not run_dir.exists():
        return bundles
    for cond in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        source = _first_existing(cond, "source")
        for child in sorted(cond.iterdir()):
            if child.is_dir() and (child / "after.png").exists():
                bundles.append(tool_bundle(run_dir, cond.name, child.name))
                continue
            if child.is_file() and child.suffix.lower() == ".png" and not child.stem.endswith("_step1"):
                # Legacy: 01_waist_reshaping/nano_banana.png
                variant = child.stem
                bundle = ToolBundle(
                    condition_dir=cond,
                    tool_dir=cond,
                    source=source or (cond / "source.jpg"),
                    after=child,
                    prompt_path=cond / f"{variant}_prompt.txt",
                    meta_path=cond / f"{variant}_meta.json",
                    extras_dir=cond / "extras",
                    judgments_dir=cond / "judgments" / variant,
                    react_dir=cond / "react" / variant,
                    variant=variant,
                )
                bundles.append(bundle)
    return bundles


def is_same_family(tool: str, judge: str) -> bool:
    return TOOL_FAMILY.get(tool) is not None and TOOL_FAMILY.get(tool) == JUDGE_FAMILY.get(judge)
