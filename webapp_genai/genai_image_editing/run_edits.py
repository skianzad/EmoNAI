#!/usr/bin/env python3
"""Run the ten GenAI image-editing conditions from the evaluation deck.

Drop a source.jpg into each folder under ./inputs/<condition>/.
Conditions 9–10 also take markup.png and edges.png in the same folder.

API keys are loaded from the environment, a local .env file, or an interactive
prompt (one key per provider).
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from getpass import getpass
from io import BytesIO
from pathlib import Path
from threading import Lock, Semaphore
from typing import Callable, Any

import requests
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "outputs"

PRESERVE = (
    " Keep everything else in the photograph unchanged: identity, pose, "
    "clothing, lighting, background, and camera framing. Do not add, remove, "
    "or restyle anything that was not requested."
)


# ---------------------------------------------------------------------------
# Conditions (from the evaluation deck)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    number: int
    slug: str
    title: str
    prompt: str
    tools: tuple[str, ...]
    followup_prompt: str | None = None
    source_stem: str = ""
    extra_stems: tuple[str, ...] = ()

    @property
    def folder(self) -> str:
        return f"{self.number:02d}_{self.slug}"


COMPARISON_TOOLS: tuple[str, ...] = (
    "nano_banana",
    "qwen_image_3",
    "seedream_5_pro",
    "gpt_image_2",
)


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        1,
        "waist_reshaping",
        "Waist Reshaping",
        "Make the waist slightly slimmer while keeping the person natural and unchanged otherwise."
        + PRESERVE,
        COMPARISON_TOOLS,
    ),
    Condition(
        2,
        "object_move",
        "Object Move",
        "Move the handbag from the chair to the table." + PRESERVE,
        COMPARISON_TOOLS,
    ),
    Condition(
        3,
        "glare_removal",
        "Glare Removal",
        "Remove the glare from the glasses while preserving the eyes and lens shape."
        + PRESERVE,
        COMPARISON_TOOLS,
    ),
    Condition(
        4,
        "plant_insertion",
        "Plant Insertion",
        "Add a small potted plant on the desk between the notebook and lamp."
        + PRESERVE,
        COMPARISON_TOOLS,
    ),
    Condition(
        5,
        "sky_replacement",
        "Sky Replacement",
        "Change the overcast sky to a golden-hour sky while preserving the landscape."
        + PRESERVE,
        COMPARISON_TOOLS,
    ),
    Condition(
        6,
        "armchair_rearrangement",
        "Armchair Rearrangement",
        "Swap the armchairs and the sofa so that the armchairs are beside the "
        "window and facing the sofa."
        + PRESERVE,
        COMPARISON_TOOLS,
    ),
    Condition(
        7,
        "label_preservation",
        "Label Preservation",
        "Change the bottle color but keep the label and all text exactly unchanged."
        + PRESERVE,
        COMPARISON_TOOLS,
    ),
    Condition(
        8,
        "compound_edit",
        "Compound Edit",
        "Remove the charger from the scene." + PRESERVE,
        COMPARISON_TOOLS,
        followup_prompt=(
            "Now enlarge the vase to 1.5 times its current size, keeping its style, "
            "material, placement, and the rest of the scene unchanged."
            + PRESERVE
        ),
    ),
    Condition(
        9,
        "input_modality_markup",
        "Input Modality — Hand-Drawn Markup",
        "Move the lamp here. Use the markup image (circled lamp and arrow) to "
        "understand the destination. Keep everything else unchanged.",
        COMPARISON_TOOLS,
        extra_stems=("markup", "edges"),
    ),
    Condition(
        10,
        "input_modality_edges",
        "Input Modality — Edge-Detection File",
        "Resize the vase to fit this outline. Use the edge-map image as the "
        "target geometry. Keep everything else unchanged.",
        COMPARISON_TOOLS,
        extra_stems=("markup", "edges"),
    ),
)

# Conditions 9–10 hold the tool constant and compare input modalities.
MODALITY_VARIANTS: dict[int, tuple[str, ...]] = {
    9: ("text_only", "hand_drawn_markup", "edge_detection"),
    10: ("text_only", "hand_drawn_markup", "edge_detection"),
}

MODALITY_PROMPTS: dict[tuple[int, str], str] = {
    (9, "text_only"): (
        "Move the lamp to the empty spot indicated by a typical desk rearrangement: "
        "shift it to the opposite side of the desk. Keep everything else unchanged."
    ),
    (9, "hand_drawn_markup"): (
        "The second image is a hand-drawn markup of the first photo. Circle marks "
        "the lamp; the arrow shows where it should go. Move the lamp here. Keep "
        "everything else unchanged."
    ),
    (9, "edge_detection"): (
        "The second image is an edge map of the scene. Move the lamp to the "
        "location indicated in that map. Keep everything else unchanged."
    ),
    (10, "text_only"): (
        "Resize the vase so it is about 20 percent taller, keeping its style, "
        "material, and placement. Keep everything else unchanged."
    ),
    (10, "hand_drawn_markup"): (
        "The second image combines an edge map with the target outline marked. "
        "Resize the vase to that marked outline. Keep everything else unchanged."
    ),
    (10, "edge_detection"): (
        "The second image is an edge map with the target outline marked. Resize "
        "the vase to fit this outline. Keep everything else unchanged."
    ),
}


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeySpec:
    env_name: str
    tool: str
    label: str
    where: str


KEY_SPECS: tuple[KeySpec, ...] = (
    KeySpec("GEMINI_API_KEY", "nano_banana", "Nano Banana (Google Gemini)", "https://aistudio.google.com/apikey"),
    KeySpec("DASHSCOPE_API_KEY", "qwen_image_3", "Qwen-Image-3 (Alibaba DashScope)", "https://modelstudio.console.alibabacloud.com/"),
    KeySpec("ARK_API_KEY", "seedream_5_pro", "Seedream 5.0 Pro (BytePlus / Volcengine Ark)", "https://console.byteplus.com/ark"),
    KeySpec("BFL_API_KEY", "flux_kontext", "FLUX Kontext (Black Forest Labs)", "https://api.bfl.ai"),
    KeySpec("IDEOGRAM_API_KEY", "ideogram", "Ideogram", "https://developer.ideogram.ai"),
    KeySpec("OPENAI_API_KEY", "gpt_image_2", "GPT Image 2.5 Sunburst (OpenAI)", "https://platform.openai.com/api-keys"),
)

TOOL_TO_KEY = {spec.tool: spec.env_name for spec in KEY_SPECS}


def env_value(name: str, default: str = "") -> str:
    value = (os.getenv(name) or "").strip()
    return value or default


def load_keys(prompt_missing: bool) -> dict[str, str]:
    load_dotenv(ROOT / ".env")
    keys: dict[str, str] = {}
    for spec in KEY_SPECS:
        value = os.getenv(spec.env_name, "").strip()
        if value:
            keys[spec.env_name] = value
            continue
        if not prompt_missing:
            continue
        print(f"\n{spec.label}")
        print(f"  Get a key: {spec.where}")
        print(f"  Env var:   {spec.env_name}")
        typed = getpass("  Paste API key (hidden, Enter to skip): ").strip()
        if typed:
            keys[spec.env_name] = typed
            os.environ[spec.env_name] = typed
    return keys


def require_key(keys: dict[str, str], tool: str) -> str:
    env_name = TOOL_TO_KEY[tool]
    key = keys.get(env_name, "").strip()
    if not key:
        spec = next(s for s in KEY_SPECS if s.tool == tool)
        raise RuntimeError(
            f"Missing {env_name} for {spec.label}. Get one at {spec.where}"
        )
    return key


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")


def find_source(inputs: Path, condition: Condition) -> Path:
    folder = inputs / condition.folder
    named = ["source", f"{condition.number:02d}", condition.slug, f"{condition.number:02d}_{condition.slug}"]
    search_dirs = [folder, inputs] if folder.is_dir() else [inputs]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for stem in named:
            for ext in IMAGE_EXTS:
                candidate = directory / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
    if folder.is_dir():
        matches = sorted(
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and p.stem.lower() not in {"markup", "edges"}
            and not p.stem.lower().endswith(("_markup", "_edges"))
        )
        if matches:
            return matches[0]
    matches = sorted(
        p
        for p in inputs.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and (
            p.stem.startswith(f"{condition.number:02d}")
            or condition.slug in p.stem.lower()
        )
        and "_markup" not in p.stem
        and "_edges" not in p.stem
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No source image for condition {condition.number} ({condition.title}). "
        f"Drop a photo named source.jpg in {folder}/"
    )


def find_extra(inputs: Path, condition: Condition, kind: str) -> Path | None:
    folder = inputs / condition.folder
    stems = (
        kind,
        kind.rstrip("s") if kind.endswith("s") else f"{kind}s",
        f"{condition.number:02d}_{kind}",
        f"{condition.slug}_{kind}",
        f"{condition.number:02d}_{condition.slug}_{kind}",
    )
    for directory in (folder, inputs):
        if not directory.is_dir():
            continue
        for stem in stems:
            for ext in IMAGE_EXTS:
                candidate = directory / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
    if kind == "markup" and condition.number == 10:
        return find_extra(inputs, condition, "edges")
    return None


def mime_of(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return {".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        path.suffix.lower(), "image/jpeg"
    )


def as_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return f"data:{mime_of(path)};base64,{base64.b64encode(data).decode('ascii')}"


def as_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def as_standard_png(path: Path) -> tuple[str, bytes, str]:
    """Re-encode to sRGB PNG so APIs reject phone MPO/HDR JPEGs less often."""
    with Image.open(path) as image:
        image.load()
        converted = image.convert("RGBA" if "A" in image.mode else "RGB")
    buffer = BytesIO()
    converted.save(buffer, format="PNG")
    return ("image.png", buffer.getvalue(), "image/png")


def download_url(url: str, dest: Path) -> Path:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def save_bytes(data: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

@dataclass
class EditJob:
    source: Path
    prompt: str
    extra_images: list[Path] = field(default_factory=list)
    followup_prompt: str | None = None


class ProviderError(RuntimeError):
    pass


def edit_nano_banana(keys: dict[str, str], job: EditJob, dest: Path) -> Path:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_key(keys, "nano_banana"))
    model = env_value("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])

    def parts_to_image(response) -> Image.Image:
        for part in response.parts:
            if getattr(part, "inline_data", None) is not None:
                image = part.as_image()
                if image is not None:
                    return image
        raise ProviderError("Nano Banana returned no image")

    source = Image.open(job.source)
    extras = [Image.open(p) for p in job.extra_images]
    contents: list = [job.prompt, source, *extras]

    if job.followup_prompt:
        chat = client.chats.create(model=model, config=config)
        first = chat.send_message(contents)
        _ = parts_to_image(first)
        second = chat.send_message(job.followup_prompt)
        image = parts_to_image(second)
    else:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        image = parts_to_image(response)

    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return dest


def edit_qwen(keys: dict[str, str], job: EditJob, dest: Path) -> Path:
    api_key = require_key(keys, "qwen_image_3")
    base = env_value("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1").rstrip("/")
    model = env_value("QWEN_IMAGE_MODEL", "qwen-image-3.0-pro")

    def one_edit(image_path: Path, prompt: str) -> bytes:
        content = [{"image": as_data_uri(image_path)}]
        for extra in job.extra_images:
            content.append({"image": as_data_uri(extra)})
        content.append({"text": prompt})
        payload = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {
                "n": 1,
                "watermark": False,
                "prompt_extend": True,
            },
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        started = requests.post(
            f"{base}/services/aigc/multimodal-generation/generation",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if started.status_code >= 400:
            # Some regions accept a synchronous call without the async header.
            headers.pop("X-DashScope-Async", None)
            started = requests.post(
                f"{base}/services/aigc/multimodal-generation/generation",
                headers=headers,
                json=payload,
                timeout=300,
            )
        started.raise_for_status()
        body = started.json()
        if "output" in body and body.get("output", {}).get("choices"):
            url = body["output"]["choices"][0]["message"]["content"][0]["image"]
            return requests.get(url, timeout=180).content

        task_id = body.get("output", {}).get("task_id") or body.get("task_id")
        if not task_id:
            raise ProviderError(f"Qwen did not return an image or task id: {body}")
        deadline = time.time() + 300
        while time.time() < deadline:
            status = requests.get(
                f"{base}/tasks/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60,
            )
            status.raise_for_status()
            task = status.json()
            state = (
                task.get("output", {}).get("task_status")
                or task.get("task_status")
                or ""
            ).upper()
            if state in {"SUCCEEDED", "SUCCESS"}:
                choices = task.get("output", {}).get("choices") or []
                url = choices[0]["message"]["content"][0]["image"]
                return requests.get(url, timeout=180).content
            if state in {"FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}:
                raise ProviderError(f"Qwen task failed: {task}")
            time.sleep(2)
        raise ProviderError("Qwen edit timed out")

    working = job.source
    data = one_edit(working, job.prompt)
    if job.followup_prompt:
        tmp = dest.with_name(dest.stem + "_step1" + dest.suffix)
        save_bytes(data, tmp)
        data = one_edit(tmp, job.followup_prompt)
    return save_bytes(data, dest)


def edit_seedream(keys: dict[str, str], job: EditJob, dest: Path) -> Path:
    api_key = require_key(keys, "seedream_5_pro")
    base = env_value("ARK_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3").rstrip("/")
    model = env_value("ARK_MODEL", "dola-seedream-5-0-pro-260628")

    def one_edit(image_path: Path, prompt: str) -> bytes:
        images: list[str] = [as_data_uri(image_path)]
        images.extend(as_data_uri(p) for p in job.extra_images)
        payload = {
            "model": model,
            "prompt": prompt,
            "image": images[0] if len(images) == 1 else images,
            "size": "2K",
            "output_format": "png",
            "response_format": "url",
            "watermark": False,
        }
        response = requests.post(
            f"{base}/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        if not response.ok:
            raise ProviderError(f"Seedream {response.status_code}: {response.text[:500]}")
        body = response.json()
        items = body.get("data") or []
        if not items:
            raise ProviderError(f"Seedream returned no image: {body}")
        item = items[0]
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        url = item.get("url")
        if not url:
            raise ProviderError(f"Seedream returned no url: {body}")
        return requests.get(url, timeout=180).content

    data = one_edit(job.source, job.prompt)
    if job.followup_prompt:
        tmp = dest.with_name(dest.stem + "_step1" + dest.suffix)
        save_bytes(data, tmp)
        data = one_edit(tmp, job.followup_prompt)
    return save_bytes(data, dest)


def edit_flux_kontext(keys: dict[str, str], job: EditJob, dest: Path) -> Path:
    api_key = require_key(keys, "flux_kontext")
    endpoint = env_value("BFL_KONTEXT_URL", "https://api.bfl.ai/v1/flux-kontext-pro")

    def one_edit(image_path: Path, prompt: str) -> bytes:
        payload = {
            "prompt": prompt,
            "input_image": as_b64(image_path),
            "output_format": "png",
            "safety_tolerance": 2,
        }
        if job.extra_images:
            payload["input_image_2"] = as_b64(job.extra_images[0])
        submitted = requests.post(
            endpoint,
            headers={
                "accept": "application/json",
                "x-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        submitted.raise_for_status()
        body = submitted.json()
        polling_url = body.get("polling_url")
        if not polling_url:
            raise ProviderError(f"FLUX Kontext missing polling_url: {body}")
        deadline = time.time() + 180
        while time.time() < deadline:
            time.sleep(0.8)
            polled = requests.get(
                polling_url,
                headers={"accept": "application/json", "x-key": api_key},
                timeout=60,
            )
            polled.raise_for_status()
            result = polled.json()
            status = result.get("status")
            if status == "Ready":
                url = (result.get("result") or {}).get("sample")
                if not url:
                    raise ProviderError(f"FLUX Kontext ready with no sample: {result}")
                return requests.get(url, timeout=180).content
            if status in {"Error", "Failed"}:
                raise ProviderError(f"FLUX Kontext failed: {result}")
        raise ProviderError("FLUX Kontext timed out")

    data = one_edit(job.source, job.prompt)
    if job.followup_prompt:
        tmp = dest.with_name(dest.stem + "_step1" + dest.suffix)
        save_bytes(data, tmp)
        data = one_edit(tmp, job.followup_prompt)
    return save_bytes(data, dest)


def edit_ideogram(keys: dict[str, str], job: EditJob, dest: Path) -> Path:
    api_key = require_key(keys, "ideogram")

    def one_edit(image_path: Path, prompt: str) -> bytes:
        files = [("images", (image_path.name, image_path.read_bytes(), mime_of(image_path)))]
        for extra in job.extra_images:
            files.append(("images", (extra.name, extra.read_bytes(), mime_of(extra))))
        response = requests.post(
            "https://api.ideogram.ai/v1/edit",
            headers={"Api-Key": api_key},
            data={"prompt": prompt, "magic_prompt": "OFF"},
            files=files,
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()
        items = body.get("data") or []
        if not items or not items[0].get("url"):
            raise ProviderError(f"Ideogram returned no image: {body}")
        return requests.get(items[0]["url"], timeout=180).content

    data = one_edit(job.source, job.prompt)
    if job.followup_prompt:
        tmp = dest.with_name(dest.stem + "_step1" + dest.suffix)
        save_bytes(data, tmp)
        data = one_edit(tmp, job.followup_prompt)
    return save_bytes(data, dest)


def edit_gpt_image_2(keys: dict[str, str], job: EditJob, dest: Path) -> Path:
    api_key = require_key(keys, "gpt_image_2")
    model = env_value("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst")

    def one_edit(image_path: Path, prompt: str, extras: list[Path] | None = None) -> bytes:
        extras = job.extra_images if extras is None else extras
        images = [as_standard_png(image_path), *(as_standard_png(p) for p in extras)]
        field = "image[]" if len(images) > 1 else "image"
        files = [(field, (name, data, mime)) for name, data, mime in images]
        response = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": model, "prompt": prompt, "output_format": "png"},
            files=files,
            timeout=300,
        )
        if not response.ok:
            raise ProviderError(f"GPT Image ({model}) {response.status_code}: {response.text[:500]}")
        body = response.json()
        items = body.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise ProviderError(f"GPT Image ({model}) returned no image payload: {body}")
        return base64.b64decode(items[0]["b64_json"])

    if job.followup_prompt:
        data = one_edit(job.source, job.prompt, extras=[])
        tmp = dest.with_name(dest.stem + "_step1" + dest.suffix)
        save_bytes(data, tmp)
        data = one_edit(tmp, job.followup_prompt, extras=job.extra_images)
    else:
        data = one_edit(job.source, job.prompt)
    return save_bytes(data, dest)


EDITORS: dict[str, Callable[[dict[str, str], EditJob, Path], Path]] = {
    "nano_banana": edit_nano_banana,
    "qwen_image_3": edit_qwen,
    "seedream_5_pro": edit_seedream,
    "flux_kontext": edit_flux_kontext,
    "ideogram": edit_ideogram,
    "gpt_image_2": edit_gpt_image_2,
}

DISPLAY_NAME = {
    "nano_banana": "Nano Banana",
    "qwen_image_3": "Qwen-Image-3",
    "seedream_5_pro": "Seedream 5.0 Pro",
    "flux_kontext": "FLUX Kontext",
    "ideogram": "Ideogram",
    "gpt_image_2": "GPT Image 2.5 Sunburst",
    "text_only": "Text-only",
    "hand_drawn_markup": "Hand-drawn markup",
    "edge_detection": "Edge-detection file",
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def jobs_for_condition(condition: Condition, inputs: Path) -> list[tuple[str, str, EditJob]]:
    source = find_source(inputs, condition)
    markup = find_extra(inputs, condition, "markup")
    edges = find_extra(inputs, condition, "edges")

    extras: list[Path] = []
    if condition.number == 9 and markup is not None:
        extras = [markup]
    elif condition.number == 10 and edges is not None:
        extras = [edges]
    elif condition.number == 10 and markup is not None:
        extras = [markup]

    cond10 = next((c for c in CONDITIONS if c.number == 10), None)
    cond10_markup = None
    if cond10 is not None:
        cond10_markup = find_extra(inputs, cond10, "markup") or find_extra(inputs, cond10, "edges")

    jobs: list[tuple[str, str, EditJob]] = []
    for tool in condition.tools:
        job_extras = list(extras)
        job_prompt = condition.prompt
        job_followup = condition.followup_prompt
        if tool == "gpt_image_2" and condition.number in {8, 9, 10}:
            if condition.number == 8 and cond10_markup is not None:
                job_extras = [cond10_markup]
                job_followup = (
                    "Now enlarge the vase to 1.5 times its current size. The additional "
                    "image is a hand-drawn markup / edge map of the target outline; resize "
                    "the vase to that marked outline, keeping its style, material, "
                    "placement, and the rest of the scene unchanged."
                    + PRESERVE
                )
            elif condition.number in {9, 10}:
                extra = markup or edges or cond10_markup
                if extra is not None:
                    job_extras = [extra]
                job_prompt = MODALITY_PROMPTS[(condition.number, "hand_drawn_markup")]
                if condition.number == 10:
                    job_prompt += (
                        " Return a full-color photograph of the original scene. "
                        "Do not convert the photo into a line drawing or edge map; "
                        "use the second image only as a geometric guide for the vase outline."
                    )
                job_followup = None
        jobs.append(
            (
                tool,
                tool,
                EditJob(
                    source=source,
                    prompt=job_prompt,
                    extra_images=job_extras,
                    followup_prompt=job_followup,
                ),
            )
        )

    if condition.number in MODALITY_VARIANTS:
        for variant in MODALITY_VARIANTS[condition.number]:
            variant_extras: list[Path] = []
            if variant == "hand_drawn_markup":
                if markup is None:
                    print(f"  skip {variant}: missing {condition.number:02d}_markup.png")
                    continue
                variant_extras = [markup]
            elif variant == "edge_detection":
                if edges is None:
                    print(f"  skip {variant}: missing {condition.number:02d}_edges.png")
                    continue
                variant_extras = [edges]
            jobs.append(
                (
                    "nano_banana",
                    variant,
                    EditJob(
                        source=source,
                        prompt=MODALITY_PROMPTS[(condition.number, variant)],
                        extra_images=variant_extras,
                    ),
                )
            )
    return jobs


def parse_int_list(raw: str | None, valid: set[int]) -> list[int]:
    if not raw:
        return sorted(valid)
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    unknown = [n for n in values if n not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown condition(s): {unknown}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--conditions", help="Comma-separated condition numbers, e.g. 1,7,8")
    parser.add_argument("--tools", help="Comma-separated tools, e.g. nano_banana,gpt_image_2")
    parser.add_argument("--skip-tools", help="Comma-separated tools to skip, e.g. flux_kontext,ideogram")
    parser.add_argument("--dry-run", action="store_true", help="Print planned calls without hitting APIs")
    parser.add_argument("--check-keys", action="store_true", help="Show which API keys are present and exit")
    parser.add_argument("--no-prompt-keys", action="store_true", help="Do not interactively ask for missing keys")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first API error")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose output PNG already exists",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Max parallel API calls across different tools/conditions (default: 4)",
    )
    args = parser.parse_args()

    keys = load_keys(prompt_missing=not args.no_prompt_keys and not args.dry_run and not args.check_keys)

    if args.check_keys:
        print("API keys")
        for spec in KEY_SPECS:
            present = "yes" if keys.get(spec.env_name) else "MISSING"
            print(f"  {present:7}  {spec.env_name:20}  {spec.label}")
        return 0

    wanted_conditions = parse_int_list(
        args.conditions, {c.number for c in CONDITIONS}
    )
    wanted_tools = (
        {part.strip() for part in args.tools.split(",") if part.strip()}
        if args.tools
        else None
    )
    skipped_tools = (
        {part.strip() for part in args.skip_tools.split(",") if part.strip()}
        if args.skip_tools
        else set()
    )
    for group, label in ((wanted_tools or set(), "--tools"), (skipped_tools, "--skip-tools")):
        unknown = group - set(EDITORS)
        if unknown:
            parser.error(f"Unknown tool(s) in {label}: {sorted(unknown)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = args.output or (OUTPUT_DIR / stamp)
    out_root.mkdir(parents=True, exist_ok=True)

    log_path = out_root / "results.json"
    previous: list[dict] = []
    if log_path.exists():
        try:
            loaded = json.loads(log_path.read_text())
            if isinstance(loaded, list):
                previous = loaded
        except json.JSONDecodeError:
            previous = []

    def result_key(row: dict) -> tuple:
        return (row.get("condition"), row.get("tool"), row.get("variant"))

    results_by_key = {result_key(row): row for row in previous if isinstance(row, dict)}
    print(f"Inputs:  {args.inputs}")
    print(f"Outputs: {out_root}")
    print(f"Nano Banana model: {env_value('GEMINI_IMAGE_MODEL', 'gemini-3.1-flash-image')}")
    print(f"OpenAI image model: {env_value('OPENAI_IMAGE_MODEL', 'gpt-image-2.5-sunburst')}")
    print(f"Parallel jobs: {max(1, args.jobs)}")

    pending: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        if condition.number not in wanted_conditions:
            continue
        print(f"\n=== Condition {condition.number}: {condition.title} ===")
        try:
            planned = jobs_for_condition(condition, args.inputs)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            results_by_key[(condition.number, None, None)] = {
                "condition": condition.number,
                "title": condition.title,
                "error": str(exc),
            }
            continue

        for tool, variant, job in planned:
            if wanted_tools and tool not in wanted_tools:
                continue
            if tool in skipped_tools:
                continue
            label = DISPLAY_NAME.get(variant, variant)
            dest = out_root / condition.folder / f"{variant}.png"
            record = {
                "condition": condition.number,
                "title": condition.title,
                "tool": tool,
                "variant": variant,
                "prompt": job.prompt,
                "followup_prompt": job.followup_prompt,
                "source": str(job.source),
                "output": str(dest),
            }
            print(f"  -> {label}")
            print(f"     prompt: {job.prompt[:140]}{'…' if len(job.prompt) > 140 else ''}")
            if args.skip_existing and dest.exists():
                record["status"] = "skipped_existing"
                print(f"     skip:   {dest} already exists")
                results_by_key[result_key(record)] = record
                continue
            if args.dry_run:
                record["status"] = "dry_run"
                results_by_key[result_key(record)] = record
                continue
            pending.append(
                {
                    "condition": condition,
                    "tool": tool,
                    "variant": variant,
                    "job": job,
                    "dest": dest,
                    "record": record,
                    "label": label,
                }
            )

    print_lock = Lock()
    results_lock = Lock()
    tool_gates = {name: Semaphore(2) for name in EDITORS}

    def execute(item: dict[str, Any]) -> dict:
        tool = item["tool"]
        dest: Path = item["dest"]
        job: EditJob = item["job"]
        record: dict = dict(item["record"])
        label = item["label"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        with tool_gates[tool]:
            for attempt in range(1, 4):
                try:
                    EDITORS[tool](keys, job, dest)
                    dest.with_name(f"{dest.stem}_prompt.txt").write_text(job.prompt)
                    dest.with_name(f"{dest.stem}_meta.json").write_text(
                        json.dumps(
                            {
                                "condition": record["condition"],
                                "title": record["title"],
                                "tool": tool,
                                "variant": item["variant"],
                                "prompt": job.prompt,
                                "followup_prompt": job.followup_prompt,
                                "extra_images": [str(path) for path in job.extra_images],
                            },
                            indent=2,
                        )
                    )
                    record["status"] = "ok"
                    record.pop("error", None)
                    with print_lock:
                        print(f"  saved [{label} / cond {record['condition']}]: {dest}")
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001 — retry transient network errors
                    last_error = exc
                    if attempt < 3:
                        wait = 5 * attempt
                        with print_lock:
                            print(f"  retry {attempt}/3 [{label} / cond {record['condition']}] in {wait}s: {exc}")
                        time.sleep(wait)
        if last_error is not None:
            record["status"] = "error"
            record["error"] = str(last_error)
            with print_lock:
                print(f"  ERROR [{label} / cond {record['condition']}]: {last_error}")
        return record

    if pending:
        workers = max(1, min(args.jobs, len(pending)))
        print(f"\nRunning {len(pending)} job(s) with {workers} worker(s)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(execute, item) for item in pending]
            for future in as_completed(futures):
                record = future.result()
                with results_lock:
                    results_by_key[result_key(record)] = record
                    log_path.write_text(json.dumps(list(results_by_key.values()), indent=2))
                if args.fail_fast and record.get("status") == "error":
                    for leftover in futures:
                        leftover.cancel()
                    print("Stopping after the first API error (--fail-fast).")
                    return 1

    results = list(results_by_key.values())
    log_path.write_text(json.dumps(results, indent=2))
    ok = sum(1 for row in results if row.get("status") == "ok")
    failed = sum(1 for row in results if row.get("status") == "error")
    print(f"\nDone. ok={ok} failed={failed} log={out_root / 'results.json'}")
    return 1 if failed and args.fail_fast else 0


if __name__ == "__main__":
    sys.exit(main())
