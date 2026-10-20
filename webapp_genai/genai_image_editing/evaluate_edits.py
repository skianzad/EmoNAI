#!/usr/bin/env python3
"""Judge before/after image edits against the evaluation-deck rubric.

Every vision editor scores every result, including its own:
Nano Banana/Gemini, Qwen-VL, and GPT Image 2 / OpenAI.

    python evaluate_edits.py --run latest

Override with --judges openai,gemini if needed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from getpass import getpass
from io import BytesIO
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any

from dotenv import load_dotenv
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layout import discover_tool_bundles, is_same_family, write_json
from run_edits import (
    COMPARISON_TOOLS,
    CONDITIONS,
    DISPLAY_NAME,
    INPUT_DIR,
    OUTPUT_DIR,
    env_value,
    find_extra,
    find_source,
)

# ---------------------------------------------------------------------------
# Rubric from the evaluation deck
# ---------------------------------------------------------------------------

CORE_ITEMS: tuple[tuple[str, str], ...] = (
    ("edit_accomplished", "The edit accomplished what was asked."),
    ("photorealism", "The result looks like a real, unedited photograph."),
    ("locality", "Only the intended part actually changed."),
    ("identity_preservation", "The subject still looks like itself."),
    ("controllability", "The degree of change could be controlled precisely."),
)

CONDITION_MEASURES: dict[int, tuple[str, ...]] = {
    1: ("edit_accuracy", "anatomy_preservation", "locality", "realism", "unintended_changes"),
    2: ("spatial_accuracy", "object_fidelity", "background_reconstruction", "shadow_consistency", "locality"),
    3: ("facial_fidelity", "fine_detail_reconstruction", "locality", "realism", "unintended_changes"),
    4: ("depth", "scale", "occlusion", "shadow", "scene_integration"),
    5: ("edge_quality", "lighting_consistency", "reflections", "global_coherence", "realism"),
    6: ("spatial_reasoning", "occlusion_handling", "shadow_regeneration", "scene_consistency", "locality"),
    7: ("typography_fidelity", "logo_preservation", "locality", "color_accuracy", "realism"),
    8: ("multi_turn_drift", "recoverability", "interaction_effort", "locality", "unintended_changes"),
    9: ("markup_comprehension", "spatial_accuracy", "locality", "controllability", "unintended_changes"),
    10: ("geometric_precision", "edge_adherence", "locality", "controllability", "realism"),
}

MEASURE_QUESTIONS: dict[str, str] = {
    "edit_accuracy": "Did it do what was asked?",
    "anatomy_preservation": "Did body shape, pose, and proportions stay intact except for the requested waist change?",
    "locality": "Were changes restricted to the intended region?",
    "realism": "Does it still look like a real photograph?",
    "unintended_changes": "How free is the result of unrequested changes? 5 = nothing unexpected, 1 = many unrelated changes.",
    "spatial_accuracy": "Did the object end up in the intended place?",
    "object_fidelity": "Did the moved object keep its identity, shape, and appearance?",
    "background_reconstruction": "Was the vacated region filled convincingly?",
    "shadow_consistency": "Are shadows coherent with the new placement?",
    "facial_fidelity": "Did the face stay recognizable with eyes and lens shape preserved?",
    "fine_detail_reconstruction": "Were fine details (lenses, frames, skin, glare region) reconstructed cleanly?",
    "depth": "Does the inserted object sit at a believable depth in the scene?",
    "scale": "Is the inserted object's size correct relative to neighbors?",
    "occlusion": "Are overlapping objects handled correctly?",
    "shadow": "Are contact shadows and lighting on the new object coherent?",
    "scene_integration": "Does the insertion belong in the photograph?",
    "edge_quality": "Are sky-to-landscape edges clean and believable?",
    "lighting_consistency": "Did global lighting update to match the new sky without wrecking the landscape?",
    "reflections": "Did water/glass/metal reflections update where they should?",
    "global_coherence": "Does the whole image still read as one photograph?",
    "spatial_reasoning": "Were the armchairs and sofa swapped so the armchairs sit beside the window and face the sofa?",
    "occlusion_handling": "Are occlusions after the rearrangement correct?",
    "shadow_regeneration": "Were shadows regenerated for the new furniture pose?",
    "scene_consistency": "Did the rest of the room stay consistent?",
    "typography_fidelity": "Did all label text stay exactly unchanged?",
    "logo_preservation": "Did logos and label artwork stay intact?",
    "color_accuracy": "Did the bottle color change as requested without coloring the label?",
    "multi_turn_drift": "After both sequential edits, how little did earlier work drift? 5 = no drift.",
    "recoverability": "Could a failed first step still leave a usable image for the second?",
    "interaction_effort": "Did a single pass per step suffice, without obvious retries or artifacts from over-editing? 5 = low effort.",
    "markup_comprehension": "Did the model follow the circled object and arrow?",
    "geometric_precision": "Did the resized object match the requested geometry?",
    "edge_adherence": "Did the result follow the provided edge/outline map?",
    "controllability": "Could the change be specified precisely — what, where, how much?",
}

SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "core_scores",
        "task_scores",
        "unintended_changes",
        "what_changed_unexpectedly",
        "findings",
    ],
    "properties": {
        "core_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "score", "rationale"],
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "rationale": {"type": "string"},
                },
            },
        },
        "task_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["measure", "score", "rationale"],
                "properties": {
                    "measure": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "rationale": {"type": "string"},
                },
            },
        },
        "unintended_changes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "what_changed_unexpectedly": {"type": "string"},
        "findings": {"type": "string"},
    },
}


def schema_without_additional_properties(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            key: schema_without_additional_properties(value)
            for key, value in schema.items()
            if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [schema_without_additional_properties(item) for item in schema]
    return schema


GEMINI_SCORE_SCHEMA = schema_without_additional_properties(SCORE_SCHEMA)


@dataclass
class EvalCase:
    before: Path
    after: Path
    prompt: str
    condition: int | None = None
    title: str = "Custom edit"
    tool: str = ""
    variant: str = ""
    extras: list[Path] | None = None
    followup_prompt: str | None = None
    tool_dir: Path | None = None
    judgments_dir: Path | None = None


# Editors that can judge (the rest are scored by these).
ALL_EDITOR_JUDGES = ("gemini", "qwen", "openai", "seedream")

JUDGE_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "seedream": "ARK_API_KEY",
}

JUDGE_KEY_WHERE = {
    "openai": "https://platform.openai.com/api-keys",
    "gemini": "https://aistudio.google.com/apikey",
    "qwen": "https://modelstudio.console.alibabacloud.com/",
    "seedream": "https://console.byteplus.com/ark",
}

JUDGE_DISPLAY = {
    "gemini": "Nano Banana / Gemini",
    "qwen": "Qwen-Image-3 / Qwen-VL",
    "openai": "GPT Image 2 / OpenAI",
    "seedream": "Seedream 5.0 Pro / ByteDance",
}


def judges_for_case(case: EvalCase, override: list[str] | None) -> list[str]:
    if override:
        return list(override)
    return list(ALL_EDITOR_JUDGES)


# ---------------------------------------------------------------------------
# Keys / images
# ---------------------------------------------------------------------------

def load_env() -> None:
    load_dotenv(ROOT / ".env")


def judge_key(provider: str, prompt_missing: bool) -> str:
    env_name = JUDGE_KEY_ENV[provider]
    value = os.getenv(env_name, "").strip()
    if value:
        return value
    if not prompt_missing:
        raise RuntimeError(f"Missing {env_name} for the {provider} judge")
    print(f"\nJudge model ({JUDGE_DISPLAY.get(provider, provider)})")
    print(f"  Get a key: {JUDGE_KEY_WHERE[provider]}")
    typed = getpass(f"  Paste {env_name} (hidden): ").strip()
    if not typed:
        raise RuntimeError(f"Missing {env_name}")
    os.environ[env_name] = typed
    return typed


def pick_provider(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.getenv("GEMINI_API_KEY", "").strip():
        return "gemini"
    return "openai"


def available_judges(requested: list[str] | None, prompt_missing: bool) -> dict[str, str]:
    names = requested or list(ALL_EDITOR_JUDGES)
    found: dict[str, str] = {}
    for name in names:
        env_name = JUDGE_KEY_ENV[name]
        if os.getenv(env_name, "").strip():
            found[name] = os.getenv(env_name, "").strip()
            continue
        if prompt_missing:
            continue
        try:
            found[name] = judge_key(name, prompt_missing=True)
        except RuntimeError:
            continue
    if not found:
        raise RuntimeError(
            "No judge API keys found. Need GEMINI_API_KEY, DASHSCOPE_API_KEY, "
            "and/or OPENAI_API_KEY so editors can score each other and themselves."
        )
    return found


def mime_of(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower(), "image/jpeg")


def prepare_image_bytes(path: Path, max_side: int = 2048, fmt: str | None = None) -> tuple[bytes, str]:
    image = Image.open(path)
    image = image.convert("RGB") if image.mode not in ("RGB", "L") else image
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = BytesIO()
    fmt = fmt or ("PNG" if path.suffix.lower() == ".png" else "JPEG")
    image.save(buf, format=fmt, quality=90)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return buf.getvalue(), mime


def data_uri(path: Path, max_side: int = 2048, fmt: str | None = None) -> str:
    data, mime = prepare_image_bytes(path, max_side=max_side, fmt=fmt)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_instruction(case: EvalCase) -> str:
    core_lines = "\n".join(f"- {key}: {question}" for key, question in CORE_ITEMS)
    task_block = ""
    if case.condition and case.condition in CONDITION_MEASURES:
        rows = []
        for measure in CONDITION_MEASURES[case.condition]:
            question = MEASURE_QUESTIONS.get(measure, measure.replace("_", " "))
            rows.append(f"- {measure}: {question}")
        task_block = "Task-specific measures (score each 1–5):\n" + "\n".join(rows) + "\n"
    followup = ""
    if case.followup_prompt:
        followup = f"Second sequential instruction:\n{case.followup_prompt}\n"
    extra_note = ""
    if case.extras:
        extra_note = (
            "Additional reference image(s) after the after-image are markup or "
            "edge-map inputs that were given to the editor. Use them to judge "
            "whether the intended target was followed.\n"
        )
    return f"""You are an expert photo-edit evaluator. Score a BEFORE/AFTER pair.

THE QUESTION THAT MATTERS MOST: "What changed that wasn't requested?"

Scoring: integers 1 (poor) to 5 (excellent). Be strict. A perfect 5 means the
requested edit happened and nothing else of consequence changed. Do not reward
a pretty image that ignored the instruction or rewrote the rest of the photo.

Requested edit:
{case.prompt}
{followup}
Condition: {case.condition or 'n/a'} — {case.title}
Tool / variant: {case.tool or 'n/a'} / {case.variant or 'n/a'}

Core items (score each 1–5):
{core_lines}

{task_block}{extra_note}
Image order: (1) BEFORE, (2) AFTER{', (3+) reference markup/edges' if case.extras else ''}.

Return JSON only, matching the schema. core_scores must include every core id.
task_scores must include every listed task-specific measure, or be [] if none.
unintended_changes is a list of concrete unrequested differences (empty if none).
what_changed_unexpectedly is a short paragraph answering that question.
findings is a 2–4 sentence overall verdict.
"""


def mean_score(verdict: dict[str, Any]) -> float:
    scores = [int(item["score"]) for item in verdict.get("core_scores") or []]
    scores += [int(item["score"]) for item in verdict.get("task_scores") or []]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


REACT_WEIGHTS = {
    "edit_accomplished": 1.0,
    "locality": 1.6,
    "identity_preservation": 1.2,
    "controllability": 1.5,
    "photorealism": 0.7,
    "recoverability": 1.6,
    "multi_turn_drift": 1.4,
    "interaction_effort": 1.1,
}


def score_map(verdict: dict[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in verdict.get("core_scores") or []:
        values[item["id"]] = int(item["score"])
    for item in verdict.get("task_scores") or []:
        values[item["measure"]] = int(item["score"])
    return values


def react_score(verdict: dict[str, Any]) -> float:
    values = score_map(verdict)
    if not values:
        return 0.0
    weighted = 0.0
    weight_sum = 0.0
    for key, weight in REACT_WEIGHTS.items():
        if key in values:
            weighted += values[key] * weight
            weight_sum += weight
    if weight_sum == 0:
        return mean_score(verdict)
    return round(weighted / weight_sum, 2)


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

def labeled_images(case: EvalCase) -> list[tuple[str, Path]]:
    labels = [("BEFORE", case.before), ("AFTER", case.after)]
    for extra in case.extras or []:
        labels.append(("REFERENCE", extra))
    return labels


def prepared_pil(path: Path) -> Image.Image:
    data, _ = prepare_image_bytes(path)
    return Image.open(BytesIO(data))


def judge_openai(case: EvalCase, api_key: str) -> dict[str, Any]:
    model = env_value("OPENAI_JUDGE_MODEL", "gpt-4.1")
    content: list[dict[str, Any]] = [{"type": "text", "text": build_instruction(case)}]
    for label, path in labeled_images(case):
        content.append({"type": "text", "text": f"{label} image:"})
        content.append({"type": "image_url", "image_url": {"url": data_uri(path)}})
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You evaluate image edits. Reply with JSON only.",
                },
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "edit_evaluation",
                    "strict": True,
                    "schema": SCORE_SCHEMA,
                },
            },
        },
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(f"OpenAI judge {response.status_code}: {response.text[:500]}")
    text = (((response.json().get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
    return json.loads(text)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    return json.loads(text[start : end + 1])


def judge_gemini(case: EvalCase, api_key: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = env_value("GEMINI_JUDGE_MODEL", "gemini-3.1-pro-preview")
    parts: list[Any] = [build_instruction(case)]
    for label, path in labeled_images(case):
        parts.append(f"{label} image:")
        parts.append(prepared_pil(path))
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_SCORE_SCHEMA,
        ),
    )
    return json.loads(response.text)


def judge_qwen(case: EvalCase, api_key: str) -> dict[str, Any]:
    base = env_value("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1").rstrip("/")
    model = env_value("QWEN_JUDGE_MODEL", "qwen-vl-max")
    content: list[dict[str, str]] = [{"text": build_instruction(case) + "\nReturn JSON only."}]
    for label, path in labeled_images(case):
        content.append({"text": f"{label} image:"})
        content.append({"image": data_uri(path)})
    response = requests.post(
        f"{base}/services/aigc/multimodal-generation/generation",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
        },
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    message = (((body.get("output") or {}).get("choices") or [{}])[0].get("message") or {})
    raw = message.get("content")
    if isinstance(raw, list):
        text = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
    else:
        text = raw or json.dumps(body)
    return parse_json_object(text)


def judge_seedream(case: EvalCase, api_key: str) -> dict[str, Any]:
    base = env_value("ARK_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3").rstrip("/")
    model = env_value("ARK_JUDGE_MODEL", "seed-2-0-lite-260428")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_instruction(case) + "\nReturn JSON only. No markdown."}
    ]
    for label, path in labeled_images(case):
        content.append({"type": "text", "text": f"{label} image:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_uri(path, max_side=768, fmt="JPEG")},
            }
        )
    last_error: Exception | None = None
    for attempt in range(12):
        try:
            response = requests.post(
                f"{base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 2500,
                    "thinking": {"type": "disabled"},
                },
                timeout=90,
            )
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(15 * (attempt + 1))
            continue
        if response.status_code == 429:
            last_error = RuntimeError(f"Seedream judge 429: {response.text[:300]}")
            time.sleep(70)
            continue
        if not response.ok:
            raise RuntimeError(f"Seedream judge {response.status_code}: {response.text[:500]}")
        body = response.json()
        message = (((body.get("choices") or [{}])[0].get("message") or {}))
        raw = message.get("content")
        if isinstance(raw, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw
            )
        else:
            text = raw or ""
        if not text.strip():
            raise RuntimeError(f"Seedream judge returned no text: {body}")
        return parse_json_object(text)
    raise last_error or RuntimeError("Seedream judge failed")


JUDGES = {"openai": judge_openai, "gemini": judge_gemini, "qwen": judge_qwen, "seedream": judge_seedream}


def coerce_score_list(value: Any, id_key: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault(id_key, key)
                rows.append(row)
            elif isinstance(item, (int, float)):
                rows.append({id_key: key, "score": int(item), "rationale": ""})
        return rows
    return []


def normalize_verdict(verdict: Any) -> dict[str, Any]:
    if not isinstance(verdict, dict):
        return {}
    normalized = dict(verdict)
    normalized["core_scores"] = coerce_score_list(normalized.get("core_scores"), "id")
    normalized["task_scores"] = coerce_score_list(normalized.get("task_scores"), "measure")
    changes = normalized.get("unintended_changes")
    if isinstance(changes, str):
        normalized["unintended_changes"] = [changes] if changes else []
    elif not isinstance(changes, list):
        normalized["unintended_changes"] = []
    normalized["what_changed_unexpectedly"] = str(normalized.get("what_changed_unexpectedly") or "")
    normalized["findings"] = str(normalized.get("findings") or "")
    return normalized


def evaluate_case(case: EvalCase, provider: str, api_key: str) -> dict[str, Any]:
    verdict = normalize_verdict(JUDGES[provider](case, api_key))
    record = {
        "condition": case.condition,
        "title": case.title,
        "tool": case.tool,
        "variant": case.variant,
        "prompt": case.prompt,
        "followup_prompt": case.followup_prompt,
        "before": str(case.before),
        "after": str(case.after),
        "extras": [str(p) for p in (case.extras or [])],
        "tool_dir": str(case.tool_dir) if case.tool_dir else None,
        "judge": provider,
        "same_family": is_same_family(case.tool or case.variant, provider),
        "verdict": verdict,
        "mean_score": mean_score(verdict),
        "react_score": react_score(verdict),
    }
    if case.judgments_dir:
        write_json(case.judgments_dir / f"{provider}.json", record)
    return record


def print_record(record: dict[str, Any]) -> None:
    label = DISPLAY_NAME.get(record.get("variant") or "", record.get("variant") or "edit")
    tool = DISPLAY_NAME.get(record.get("tool") or "", record.get("tool") or "")
    judge = record.get("judge")
    kin = "same-family" if record.get("same_family") else "cross-exam"
    header = f"{record.get('title')} — {tool or label}  [judge: {judge}, {kin}]"
    print(f"\n{header}")
    print(f"  mean: {record['mean_score']:.2f} / 5   react: {record.get('react_score', 0):.2f} / 5")
    verdict = record["verdict"]
    for item in verdict.get("core_scores") or []:
        print(f"  {item['id']:24} {item['score']}  {item.get('rationale', '')[:90]}")
    for item in verdict.get("task_scores") or []:
        print(f"  {item['measure']:24} {item['score']}  {item.get('rationale', '')[:90]}")
    unexpected = verdict.get("what_changed_unexpectedly") or ""
    if unexpected:
        print(f"  unexpected: {unexpected}")
    findings = verdict.get("findings") or ""
    if findings:
        print(f"  findings:   {findings}")


def extras_for_case(condition, variant: str, existing: list[Path] | None, inputs: Path) -> list[Path]:
    extras = list(existing or [])
    if not condition:
        return extras
    if variant == "gpt_image_2" and condition.number in {8, 9, 10}:
        extra = find_extra(inputs, condition, "markup") or find_extra(inputs, condition, "edges")
        if extra is None and condition.number == 8:
            cond10 = next((c for c in CONDITIONS if c.number == 10), None)
            if cond10 is not None:
                extra = find_extra(inputs, cond10, "markup") or find_extra(inputs, cond10, "edges")
        return [extra] if extra else extras
    if condition.number == 9:
        extra = find_extra(inputs, condition, "markup")
        return [extra] if extra else extras
    if condition.number == 10:
        extra = find_extra(inputs, condition, "edges") or find_extra(inputs, condition, "markup")
        return [extra] if extra else extras
    if variant == "hand_drawn_markup":
        extra = find_extra(inputs, condition, "markup")
        return [extra] if extra else extras
    if variant == "edge_detection":
        extra = find_extra(inputs, condition, "edges")
        return [extra] if extra else extras
    return extras


def _excel_label(value: str | None, mapping: dict[str, str]) -> str:
    if not value:
        return ""
    return mapping.get(value, value)


def _score_map(items: list[dict[str, Any]], id_key: str) -> dict[str, tuple[Any, str]]:
    mapped: dict[str, tuple[Any, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get(id_key)
        if not key:
            continue
        mapped[str(key)] = (item.get("score"), str(item.get("rationale") or ""))
    return mapped


def write_excel(records: list[dict[str, Any]], dest_dir: Path) -> Path:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, RadarChart, Reference
    from openpyxl.chart.data_source import NumDataSource, NumRef
    from openpyxl.chart.error_bar import ErrorBars
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.worksheet import Worksheet

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "scores.xlsx"
    wb = Workbook()

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1F4E79")
    wrap = Alignment(wrap_text=True, vertical="top")
    fill_fail = PatternFill("solid", fgColor="F5B7B1")
    fill_poor = PatternFill("solid", fgColor="FAD7A0")
    fill_weak = PatternFill("solid", fgColor="FCF3CF")

    def style_header(ws: Worksheet) -> None:
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 22

    def autosize(ws: Worksheet, max_width: int = 48) -> None:
        for column in ws.columns:
            letter = get_column_letter(column[0].column)
            longest = 0
            for cell in column[:80]:
                value = "" if cell.value is None else str(cell.value)
                longest = max(longest, min(len(value.split("\n", 1)[0]), max_width))
            ws.column_dimensions[letter].width = min(max(12, longest + 2), max_width)

    def write_sheet(ws: Worksheet, headers: list[str], rows: list[dict[str, Any]], wrap_cols: set[str] | None = None) -> None:
        ws.append(headers)
        wrap_cols = wrap_cols or set()
        for row in rows:
            ws.append([row.get(key) for key in headers])
        style_header(ws)
        for excel_row in ws.iter_rows(min_row=2, max_col=len(headers)):
            for cell, key in zip(excel_row, headers):
                cell.alignment = wrap if key in wrap_cols else Alignment(vertical="top")
        autosize(ws)
        for key in wrap_cols:
            idx = headers.index(key) + 1
            ws.column_dimensions[get_column_letter(idx)].width = 56

    measure_labels = {key: question for key, question in CORE_ITEMS}
    measure_labels.update(MEASURE_QUESTIONS)

    ordered = sorted(
        records,
        key=lambda r: (
            r.get("condition") if r.get("condition") is not None else 99,
            str(r.get("variant") or r.get("tool") or ""),
            str(r.get("judge") or ""),
        ),
    )

    scored_items: list[dict[str, Any]] = []
    notes_rows: list[dict[str, Any]] = []
    for record in ordered:
        verdict = record.get("verdict") or {}
        unexpected = str(verdict.get("what_changed_unexpectedly") or "")
        findings = str(verdict.get("findings") or "")
        unintended = "; ".join(verdict.get("unintended_changes") or [])
        editor = _excel_label(record.get("variant") or record.get("tool"), DISPLAY_NAME)
        judge = _excel_label(record.get("judge"), JUDGE_DISPLAY)
        exam = "self" if record.get("same_family") else "cross"
        base = {
            "condition": record.get("condition"),
            "title": record.get("title"),
            "editor": editor,
            "judge": judge,
            "exam": exam,
            "unexpected_changes": unexpected,
            "findings": findings,
            "unintended_changes": unintended,
            "after": record.get("after"),
        }
        item_notes: list[str] = []
        for kind, id_key, items in (
            ("core", "id", verdict.get("core_scores") or []),
            ("task", "measure", verdict.get("task_scores") or []),
        ):
            for item in items:
                if not isinstance(item, dict):
                    continue
                measure = str(item.get(id_key) or "")
                score = item.get("score")
                rationale = str(item.get("rationale") or "")
                if score is None:
                    continue
                try:
                    score_n = int(score)
                except (TypeError, ValueError):
                    continue
                severity = {1: "fail", 2: "poor", 3: "weak"}.get(score_n, "ok")
                row = {
                    **base,
                    "severity": severity,
                    "score": score_n,
                    "kind": kind,
                    "measure": measure,
                    "what_was_scored": measure_labels.get(measure, measure.replace("_", " ")),
                    "rationale": rationale,
                }
                scored_items.append(row)
                if rationale:
                    item_notes.append(f"{measure}: {score_n}/5 — {rationale}")
                else:
                    item_notes.append(f"{measure}: {score_n}/5")
        notes_rows.append(
            {
                **base,
                "score_notes": "\n".join(item_notes),
                "all_notes": "\n\n".join(
                    part
                    for part in (
                        f"Unexpected: {unexpected}" if unexpected else "",
                        f"Unintended: {unintended}" if unintended else "",
                        f"Findings: {findings}" if findings else "",
                        "\n".join(item_notes),
                    )
                    if part
                ),
            }
        )

    from collections import defaultdict

    editors = [_excel_label(tool, DISPLAY_NAME) for tool in COMPARISON_TOOLS]
    judges = [JUDGE_DISPLAY[name] for name in ALL_EDITOR_JUDGES]
    core_keys = [key for key, _ in CORE_ITEMS]
    param_keys = [*core_keys, "overall"]
    param_titles = {
        "edit_accomplished": "Edit accomplished",
        "photorealism": "Photorealism",
        "locality": "Locality",
        "identity_preservation": "Identity preserved",
        "controllability": "Controllability",
        "overall": "Overall (5 measures)",
    }

    def add_minmax_error_bars(chart: BarChart, sheet_title: str, n_series: int, n_cats: int, plus_start_col: int) -> None:
        quoted = f"'{sheet_title}'"
        for index in range(n_series):
            plus_col = get_column_letter(plus_start_col + index * 2)
            minus_col = get_column_letter(plus_start_col + index * 2 + 1)
            plus_ref = f"{quoted}!${plus_col}$2:${plus_col}${1 + n_cats}"
            minus_ref = f"{quoted}!${minus_col}$2:${minus_col}${1 + n_cats}"
            chart.series[index].errBars = ErrorBars(
                errDir="y",
                errValType="cust",
                plus=NumDataSource(numRef=NumRef(f=plus_ref)),
                minus=NumDataSource(numRef=NumRef(f=minus_ref)),
            )

    # Per editor × core measure × condition: average across judges, then mean / min / max across 10 scenarios.
    cond_scores: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for item in scored_items:
        if item.get("kind") != "core" or item.get("measure") not in core_keys:
            continue
        if not isinstance(item.get("condition"), int):
            continue
        cond_scores[(str(item.get("editor")), str(item.get("measure")), int(item["condition"]))].append(int(item["score"]))
    cond_mean: dict[tuple[str, str, int], float] = {
        key: sum(vals) / len(vals) for key, vals in cond_scores.items() if vals
    }
    for editor in editors:
        for condition in range(1, 11):
            parts = [cond_mean.get((editor, measure, condition)) for measure in core_keys]
            parts = [value for value in parts if value is not None]
            if parts:
                cond_mean[(editor, "overall", condition)] = sum(parts) / len(parts)

    def scenario_stats(editor: str, measure: str) -> tuple[float, float, float, float, float] | None:
        values = [
            cond_mean[(editor, measure, condition)]
            for condition in range(1, 11)
            if (editor, measure, condition) in cond_mean
        ]
        if not values:
            return None
        mean = sum(values) / len(values)
        lowest, highest = min(values), max(values)
        return mean, lowest, highest, mean - lowest, highest - mean

    perf_ws = wb.active
    perf_ws.title = "Editor performance"
    perf_ws["A1"] = "How well each editor did (5 core measures + overall, across 10 scenarios)"
    perf_ws["A1"].font = Font(bold=True, size=16, color="1F4E79")
    perf_ws.merge_cells("A1:L1")
    perf_ws["A2"] = (
        "Bar = mean of the 10 scenario scores (each scenario averaged over 4 judges). "
        "Error bars = lowest and highest scenario. This is edit quality, not judge reasoning."
    )
    perf_ws["A2"].font = Font(italic=True, color="5D6D7E")
    perf_ws.merge_cells("A2:L2")

    detail_headers = ["parameter", "editor", "mean", "lowest_scenario", "highest_scenario"]
    for col, header in enumerate(detail_headers, start=1):
        cell = perf_ws.cell(4, col, header)
        cell.font = header_font
        cell.fill = header_fill
    detail_row = 5
    for measure in param_keys:
        for editor in editors:
            stats = scenario_stats(editor, measure)
            if stats is None:
                continue
            mean, lowest, highest, _, _ = stats
            perf_ws.cell(detail_row, 1, param_titles[measure])
            perf_ws.cell(detail_row, 2, editor)
            for col, value in enumerate((mean, lowest, highest), start=3):
                cell = perf_ws.cell(detail_row, col, round(value, 2))
                cell.number_format = "0.00"
            detail_row += 1

    chart_header_row = detail_row + 1
    perf_ws.cell(chart_header_row, 1, "Chart data")
    perf_ws.cell(chart_header_row, 1).font = Font(bold=True, color="1F4E79")
    chart_row = chart_header_row + 1
    chart_headers = ["parameter", *editors]
    plus_minus_headers = []
    for editor in editors:
        plus_minus_headers.extend([f"{editor} plus", f"{editor} minus"])
    for col, header in enumerate(chart_headers + plus_minus_headers, start=1):
        cell = perf_ws.cell(chart_row, col, header)
        cell.font = header_font
        cell.fill = header_fill
    first_data_row = chart_row + 1
    for offset, measure in enumerate(param_keys):
        row_idx = first_data_row + offset
        perf_ws.cell(row_idx, 1, param_titles[measure])
        for editor_idx, editor in enumerate(editors):
            stats = scenario_stats(editor, measure)
            if stats is None:
                continue
            mean, lowest, highest, minus, plus = stats
            mean_cell = perf_ws.cell(row_idx, 2 + editor_idx, round(mean, 2))
            mean_cell.number_format = "0.00"
            plus_cell = perf_ws.cell(row_idx, 2 + len(editors) + editor_idx * 2, round(plus, 2))
            minus_cell = perf_ws.cell(row_idx, 3 + len(editors) + editor_idx * 2, round(minus, 2))
            plus_cell.number_format = "0.00"
            minus_cell.number_format = "0.00"
    last_data_row = first_data_row + len(param_keys) - 1

    editor_chart = BarChart()
    editor_chart.type = "col"
    editor_chart.grouping = "clustered"
    editor_chart.title = "Editor quality across 10 scenarios"
    editor_chart.y_axis.title = "Score (1–5)"
    editor_chart.x_axis.title = "Parameter"
    editor_chart.y_axis.scaling.min = 1
    editor_chart.y_axis.scaling.max = 5
    editor_chart.shape = 4
    editor_chart.legend.position = "b"
    editor_chart.width = 22
    editor_chart.height = 11
    editor_data = Reference(perf_ws, min_col=2, min_row=chart_row, max_col=1 + len(editors), max_row=last_data_row)
    editor_cats = Reference(perf_ws, min_col=1, min_row=first_data_row, max_row=last_data_row)
    editor_chart.add_data(editor_data, titles_from_data=True)
    editor_chart.set_categories(editor_cats)
    add_minmax_error_bars(editor_chart, perf_ws.title, len(editors), len(param_keys), 2 + len(editors))
    perf_ws.add_chart(editor_chart, "G4")
    perf_ws.column_dimensions["A"].width = 28
    perf_ws.column_dimensions["B"].width = 20
    perf_ws.sheet_properties.tabColor = "1F4E79"

    radar_ws = wb.create_sheet("Radar")
    radar_ws["A1"] = "Radial profiles of the 5 core dimensions"
    radar_ws["A1"].font = Font(bold=True, size=16, color="1F4E79")
    radar_ws.merge_cells("A1:F1")
    radar_ws["A2"] = (
        "One radar per dimension. Spokes are the 10 scenarios. Lines are editors. "
        "Scale is 1–5. The first chart is the 5-dimension mean profile."
    )
    radar_ws["A2"].font = Font(italic=True, color="5D6D7E")
    radar_ws.merge_cells("A2:F2")

    def style_block_header(ws, row: int, headers: list[str]) -> None:
        for col, header in enumerate(headers, start=1):
            cell = ws.cell(row, col, header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")

    def add_radar(title: str, header_row: int, first_row: int, last_row: int, anchor: str) -> None:
        chart = RadarChart()
        chart.type = "marker"
        chart.title = title
        chart.style = 10
        chart.y_axis.scaling.min = 1
        chart.y_axis.scaling.max = 5
        if chart.legend is not None:
            chart.legend.position = "b"
        chart.width = 14
        chart.height = 10
        data = Reference(radar_ws, min_col=2, min_row=header_row, max_col=1 + len(editors), max_row=last_row)
        cats = Reference(radar_ws, min_col=1, min_row=first_row, max_row=last_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        radar_ws.add_chart(chart, anchor)

    radar_ws["A4"] = "All 5 dimensions (mean across scenarios)"
    radar_ws["A4"].font = Font(bold=True, color="1F4E79")
    style_block_header(radar_ws, 5, ["dimension", *editors])
    for offset, measure in enumerate(core_keys):
        row_idx = 6 + offset
        radar_ws.cell(row_idx, 1, param_titles[measure])
        for editor_idx, editor in enumerate(editors):
            stats = scenario_stats(editor, measure)
            cell = radar_ws.cell(row_idx, 2 + editor_idx, round(stats[0], 2) if stats else None)
            cell.number_format = "0.00"
    add_radar("5 dimensions (mean)", 5, 6, 10, "G4")

    cond_titles: dict[int, str] = {}
    for item in scored_items:
        number = item.get("condition")
        if isinstance(number, int) and number not in cond_titles:
            cond_titles[number] = str(item.get("title") or number)

    block_start = 13
    chart_anchors = ["G13", "G30", "G47", "G64", "G81"]
    for measure_idx, measure in enumerate(core_keys):
        title_row = block_start + measure_idx * 17
        radar_ws.cell(title_row, 1, param_titles[measure])
        radar_ws.cell(title_row, 1).font = Font(bold=True, color="1F4E79")
        header_row = title_row + 1
        style_block_header(radar_ws, header_row, ["scenario", *editors])
        first_row = header_row + 1
        for condition in range(1, 11):
            row_idx = first_row + condition - 1
            label = f"{condition}. {cond_titles.get(condition, '')}".strip()
            radar_ws.cell(row_idx, 1, label)
            for editor_idx, editor in enumerate(editors):
                value = cond_mean.get((editor, measure, condition))
                cell = radar_ws.cell(
                    row_idx, 2 + editor_idx, round(value, 2) if value is not None else None
                )
                cell.number_format = "0.00"
        last_row = first_row + 9
        add_radar(param_titles[measure], header_row, first_row, last_row, chart_anchors[measure_idx])

    radar_ws.column_dimensions["A"].width = 36
    for idx in range(2, 6):
        radar_ws.column_dimensions[get_column_letter(idx)].width = 16
    radar_ws.sheet_properties.tabColor = "1F4E79"

    judge_all_scores: dict[str, list[int]] = defaultdict(list)
    for item in scored_items:
        if item.get("kind") != "core" or not isinstance(item.get("score"), int):
            continue
        judge_all_scores[str(item.get("judge") or "")].append(int(item["score"]))
    judge_own_mean = {
        judge: (sum(scores) / len(scores) if scores else 0.0)
        for judge, scores in judge_all_scores.items()
    }

    below_mean_rows: list[dict[str, Any]] = []
    opportunities: dict[tuple[str, int], int] = defaultdict(int)
    found: dict[tuple[str, int], int] = defaultdict(int)

    for item in scored_items:
        if item.get("kind") != "core" or not isinstance(item.get("score"), int):
            continue
        if not isinstance(item.get("condition"), int):
            continue
        judge = str(item.get("judge") or "")
        score_n = int(item["score"])
        own_mean = judge_own_mean.get(judge, 0.0)
        condition = int(item["condition"])
        opportunities[(judge, condition)] += 1
        if score_n >= own_mean - 1e-9:
            continue
        found[(judge, condition)] += 1
        below_mean_rows.append(
            {
                "condition": condition,
                "title": item.get("title"),
                "editor": item.get("editor"),
                "judge": judge,
                "measure": item.get("measure"),
                "what_was_scored": item.get("what_was_scored"),
                "judge_score": score_n,
                "judge_own_mean": round(own_mean, 2),
                "below_own_mean_by": round(own_mean - score_n, 2),
                "rationale": str(item.get("rationale") or "").strip(),
                "findings": item.get("findings") or "",
            }
        )

    def judge_find_stats(judge: str) -> tuple[float, float, float, float, float] | None:
        values = [
            (100.0 * found.get((judge, condition), 0) / opportunities[(judge, condition)])
            for condition in range(1, 11)
            if opportunities.get((judge, condition))
        ]
        if not values:
            return None
        mean = sum(values) / len(values)
        lowest, highest = min(values), max(values)
        return mean, lowest, highest, mean - lowest, highest - mean

    below_mean_rows.sort(
        key=lambda r: (
            -(r.get("below_own_mean_by") or 0),
            r.get("judge_score") or 9,
            r.get("condition") if r.get("condition") is not None else 99,
        )
    )

    reason_ws = wb.create_sheet("Judge reasoning")
    reason_ws["A1"] = "Did the judge find a problem vs their own usual score?"
    reason_ws["A1"].font = Font(bold=True, size=16, color="1F4E79")
    reason_ws.merge_cells("A1:L1")
    reason_ws["A2"] = (
        "Gemini almost never gives 5 and usually gives 4, so beating the group mean is the wrong test. "
        "A problem find is a score below that judge’s own mean. 4 from Gemini is their default; 3 or lower is a flag. "
        "4 from OpenAI is a flag because they usually give 5."
    )
    reason_ws["A2"].font = Font(italic=True, color="5D6D7E")
    reason_ws.merge_cells("A2:L2")

    reason_ws["A4"] = "Rating habit (core measures)"
    reason_ws["A4"].font = Font(bold=True, color="1F4E79")
    calib_headers = ["judge", "own_mean", "%1", "%2", "%3", "%4", "%5", "n"]
    for col, header in enumerate(calib_headers, start=1):
        cell = reason_ws.cell(5, col, header)
        cell.font = header_font
        cell.fill = header_fill
    from collections import Counter
    for idx, judge in enumerate(judges):
        scores = judge_all_scores.get(judge, [])
        counts = Counter(scores)
        n = len(scores) or 1
        reason_ws.cell(6 + idx, 1, judge)
        reason_ws.cell(6 + idx, 2, round(judge_own_mean.get(judge, 0.0), 2))
        for score in range(1, 6):
            cell = reason_ws.cell(6 + idx, 2 + score, round(100.0 * counts.get(score, 0) / n, 1))
            cell.number_format = "0.0"
        reason_ws.cell(6 + idx, 8, len(scores))

    reason_ws["A11"] = "Problem finds vs own mean"
    reason_ws["A11"].font = Font(bold=True, color="1F4E79")
    for col, header in enumerate(
        ["judge", "mean_%_below_own_mean", "lowest_scenario", "highest_scenario", "finds"],
        start=1,
    ):
        cell = reason_ws.cell(12, col, header)
        cell.font = header_font
        cell.fill = header_fill
    find_counts = defaultdict(int)
    for row in below_mean_rows:
        find_counts[str(row.get("judge") or "")] += 1
    for idx, judge in enumerate(judges):
        stats = judge_find_stats(judge)
        if stats is None:
            continue
        mean, lowest, highest, minus, plus = stats
        reason_ws.cell(13 + idx, 1, judge)
        for col, value in enumerate((mean, lowest, highest), start=2):
            cell = reason_ws.cell(13 + idx, col, round(value, 1))
            cell.number_format = "0.0"
        reason_ws.cell(13 + idx, 5, find_counts.get(judge, 0))
        reason_ws.cell(13 + idx, 6, round(plus, 1))
        reason_ws.cell(13 + idx, 7, round(minus, 1))
    reason_ws.cell(12, 6, "plus")
    reason_ws.cell(12, 7, "minus")

    habit_chart = BarChart()
    habit_chart.type = "col"
    habit_chart.grouping = "stacked"
    habit_chart.title = "Score habit: share of 1–5"
    habit_chart.y_axis.title = "% of core ratings"
    habit_chart.x_axis.title = "Judge"
    habit_chart.y_axis.scaling.min = 0
    habit_chart.y_axis.scaling.max = 100
    habit_chart.shape = 4
    habit_chart.legend.position = "b"
    habit_chart.width = 16
    habit_chart.height = 9
    habit_data = Reference(reason_ws, min_col=3, min_row=5, max_col=7, max_row=5 + len(judges))
    habit_cats = Reference(reason_ws, min_col=1, min_row=6, max_row=5 + len(judges))
    habit_chart.add_data(habit_data, titles_from_data=True)
    habit_chart.set_categories(habit_cats)
    reason_ws.add_chart(habit_chart, "J4")

    reason_chart = BarChart()
    reason_chart.type = "col"
    reason_chart.title = "% of ratings below that judge’s own mean"
    reason_chart.y_axis.title = "Below own mean (%)"
    reason_chart.x_axis.title = "Judge"
    reason_chart.y_axis.scaling.min = 0
    reason_chart.y_axis.scaling.max = 100
    reason_chart.shape = 4
    reason_chart.legend = None
    reason_chart.width = 16
    reason_chart.height = 9
    reason_data = Reference(reason_ws, min_col=2, min_row=12, max_row=12 + len(judges))
    reason_cats = Reference(reason_ws, min_col=1, min_row=13, max_row=12 + len(judges))
    reason_chart.add_data(reason_data, titles_from_data=True)
    reason_chart.set_categories(reason_cats)
    plus_ref = f"'Judge reasoning'!$F$13:$F${12 + len(judges)}"
    minus_ref = f"'Judge reasoning'!$G$13:$G${12 + len(judges)}"
    reason_chart.series[0].errBars = ErrorBars(
        errDir="y",
        errValType="cust",
        plus=NumDataSource(numRef=NumRef(f=plus_ref)),
        minus=NumDataSource(numRef=NumRef(f=minus_ref)),
    )
    reason_ws.add_chart(reason_chart, "J20")
    reason_ws.column_dimensions["A"].width = 32
    reason_ws.column_dimensions["B"].width = 24
    reason_ws.sheet_properties.tabColor = "6C3483"

    finds_ws = wb.create_sheet("Below-own-mean finds")
    write_sheet(
        finds_ws,
        [
            "below_own_mean_by",
            "judge_score",
            "judge_own_mean",
            "condition",
            "title",
            "editor",
            "judge",
            "measure",
            "what_was_scored",
            "rationale",
            "findings",
        ],
        below_mean_rows,
        wrap_cols={"what_was_scored", "rationale", "findings"},
    )
    finds_ws.column_dimensions["J"].width = 78
    finds_ws.column_dimensions["K"].width = 64
    for excel_row in finds_ws.iter_rows(min_row=2, max_row=finds_ws.max_row):
        finds_ws.row_dimensions[excel_row[0].row].height = 72
        excel_row[0].fill = fill_fail if (excel_row[1].value or 5) <= 2 else fill_poor

    diagnosis_ws = wb.create_sheet("Best diagnoses")

    worst_rows = [
        item
        for item in scored_items
        if isinstance(item.get("score"), int) and item["score"] <= 2
    ]

    def diagnosis_weight(item: dict[str, Any]) -> int:
        rationale = str(item.get("rationale") or "")
        findings = str(item.get("findings") or "")
        unexpected = str(item.get("unexpected_changes") or "")
        weight = len(rationale) + len(findings) + min(len(unexpected), 400)
        if findings.strip():
            weight += 80
        if item.get("score") == 1:
            weight += 60
        if len(rationale.strip()) < 25:
            weight -= 50
        if unexpected.lower().startswith("no ") or "no unexpected" in unexpected.lower():
            weight -= 15
        return weight

    for item in worst_rows:
        item["diagnosis_weight"] = diagnosis_weight(item)
    worst_rows.sort(
        key=lambda r: (
            r.get("score") or 9,
            -(r.get("diagnosis_weight") or 0),
            r.get("condition") if r.get("condition") is not None else 99,
            str(r.get("editor") or ""),
        )
    )

    grouped_by_edit: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for item in worst_rows:
        grouped_by_edit.setdefault((item.get("condition"), item.get("editor") or ""), []).append(item)

    diagnosis_rows: list[dict[str, Any]] = []
    for (condition, editor), items in grouped_by_edit.items():
        by_judge: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_judge.setdefault(str(item.get("judge") or ""), []).append(item)
        best_judge = ""
        best_items: list[dict[str, Any]] = []
        best_weight = -10**9
        for judge, judge_items in by_judge.items():
            weight = sum(int(row.get("diagnosis_weight") or 0) for row in judge_items)
            lowest = min(int(row.get("score") or 9) for row in judge_items)
            if lowest < min((int(row.get("score") or 9) for row in best_items), default=9) or (
                lowest == min((int(row.get("score") or 9) for row in best_items), default=9)
                and weight > best_weight
            ):
                best_judge = judge
                best_items = judge_items
                best_weight = weight
        best_items = sorted(best_items, key=lambda r: (r.get("score") or 9, -(r.get("diagnosis_weight") or 0)))
        head = best_items[0]
        diagnosis_rows.append(
            {
                "lowest_score": head.get("score"),
                "condition": condition,
                "title": head.get("title"),
                "editor": editor,
                "best_judge": best_judge,
                "issues": "; ".join(
                    f"{item.get('measure')} {item.get('score')}/5"
                    for item in best_items[:6]
                ),
                "rationale": "\n\n".join(
                    f"{item.get('measure')} ({item.get('score')}/5): {item.get('rationale') or '(no rationale)'}"
                    for item in best_items
                    if item.get("rationale") or True
                ),
                "findings": head.get("findings") or "",
                "unexpected_changes": head.get("unexpected_changes") or "",
                "unintended_changes": head.get("unintended_changes") or "",
            }
        )
    diagnosis_rows.sort(
        key=lambda r: (
            r.get("lowest_score") or 9,
            r.get("condition") if r.get("condition") is not None else 99,
            str(r.get("editor") or ""),
        )
    )

    write_sheet(
        diagnosis_ws,
        [
            "lowest_score",
            "condition",
            "title",
            "editor",
            "best_judge",
            "issues",
            "rationale",
            "findings",
            "unexpected_changes",
            "unintended_changes",
        ],
        diagnosis_rows,
        wrap_cols={"issues", "rationale", "findings", "unexpected_changes", "unintended_changes"},
    )
    diagnosis_ws.column_dimensions["G"].width = 78
    diagnosis_ws.column_dimensions["H"].width = 72
    diagnosis_ws.column_dimensions["I"].width = 56
    for excel_row in diagnosis_ws.iter_rows(min_row=2, max_row=diagnosis_ws.max_row):
        diagnosis_ws.row_dimensions[excel_row[0].row].height = 110
        score = excel_row[0].value
        excel_row[0].fill = fill_fail if score == 1 else fill_poor
    diagnosis_ws.sheet_properties.tabColor = "C0392B"

    worst_headers = [
        "score",
        "rationale",
        "findings",
        "unexpected_changes",
        "condition",
        "title",
        "editor",
        "judge",
        "measure",
        "what_was_scored",
        "unintended_changes",
        "after",
    ]
    worst_ws = wb.create_sheet("Lowest scores")
    write_sheet(
        worst_ws,
        worst_headers,
        worst_rows,
        wrap_cols={
            "rationale",
            "findings",
            "unexpected_changes",
            "what_was_scored",
            "unintended_changes",
        },
    )
    worst_ws.column_dimensions["B"].width = 78
    worst_ws.column_dimensions["C"].width = 72
    worst_ws.column_dimensions["D"].width = 56
    for excel_row in worst_ws.iter_rows(min_row=2, max_row=worst_ws.max_row):
        score = excel_row[0].value
        excel_row[0].fill = fill_fail if score == 1 else fill_poor
        worst_ws.row_dimensions[excel_row[0].row].height = 88
    worst_ws.conditional_formatting.add(
        f"A2:A{max(2, worst_ws.max_row)}",
        ColorScaleRule(
            start_type="num",
            start_value=1,
            start_color="C0392B",
            mid_type="num",
            mid_value=1.5,
            mid_color="E67E22",
            end_type="num",
            end_value=2,
            end_color="FAD7A0",
        ),
    )

    grouped_worst = grouped_by_edit
    per_edit_rows: list[dict[str, Any]] = []
    for (condition, editor), items in grouped_worst.items():
        items = sorted(items, key=lambda r: (r.get("score") or 9, r.get("measure") or ""))
        seen: set[str] = set()
        picked: list[dict[str, Any]] = []
        for item in items:
            measure = str(item.get("measure") or "")
            if measure in seen:
                continue
            seen.add(measure)
            picked.append(item)
            if len(picked) == 3:
                break
        if not picked:
            continue
        per_edit_rows.append(
            {
                "condition": condition,
                "title": picked[0].get("title"),
                "editor": editor,
                "lowest_score": picked[0].get("score"),
                "fail_count": sum(1 for item in items if item.get("score") == 1),
                "poor_count": sum(1 for item in items if item.get("score") == 2),
                "findings": next((item.get("findings") for item in items if item.get("findings")), ""),
                "unexpected_changes": next((item.get("unexpected_changes") for item in items if item.get("unexpected_changes")), ""),
                "worst_measures": "; ".join(
                    f"{item.get('measure')} {item.get('score')}/5"
                    for item in picked
                ),
                "rationales": "\n\n".join(
                    f"{item.get('judge')} on {item.get('measure')} ({item.get('score')}/5): "
                    f"{item.get('rationale') or 'No rationale'}"
                    for item in picked
                ),
            }
        )
    per_edit_rows.sort(
        key=lambda r: (
            r.get("lowest_score") or 9,
            -(r.get("fail_count") or 0),
            r.get("condition") if r.get("condition") is not None else 99,
            str(r.get("editor") or ""),
        )
    )
    per_edit_ws = wb.create_sheet("Worst per edit")
    write_sheet(
        per_edit_ws,
        [
            "lowest_score",
            "condition",
            "title",
            "editor",
            "fail_count",
            "poor_count",
            "findings",
            "rationales",
            "unexpected_changes",
            "worst_measures",
        ],
        per_edit_rows,
        wrap_cols={"findings", "rationales", "unexpected_changes", "worst_measures"},
    )
    per_edit_ws.column_dimensions["G"].width = 72
    per_edit_ws.column_dimensions["H"].width = 80
    for excel_row in per_edit_ws.iter_rows(min_row=2, max_row=per_edit_ws.max_row):
        per_edit_ws.row_dimensions[excel_row[0].row].height = 96
        score = excel_row[0].value
        fill = fill_fail if score == 1 else fill_poor if score == 2 else fill_weak if score == 3 else None
        if fill is not None:
            excel_row[0].fill = fill

    from collections import Counter

    fail_by_editor: Counter[str] = Counter()
    fail_by_condition: Counter[int] = Counter()
    fail_by_measure: Counter[str] = Counter()
    for item in scored_items:
        if item.get("score") not in {1, 2}:
            continue
        fail_by_editor[str(item.get("editor") or "")] += 1
        if isinstance(item.get("condition"), int):
            fail_by_condition[item["condition"]] += 1
        fail_by_measure[str(item.get("measure") or "")] += 1

    overview_ws = wb.create_sheet("Where it failed")
    overview_ws["A1"] = "How often judges gave a 1 or 2"
    overview_ws["A1"].font = Font(bold=True, size=16, color="1F4E79")
    overview_ws.merge_cells("A1:F1")
    overview_ws["A2"] = (
        "Counts only. Use “Best diagnoses” for the actual issue write-ups, sorted by lowest score."
    )
    overview_ws["A2"].font = Font(italic=True, color="5D6D7E")
    overview_ws.merge_cells("A2:F2")
    overview_ws["A4"] = "By editor"
    overview_ws["A4"].font = Font(bold=True, color="1F4E79")
    overview_ws["A5"] = "Editor"
    overview_ws["B5"] = "Scores of 1–2"
    overview_ws["A5"].font = header_font
    overview_ws["B5"].font = header_font
    overview_ws["A5"].fill = header_fill
    overview_ws["B5"].fill = header_fill
    editor_start = 6
    for i, (editor, count) in enumerate(fail_by_editor.most_common()):
        overview_ws.cell(editor_start + i, 1, editor)
        overview_ws.cell(editor_start + i, 2, count)
    editor_end = editor_start + max(len(fail_by_editor) - 1, 0)

    overview_ws["D4"] = "By condition"
    overview_ws["D4"].font = Font(bold=True, color="1F4E79")
    overview_ws["D5"] = "Condition"
    overview_ws["E5"] = "Scores of 1–2"
    overview_ws["D5"].font = header_font
    overview_ws["E5"].font = header_font
    overview_ws["D5"].fill = header_fill
    overview_ws["E5"].fill = header_fill
    cond_start = 6
    for i, number in enumerate(sorted(fail_by_condition)):
        overview_ws.cell(cond_start + i, 4, number)
        overview_ws.cell(cond_start + i, 5, fail_by_condition[number])
    cond_end = cond_start + max(len(fail_by_condition) - 1, 0)

    overview_ws["A16"] = "By measure"
    overview_ws["A16"].font = Font(bold=True, color="1F4E79")
    overview_ws["A17"] = "Measure"
    overview_ws["B17"] = "What was scored"
    overview_ws["C17"] = "Scores of 1–2"
    for col in range(1, 4):
        cell = overview_ws.cell(17, col)
        cell.font = header_font
        cell.fill = header_fill
    measure_start = 18
    for i, (measure, count) in enumerate(fail_by_measure.most_common()):
        overview_ws.cell(measure_start + i, 1, measure)
        overview_ws.cell(measure_start + i, 2, measure_labels.get(measure, measure.replace("_", " ")))
        overview_ws.cell(measure_start + i, 3, count)
        overview_ws.row_dimensions[measure_start + i].height = 32
        overview_ws.cell(measure_start + i, 2).alignment = wrap
    measure_end = measure_start + max(len(fail_by_measure) - 1, 0)

    overview_ws.column_dimensions["A"].width = 22
    overview_ws.column_dimensions["B"].width = 64
    overview_ws.column_dimensions["C"].width = 16
    overview_ws.column_dimensions["D"].width = 14
    overview_ws.column_dimensions["E"].width = 16
    overview_ws.sheet_properties.tabColor = "1F4E79"

    if fail_by_editor:
        editor_chart = BarChart()
        editor_chart.type = "col"
        editor_chart.title = "Failing scores (1–2) by editor"
        editor_chart.y_axis.title = "Count of 1s and 2s"
        editor_chart.x_axis.title = "Editor"
        editor_data = Reference(overview_ws, min_col=2, min_row=5, max_row=editor_end)
        editor_cats = Reference(overview_ws, min_col=1, min_row=editor_start, max_row=editor_end)
        editor_chart.add_data(editor_data, titles_from_data=True)
        editor_chart.set_categories(editor_cats)
        editor_chart.shape = 4
        editor_chart.legend = None
        editor_chart.width = 15
        editor_chart.height = 8
        overview_ws.add_chart(editor_chart, "G4")
    if fail_by_condition:
        cond_chart = BarChart()
        cond_chart.type = "col"
        cond_chart.title = "Failing scores (1–2) by condition"
        cond_chart.y_axis.title = "Count of 1s and 2s"
        cond_chart.x_axis.title = "Condition"
        cond_data = Reference(overview_ws, min_col=5, min_row=5, max_row=cond_end)
        cond_cats = Reference(overview_ws, min_col=4, min_row=cond_start, max_row=cond_end)
        cond_chart.add_data(cond_data, titles_from_data=True)
        cond_chart.set_categories(cond_cats)
        cond_chart.shape = 4
        cond_chart.legend = None
        cond_chart.width = 15
        cond_chart.height = 8
        overview_ws.add_chart(cond_chart, "G20")

    notes_ws = wb.create_sheet("Notes")
    write_sheet(
        notes_ws,
        [
            "condition",
            "title",
            "editor",
            "judge",
            "exam",
            "unexpected_changes",
            "findings",
            "unintended_changes",
            "score_notes",
            "all_notes",
        ],
        notes_rows,
        wrap_cols={
            "unexpected_changes",
            "findings",
            "unintended_changes",
            "score_notes",
            "all_notes",
        },
    )
    notes_ws.column_dimensions["I"].width = 70
    notes_ws.column_dimensions["J"].width = 80
    for excel_row in notes_ws.iter_rows(min_row=2, max_row=notes_ws.max_row):
        notes_ws.row_dimensions[excel_row[0].row].height = 90

    wb.save(path)
    return path

def flatten_row(record: dict[str, Any]) -> dict[str, Any]:
    verdict = record.get("verdict") or {}
    row = {
        "condition": record.get("condition"),
        "title": record.get("title"),
        "tool": record.get("tool"),
        "variant": record.get("variant"),
        "judge": record.get("judge"),
        "same_family": record.get("same_family"),
        "exam": "self" if record.get("same_family") else "cross",
        "mean_score": record.get("mean_score"),
        "react_score": record.get("react_score"),
        "before": record.get("before"),
        "after": record.get("after"),
        "what_changed_unexpectedly": verdict.get("what_changed_unexpectedly"),
        "findings": verdict.get("findings"),
        "unintended_changes": "; ".join(verdict.get("unintended_changes") or []),
        "notes": " | ".join(
            part for part in (
                verdict.get("what_changed_unexpectedly"),
                verdict.get("findings"),
            ) if part
        ),
        "error": record.get("error"),
    }
    for item in verdict.get("core_scores") or []:
        row[item["id"]] = item["score"]
        row[f"{item['id']}_rationale"] = item.get("rationale", "")
    for item in verdict.get("task_scores") or []:
        row[item["measure"]] = item["score"]
        row[f"{item['measure']}_rationale"] = item.get("rationale", "")
    return row


def write_notes(records: list[dict[str, Any]], dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "notes.md"
    lines = [
        "# Evaluation notes",
        "",
        "Each AFTER is scored against the original input BEFORE image.",
        "Judges: Gemini, Qwen-VL, OpenAI, and Seedream. Each scores every editor, including itself when it is the same company.",
        "",
        "Quantitative values are 1–5 integers. Qualitative notes are the per-item rationales, unexpected-change list, and findings.",
        "",
    ]
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        key = (record.get("condition"), record.get("variant") or record.get("tool"))
        grouped.setdefault(key, []).append(record)
    for (condition, variant), group in grouped.items():
        title = group[0].get("title") or ""
        lines.append(f"## {condition}. {title} — {DISPLAY_NAME.get(variant, variant)}")
        lines.append("")
        for record in group:
            judge = JUDGE_DISPLAY.get(record.get("judge"), record.get("judge"))
            kin = "self" if record.get("same_family") else "cross"
            if record.get("error"):
                lines.append(f"- **{judge}** ({kin}): ERROR {record['error']}")
                continue
            verdict = record.get("verdict") or {}
            lines.append(
                f"- **{judge}** ({kin}): mean {record.get('mean_score')} / 5, react {record.get('react_score')} / 5"
            )
            unexpected = verdict.get("what_changed_unexpectedly") or ""
            if unexpected:
                lines.append(f"  - Unexpected: {unexpected}")
            changes = verdict.get("unintended_changes") or []
            if changes:
                lines.append("  - Unintended: " + "; ".join(changes))
            findings = verdict.get("findings") or ""
            if findings:
                lines.append(f"  - Findings: {findings}")
            for item in (verdict.get("core_scores") or []) + (verdict.get("task_scores") or []):
                name = item.get("id") or item.get("measure")
                rationale = item.get("rationale") or ""
                lines.append(f"  - `{name}` {item.get('score')}/5 — {rationale}")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def record_key(record: dict[str, Any]) -> tuple:
    return (record.get("condition"), record.get("variant") or record.get("tool"), record.get("judge"))


def merge_records(previous: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {record_key(row): i for i, row in enumerate(previous)}
    merged = list(previous)
    for row in incoming:
        key = record_key(row)
        if key in index:
            merged[index[key]] = row
        else:
            index[key] = len(merged)
            merged.append(row)
    return merged


def write_outputs(records: list[dict[str, Any]], dest_dir: Path, merge: bool = False) -> list[dict[str, Any]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    json_path = dest_dir / "scores.json"
    csv_path = dest_dir / "scores.csv"
    if merge and json_path.exists():
        try:
            loaded = json.loads(json_path.read_text())
            if isinstance(loaded, list):
                records = merge_records(loaded, records)
        except json.JSONDecodeError:
            pass
    json_path.write_text(json.dumps(records, indent=2))
    rows = [flatten_row(r) for r in records]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    notes_path = write_notes(records, dest_dir)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {notes_path}")
    try:
        excel_path = write_excel(records, dest_dir)
        print(f"Wrote {excel_path}")
    except ImportError:
        print("Skipped Excel export (pip install openpyxl)")
    return records


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def condition_by_number(number: int):
    for condition in CONDITIONS:
        if condition.number == number:
            return condition
    raise argparse.ArgumentTypeError(f"Unknown condition {number}")


def cases_from_run(run_dir: Path, inputs: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    by_folder = {c.folder: c for c in CONDITIONS}
    by_number = {c.number: c for c in CONDITIONS}

    bundles = discover_tool_bundles(run_dir)
    if bundles:
        for bundle in bundles:
            condition = by_folder.get(bundle.condition_dir.name)
            prompt = bundle.prompt_path.read_text().strip() if bundle.prompt_path.exists() else ""
            followup = None
            meta = {}
            if bundle.meta_path.exists():
                meta = json.loads(bundle.meta_path.read_text())
                prompt = prompt or meta.get("prompt") or ""
                followup = meta.get("followup_prompt")
            before = bundle.source
            if condition:
                try:
                    before = find_source(inputs, condition)
                except FileNotFoundError:
                    if not before.exists() or before.resolve() == bundle.after.resolve():
                        continue
            elif not before.exists() or before.resolve() == bundle.after.resolve():
                continue
            if not bundle.after.exists() or not before.exists():
                continue
            extras = extras_for_case(condition, bundle.variant, bundle.extras, inputs)
            if not prompt and condition:
                prompt = condition.prompt
                followup = condition.followup_prompt
            cases.append(
                EvalCase(
                    before=before,
                    after=bundle.after,
                    prompt=prompt,
                    condition=condition.number if condition else meta.get("condition"),
                    title=(condition.title if condition else meta.get("title") or bundle.condition_dir.name),
                    tool=meta.get("tool") or bundle.variant,
                    variant=bundle.variant,
                    extras=extras,
                    followup_prompt=followup,
                    tool_dir=bundle.tool_dir,
                    judgments_dir=bundle.judgments_dir,
                )
            )
        if cases:
            return cases

    log_path = run_dir / "results.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        for row in log:
            after = Path(row.get("output") or "")
            before = Path(row.get("source") or "")
            if row.get("status") not in {None, "ok"}:
                continue
            if not after.exists() or not before.exists():
                continue
            number = row.get("condition")
            extras: list[Path] = extras_for_case(by_number.get(number), row.get("variant") or "", [], inputs)
            tool_dir = Path(row["tool_dir"]) if row.get("tool_dir") else after.parent
            cases.append(
                EvalCase(
                    before=before,
                    after=after,
                    prompt=row.get("prompt") or "",
                    condition=number,
                    title=row.get("title") or "",
                    tool=row.get("tool") or "",
                    variant=row.get("variant") or "",
                    extras=extras,
                    followup_prompt=row.get("followup_prompt"),
                    tool_dir=tool_dir,
                    judgments_dir=tool_dir / "judgments",
                )
            )
    if not cases:
        raise FileNotFoundError(f"No before/after pairs found in {run_dir}")
    return cases


def latest_run() -> Path:
    latest = OUTPUT_DIR / "latest"
    if latest.exists():
        return latest.resolve()
    runs = [p for p in OUTPUT_DIR.iterdir() if p.is_dir()] if OUTPUT_DIR.exists() else []
    if not runs:
        raise FileNotFoundError(f"No runs in {OUTPUT_DIR}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def summarize_cross_exam(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if r.get("mean_score") is not None]
    per_condition: dict[int, dict[str, Any]] = {}
    overall: dict[str, dict[str, list[float]]] = {}

    for record in scored:
        tool = record.get("variant") or record.get("tool") or "custom"
        condition = record.get("condition") if record.get("condition") is not None else -1
        slot = per_condition.setdefault(
            condition,
            {"title": record.get("title"), "tools": {}},
        )
        tool_slot = slot["tools"].setdefault(
            tool,
            {"all": [], "cross": [], "react": [], "cross_react": []},
        )
        mean = float(record["mean_score"])
        react = float(record.get("react_score") or mean)
        tool_slot["all"].append(mean)
        tool_slot["react"].append(react)
        overall.setdefault(tool, {"all": [], "cross": [], "react": [], "cross_react": []})
        overall[tool]["all"].append(mean)
        overall[tool]["react"].append(react)
        if not record.get("same_family"):
            tool_slot["cross"].append(mean)
            tool_slot["cross_react"].append(react)
            overall[tool]["cross"].append(mean)
            overall[tool]["cross_react"].append(react)

    def avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    def pack(bucket: dict[str, list[float]]) -> dict[str, float | None]:
        return {
            "mean_all_judges": avg(bucket["all"]),
            "mean_cross_exam": avg(bucket["cross"]) or avg(bucket["all"]),
            "react_all_judges": avg(bucket["react"]),
            "react_cross_exam": avg(bucket["cross_react"]) or avg(bucket["react"]),
        }

    condition_winners = []
    for number, slot in sorted(per_condition.items(), key=lambda item: item[0]):
        ranked = sorted(
            (
                {"tool": tool, **pack(bucket)}
                for tool, bucket in slot["tools"].items()
            ),
            key=lambda row: row["react_all_judges"] or 0,
            reverse=True,
        )
        condition_winners.append(
            {
                "condition": number,
                "title": slot["title"],
                "winner": ranked[0]["tool"] if ranked else None,
                "ranking": ranked,
            }
        )

    overall_ranked = sorted(
        ({"tool": tool, **pack(bucket)} for tool, bucket in overall.items()),
        key=lambda row: row["react_all_judges"] or 0,
        reverse=True,
    )
    return {
        "best_for_react_loop": overall_ranked[0]["tool"] if overall_ranked else None,
        "overall": overall_ranked,
        "by_condition": condition_winners,
    }


def print_ranking(summary: dict[str, Any]) -> None:
    print("\n=== Ranking (every editor judges every result, including itself) ===")
    print("all = Gemini + Qwen + OpenAI + Seedream. cross = judges from a different company.")
    print("react weights locality, controllability, identity, recoverability.\n")
    print(f"{'Tool':22} {'all':>6} {'cross':>6} {'react':>6}")
    for row in summary.get("overall") or []:
        name = DISPLAY_NAME.get(row["tool"], row["tool"])
        print(
            f"{name:22} "
            f"{row['mean_all_judges'] or 0:6.2f} "
            f"{row['mean_cross_exam'] or 0:6.2f} "
            f"{row['react_all_judges'] or 0:6.2f}"
        )
    winner = summary.get("best_for_react_loop")
    if winner:
        print(f"\nBest for a reason-and-act loop: {DISPLAY_NAME.get(winner, winner)}")
    print("\nPer condition")
    for slot in summary.get("by_condition") or []:
        winner = slot.get("winner")
        print(f"  {slot.get('condition')}. {slot.get('title')}: {DISPLAY_NAME.get(winner, winner)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, help="Original image")
    parser.add_argument("--after", type=Path, help="Edited image")
    parser.add_argument("--prompt", help="Edit instruction that was requested")
    parser.add_argument("--condition", type=int, help="Deck condition number 1–10 (sets prompt and measures)")
    parser.add_argument("--extra", type=Path, action="append", default=[], help="Markup or edge-map reference image")
    parser.add_argument("--run", type=Path, help="run_edits.py output folder, or 'latest'")
    parser.add_argument("--inputs", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, help="Where to write scores.json / scores.csv")
    parser.add_argument("--judge", choices=("auto", "openai", "gemini", "qwen", "seedream"), help="Single judge override")
    parser.add_argument("--judges", help="Override PPT defaults, e.g. openai,gemini,qwen")
    parser.add_argument("--conditions", help="Comma-separated condition numbers, e.g. 8,9,10")
    parser.add_argument("--tools", help="Only evaluate these editor variants, e.g. nano_banana,gpt_image_2")
    parser.add_argument(
        "--skip-tools",
        default="flux_kontext,ideogram",
        help="Editor variants to skip (default: flux_kontext,ideogram)",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Max parallel judge calls")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse judgment JSON files that already exist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-prompt-keys", action="store_true")
    args = parser.parse_args()

    load_env()
    override: list[str] | None = None
    if args.judges:
        override = [part.strip() for part in args.judges.split(",") if part.strip()]
    elif args.judge and args.judge != "auto":
        override = [args.judge]

    cases: list[EvalCase] = []
    run_dir: Path | None = None
    if args.run:
        try:
            run_dir = latest_run() if str(args.run) == "latest" else args.run
            cases = cases_from_run(run_dir, args.inputs)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        dest = args.output or (run_dir / "evaluation")
    elif args.before and args.after:
        prompt = args.prompt
        title = "Custom edit"
        followup = None
        if args.condition:
            condition = condition_by_number(args.condition)
            title = condition.title
            prompt = prompt or condition.prompt
            followup = condition.followup_prompt
        if not prompt:
            parser.error("Provide --prompt or --condition so the judge knows what was requested")
        if not args.before.exists() or not args.after.exists():
            parser.error("Both --before and --after must exist")
        cases = [
            EvalCase(
                before=args.before,
                after=args.after,
                prompt=prompt,
                condition=args.condition,
                title=title,
                extras=list(args.extra),
                followup_prompt=followup,
            )
        ]
        dest = args.output or (
            ROOT / "evaluations" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    else:
        parser.error("Pass --before and --after, or --run <output-folder>")

    wanted_tools = (
        {part.strip() for part in args.tools.split(",") if part.strip()}
        if args.tools
        else set(COMPARISON_TOOLS)
    )
    skipped_tools = {part.strip() for part in (args.skip_tools or "").split(",") if part.strip()}
    skipped_tools.update({"flux_kontext", "ideogram"})
    cases = [
        case
        for case in cases
        if (case.variant or case.tool) in wanted_tools
        and (case.variant or case.tool) not in skipped_tools
    ]
    if args.conditions:
        wanted_conditions = {int(part.strip()) for part in args.conditions.split(",") if part.strip()}
        cases = [case for case in cases if case.condition in wanted_conditions]
    partial_run = bool(args.tools or args.conditions or args.judges or (args.judge and args.judge != "auto"))

    print(
        "Each AFTER is compared to the original input BEFORE image, then scored by "
        "Gemini, Qwen-VL, OpenAI, and Seedream (self + cross)."
    )
    print("FLUX and Ideogram are excluded.")
    print(f"Cases: {len(cases)}")
    if override:
        print(f"Override judges: {', '.join(override)}")
    if args.dry_run:
        for case in cases:
            planned = judges_for_case(case, override)
            print(f"  {case.title}  {case.variant or case.after.name}")
            print(f"    judges: {', '.join(JUDGE_DISPLAY.get(j, j) for j in planned)}")
            print(f"    before: {case.before}")
            print(f"    after:  {case.after}")
            if case.extras:
                print(f"    extras: {', '.join(str(p) for p in case.extras)}")
        return 0

    needed: list[str] = []
    for case in cases:
        for name in judges_for_case(case, override):
            if name not in needed:
                needed.append(name)
    judges = available_judges(needed, prompt_missing=not args.no_prompt_keys)
    print(f"Active judges: {', '.join(JUDGE_DISPLAY.get(j, j) for j in judges)}")

    work: list[tuple[EvalCase, str, str]] = []
    records: list[dict[str, Any]] = []
    for case in cases:
        planned = [name for name in judges_for_case(case, override) if name in judges]
        if not planned:
            print(f"Skip {case.variant or case.after.name}: no API key for judges")
            continue
        for provider in planned:
            existing = None
            if args.skip_existing and case.judgments_dir:
                existing_path = case.judgments_dir / f"{provider}.json"
                if existing_path.exists():
                    existing = json.loads(existing_path.read_text())
                    records.append(existing)
                    print(f"  skip existing {case.variant} / {provider}")
                    continue
            work.append((case, provider, judges[provider]))

    print_lock = Lock()
    judge_gates = {name: Semaphore(2) for name in JUDGES}
    judge_gates["seedream"] = Semaphore(4)

    def run_one(item: tuple[EvalCase, str, str]) -> dict[str, Any]:
        case, provider, api_key = item
        label = f"{case.variant or case.after.name} / {JUDGE_DISPLAY.get(provider, provider)}"
        last_error: Exception | None = None
        with judge_gates[provider]:
            for attempt in range(1, 4):
                try:
                    with print_lock:
                        print(f"Scoring {label}...")
                    record = evaluate_case(case, provider, api_key)
                    with print_lock:
                        print_record(record)
                    return record
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < 3:
                        wait = 5 * attempt
                        with print_lock:
                            print(f"  retry {attempt}/3 [{label}] in {wait}s: {exc}")
                        time.sleep(wait)
        with print_lock:
            print(f"  ERROR [{label}]: {last_error}")
        return {
            "condition": case.condition,
            "title": case.title,
            "tool": case.tool,
            "variant": case.variant,
            "judge": provider,
            "same_family": is_same_family(case.tool or case.variant, provider),
            "before": str(case.before),
            "after": str(case.after),
            "error": str(last_error),
            "mean_score": None,
            "react_score": None,
            "verdict": {},
        }

    if work:
        workers = max(1, min(args.jobs, len(work)))
        print(f"Running {len(work)} judge call(s) with {workers} worker(s)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, item) for item in work]
            for future in as_completed(futures):
                records.append(future.result())

    records = write_outputs(records, dest, merge=partial_run)
    summary = summarize_cross_exam(records)
    write_json(dest / "ranking.json", summary)
    print_ranking(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
