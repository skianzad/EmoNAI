#!/usr/bin/env python3
"""Reason-and-act loop over a finished experiment folder.

Reads cross-exam judgments from evaluate_edits.py, turns them into a correction
prompt (reason), then asks the editor to fix its own after-image (act).

    python run_edits.py
    python evaluate_edits.py --run latest --judges openai,gemini
    python react_loop.py --run latest

Default: every editor is corrected from the other AIs' critique of its output.
Use --only-winner to re-run only the tool ranked best for the ReAct loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_edits import latest_run
from layout import discover_tool_bundles, write_json
from run_edits import DISPLAY_NAME, EDITORS, EditJob, load_keys


def critiques(judgments_dir: Path, cross_only: bool) -> list[dict]:
    notes = []
    if not judgments_dir.exists():
        return notes
    for path in sorted(judgments_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if cross_only and data.get("same_family"):
            continue
        notes.append(data)
    if not notes and judgments_dir.exists():
        notes = [json.loads(p.read_text()) for p in sorted(judgments_dir.glob("*.json"))]
    return notes


def correction_prompt(original: str, reviews: list[dict]) -> str:
    blocks = []
    for review in reviews:
        verdict = review.get("verdict") or {}
        kin = "independent judge" if not review.get("same_family") else "same-family judge"
        unexpected = verdict.get("what_changed_unexpectedly") or "none listed"
        findings = verdict.get("findings") or ""
        extras = verdict.get("unintended_changes") or []
        extra_txt = "; ".join(extras) if extras else "none listed"
        blocks.append(
            f"- {review.get('judge')} ({kin}): {findings}\n"
            f"  Unexpected: {unexpected}\n"
            f"  Unintended: {extra_txt}"
        )
    critique = "\n".join(blocks) if blocks else "No written critique; make the original edit more local and precise."
    return (
        "This photograph was already edited. Independent evaluators reviewed it.\n\n"
        f"Original requested edit:\n{original}\n\n"
        f"Evaluator critique:\n{critique}\n\n"
        "REASON then ACT: keep the original requested edit, reverse only the "
        "unintended changes, and leave everything else untouched. Do not restyle "
        "the photo or introduce new objects."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="latest", help="Experiment folder or 'latest'")
    parser.add_argument("--round", type=int, default=1, help="ReAct round number (folder name)")
    parser.add_argument("--only-winner", action="store_true", help="Only re-edit the ranked ReAct winner")
    parser.add_argument("--include-same-family", action="store_true", help="Use same-family judges in the critique")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-prompt-keys", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    run_dir = latest_run() if str(args.run) == "latest" else Path(args.run)
    ranking_path = run_dir / "evaluation" / "ranking.json"
    winner = None
    if ranking_path.exists():
        ranking = json.loads(ranking_path.read_text())
        winner = ranking.get("best_for_react_loop")
        print(f"Ranked ReAct winner: {DISPLAY_NAME.get(winner, winner)}")
    else:
        print("No evaluation/ranking.json yet — correcting every editor that has judgments.")

    bundles = discover_tool_bundles(run_dir)
    jobs = []
    for bundle in bundles:
        if not bundle.after.exists():
            continue
        tool = bundle.variant
        if bundle.meta_path.exists():
            meta = json.loads(bundle.meta_path.read_text())
            tool = meta.get("tool") or bundle.variant
        if args.only_winner and winner and tool != winner and bundle.variant != winner:
            continue
        reviews = critiques(bundle.judgments_dir, cross_only=not args.include_same_family)
        if not reviews:
            print(f"  skip {bundle.variant}: no judgments (run evaluate_edits.py first)")
            continue
        original = bundle.prompt_path.read_text().strip() if bundle.prompt_path.exists() else ""
        prompt = correction_prompt(original, reviews)
        dest = bundle.react_dir / f"round_{args.round:02d}" / "after.png"
        jobs.append((tool, bundle, prompt, dest, reviews))

    if not jobs:
        print("Nothing to correct.")
        return 1

    print(f"ReAct round {args.round}: {len(jobs)} editor(s)")
    if args.dry_run:
        for tool, bundle, prompt, dest, _ in jobs:
            print(f"  {DISPLAY_NAME.get(tool, tool)}  {bundle.after} -> {dest}")
            print(f"    {prompt.splitlines()[0]}")
        return 0

    keys = load_keys(prompt_missing=not args.no_prompt_keys)
    for tool, bundle, prompt, dest, reviews in jobs:
        if tool not in EDITORS:
            print(f"  skip {tool}: not an editor")
            continue
        print(f"\nACT  {DISPLAY_NAME.get(tool, tool)}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        (dest.parent / "prompt.txt").write_text(prompt)
        write_json(
            dest.parent / "critique.json",
            [{"judge": r.get("judge"), "same_family": r.get("same_family"), "verdict": r.get("verdict")} for r in reviews],
        )
        try:
            EDITORS[tool](
                keys,
                EditJob(source=bundle.after, prompt=prompt, extra_images=bundle.extras),
                dest,
            )
            write_json(dest.parent / "meta.json", {"tool": tool, "source": str(bundle.after), "output": str(dest)})
            print(f"  saved {dest}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            write_json(dest.parent / "meta.json", {"tool": tool, "error": str(exc)})
    print("\nRe-run evaluation on the react after.png files if you want another judge pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
