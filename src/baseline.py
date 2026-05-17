from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.vlm import ask_vlm, ask_vlm_json


def baseline_answer(image_path: str, question: str) -> dict:
    """Single-shot: full image + question -> answer. No agents."""
    image_file = _resolve_path(image_path)
    json_prompt = (
        "You are an expert process engineer. Looking at this engineering "
        "diagram, answer this question precisely and briefly.\n"
        f"Question: {question}\n"
        "Return JSON only with this shape:\n"
        '{"answer": "short answer", "confidence": 0.0}\n'
        "The confidence must be your self-rated certainty from 0.0 to 1.0."
    )
    started = time.perf_counter()
    with Image.open(image_file) as opened:
        image = opened.copy()

    try:
        result = ask_vlm_json(json_prompt, image=image, retries=1, temperature=0.0, max_tokens=192)
        if isinstance(result, dict):
            answer = str(result.get("answer", "")).strip()
            if answer:
                return {
                    "answer": answer,
                    "confidence": _clamp_confidence(result.get("confidence"), _fallback_confidence(answer)),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
    except Exception as exc:
        print(f"[baseline] JSON confidence response failed; falling back to answer-only mode: {exc}")

    prompt = (
        "You are an expert process engineer. Looking at this engineering "
        "diagram, answer this question precisely and briefly.\n"
        f"Question: {question}\n"
        "Answer with just the answer, no explanation."
    )
    answer = ask_vlm(prompt, image=image, temperature=0.0, max_tokens=128).strip()
    return {
        "answer": answer,
        "confidence": _fallback_confidence(answer),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def baseline_batch(questions: list[dict]) -> list[dict]:
    """Run baseline on a list of question dicts. Adds predicted_answer."""
    rows: list[dict] = []
    for item in questions:
        row = dict(item)
        image_path = row.get("image_path") or ROOT / "data" / "diagrams" / str(row.get("diagram", ""))
        result = baseline_answer(str(image_path), str(row.get("question", "")))
        row["predicted_answer"] = result["answer"]
        row["elapsed_seconds"] = result["elapsed_seconds"]
        rows.append(row)
    return rows


def _resolve_path(path: str) -> Path:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def _clamp_confidence(value: Any, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))


def _fallback_confidence(answer: str) -> float:
    text = str(answer or "").strip().lower()
    if not text or text in {"unknown", "none", "n/a", "not visible"}:
        return 0.0
    if any(token in text for token in ("unclear", "cannot determine", "not sure", "appears")):
        return 0.45
    return 0.6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the single-shot SchemaSense baseline.")
    parser.add_argument("--image", required=True, help="Path to a diagram image.")
    parser.add_argument("--question", required=True, help="Question to answer about the diagram.")
    args = parser.parse_args()
    print(json.dumps(baseline_answer(args.image, args.question), indent=2))
