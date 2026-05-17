from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import baseline_answer
from src.orchestrator import schemasense
from src.vlm import ask_vlm

QUESTIONS_PATH = ROOT / "data" / "questions.json"
DIAGRAMS_DIR = ROOT / "data" / "diagrams"
OUTPUTS_DIR = ROOT / "outputs"
EVAL_OUTPUTS_DIR = OUTPUTS_DIR / "eval"
PARTIAL_PATH = EVAL_OUTPUTS_DIR / "partial.json"
RESULTS_PATH = EVAL_OUTPUTS_DIR / "results.json"
TABLE_PATH = OUTPUTS_DIR / "results_table.md"

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

RULE_GRADED_TYPES = {"counting", "specification"}


def grade(question: str, gold: str, predicted: str, qtype: str = "") -> tuple[bool, str]:
    """Return (correct, judge_kind). Rule-based for counting/specification, LLM otherwise.

    The LLM judge is the same Qwen3-VL that produced the answers, so we route
    every question we can to a deterministic rule first and only ask the model
    when the answer space is open-ended.
    """
    if qtype in RULE_GRADED_TYPES:
        rule_result = _rule_grade(qtype, gold, predicted)
        if rule_result is not None:
            return rule_result, f"rule:{qtype}"

    return _llm_grade(question, gold, predicted), "llm"


def _rule_grade(qtype: str, gold: str, predicted: str) -> bool | None:
    if qtype == "counting":
        gold_n = _first_number(gold)
        pred_n = _first_number(predicted)
        if gold_n is None or pred_n is None:
            return None
        return gold_n == pred_n

    if qtype == "specification":
        gold_clean = _normalize(gold)
        pred_clean = _normalize(predicted)
        if not gold_clean or not pred_clean:
            return None
        gold_value, gold_unit = _split_value_unit(gold_clean)
        pred_value, pred_unit = _split_value_unit(pred_clean)
        if gold_value is not None and pred_value is not None:
            if not _close(gold_value, pred_value):
                return False
            if gold_unit and pred_unit and gold_unit != pred_unit:
                return False
            return True
        return gold_clean in pred_clean or pred_clean in gold_clean

    return None


def _llm_grade(question: str, gold: str, predicted: str) -> bool:
    prompt = (
        "You are grading a short-answer prediction for an engineering diagram question.\n"
        "Reply with only YES or NO. Treat synonyms, abbreviations, and minor wording "
        "differences as correct. Treat substantively wrong answers as NO.\n"
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {predicted}\n"
    )
    result = ask_vlm(prompt, image=None, temperature=0)
    return result.strip().upper().startswith("YES")


def _first_number(value: str) -> int | None:
    text = str(value or "").lower()
    match = re.search(r"-?\d+", text)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return number
    return None


def _normalize(value: str) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _split_value_unit(text: str) -> tuple[float | None, str]:
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-z%°/]*[a-z%/]?)\s*$", text)
    if not match:
        return None, ""
    try:
        return float(match.group(1)), match.group(2).strip()
    except ValueError:
        return None, ""


def _close(a: float, b: float, tol: float = 0.05) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tol


def run_eval() -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []
    OUTPUTS_DIR.mkdir(exist_ok=True)
    EVAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(questions, start=1):
        image_path = DIAGRAMS_DIR / item["diagram"]
        question = str(item["question"])
        gold = str(item["answer"])
        qtype = str(item.get("type", ""))
        started = time.perf_counter()
        ss = schemasense(str(image_path), question, question_type=qtype)
        bl = _baseline_from_schemasense(ss) or baseline_answer(str(image_path), question)
        ss_correct, ss_judge = grade(question, gold, ss["answer"], qtype)
        bl_correct, bl_judge = grade(question, gold, bl["answer"], qtype)
        row = {
            **item,
            "predicted_ss": ss["answer"],
            "predicted_bl": bl["answer"],
            "ss_correct": ss_correct,
            "bl_correct": bl_correct,
            "ss_judge": ss_judge,
            "bl_judge": bl_judge,
            "ss_seconds": ss.get("timing", {}).get("total_seconds", 0),
            "bl_seconds": bl.get("elapsed_seconds", 0),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        rows.append(row)
        if index % 5 == 0:
            PARTIAL_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    TABLE_PATH.write_text(results_table(rows), encoding="utf-8")
    PARTIAL_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def results_table(rows: list[dict]) -> str:
    ss_total = sum(1 for row in rows if row["ss_correct"])
    bl_total = sum(1 for row in rows if row["bl_correct"])
    lines = [
        "| ID | Diagram | Type | Gold | SchemaSense | SS OK | Baseline | BL OK | Judge |",
        "|---|---|---|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {diagram} | {type} | {gold} | {ss} | {ss_ok} | {bl} | {bl_ok} | {judge} |".format(
                id=_cell(row.get("id", "")),
                diagram=_cell(row.get("diagram", "")),
                type=_cell(row.get("type", "")),
                gold=_cell(row.get("answer", "")),
                ss=_cell(row.get("predicted_ss", "")),
                ss_ok="YES" if row.get("ss_correct") else "NO",
                bl=_cell(row.get("predicted_bl", "")),
                bl_ok="YES" if row.get("bl_correct") else "NO",
                judge=_cell(row.get("ss_judge", "")),
            )
        )
    lines.append("")
    lines.append(f"SchemaSense accuracy: {ss_total}/{len(rows)}")
    lines.append(f"Baseline accuracy: {bl_total}/{len(rows)}")
    lines.extend(["", "## Per-type accuracy", "", "| Type | SchemaSense | Baseline | N |", "|---|---:|---:|---:|"])
    for qtype, ss_hits, bl_hits, total in _per_type_breakdown(rows):
        lines.append(f"| {qtype} | {ss_hits}/{total} | {bl_hits}/{total} | {total} |")
    return "\n".join(lines) + "\n"


def _baseline_from_schemasense(result: dict) -> dict | None:
    visual = result.get("visual_answer") if isinstance(result, dict) else None
    if not isinstance(visual, dict):
        return None
    answer = str(visual.get("answer", "")).strip()
    if not answer:
        return None
    timing = result.get("timing", {}) if isinstance(result.get("timing"), dict) else {}
    return {
        "answer": answer,
        "elapsed_seconds": visual.get("elapsed_seconds", timing.get("visual_fallback_seconds", 0)),
    }


def _per_type_breakdown(rows: list[dict]) -> list[tuple[str, int, int, int]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("type") or "unknown")].append(row)
    return [
        (qtype, sum(1 for r in items if r.get("ss_correct")), sum(1 for r in items if r.get("bl_correct")), len(items))
        for qtype, items in sorted(by_type.items())
    ]


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


if __name__ == "__main__":
    print(json.dumps({"results": len(run_eval()), "table": str(TABLE_PATH)}, indent=2))
