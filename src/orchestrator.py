from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.connector import build_graph, graph_to_dict
from src.agents.reasoner import answer_question
from src.pipeline_spotting import spot_all_symbols


CACHE_DIR = ROOT / "data" / "cache"
DIAGRAMS_DIR = ROOT / "data" / "diagrams"
VISUAL_FIRST_TYPES = {"counting", "specification", "anomaly"}


def schemasense(
    image_path: str,
    question: str,
    use_cache: bool = True,
    question_type: str = "",
    use_hybrid: bool = True,
) -> dict[str, Any]:
    """Run the full SchemaSense pipeline. Returns answer + metadata."""
    started_total = time.perf_counter()
    image_file = _resolve_path(image_path)
    image_hash = _md5_file(image_file)
    detections_path = CACHE_DIR / f"{image_hash}_detections.json"
    graph_path = CACHE_DIR / f"{image_hash}_graph.pkl"
    cache_hit = False
    spotting_seconds = 0.0
    graph_seconds = 0.0

    with Image.open(image_file) as opened:
        image = opened.copy()
        image.filename = str(image_file)

    if use_cache and detections_path.exists() and graph_path.exists():
        detections = _load_json(detections_path)
        graph = _load_graph(graph_path)
        cache_hit = True
    else:
        spot_image = image.copy()
        spot_image.filename = ""
        started = time.perf_counter()
        detections = spot_all_symbols(spot_image, use_cache=False)
        spotting_seconds = time.perf_counter() - started
        _save_json(detections_path, detections)

        started = time.perf_counter()
        graph = build_graph(image, detections)
        graph_seconds = time.perf_counter() - started
        _save_graph(graph_path, graph)

    started = time.perf_counter()
    graph_answer = answer_question(question, graph, image)
    answer = graph_answer
    visual_answer: dict[str, Any] | None = None
    hybrid_decision = {
        "enabled": use_hybrid,
        "question_type": _question_type(question, question_type),
        "graph_weak": _graph_is_weak(graph, _question_type(question, question_type), graph_answer),
        "used_visual_fallback": False,
        "final_answer_source": "graph",
        "reason": "Graph answer used.",
    }
    baseline_seconds = 0.0

    if use_hybrid:
        started_visual = time.perf_counter()
        answer, visual_answer, hybrid_decision = _hybrid_answer(
            image_file=image_file,
            question=question,
            question_type=hybrid_decision["question_type"],
            graph=graph,
            graph_answer=graph_answer,
            detections=detections,
            graph_weak=hybrid_decision["graph_weak"],
        )
        baseline_seconds = time.perf_counter() - started_visual if visual_answer is not None else 0.0

    reasoning_seconds = time.perf_counter() - started
    total_seconds = time.perf_counter() - started_total

    connector_warnings = list(graph.graph.get("warnings", []))
    connector_queries = dict(graph.graph.get("queries", {"attempted": 0, "failed": 0}))

    return {
        "answer": str(answer.get("answer", "unknown")),
        "reasoning": str(answer.get("reasoning", "")),
        "confidence": float(answer.get("confidence", 0.0)),
        "used_image_lookup": bool(answer.get("used_image_lookup", False)),
        "graph_answer": graph_answer,
        "visual_answer": visual_answer,
        "hybrid_decision": hybrid_decision,
        "graph_stats": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        "n_detections": len(detections),
        "detections": detections,
        "graph": graph_to_dict(graph),
        "image_hash": image_hash,
        "timing": {
            "spotting_seconds": round(spotting_seconds, 3),
            "graph_seconds": round(graph_seconds, 3),
            "reasoning_seconds": round(reasoning_seconds, 3),
            "visual_fallback_seconds": round(baseline_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        },
        "cache_hit": cache_hit,
        "connector_warnings": connector_warnings,
        "connector_queries": connector_queries,
    }


def batch_run(questions: list[dict[str, Any]], use_cache: bool = True) -> list[dict[str, Any]]:
    """Run schemasense on a list of question dicts from questions.json."""
    augmented: list[dict[str, Any]] = []
    for item in tqdm(questions, desc="SchemaSense batch"):
        row = dict(item)
        diagram = str(row.get("diagram", ""))
        image_path = str(DIAGRAMS_DIR / diagram)
        result = schemasense(
            image_path=image_path,
            question=str(row.get("question", "")),
            use_cache=use_cache,
            question_type=str(row.get("type", "")),
        )
        row["predicted_answer"] = result["answer"]
        row["timing"] = result["timing"]
        row["confidence"] = result["confidence"]
        row["used_image_lookup"] = result["used_image_lookup"]
        row["graph_stats"] = result["graph_stats"]
        row["cache_hit"] = result["cache_hit"]
        augmented.append(row)
    return augmented


def _hybrid_answer(
    image_file: Path,
    question: str,
    question_type: str,
    graph: Any,
    graph_answer: dict[str, Any],
    detections: list[dict[str, Any]],
    graph_weak: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    should_use_visual, reason = _should_use_visual_fallback(
        question_type=question_type,
        graph=graph,
        graph_answer=graph_answer,
        detections=detections,
        graph_weak=graph_weak,
    )
    decision = {
        "enabled": True,
        "question_type": question_type,
        "graph_weak": graph_weak,
        "used_visual_fallback": should_use_visual,
        "final_answer_source": "graph",
        "reason": reason,
    }
    if not should_use_visual:
        return graph_answer, None, decision

    from src.baseline import baseline_answer

    visual = baseline_answer(str(image_file), question)
    visual_payload = {
        "answer": str(visual.get("answer", "unknown")).strip() or "unknown",
        "reasoning": "Full-image visual fallback from the local VLM baseline.",
        "confidence": _visual_confidence(visual, graph_answer, graph_weak),
        "used_image_lookup": True,
        "elapsed_seconds": visual.get("elapsed_seconds", 0),
    }
    final = _select_hybrid_answer(question_type, graph_answer, visual_payload, graph_weak)
    decision["final_answer_source"] = "visual_fallback" if final is visual_payload else "graph"
    if final is visual_payload:
        decision["reason"] = f"{reason} The final answer uses full-image visual evidence."
    else:
        decision["reason"] = f"{reason} The graph answer was kept because it was stronger."
    return final, visual_payload, decision


def _should_use_visual_fallback(
    question_type: str,
    graph: Any,
    graph_answer: dict[str, Any],
    detections: list[dict[str, Any]],
    graph_weak: bool,
) -> tuple[bool, str]:
    answer_text = str(graph_answer.get("answer", "")).strip().lower()
    confidence = _confidence(graph_answer)
    if question_type in VISUAL_FIRST_TYPES:
        return True, f"{question_type} questions are routed to full-image visual evidence."
    if question_type == "identification" and (graph_weak or confidence < 0.75):
        return True, "Identification answer needs visual verification because graph evidence is weak or low-confidence."
    if question_type == "connectivity" and (graph.number_of_edges() == 0 or confidence < 0.7 or answer_text in {"", "unknown", "none"}):
        return True, "Connectivity graph has too little edge evidence or low confidence."
    if graph_weak or confidence < 0.5 or answer_text in {"", "unknown"}:
        return True, "Graph answer is weak, unknown, or low-confidence."
    if len(detections) == 0:
        return True, "No detections were produced, so full-image evidence is required."
    return True, "Full-image visual verifier is used to prevent graph-only error propagation."


def _select_hybrid_answer(
    question_type: str,
    graph_answer: dict[str, Any],
    visual_answer: dict[str, Any],
    graph_weak: bool,
) -> dict[str, Any]:
    visual_text = str(visual_answer.get("answer", "")).strip()
    graph_text = str(graph_answer.get("answer", "")).strip()
    if not visual_text:
        return graph_answer
    if visual_text.lower() in {"unknown", "none"} and graph_text:
        return graph_answer
    if not graph_text or graph_text.lower() in {"unknown", "none"}:
        return visual_answer
    return visual_answer


def _question_type(question: str, explicit: str = "") -> str:
    normalized = str(explicit or "").strip().lower()
    if normalized:
        return normalized
    text = question.strip().lower()
    if text.startswith("how many") or "number of" in text:
        return "counting"
    if any(token in text for token in ("pressure", "voltage", "value", "rating", "label")) and (
        "what is" in text or "written" in text
    ):
        return "specification"
    if "missing" in text or text.startswith("is "):
        return "anomaly"
    if any(token in text for token in ("connect", "connected", "receives", "feeds", "follows", "after", "discharge", "leaves", "sends")):
        return "connectivity"
    if any(token in text for token in ("which", "what", "where")):
        return "identification"
    return "unknown"


def _graph_is_weak(graph: Any, question_type: str, answer: dict[str, Any]) -> bool:
    if graph.number_of_nodes() == 0:
        return True
    if question_type == "connectivity" and graph.number_of_edges() == 0:
        return True
    if graph.number_of_nodes() < 2 and question_type in {"connectivity", "counting", "identification"}:
        return True
    return _confidence(answer) < 0.45


def _confidence(answer: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(answer.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _visual_confidence(visual: dict[str, Any], graph_answer: dict[str, Any], graph_weak: bool) -> float:
    if "confidence" in visual:
        return _confidence(visual)

    visual_text = str(visual.get("answer", "")).strip().lower()
    if not visual_text or visual_text in {"unknown", "none", "n/a"}:
        return 0.0

    graph_text = str(graph_answer.get("answer", "")).strip().lower()
    if graph_text and graph_text == visual_text:
        return round(max(0.65, min(0.9, _confidence(graph_answer))), 2)

    return 0.52 if graph_weak else 0.6


def _resolve_path(path: str) -> Path:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_graph(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_graph(path: Path, graph: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SchemaSense pipeline.")
    parser.add_argument("--image", required=True, help="Path to a diagram image.")
    parser.add_argument("--question", required=True, help="Question to answer about the diagram.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing MD5 cache and recompute.")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable full-image visual fallback and verification.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = schemasense(
        image_path=args.image,
        question=args.question,
        use_cache=not args.no_cache,
        use_hybrid=not args.no_hybrid,
    )
    print(json.dumps(result, indent=2))
