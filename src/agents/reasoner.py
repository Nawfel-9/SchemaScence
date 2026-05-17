from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import networkx as nx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.connector import graph_to_text
from src.vlm import ask_vlm_json


def answer_question(
    question: str,
    graph: nx.Graph,
    image: Image.Image,
    max_reflections: int = 2,
) -> dict[str, Any]:
    try:
        graph_text = graph_to_text(graph)
        first_result = _ask_graph_only(question, graph_text)
        action = _action(first_result)
        reasoning_parts = [_reasoning(first_result)]

        if action == "answer":
            return _answer_payload(first_result, used_image_lookup=False, n_reflections=0, reasoning_parts=reasoning_parts)

        lookup_target = str(first_result.get("lookup_target", "")).strip()
        used_image_lookup = False

        for reflection in range(1, max_reflections + 1):
            used_image_lookup = True
            matched_nodes, match_quality = _match_nodes(graph, lookup_target)
            crop = _crop_for_lookup(image, graph, matched_nodes, match_quality)
            excerpt = _graph_excerpt(graph, matched_nodes) if matched_nodes else graph_text
            lookup_result = _ask_visual_lookup(question, lookup_target, excerpt, crop, match_quality)
            reasoning_parts.append(_reasoning(lookup_result))

            if _action(lookup_result) == "answer":
                return _answer_payload(
                    lookup_result,
                    used_image_lookup=used_image_lookup,
                    n_reflections=reflection,
                    reasoning_parts=reasoning_parts,
                )

            lookup_target = str(lookup_result.get("lookup_target") or lookup_target).strip()

        return {
            "answer": "unknown",
            "reasoning": " ".join(part for part in reasoning_parts if part).strip()
            or "Not enough evidence in the graph/crop.",
            "used_image_lookup": True,
            "n_reflections": max_reflections,
            "confidence": 0.0,
        }
    except Exception as exc:
        return {
            "answer": "unknown",
            "reasoning": f"Reasoner failed safely: {str(exc).splitlines()[0]}",
            "used_image_lookup": False,
            "n_reflections": 0,
            "confidence": 0.0,
        }


def _ask_graph_only(question: str, graph_text: str) -> dict[str, Any]:
    prompt = f"""You are an expert process engineer analyzing a diagram.
You have a structural graph of the diagram components.
Use the graph to answer questions. If the graph does not contain enough information, request a visual lookup.

GRAPH DESCRIPTION:
{graph_text}

QUESTION: {question}

Instructions:
- If you can answer from the graph alone, return:
  {{"action": "answer", "answer": "...", "confidence": 0.9, "reasoning": "..."}}
- If you need to look at the image, return:
  {{"action": "lookup", "lookup_target": "node label or region description", "reasoning": "why you need to look"}}
Return ONLY JSON."""
    result = ask_vlm_json(prompt, image=None, retries=1, temperature=0.0, max_tokens=512)
    return result if isinstance(result, dict) else {}


def _ask_visual_lookup(
    question: str,
    lookup_target: str,
    graph_excerpt: str,
    crop: Image.Image,
    match_quality: str,
) -> dict[str, Any]:
    scope = "full diagram" if match_quality == "full_image" else "diagram region"
    prompt = f"""Given this {scope} of the diagram, answer: {question}

Lookup target: {lookup_target or "unspecified"}

Context from graph:
{graph_excerpt}

Return JSON:
{{"action": "answer", "answer": "...", "confidence": 0.8, "reasoning": "..."}}"""
    result = ask_vlm_json(prompt, image=crop, retries=1, temperature=0.0, max_tokens=512)
    return result if isinstance(result, dict) else {}


def _answer_payload(
    result: dict[str, Any],
    used_image_lookup: bool,
    n_reflections: int,
    reasoning_parts: list[str],
) -> dict[str, Any]:
    return {
        "answer": str(result.get("answer", "unknown")).strip() or "unknown",
        "reasoning": " ".join(part for part in reasoning_parts if part).strip(),
        "used_image_lookup": used_image_lookup,
        "n_reflections": n_reflections,
        "confidence": _clamp_confidence(result.get("confidence", 0.0)),
    }


def _match_nodes(graph: nx.Graph, lookup_target: str) -> tuple[list[Any], str]:
    """Return (matched_nodes, match_quality).

    match_quality is one of: "matched", "weak", "full_image".
    "weak" means the best score was zero (no useful textual overlap).
    "full_image" means there were no nodes or no usable target — the caller
    should crop the full image rather than fabricate a region.
    """
    if graph.number_of_nodes() == 0:
        return [], "full_image"

    target = lookup_target.casefold().strip()
    if not target:
        return [], "full_image"

    target_tokens = _tokens(target)
    scored: list[tuple[float, Any]] = []
    for node, data in graph.nodes(data=True):
        label = str(data.get("label", "")).casefold()
        symbol_type = str(data.get("symbol_type", "")).casefold()
        haystack = f"{node} {label} {symbol_type}".strip()
        haystack_tokens = _tokens(haystack)
        score = 0.0

        if target in haystack:
            score += 3.0
        if label and label in target:
            score += 2.5
        if symbol_type and symbol_type in target:
            score += 1.5
        if f"node {node}" in target or str(node) == target:
            score += 4.0
        if target_tokens:
            score += len(target_tokens & haystack_tokens) / len(target_tokens)

        scored.append((score, node))

    best_score = max(score for score, _ in scored)
    if best_score <= 0:
        return [], "full_image"
    return [node for score, node in scored if score == best_score], "matched"


def _crop_for_lookup(
    image: Image.Image,
    graph: nx.Graph,
    nodes: list[Any],
    match_quality: str,
    expansion: float = 0.8,
) -> Image.Image:
    if match_quality == "full_image" or not nodes:
        return image.copy()

    boxes = []
    for node in nodes:
        bbox = graph.nodes[node].get("bbox")
        if _valid_bbox(bbox):
            boxes.append([int(round(value)) for value in bbox])

    if not boxes:
        return image.copy()

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    expand_x = int(round(width * expansion))
    expand_y = int(round(height * expansion))
    crop_box = (
        max(0, x1 - expand_x),
        max(0, y1 - expand_y),
        min(image.width, x2 + expand_x),
        min(image.height, y2 + expand_y),
    )
    return image.crop(crop_box)


def _graph_excerpt(graph: nx.Graph, nodes: list[Any]) -> str:
    if not nodes:
        return graph_to_text(graph)

    lines: list[str] = []
    included = set(nodes)
    for node in nodes:
        included.update(graph.neighbors(node))

    for node in sorted(included, key=str):
        data = graph.nodes[node]
        label = data.get("label") or ""
        neighbors = []
        for neighbor in sorted(graph.neighbors(node), key=str):
            if neighbor in included:
                neighbor_type = graph.nodes[neighbor].get("symbol_type", "unknown")
                neighbors.append(f"Node {neighbor} [{neighbor_type}]")
        connected = ", ".join(neighbors) if neighbors else "none"
        lines.append(f"Node {node} [{data.get('symbol_type', 'unknown')}] label={label}: connected to {connected}")
    return "\n".join(lines)


def _action(result: dict[str, Any]) -> str:
    return str(result.get("action", "")).strip().lower()


def _reasoning(result: dict[str, Any]) -> str:
    return str(result.get("reasoning", "")).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in value.replace("_", " ").replace("-", " ").split() if token}


def _valid_bbox(value: Any) -> bool:
    return isinstance(value, list | tuple) and len(value) == 4 and all(isinstance(item, int | float) for item in value)


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
