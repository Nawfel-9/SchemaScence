from __future__ import annotations

import hashlib
import json
import pickle
import sys
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.cartographer import IMAGE_EXTENSIONS, tile_image
from src.agents.connector import build_graph, graph_to_dict
from src.orchestrator import schemasense
from src.pipeline_spotting import PALETTE, spot_all_symbols

CACHE_DIR = ROOT / "data" / "cache"
DIAGRAMS_DIR = ROOT / "data" / "diagrams"
QUESTIONS_PATH = ROOT / "data" / "questions.json"
FIGURES_DIR = ROOT / "outputs" / "figures"


def make_pipeline_figure(image_path: str, out_path: str):
    """4-panel figure showing the agent pipeline visually."""
    path = _resolve_path(image_path)
    image, detections, graph = _pipeline_data(path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    _show_image(axes[0, 0], image, "Input")
    _show_image(axes[0, 1], image, "Cartographer")
    _draw_tiles(axes[0, 1], image)
    _show_image(axes[1, 0], image, "Symbol Spotter")
    _draw_detections(axes[1, 0], detections)
    _show_image(axes[1, 1], image, "Connector")
    _draw_graph_overlay(axes[1, 1], graph)

    fig.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def make_answer_figure(image_path: str, question: str, result: dict, out_path: str):
    """Single-diagram figure showing Q&A with reasoning trace."""
    path = _resolve_path(image_path)
    with Image.open(path) as opened:
        image = opened.convert("RGB").copy()
    graph = result.get("graph", {}) if isinstance(result, dict) else {}
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    reasoning = str(result.get("reasoning", ""))

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14, 8), facecolor="white", gridspec_kw={"width_ratios": [3, 2]}
    )
    _show_image(left, image, "")
    _draw_relevant_nodes(left, nodes, reasoning)
    _text_panel(right, question, result, len(nodes), len(edges))

    fig.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def _pipeline_data(path: Path) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any]]:
    with Image.open(path) as opened:
        image = opened.convert("RGB").copy()
        image.filename = str(path)
    digest = _md5_file(path)
    detections_path = CACHE_DIR / f"{digest}_detections.json"
    graph_path = CACHE_DIR / f"{digest}_graph.pkl"
    if detections_path.exists() and graph_path.exists():
        detections = json.loads(detections_path.read_text(encoding="utf-8"))
        with graph_path.open("rb") as handle:
            graph = graph_to_dict(pickle.load(handle))
        return image, detections, graph
    detections = spot_all_symbols(image, use_cache=True)
    graph = graph_to_dict(build_graph(image, detections))
    return image, detections, graph


def _show_image(ax, image: Image.Image, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_tiles(ax, image: Image.Image) -> None:
    for tile in tile_image(image):
        x1, y1, x2, y2 = tile["x1"], tile["y1"], tile["x2"], tile["y2"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#2563eb", linewidth=1.8))


def _draw_detections(ax, detections: list[dict[str, Any]]) -> None:
    color_map = _color_map(detections)
    for item in detections:
        bbox = item.get("bbox_absolute") or item.get("bbox")
        if not _valid_bbox(bbox):
            continue
        color = color_map.get(str(item.get("symbol_type", "unknown")), PALETTE[0])
        x1, y1, x2, y2 = [float(v) for v in bbox]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.2))
        ax.text(x1, max(0, y1 - 5), str(item.get("symbol_type", "unknown")), color=color, fontsize=8, weight="bold")


def _draw_graph_overlay(ax, graph: dict[str, Any]) -> None:
    nodes = {str(node.get("id")): node for node in graph.get("nodes", [])}
    centers = {node_id: _center(node.get("bbox")) for node_id, node in nodes.items() if _valid_bbox(node.get("bbox"))}
    for edge in graph.get("edges", []):
        a, b = centers.get(str(edge.get("source"))), centers.get(str(edge.get("target")))
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#111827", linewidth=2, alpha=0.75)
    for index, (node_id, point) in enumerate(centers.items()):
        ax.scatter(point[0], point[1], s=90, color=PALETTE[index % len(PALETTE)], edgecolor="white", linewidth=1.5, zorder=5)
        ax.text(point[0] + 6, point[1] - 6, node_id, color="#111827", fontsize=8, weight="bold")


def _draw_relevant_nodes(ax, nodes: list[dict[str, Any]], reasoning: str) -> None:
    text = reasoning.casefold()
    for node in nodes:
        node_id = str(node.get("id", ""))
        label = str(node.get("label", "")).casefold()
        symbol = str(node.get("symbol_type", "")).casefold()
        relevant = f"node {node_id}".casefold() in text or (label and label in text) or (symbol and symbol in text)
        if not relevant or not _valid_bbox(node.get("bbox")):
            continue
        cx, cy = _center(node["bbox"])
        ax.scatter(cx, cy, s=170, color="#f97316", edgecolor="#dc2626", linewidth=2.5, zorder=6)
        ax.text(cx + 8, cy - 8, f"Node {node_id}", color="#dc2626", fontsize=9, weight="bold")


def _text_panel(ax, question: str, result: dict, n_nodes: int, n_edges: int) -> None:
    ax.axis("off")
    answer = str(result.get("answer", "unknown"))
    confidence = float(result.get("confidence", 0.0) or 0.0)
    answer_color = "#15803d" if confidence >= 0.65 else "#92400e"
    ax.text(0, 0.98, "Question", transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    ax.text(0, 0.91, _wrap(question, 44), transform=ax.transAxes, fontsize=10, va="top")
    ax.text(0, 0.73, "Answer", transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    ax.text(0, 0.66, _wrap(answer, 34), transform=ax.transAxes, fontsize=18, weight="bold", color=answer_color, va="top")
    ax.text(0, 0.53, f"{n_nodes} nodes, {n_edges} edges", transform=ax.transAxes, fontsize=10, color="#1d4ed8", bbox={"boxstyle": "round,pad=0.35", "facecolor": "#dbeafe", "edgecolor": "none"})
    ax.text(0, 0.43, "Reasoning trace", transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    ax.text(0, 0.36, _wrap(str(result.get("reasoning", "")), 52), transform=ax.transAxes, fontsize=8.5, color="#687281", va="top")


def _first_diagram() -> Path:
    images = sorted((p for p in DIAGRAMS_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS), key=_sort_key)
    if not images:
        raise FileNotFoundError("No diagrams found in data/diagrams.")
    return images[0]


def _first_question() -> dict[str, Any]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[0]


def _resolve_path(path: str | Path) -> Path:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def _color_map(detections: list[dict[str, Any]]) -> dict[str, str]:
    types = sorted({str(item.get("symbol_type", "unknown")) for item in detections})
    return {symbol_type: PALETTE[index % len(PALETTE)] for index, symbol_type in enumerate(types)}


def _center(bbox: Any) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _valid_bbox(value: Any) -> bool:
    return isinstance(value, list | tuple) and len(value) == 4 and all(isinstance(item, int | float) for item in value)


def _wrap(value: str, width: int) -> str:
    return "\n".join(textwrap.wrap(value, width=width, replace_whitespace=False))


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_key(path: Path) -> tuple[int, str]:
    return (0, f"{int(path.stem):08d}") if path.stem.isdigit() else (1, path.name.lower())


if __name__ == "__main__":
    diagram = _first_diagram()
    question = _first_question()
    answer_image = DIAGRAMS_DIR / question["diagram"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = FIGURES_DIR / f"pipeline_{diagram.stem}.png"
    answer_path = FIGURES_DIR / f"answer_{question['id']}.png"
    make_pipeline_figure(str(diagram), str(pipeline_path))
    result = schemasense(str(answer_image), str(question["question"]))
    make_answer_figure(str(answer_image), str(question["question"]), result, str(answer_path))
    print(pipeline_path)
    print(answer_path)
