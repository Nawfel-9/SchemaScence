from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import networkx as nx
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.cartographer import IMAGE_EXTENSIONS
from src.vlm import ask_vlm_json

CACHE_DIR = ROOT / "data" / "cache"
OUTPUTS_DIR = ROOT / "outputs"
GRAPH_OUTPUTS_DIR = OUTPUTS_DIR / "graphs"

PALETTE = [
    "#dc2626",
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#a16207",
]


def build_graph(
    image: Image.Image,
    detections: list[dict[str, Any]],
) -> nx.Graph:
    graph = nx.Graph()
    warnings: list[str] = []
    queried = 0
    failed = 0
    absolute_detections = [_with_absolute_bbox(image, detection) for detection in detections]

    for index, detection in enumerate(absolute_detections):
        graph.add_node(
            index,
            symbol_type=str(detection.get("symbol_type", "unknown")),
            label=str(detection.get("label", "")).strip(),
            bbox=detection["bbox_absolute"],
            confidence=float(detection.get("confidence", 0.0)),
        )

    for target_id, detection in enumerate(absolute_detections):
        context = _context_window(detection["bbox_absolute"], image.size)
        candidates = [
            (node_id, other)
            for node_id, other in enumerate(absolute_detections)
            if node_id != target_id and _intersects(context, other["bbox_absolute"])
        ]
        if not candidates:
            continue

        crop = _annotated_context_crop(image, context, target_id, detection, candidates)
        prompt = _connection_prompt(target_id, detection, candidates)
        queried += 1

        try:
            result = ask_vlm_json(prompt, image=crop, retries=1, temperature=0.0, max_tokens=512)
        except Exception as exc:
            failed += 1
            message = f"node {target_id} query failed: {str(exc).splitlines()[0]}"
            warnings.append(message)
            print(f"Connector warning: {message}")
            continue

        for neighbor_id, connection_type in _parse_connections(result, candidates):
            _add_or_update_edge(graph, target_id, neighbor_id, connection_type)

    graph.graph["warnings"] = warnings
    graph.graph["queries"] = {"attempted": queried, "failed": failed}
    print(
        f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
        f"(connector queries: {queried}, failed: {failed})"
    )
    return graph


def graph_to_dict(g: nx.Graph) -> dict[str, list[dict[str, Any]]]:
    """Serialize graph to JSON-friendly structure."""
    return {
        "nodes": [{"id": str(node), **data} for node, data in g.nodes(data=True)],
        "edges": [{"source": str(source), "target": str(target), **data} for source, target, data in g.edges(data=True)],
    }


def graph_to_text(g: nx.Graph) -> str:
    """
    Compact text description for use in Reasoner prompts.
    Format:
      Node 0 [centrifugal_pump] label=P-101: connected to Node 1 [gate_valve], Node 5 [heat_exchanger]
      Node 1 [gate_valve] label=V-101: connected to Node 0 [centrifugal_pump], Node 2 [pipe_line]
    """
    lines: list[str] = []
    for node in sorted(g.nodes):
        data = g.nodes[node]
        label = data.get("label") or ""
        neighbors = []
        for neighbor in sorted(g.neighbors(node)):
            neighbor_type = g.nodes[neighbor].get("symbol_type", "unknown")
            neighbors.append(f"Node {neighbor} [{neighbor_type}]")
        connected = ", ".join(neighbors) if neighbors else "none"
        lines.append(f"Node {node} [{data.get('symbol_type', 'unknown')}] label={label}: connected to {connected}")
    return "\n".join(lines)


def _with_absolute_bbox(image: Image.Image, detection: dict[str, Any]) -> dict[str, Any]:
    updated = dict(detection)
    if "bbox_absolute" in updated and _valid_bbox(updated["bbox_absolute"]):
        updated["bbox_absolute"] = [int(round(value)) for value in updated["bbox_absolute"]]
        return updated

    bbox = updated.get("bbox") or updated.get("bbox_percent")
    if not _valid_bbox(bbox):
        raise ValueError(f"Detection is missing a valid bbox: {detection}")

    values = [float(value) for value in bbox]
    if all(0 <= value <= 100 for value in values):
        width, height = image.size
        x1, y1, x2, y2 = values
        updated["bbox_absolute"] = [
            int(round(x1 / 100 * width)),
            int(round(y1 / 100 * height)),
            int(round(x2 / 100 * width)),
            int(round(y2 / 100 * height)),
        ]
    else:
        updated["bbox_absolute"] = [int(round(value)) for value in values]

    return updated


def _context_window(
    bbox: list[int],
    image_size: tuple[int, int],
    expansion: float = 0.6,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    expand_x = int(round(box_width * expansion))
    expand_y = int(round(box_height * expansion))
    return (
        max(0, x1 - expand_x),
        max(0, y1 - expand_y),
        min(width, x2 + expand_x),
        min(height, y2 + expand_y),
    )


def _annotated_context_crop(
    image: Image.Image,
    context: tuple[int, int, int, int],
    target_id: int,
    target: dict[str, Any],
    candidates: list[tuple[int, dict[str, Any]]],
) -> Image.Image:
    crop = image.crop(context).convert("RGB")
    draw = ImageDraw.Draw(crop)
    font = ImageFont.load_default()
    offset_x, offset_y = context[0], context[1]

    for local_id, (node_id, detection) in enumerate(candidates, start=1):
        box = _local_box(detection["bbox_absolute"], offset_x, offset_y)
        draw.rectangle(box, outline="#2563eb", width=3)
        _draw_text_box(draw, (box[0] + 3, box[1] + 3), str(local_id), "#2563eb", font)

    target_box = _local_box(target["bbox_absolute"], offset_x, offset_y)
    draw.rectangle(target_box, outline="#dc2626", width=4)
    _draw_text_box(draw, (target_box[0] + 3, max(0, target_box[1] - 18)), f"★ target Node {target_id}", "#dc2626", font)
    return crop


def _connection_prompt(
    target_id: int,
    target: dict[str, Any],
    candidates: list[tuple[int, dict[str, Any]]],
) -> str:
    candidate_lines = []
    for local_id, (node_id, detection) in enumerate(candidates, start=1):
        label = str(detection.get("label", "")).strip()
        label_text = f", label={label}" if label else ""
        candidate_lines.append(f"{local_id}: Node {node_id} [{detection.get('symbol_type', 'unknown')}{label_text}]")

    target_label = str(target.get("label", "")).strip()
    target_label_text = f", label={target_label}" if target_label else ""
    return f"""In this engineering diagram region, determine direct connectivity.

The highlighted component marked with ★ is Node {target_id} [{target.get("symbol_type", "unknown")}{target_label_text}].

Numbered candidate components:
{chr(10).join(candidate_lines)}

Which of the numbered components are DIRECTLY connected to the highlighted component by a pipe, wire, or line?
Only include direct physical/diagram connections, not nearby unrelated symbols or text labels.

Return JSON exactly:
{{
  "connected_to": [list of numbers],
  "connection_types": ["pipe"|"wire"|"line"|"unknown"]
}}"""


def _parse_connections(
    result: dict[str, Any] | list[Any],
    candidates: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, str]]:
    if not isinstance(result, dict):
        return []

    local_to_node = {local_id: node_id for local_id, (node_id, _) in enumerate(candidates, start=1)}
    connected_to = result.get("connected_to", [])
    connection_types = result.get("connection_types", [])
    if not isinstance(connected_to, list):
        return []
    if not isinstance(connection_types, list):
        connection_types = []

    parsed: list[tuple[int, str]] = []
    for index, raw_local_id in enumerate(connected_to):
        local_id = _as_int(raw_local_id)
        if local_id not in local_to_node:
            continue
        connection_type = _normalize_connection_type(connection_types[index] if index < len(connection_types) else "unknown")
        parsed.append((local_to_node[local_id], connection_type))
    return parsed


def _add_or_update_edge(graph: nx.Graph, source: int, target: int, connection_type: str) -> None:
    if source == target:
        return
    if graph.has_edge(source, target):
        existing = graph.edges[source, target].get("connection_type", "unknown")
        if existing == "unknown" and connection_type != "unknown":
            graph.edges[source, target]["connection_type"] = connection_type
        return
    graph.add_edge(source, target, connection_type=connection_type)


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    text_box = draw.textbbox(xy, text, font=font)
    draw.rectangle(
        [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
        fill="white",
        outline=color,
    )
    draw.text(xy, text, fill=color, font=font)


def _local_box(bbox: list[int], offset_x: int, offset_y: int) -> list[int]:
    return [bbox[0] - offset_x, bbox[1] - offset_y, bbox[2] - offset_x, bbox[3] - offset_y]


def _intersects(a: tuple[int, int, int, int] | list[int], b: tuple[int, int, int, int] | list[int]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _valid_bbox(value: Any) -> bool:
    return isinstance(value, list | tuple) and len(value) == 4 and all(isinstance(item, int | float) for item in value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_connection_type(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in {"pipe", "wire", "line", "unknown"} else "unknown"


def _load_cached_detections(diagram_path: Path) -> list[dict[str, Any]] | None:
    cache_path = CACHE_DIR / f"{diagram_path.stem}_detections.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _first_diagram_with_cache() -> Path | None:
    diagrams_dir = ROOT / "data" / "diagrams"
    images = sorted(
        (path for path in diagrams_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.name.lower()),
    )
    for path in images:
        if (CACHE_DIR / f"{path.stem}_detections.json").exists():
            return path
    return images[0] if images else None


def _save_graph_json(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph_to_dict(graph), indent=2), encoding="utf-8")


def _save_graph_pickle(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle)


def _draw_graph_topology(graph: nx.Graph, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    if graph.number_of_nodes() == 0:
        plt.text(0.5, 0.5, "No graph nodes", ha="center", va="center")
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight", dpi=160)
        plt.close()
        return

    symbol_types = sorted({data.get("symbol_type", "unknown") for _, data in graph.nodes(data=True)})
    color_map = {symbol_type: PALETTE[index % len(PALETTE)] for index, symbol_type in enumerate(symbol_types)}
    colors = [color_map[graph.nodes[node].get("symbol_type", "unknown")] for node in graph.nodes]
    labels = {
        node: graph.nodes[node].get("label") or graph.nodes[node].get("symbol_type", "unknown")
        for node in graph.nodes
    }
    layout = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, layout, node_color=colors, node_size=1300, edgecolors="#111827", linewidths=1)
    nx.draw_networkx_edges(graph, layout, width=2, edge_color="#6b7280")
    nx.draw_networkx_labels(graph, layout, labels=labels, font_size=8)
    edge_labels = nx.get_edge_attributes(graph, "connection_type")
    nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    diagram_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (_first_diagram_with_cache() or Path())
    if not diagram_path.is_absolute():
        diagram_path = ROOT / diagram_path
    if not diagram_path.exists():
        raise SystemExit("No diagram image found in data/diagrams/.")

    with Image.open(diagram_path) as opened:
        image = opened.copy()
        image.filename = str(diagram_path)

    detections = _load_cached_detections(diagram_path)
    if detections is None:
        from src.pipeline_spotting import spot_all_symbols

        detections = spot_all_symbols(image)
    else:
        print(f"Loaded {len(detections)} cached detections for {diagram_path.name}.")

    graph = build_graph(image, detections)
    json_path = GRAPH_OUTPUTS_DIR / f"{diagram_path.stem}.json"
    png_path = GRAPH_OUTPUTS_DIR / f"{diagram_path.stem}.png"
    pickle_path = CACHE_DIR / f"{diagram_path.stem}_graph.pkl"

    _save_graph_json(graph, json_path)
    _draw_graph_topology(graph, png_path)
    _save_graph_pickle(graph, pickle_path)

    print(graph_to_text(graph))
    print(f"Saved graph JSON to {_display_path(json_path)}")
    print(f"Saved graph image to {_display_path(png_path)}")
    print(f"Saved graph pickle to {_display_path(pickle_path)}")
