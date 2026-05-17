from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.agents.cartographer import IMAGE_EXTENSIONS, merge_detections, tile_image
from src.agents.spotter import spot_symbols


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
OUTPUTS_DIR = ROOT / "outputs"
SPOTTING_OUTPUTS_DIR = OUTPUTS_DIR / "spotting"

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


def spot_all_symbols(
    image: Image.Image,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Run full spotting: tile -> spot -> merge -> return."""
    cache_path = _cache_path_for_image(image)
    if use_cache and cache_path.exists():
        detections = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(detections)} cached detections from {_display_path(cache_path)}")
        return detections

    tiles = tile_image(image)
    tile_results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    raw_count = 0

    for tile in tqdm(tiles, desc="Spotting tiles"):
        detections = spot_symbols(tile["image"])
        raw_count += len(detections)
        tile_results.append((tile, detections))

    merged = merge_detections(tile_results)
    print(f"Found {len(merged)} symbols after merging {raw_count} raw detections from {len(tiles)} tiles")
    if use_cache:
        _save_json(cache_path, merged)
    return merged


def draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    out_path: str,
) -> None:
    """Draw bounding boxes on image and save."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    color_map = _color_map(detections)

    for detection in detections:
        bbox = detection.get("bbox_absolute") or detection.get("bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            continue
        symbol_type = str(detection.get("symbol_type", "unknown"))
        label = str(detection.get("label", "")).strip()
        text = symbol_type if not label else f"{symbol_type} {label}"
        color = color_map.get(symbol_type, PALETTE[0])
        box = [int(round(value)) for value in bbox]

        draw.rectangle(box, outline=color, width=3)
        _draw_label(draw, _label_position(box, text, draw, font), text, color, font)

    _draw_legend(draw, color_map, font)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def _cache_path_for_image(image: Image.Image) -> Path:
    stem = _image_stem(image)
    return CACHE_DIR / f"{stem}_detections.json"


def _image_stem(image: Image.Image) -> str:
    filename = getattr(image, "filename", "")
    if filename:
        return Path(filename).stem
    return f"image_{image.width}x{image.height}"


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _color_map(detections: list[dict[str, Any]]) -> dict[str, str]:
    symbol_types = sorted({str(item.get("symbol_type", "unknown")) for item in detections})
    return {symbol_type: PALETTE[index % len(PALETTE)] for index, symbol_type in enumerate(symbol_types)}


def _draw_label(
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


def _label_position(
    box: list[int],
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    x = box[0] + 3
    y = max(0, box[1] - 16)
    text_box = draw.textbbox((x, y), text, font=font)
    legend_region = (0, 0, 180, 90)
    overlaps_legend = not (
        text_box[2] < legend_region[0]
        or text_box[0] > legend_region[2]
        or text_box[3] < legend_region[1]
        or text_box[1] > legend_region[3]
    )
    if overlaps_legend:
        y = box[3] + 4
    return (x, y)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    color_map: dict[str, str],
    font: ImageFont.ImageFont,
) -> None:
    if not color_map:
        return

    padding = 8
    swatch = 12
    row_height = 18
    x = 12
    y = 12
    title = "Legend"
    rows = [title, *color_map.keys()]
    width = max(draw.textlength(row, font=font) for row in rows) + padding * 3 + swatch
    height = padding * 2 + row_height * len(rows)
    draw.rectangle([x, y, x + width, y + height], fill="white", outline="#111827", width=1)
    draw.text((x + padding, y + padding), title, fill="#111827", font=font)

    for index, (symbol_type, color) in enumerate(color_map.items(), start=1):
        row_y = y + padding + row_height * index
        draw.rectangle([x + padding, row_y + 2, x + padding + swatch, row_y + 2 + swatch], fill=color, outline=color)
        draw.text((x + padding * 2 + swatch, row_y), symbol_type, fill="#111827", font=font)


def _first_diagram() -> Path | None:
    diagrams_dir = ROOT / "data" / "diagrams"
    images = sorted(
        (path for path in diagrams_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.name.lower()),
    )
    return images[0] if images else None


def _summary(detections: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(item.get("symbol_type", "unknown")) for item in detections))


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    diagram_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (_first_diagram() or Path())
    if not diagram_path.is_absolute():
        diagram_path = ROOT / diagram_path
    if not diagram_path.exists():
        raise SystemExit("No diagram found in data/diagrams/.")

    with Image.open(diagram_path) as opened:
        image = opened.copy()
        image.filename = str(diagram_path)

    detections = spot_all_symbols(image)
    print(_summary(detections))

    output_image = SPOTTING_OUTPUTS_DIR / f"{diagram_path.stem}.png"
    output_json = SPOTTING_OUTPUTS_DIR / f"{diagram_path.stem}.json"
    draw_detections(image, detections, str(output_image))
    _save_json(output_json, detections)
    print(f"Saved visualization to {_display_path(output_image)}")
    print(f"Saved detections to {_display_path(output_json)}")
