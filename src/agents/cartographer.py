from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def tile_image(
    image: Image.Image,
    tile_size: int = 768,
    overlap: float = 0.2,
) -> list[dict[str, Any]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than 0.")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the range [0, 1).")

    width, height = image.size
    if width <= tile_size and height <= tile_size:
        return [
            {
                "tile_id": "t_00_00",
                "image": image.copy(),
                "x1": 0,
                "y1": 0,
                "x2": width,
                "y2": height,
            }
        ]

    stride = max(1, int(round(tile_size * (1 - overlap))))
    x_starts = _tile_starts(width, tile_size, stride)
    y_starts = _tile_starts(height, tile_size, stride)

    tiles: list[dict[str, Any]] = []
    for row, y1 in enumerate(y_starts):
        for col, x1 in enumerate(x_starts):
            x2 = min(x1 + tile_size, width)
            y2 = min(y1 + tile_size, height)
            crop = image.crop((x1, y1, x2, y2))
            tile = _pad_tile(crop, tile_size)
            tiles.append(
                {
                    "tile_id": f"t_{row:02d}_{col:02d}",
                    "image": tile,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return tiles


def bbox_tile_to_full(
    bbox_percent: list[float],
    tile: dict[str, Any],
    full_image: Image.Image,
) -> list[int]:
    if len(bbox_percent) != 4:
        raise ValueError("bbox_percent must contain four values.")

    tile_image_obj = tile.get("image")
    if not isinstance(tile_image_obj, Image.Image):
        raise ValueError("tile must include a PIL.Image under the 'image' key.")

    tile_width, tile_height = tile_image_obj.size
    full_width, full_height = full_image.size
    x1_pct, y1_pct, x2_pct, y2_pct = [_as_float(value) for value in bbox_percent]
    left_pct, right_pct = sorted((_clamp_percent(x1_pct), _clamp_percent(x2_pct)))
    top_pct, bottom_pct = sorted((_clamp_percent(y1_pct), _clamp_percent(y2_pct)))

    x1 = int(round(tile["x1"] + left_pct / 100 * tile_width))
    y1 = int(round(tile["y1"] + top_pct / 100 * tile_height))
    x2 = int(round(tile["x1"] + right_pct / 100 * tile_width))
    y2 = int(round(tile["y1"] + bottom_pct / 100 * tile_height))

    return [
        max(0, min(full_width, x1)),
        max(0, min(full_height, y1)),
        max(0, min(full_width, x2)),
        max(0, min(full_height, y2)),
    ]


def merge_detections(
    tile_results: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    iou_threshold: float = 0.4,
) -> list[dict[str, Any]]:
    if not tile_results:
        return []

    merged: list[dict[str, Any]] = []
    full_image = _full_image_from_tiles(tile_results)

    for tile, detections in tile_results:
        for detection in detections:
            bbox_percent = detection.get("bbox")
            if not isinstance(bbox_percent, list | tuple) or len(bbox_percent) != 4:
                continue

            item = dict(detection)
            item["bbox_percent"] = [float(value) for value in bbox_percent]
            item["bbox_absolute"] = bbox_tile_to_full(item["bbox_percent"], tile, full_image)
            item["tile_id"] = tile.get("tile_id", "")
            merged = _insert_detection(merged, item, iou_threshold)

    return sorted(
        merged,
        key=lambda item: (
            item["bbox_absolute"][1],
            item["bbox_absolute"][0],
            -float(item.get("confidence", 0.0)),
        ),
    )


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = [0]
    edge_start = length - tile_size
    while starts[-1] < edge_start:
        next_start = starts[-1] + stride
        if next_start + tile_size >= length:
            next_start = edge_start
        if next_start == starts[-1]:
            break
        starts.append(next_start)
    return starts


def _pad_tile(tile: Image.Image, tile_size: int) -> Image.Image:
    if tile.width == tile_size and tile.height == tile_size:
        return tile.copy()

    background_color = _background_color_for_mode(tile)
    padded = Image.new(tile.mode, (tile_size, tile_size), background_color)
    if tile.mode == "RGBA":
        padded.alpha_composite(tile, (0, 0))
    else:
        padded.paste(tile, (0, 0))
    return padded


def _background_color_for_mode(image: Image.Image) -> int | tuple[int, ...]:
    if image.mode == "RGBA":
        return (255, 255, 255, 255)
    if image.mode == "LA":
        return (255, 255)
    if image.mode == "L":
        return 255
    if image.mode == "P":
        return 255
    return (255, 255, 255)


def _insert_detection(
    merged: list[dict[str, Any]],
    detection: dict[str, Any],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    for index, existing in enumerate(merged):
        same_type = detection.get("symbol_type") == existing.get("symbol_type")
        if same_type and iou(detection["bbox_absolute"], existing["bbox_absolute"]) > iou_threshold:
            if float(detection.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                merged[index] = detection
            return merged

    merged.append(detection)
    return merged


def _full_image_from_tiles(
    tile_results: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> Image.Image:
    if not tile_results:
        raise ValueError("tile_results cannot be empty.")

    width = max(int(tile["x2"]) for tile, _ in tile_results)
    height = max(int(tile["y2"]) for tile, _ in tile_results)
    return Image.new("RGB", (width, height), "white")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox value is not numeric: {value!r}") from exc


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def _first_large_diagram(tile_size: int = 768) -> Path | None:
    diagrams_dir = ROOT / "data" / "diagrams"
    images = sorted(
        (path for path in diagrams_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.name.lower()),
    )
    candidates: list[tuple[int, Path]] = []
    for path in images:
        with Image.open(path) as image:
            if image.width > tile_size or image.height > tile_size:
                candidates.append((image.width * image.height, path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return images[0] if images else None


def _draw_tiling_grid(image: Image.Image, tiles: list[dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for tile in tiles:
        box = [tile["x1"], tile["y1"], tile["x2"], tile["y2"]]
        draw.rectangle(box, outline="#dc2626", width=3)
        label = str(tile["tile_id"])
        text_box = draw.textbbox((box[0] + 4, box[1] + 4), label, font=font)
        draw.rectangle(
            [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill="white",
            outline="#dc2626",
        )
        draw.text((box[0] + 4, box[1] + 4), label, fill="#dc2626", font=font)

    return annotated


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    diagram_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (_first_large_diagram() or Path())
    if not diagram_path.is_absolute():
        diagram_path = ROOT / diagram_path
    if not diagram_path.exists():
        raise SystemExit("No diagram image found in data/diagrams/.")

    with Image.open(diagram_path) as opened:
        diagram = opened.copy()

    tiles = tile_image(diagram)
    print(f"Loaded {_display_path(diagram_path)} at {diagram.width}x{diagram.height}.")
    print(f"Created {len(tiles)} tiles.")

    tiles_dir = ROOT / "outputs" / "tiles_preview"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for old_tile in tiles_dir.glob("t_*.png"):
        old_tile.unlink()
    for tile in tiles:
        tile["image"].save(tiles_dir / f"{tile['tile_id']}.png")

    grid_path = ROOT / "outputs" / "cartography" / "tiling_grid.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    _draw_tiling_grid(diagram, tiles).save(grid_path)
    print(f"Saved tiles to {_display_path(tiles_dir)}")
    print(f"Saved grid overlay to {_display_path(grid_path)}")
