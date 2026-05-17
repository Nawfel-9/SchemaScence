from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageStat

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm import ask_vlm_json


BBOX_FORMAT = "percent"
MODEL_BBOX_SCALE = "qwen_0_1000"
CONFIDENCE_THRESHOLD = 0.3
MIN_BBOX_SIDE_PERCENT = 0.35
MIN_BBOX_AREA_PERCENT = 0.25


SYMBOL_TYPES = {
    "pid": [
        "gate_valve",
        "globe_valve",
        "check_valve",
        "ball_valve",
        "butterfly_valve",
        "control_valve",
        "safety_relief_valve",
        "centrifugal_pump",
        "positive_displacement_pump",
        "compressor",
        "turbine",
        "heat_exchanger",
        "vessel_tank",
        "column_tower",
        "filter_strainer",
        "flow_meter",
        "pressure_indicator",
        "temperature_indicator",
        "level_indicator",
        "flow_controller",
        "pressure_controller",
        "motor",
        "pipe_line",
    ],
    "electrical": [
        "resistor",
        "capacitor",
        "inductor",
        "diode",
        "transistor",
        "voltage_source",
        "current_source",
        "ground",
        "switch",
        "transformer",
        "motor",
        "generator",
        "op_amp",
        "logic_gate",
    ],
    "hydraulic": [
        "hydraulic_pump",
        "hydraulic_motor",
        "hydraulic_cylinder",
        "directional_valve",
        "check_valve",
        "pressure_relief_valve",
        "filter",
        "reservoir",
        "accumulator",
        "flow_control_valve",
    ],
}

REFINABLE_SYMBOL_TYPES = {
    "motor",
    "pressure_indicator",
    "temperature_indicator",
    "level_indicator",
    "flow_meter",
    "vessel_tank",
    "centrifugal_pump",
    "positive_displacement_pump",
}

SYMBOL_ALIASES = {
    "tank": "vessel_tank",
    "vessel": "vessel_tank",
    "vessel_or_tank": "vessel_tank",
    "pressure_gauge": "pressure_indicator",
    "pressure_instrument": "pressure_indicator",
    "temperature_instrument": "temperature_indicator",
    "level_instrument": "level_indicator",
    "flow_indicator": "flow_meter",
    "pump": "centrifugal_pump",
    "relief_valve": "safety_relief_valve",
    "safety_valve": "safety_relief_valve",
    "pipeline": "pipe_line",
    "pipe": "pipe_line",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ShapeCandidate:
    bbox: tuple[float, float, float, float]
    kind: str
    area: float
    aspect: float
    fill: float
    radius: float = 0.0
    horizontal_density: float = 0.0
    vertical_density: float = 0.0


def spot_symbols(
    image: Image.Image,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Detect visible engineering symbols and return percent-normalized boxes."""
    symbol_list = _canonical_symbol_list()
    prompt = _build_prompt(symbol_list)

    result = ask_vlm_json(prompt, image=image, temperature=0.0, max_tokens=1536)
    if debug:
        print("RAW SPOTTER RESULT:")
        print(json.dumps(result, indent=2))
    symbols = _refine_symbols_with_image(image, _post_process(result, symbol_list))

    if not symbols and _image_stddev(image) > 10:
        retry_prompt = (
            prompt
            + "\n\nLook again for small but clear engineering symbols. Return [] only if no "
            "canonical symbol is visible."
        )
        result = ask_vlm_json(retry_prompt, image=image, retries=1, temperature=0.0, max_tokens=1536)
        if debug:
            print("RAW SPOTTER RETRY RESULT:")
            print(json.dumps(result, indent=2))
        symbols = _refine_symbols_with_image(image, _post_process(result, symbol_list))

    return symbols


def _canonical_symbol_list() -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for group in SYMBOL_TYPES.values():
        for symbol in group:
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols


def _build_prompt(symbol_list: list[str]) -> str:
    return f"""You are the SchemaSense Symbol Spotter agent.

Canonical symbol types:
{json.dumps(symbol_list, indent=2)}

Identify visible engineering symbols in the target image. Return ONLY valid JSON in this exact schema:
[
  {{
    "symbol_type": "gate_valve",
    "label": "V-101",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.85
  }}
]

Bounding-box coordinate contract:
- bbox MUST use normalized image coordinates from 0 to 1000.
- [0, 0] is the top-left of the target image and [1000, 1000] is the bottom-right.
- Do not use screen coordinates, pixels, crop coordinates, or percentages.
- Use tight boxes around the graphical symbol body only.
- Do not include surrounding label text, tag callout boxes, legend text, pipe lines, signal lines, or leader lines unless they are inside the symbol body.
- For a motor, box the circle containing "M"; do not include the word "Motor" or the vertical lead line.
- For pressure, temperature, level, and flow indicators, box the circular or oval instrument body.
- Level indicators often appear as oval bodies containing "L" and a numeric tag; include them when visible.
- For a tank or vessel, box the vessel body; do not include external nozzles, labels, or connected piping.
- For a centrifugal pump, box the pump body; do not confuse a nearby motor/manway circle for the pump.
- Ensure x1 < x2 and y1 < y2.

Do not copy the schema example unless that exact symbol is visible.
Do not return duplicate detections for the same visible symbol.
Return [] if no canonical engineering symbol is clearly visible.
If a symbol has no visible label, use an empty string for label.
Only include symbols you are reasonably confident about.
Use only symbol_type values from the canonical list."""


def _post_process(
    result: dict[str, Any] | list[Any],
    allowed_symbols: list[str] | None = None,
) -> list[dict[str, Any]]:
    raw_items = _extract_symbol_items(result)
    if not raw_items:
        return []

    allowed = set(allowed_symbols or [])
    symbols: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        symbol_type = _normalize_symbol_type(item.get("symbol_type"), allowed)
        if not symbol_type:
            continue

        confidence = _as_float(item.get("confidence"), 0.0)
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        bbox = _normalize_bbox(_extract_bbox(item))
        if bbox is None:
            continue

        symbols.append(
            {
                "symbol_type": symbol_type,
                "label": str(item.get("label", "")).strip(),
                "bbox": bbox,
                "confidence": round(confidence, 3),
            }
        )

    return _dedupe_symbols(symbols)


def _extract_symbol_items(result: dict[str, Any] | list[Any]) -> list[Any]:
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []
    for key in ("symbols", "detections", "items", "objects"):
        value = result.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_symbol_type(value: Any, allowed: set[str]) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    normalized = SYMBOL_ALIASES.get(normalized, normalized)
    if allowed and normalized not in allowed:
        return ""
    return normalized


def _extract_bbox(item: dict[str, Any]) -> Any:
    for key in ("bbox", "box", "bounding_box"):
        if key in item:
            return item[key]
    return None


def _normalize_bbox(value: Any) -> list[float] | None:
    raw = _coerce_bbox_parts(value)
    if raw is None:
        return None

    percent_bbox = _to_percent_coordinates(raw)
    if percent_bbox is None:
        return None

    left, right = sorted((_clamp(percent_bbox[0]), _clamp(percent_bbox[2])))
    top, bottom = sorted((_clamp(percent_bbox[1]), _clamp(percent_bbox[3])))
    width = right - left
    height = bottom - top

    if width < MIN_BBOX_SIDE_PERCENT or height < MIN_BBOX_SIDE_PERCENT:
        return None
    if width * height < MIN_BBOX_AREA_PERCENT:
        return None

    return [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]


def _coerce_bbox_parts(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        lowered = {str(key).lower(): val for key, val in value.items()}
        if all(key in lowered for key in ("x1", "y1", "x2", "y2")):
            value = [lowered["x1"], lowered["y1"], lowered["x2"], lowered["y2"]]
        elif all(key in lowered for key in ("left", "top", "right", "bottom")):
            value = [lowered["left"], lowered["top"], lowered["right"], lowered["bottom"]]
        elif all(key in lowered for key in ("x", "y", "width", "height")):
            x = _as_float(lowered["x"], math.nan)
            y = _as_float(lowered["y"], math.nan)
            width = _as_float(lowered["width"], math.nan)
            height = _as_float(lowered["height"], math.nan)
            value = [x, y, x + width, y + height]

    if not isinstance(value, list | tuple) or len(value) != 4:
        return None

    parts = [_as_float(part, math.nan) for part in value]
    if any(not math.isfinite(part) for part in parts):
        return None
    return parts


def _to_percent_coordinates(raw: list[float]) -> list[float] | None:
    lower = min(raw)
    upper = max(raw)

    if lower >= -0.05 and upper <= 1.05:
        return [part * 100 for part in raw]

    if lower >= -2 and upper <= 102:
        return raw

    if lower >= -20 and upper <= 1020:
        return [part / 10 for part in raw]

    return None


def _dedupe_symbols(symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for symbol in sorted(symbols, key=_symbol_quality, reverse=True):
        duplicate = False
        for index, existing in enumerate(deduped):
            same_type = symbol["symbol_type"] == existing["symbol_type"]
            same_label = symbol["label"].casefold() == existing["label"].casefold()
            iou = _bbox_iou(symbol["bbox"], existing["bbox"])
            if same_type and same_label and iou > 0.7:
                duplicate = True
                break
            if same_type and iou > 0.92:
                duplicate = True
                if _symbol_quality(symbol) > _symbol_quality(existing):
                    deduped[index] = symbol
                break
        if not duplicate:
            deduped.append(symbol)
    return sorted(deduped, key=lambda item: (item["bbox"][1], item["bbox"][0], -item["confidence"]))


def _symbol_quality(symbol: dict[str, Any]) -> float:
    score = float(symbol.get("confidence", 0.0))
    label = str(symbol.get("label", "")).strip().casefold()
    symbol_type = str(symbol.get("symbol_type", ""))
    if label:
        score += 0.06
    if symbol.get("bbox_source") == "vlm+geometry":
        score += 0.04
    if symbol_type == "motor" and label in {"m", "motor"}:
        score += 0.15
    if symbol_type == "flow_meter" and not label:
        score -= 0.04
    return score


def _refine_symbols_with_image(image: Image.Image, symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if cv2 is None or np is None or not symbols:
        return symbols

    candidates = _shape_candidates(image)
    if not candidates:
        return symbols

    refined: list[dict[str, Any]] = []
    for symbol in symbols:
        if symbol["symbol_type"] not in REFINABLE_SYMBOL_TYPES:
            refined.append(symbol)
            continue

        candidate = _best_shape_candidate(symbol, candidates, image.size)
        if candidate is None:
            refined.append(symbol)
            continue

        updated = dict(symbol)
        updated["bbox"] = _pixel_bbox_to_percent(candidate.bbox, image.size)
        updated["bbox_source"] = "vlm+geometry"
        refined.append(updated)

    return _dedupe_symbols(refined)


def _shape_candidates(image: Image.Image) -> list[ShapeCandidate]:
    gray = np.array(image.convert("L"))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    candidates: list[ShapeCandidate] = []
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.size[0] * image.size[1]
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 8 or height < 8:
            continue
        area = float(cv2.contourArea(contour))
        if area < 35:
            continue
        bbox_area = float(width * height)
        if bbox_area > image_area * 0.35:
            continue
        aspect = width / height
        fill = area / bbox_area if bbox_area else 0.0
        candidates.append(
            ShapeCandidate(
                bbox=(float(x), float(y), float(x + width), float(y + height)),
                kind="contour",
                area=area,
                aspect=aspect,
                fill=fill,
                horizontal_density=_line_density(binary, (x, y, x + width, y + height), "horizontal"),
                vertical_density=_line_density(binary, (x, y, x + width, y + height), "vertical"),
            )
        )

    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=24,
        param1=80,
        param2=20,
        minRadius=8,
        maxRadius=max(10, min(image.size) // 5),
    )
    if circles is not None:
        for center_x, center_y, radius in np.round(circles[0, :]).astype(int):
            x1 = float(center_x - radius)
            y1 = float(center_y - radius)
            x2 = float(center_x + radius)
            y2 = float(center_y + radius)
            if x2 <= 0 or y2 <= 0 or x1 >= image.size[0] or y1 >= image.size[1]:
                continue
            area = math.pi * float(radius * radius)
            candidates.append(
                ShapeCandidate(
                    bbox := (
                        max(0.0, x1),
                        max(0.0, y1),
                        min(float(image.size[0]), x2),
                        min(float(image.size[1]), y2),
                    ),
                    kind="circle",
                    area=area,
                    aspect=1.0,
                    fill=math.pi / 4,
                    radius=float(radius),
                    horizontal_density=_line_density(binary, bbox, "horizontal"),
                    vertical_density=_line_density(binary, bbox, "vertical"),
                )
            )

    return candidates


def _line_density(
    binary: Any,
    bbox: tuple[float, float, float, float],
    orientation: str,
) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    height, width = binary.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    if orientation == "horizontal":
        y = max(y1, min(y2 - 1, (y1 + y2) // 2))
        return float(np.count_nonzero(binary[y, x1:x2]) / max(1, x2 - x1))

    x = max(x1, min(x2 - 1, (x1 + x2) // 2))
    return float(np.count_nonzero(binary[y1:y2, x]) / max(1, y2 - y1))


def _best_shape_candidate(
    symbol: dict[str, Any],
    candidates: list[ShapeCandidate],
    image_size: tuple[int, int],
) -> ShapeCandidate | None:
    rough = _percent_bbox_to_pixels(symbol["bbox"], image_size)
    search = _search_bbox(symbol["symbol_type"], rough, image_size)
    best: tuple[float, ShapeCandidate] | None = None

    for candidate in candidates:
        if _bbox_area(candidate.bbox) < 20:
            continue
        if not _candidate_near_search(candidate.bbox, search):
            continue

        type_score = _candidate_type_score(symbol["symbol_type"], candidate)
        if type_score <= 0:
            continue

        center_score = _center_score(candidate.bbox, rough, search)
        overlap_score = _overlap_score(candidate.bbox, rough, search)
        size_score = _size_score(symbol["symbol_type"], candidate)
        score = type_score + center_score + overlap_score + size_score

        if best is None or score > best[0]:
            best = (score, candidate)

    if best is None or best[0] < 1.75:
        return None
    return best[1]


def _candidate_type_score(symbol_type: str, candidate: ShapeCandidate) -> float:
    aspect = candidate.aspect
    fill = candidate.fill
    width = candidate.bbox[2] - candidate.bbox[0]
    height = candidate.bbox[3] - candidate.bbox[1]

    if symbol_type == "motor":
        if max(width, height) > 130:
            return 0.0
        if 0.7 <= aspect <= 1.45 and fill >= 0.35 and width >= 20 and height >= 20:
            return 3.1 if candidate.kind == "contour" and fill >= 0.55 else 2.2
        if candidate.kind == "circle" and candidate.radius >= 12:
            return 2.4
        return 0.0

    if symbol_type in {"pressure_indicator", "temperature_indicator", "level_indicator", "flow_meter"}:
        if candidate.kind == "circle" and candidate.radius >= 12:
            return 1.5
        if 0.75 <= aspect <= 4.5 and fill >= 0.35 and width >= 18 and height >= 14:
            return 2.8 if candidate.kind == "contour" else 2.1
        return 0.0

    if symbol_type in {"centrifugal_pump", "positive_displacement_pump"}:
        if candidate.kind == "circle" and 16 <= candidate.radius <= 45:
            return 3.0 + min(1.6, candidate.horizontal_density * 1.6)
        if 0.7 <= aspect <= 1.6 and fill >= 0.3 and width >= 24 and height >= 20:
            return 1.8 + min(1.2, candidate.horizontal_density * 1.2)
        return 0.0

    if symbol_type == "vessel_tank":
        if candidate.kind != "contour":
            return 0.0
        if 0.25 <= aspect <= 0.85 and fill >= 0.45 and height >= 60:
            return 2.8
        return 0.0

    return 0.0


def _search_bbox(
    symbol_type: str,
    bbox: list[float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    left_factor = right_factor = 1.25
    top_factor = bottom_factor = 1.25

    if symbol_type in {"centrifugal_pump", "positive_displacement_pump"}:
        top_factor = 0.9
        bottom_factor = 2.25
    elif symbol_type in {"motor", "pressure_indicator", "temperature_indicator", "level_indicator", "flow_meter"}:
        top_factor = 1.7
        bottom_factor = 1.2
    elif symbol_type == "vessel_tank":
        left_factor = right_factor = 0.9
        top_factor = bottom_factor = 0.5

    return (
        max(0.0, x1 - width * left_factor),
        max(0.0, y1 - height * top_factor),
        min(float(image_size[0]), x2 + width * right_factor),
        min(float(image_size[1]), y2 + height * bottom_factor),
    )


def _candidate_near_search(
    candidate_bbox: tuple[float, float, float, float],
    search_bbox: tuple[float, float, float, float],
) -> bool:
    if _bbox_intersection_area(candidate_bbox, search_bbox) > 0:
        return True
    cx, cy = _bbox_center(candidate_bbox)
    sx1, sy1, sx2, sy2 = search_bbox
    return sx1 <= cx <= sx2 and sy1 <= cy <= sy2


def _center_score(
    candidate_bbox: tuple[float, float, float, float],
    rough_bbox: list[float],
    search_bbox: tuple[float, float, float, float],
) -> float:
    cx, cy = _bbox_center(candidate_bbox)
    rx, ry = _bbox_center(tuple(rough_bbox))
    search_width = max(1.0, search_bbox[2] - search_bbox[0])
    search_height = max(1.0, search_bbox[3] - search_bbox[1])
    normalized_distance = math.hypot((cx - rx) / search_width, (cy - ry) / search_height)
    return max(0.0, 1.0 - min(1.0, normalized_distance))


def _overlap_score(
    candidate_bbox: tuple[float, float, float, float],
    rough_bbox: list[float],
    search_bbox: tuple[float, float, float, float],
) -> float:
    rough_overlap = _bbox_intersection_area(candidate_bbox, tuple(rough_bbox)) / max(1.0, _bbox_area(candidate_bbox))
    search_overlap = _bbox_intersection_area(candidate_bbox, search_bbox) / max(1.0, _bbox_area(candidate_bbox))
    return min(1.0, rough_overlap + 0.35 * search_overlap)


def _size_score(symbol_type: str, candidate: ShapeCandidate) -> float:
    width = candidate.bbox[2] - candidate.bbox[0]
    height = candidate.bbox[3] - candidate.bbox[1]
    if symbol_type in {"centrifugal_pump", "positive_displacement_pump"}:
        return min(0.8, max(width, height) / 70)
    if symbol_type == "vessel_tank":
        return min(0.8, height / 180)
    if symbol_type == "motor":
        return min(0.6, min(width, height) / 80)
    return min(0.6, min(width, height) / 60)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _image_stddev(image: Image.Image) -> float:
    stat = ImageStat.Stat(image.convert("L"))
    return float(stat.stddev[0])


def _first_diagram() -> Path | None:
    diagrams_dir = ROOT / "data" / "diagrams"
    images = sorted(
        diagrams_dir.glob("*"),
        key=lambda path: (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.name),
    )
    return next((path for path in images if path.suffix.lower() in IMAGE_EXTENSIONS), None)


def _diagram_from_args() -> tuple[Path | None, bool]:
    args = [arg for arg in sys.argv[1:] if arg != "--debug"]
    debug = "--debug" in sys.argv[1:]
    if args:
        path = Path(args[0])
        if not path.is_absolute():
            path = ROOT / path
        return path, debug
    return _first_diagram(), debug


def _percent_bbox_to_pixels(bbox: list[float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [x1 / 100 * width, y1 / 100 * height, x2 / 100 * width, y2 / 100 * height]


def _pixel_bbox_to_percent(bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [
        round(_clamp(x1 / width * 100), 3),
        round(_clamp(y1 / height * 100), 3),
        round(_clamp(x2 / width * 100), 3),
        round(_clamp(y2 / height * 100), 3),
    ]


def annotate_image(image: Image.Image, symbols: list[dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for symbol in symbols:
        box = _percent_bbox_to_pixels(symbol["bbox"], annotated.size)
        label = symbol["symbol_type"]
        if symbol["label"]:
            label = f"{label} {symbol['label']}"
        draw.rectangle(box, outline="#dc2626", width=3)
        text_x = box[0] + 3
        text_y = max(0, box[1] - 16)
        text_box = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(
            [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill="white",
            outline="#dc2626",
        )
        draw.text((text_x, text_y), label, fill="#dc2626", font=font)

    return annotated


def save_annotated(image: Image.Image, symbols: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotate_image(image, symbols).save(output_path)


def save_spotter_result(
    image_name: str,
    image_size: tuple[int, int],
    symbols: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": image_name,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "bbox_format": BBOX_FORMAT,
        "model_bbox_scale": MODEL_BBOX_SCALE,
        "symbols": symbols,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    diagram_path, debug_mode = _diagram_from_args()
    if diagram_path is None:
        raise SystemExit("No image found in data/diagrams/")
    if not diagram_path.exists():
        raise SystemExit(f"Image not found: {diagram_path}")

    print(f"Testing Spotter on: {_display_path(diagram_path)}")
    with Image.open(diagram_path) as opened:
        diagram = opened.copy()

    spotted = spot_symbols(diagram, debug=debug_mode)
    print(json.dumps(spotted, indent=2))

    output_dir = ROOT / "outputs" / "spotter"
    image_output = output_dir / f"{diagram_path.stem}_symbols.png"
    json_output = output_dir / f"{diagram_path.stem}_symbols.json"
    save_annotated(diagram, spotted, image_output)
    save_spotter_result(diagram_path.name, diagram.size, spotted, json_output)
    print(f"Annotated image saved to: {_display_path(image_output)}")
    print(f"Detection JSON saved to: {_display_path(json_output)}")
