from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
OUTPUTS_DIR = ROOT / "outputs"
ANALYSIS_OUTPUTS_DIR = OUTPUTS_DIR / "analysis"
EVAL_OUTPUTS_DIR = OUTPUTS_DIR / "eval"
GRAPH_OUTPUTS_DIR = OUTPUTS_DIR / "graphs"
SPOTTING_OUTPUTS_DIR = OUTPUTS_DIR / "spotting"
DIAGRAMS_DIR = ROOT / "data" / "diagrams"
CACHE_DIR = ROOT / "data" / "cache"
UPLOADS_DIR = CACHE_DIR / "uploads"
QUESTIONS_PATH = ROOT / "data" / "questions.json"
BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
VLM_MODEL = os.getenv("VLM_MODEL", "local-vlm")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
LAST_TELEMETRY: dict[str, Any] = {
    "active_model": VLM_MODEL,
    "model_server_url": BASE_URL,
    "current_agent": "Idle",
    "current_task": "Awaiting analysis",
    "current_input_type": "text-only",
    "cache": "unknown",
    "elapsed_seconds": 0.0,
    "last_warning_error": "",
    "progress_stage": "idle",
}
EVAL_LOCK = threading.Lock()
EVAL_STOP_EVENT = threading.Event()
EVAL_THREAD: threading.Thread | None = None
EVAL_STATE: dict[str, Any] = {
    "status": "idle",
    "total": 0,
    "completed": 0,
    "current_id": "",
    "current_diagram": "",
    "current_question": "",
    "current_stage": "Idle",
    "elapsed_seconds": 0.0,
    "started_at": 0.0,
    "finished_at": 0.0,
    "stop_requested": False,
    "error": "",
    "rows": [],
    "summary": {},
    "outputs": {},
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="SchemaSense Local Demo")
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


def api_error(exc: Exception) -> JSONResponse:
    _update_telemetry(last_warning_error=str(exc), progress_stage="error")
    return JSONResponse({"ok": False, "error": str(exc), "telemetry": dict(LAST_TELEMETRY)}, status_code=200)


def _update_telemetry(**values: Any) -> dict[str, Any]:
    LAST_TELEMETRY.update(values)
    return dict(LAST_TELEMETRY)


def numeric_sort(path: Path) -> tuple[int, str]:
    return (0, f"{int(path.stem):08d}") if path.stem.isdigit() else (1, path.name.lower())


def safe_stem(name: str, fallback: str = "uploaded") -> str:
    stem = Path(name or fallback).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return cleaned or fallback


def relative_url(path: Path) -> str:
    return "/" + path.relative_to(ROOT).as_posix()


def diagrams() -> list[dict[str, Any]]:
    items = []
    for path in sorted(DIAGRAMS_DIR.glob("*"), key=numeric_sort):
        if path.suffix.lower() in IMAGE_EXTS:
            items.append({"name": path.name, "size": path.stat().st_size, "url": f"/api/diagrams/{path.name}"})
    return items


def read_questions() -> list[dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        return []
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("data/questions.json must contain a JSON array.")
    return data


def cache_count() -> int:
    total = 0
    for cache_dir in (CACHE_DIR, OUTPUTS_DIR / "cache"):
        total += len([p for p in cache_dir.rglob("*") if p.is_file()]) if cache_dir.exists() else 0
    return total


def save_uploaded_file(raw: bytes, uploaded_name: str) -> Path:
    if not raw:
        raise ValueError("Uploaded diagram is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded diagram is larger than the local 20 MB limit.")
    suffix = Path(uploaded_name or "").suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise ValueError("Upload must be a PNG, JPG, WEBP, BMP, or TIFF file.")

    digest = hashlib.md5(raw).hexdigest()
    stem = safe_stem(uploaded_name, "uploaded")
    path = UPLOADS_DIR / f"{digest}_{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return path


async def parse_analysis_request(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("diagram")
        sample_name = str(form.get("sample_name") or "")
        payload: dict[str, Any] = {
            "question": str(form.get("question") or ""),
            "compare_baseline": _truthy(form.get("compare_baseline")),
            "sample_name": sample_name,
        }
        if upload is not None and getattr(upload, "filename", ""):
            payload["upload_bytes"] = await upload.read()
            payload["uploaded_name"] = upload.filename
            payload["uploaded"] = True
        else:
            payload["diagram"] = sample_name
            payload["uploaded"] = False
        return payload
    if content_type.startswith("application/json"):
        return await request.json()
    return {}


def analysis_target_from_payload(body: dict[str, Any]) -> tuple[Path, str, bool]:
    diagram_name = body.get("diagram") or body.get("sample_name") or ""
    uploaded = bool(body.get("uploaded"))
    uploaded_name = body.get("uploaded_name") or ""
    if uploaded and "upload_bytes" in body:
        path = save_uploaded_file(body["upload_bytes"], uploaded_name)
        return path, uploaded_name or path.name, True
    if uploaded:
        raise ValueError(
            "Uploaded diagrams must be sent as multipart/form-data with a 'diagram' file field."
        )
    if diagram_name:
        path = (DIAGRAMS_DIR / diagram_name).resolve()
        if DIAGRAMS_DIR.resolve() not in path.parents or not path.exists():
            raise ValueError("Selected sample diagram was not found.")
        return path, str(diagram_name), False
    raise ValueError("No diagram was selected.")


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def llama_models() -> dict[str, Any]:
    response = requests.get(f"{BASE_URL}/models", timeout=2)
    response.raise_for_status()
    data = response.json()
    model_ids = [item.get("id") for item in data.get("data", []) if item.get("id")]
    return {"reachable": True, "active_model": model_ids[0] if model_ids else VLM_MODEL, "models": model_ids}


def safe_llama_models() -> dict[str, Any]:
    try:
        return llama_models()
    except Exception as exc:
        return {"reachable": False, "active_model": VLM_MODEL, "models": [], "error": str(exc)}


def output_files() -> list[dict[str, Any]]:
    patterns = [
        "spotter/*.png",
        "spotter/*.json",
        "spotting/*.png",
        "spotting/*.json",
        "cartography/*.png",
        "graphs/*.png",
        "graphs/*.json",
        "analysis/*.json",
        "eval/*.json",
        "results_table.md",
        "figures/*.png",
        "figures/*.svg",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(OUTPUTS_DIR.glob(pattern))
    unique = sorted({p.resolve(): p for p in files if p.is_file()}.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "name": path.name,
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size": path.stat().st_size,
            "modified": path.stat().st_mtime,
            "url": relative_url(path),
        }
        for path in unique[:20]
    ]


def _eval_snapshot() -> dict[str, Any]:
    with EVAL_LOCK:
        snapshot = dict(EVAL_STATE)
        snapshot["rows"] = list(EVAL_STATE.get("rows", []))
        snapshot["summary"] = dict(EVAL_STATE.get("summary", {}))
        snapshot["outputs"] = dict(EVAL_STATE.get("outputs", {}))
    return snapshot


def _set_eval_state(**values: Any) -> dict[str, Any]:
    with EVAL_LOCK:
        EVAL_STATE.update(values)
        return dict(EVAL_STATE)


def _reset_eval_state(total: int) -> None:
    now = time.time()
    with EVAL_LOCK:
        EVAL_STATE.update(
            {
                "status": "running",
                "total": total,
                "completed": 0,
                "current_id": "",
                "current_diagram": "",
                "current_question": "",
                "current_stage": "Preparing",
                "elapsed_seconds": 0.0,
                "started_at": now,
                "finished_at": 0.0,
                "stop_requested": False,
                "error": "",
                "rows": [],
                "summary": _eval_summary([], total),
                "outputs": {},
            }
        )


def _eval_summary(rows: list[dict[str, Any]], total: int | None = None) -> dict[str, Any]:
    total_questions = total if total is not None else len(rows)
    completed = len(rows)
    ss_correct = sum(1 for row in rows if row.get("ss_correct"))
    bl_correct = sum(1 for row in rows if row.get("bl_correct"))
    ss_seconds = sum(float(row.get("ss_seconds") or 0) for row in rows)
    bl_seconds = sum(float(row.get("bl_seconds") or 0) for row in rows)
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        qtype = str(row.get("type") or "unknown")
        bucket = by_type.setdefault(qtype, {"total": 0, "schemasense": 0, "baseline": 0})
        bucket["total"] += 1
        bucket["schemasense"] += 1 if row.get("ss_correct") else 0
        bucket["baseline"] += 1 if row.get("bl_correct") else 0

    if ss_correct > bl_correct:
        leader = "SchemaSense"
    elif bl_correct > ss_correct:
        leader = "Baseline"
    elif completed:
        leader = "Tie"
    else:
        leader = "Pending"

    return {
        "completed": completed,
        "total": total_questions,
        "schemasense_correct": ss_correct,
        "baseline_correct": bl_correct,
        "schemasense_accuracy": round(ss_correct / completed * 100, 1) if completed else 0.0,
        "baseline_accuracy": round(bl_correct / completed * 100, 1) if completed else 0.0,
        "schemasense_avg_seconds": round(ss_seconds / completed, 2) if completed else 0.0,
        "baseline_avg_seconds": round(bl_seconds / completed, 2) if completed else 0.0,
        "leader": leader,
        "by_type": [
            {"type": qtype, **values}
            for qtype, values in sorted(by_type.items())
        ],
    }


def _write_eval_artifacts(rows: list[dict[str, Any]], final: bool = False) -> dict[str, str]:
    from eval import PARTIAL_PATH, RESULTS_PATH, TABLE_PATH, results_table

    EVAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    outputs = {"partial": relative_url(PARTIAL_PATH), "table": relative_url(TABLE_PATH)}
    if final:
        RESULTS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        TABLE_PATH.write_text(results_table(rows), encoding="utf-8")
        outputs["results"] = relative_url(RESULTS_PATH)
    return outputs


def _run_eval_comparison() -> None:
    from eval import grade
    from src.baseline import baseline_answer
    from src.orchestrator import schemasense

    rows: list[dict[str, Any]] = []
    questions = read_questions()
    started = time.perf_counter()

    try:
        for index, item in enumerate(questions, start=1):
            if EVAL_STOP_EVENT.is_set():
                _set_eval_state(status="stopped", current_stage="Stopped", stop_requested=True)
                break

            question = str(item.get("question", ""))
            gold = str(item.get("answer", ""))
            qtype = str(item.get("type", ""))
            image_path = DIAGRAMS_DIR / str(item.get("diagram", ""))
            row_started = time.perf_counter()
            _set_eval_state(
                current_id=str(item.get("id", "")),
                current_diagram=str(item.get("diagram", "")),
                current_question=question,
                current_stage="SchemaSense",
                elapsed_seconds=round(time.perf_counter() - started, 1),
            )

            ss = schemasense(str(image_path), question, question_type=qtype)
            _set_eval_state(
                current_stage="Baseline",
                elapsed_seconds=round(time.perf_counter() - started, 1),
                stop_requested=EVAL_STOP_EVENT.is_set(),
            )

            bl = _baseline_from_schemasense_result(ss) or baseline_answer(str(image_path), question)
            _set_eval_state(current_stage="Judging", elapsed_seconds=round(time.perf_counter() - started, 1))

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
                "elapsed_seconds": round(time.perf_counter() - row_started, 3),
            }
            rows.append(row)
            outputs = _write_eval_artifacts(rows, final=False)
            _set_eval_state(
                completed=index,
                rows=rows[-8:],
                summary=_eval_summary(rows, len(questions)),
                outputs=outputs,
                current_stage="Completed question",
                elapsed_seconds=round(time.perf_counter() - started, 1),
            )

        status = _eval_snapshot().get("status")
        if status != "stopped":
            outputs = _write_eval_artifacts(rows, final=True)
            _set_eval_state(
                status="completed",
                current_stage="Complete",
                finished_at=time.time(),
                elapsed_seconds=round(time.perf_counter() - started, 1),
                summary=_eval_summary(rows, len(questions)),
                outputs=outputs,
                stop_requested=False,
            )
        else:
            outputs = _write_eval_artifacts(rows, final=False)
            _set_eval_state(
                finished_at=time.time(),
                elapsed_seconds=round(time.perf_counter() - started, 1),
                summary=_eval_summary(rows, len(questions)),
                outputs=outputs,
            )
    except Exception as exc:
        outputs = _write_eval_artifacts(rows, final=False) if rows else {}
        _set_eval_state(
            status="error",
            error=str(exc),
            current_stage="Error",
            finished_at=time.time(),
            elapsed_seconds=round(time.perf_counter() - started, 1),
            summary=_eval_summary(rows, len(questions)),
            outputs=outputs,
        )


def _baseline_from_schemasense_result(result: dict[str, Any]) -> dict[str, Any] | None:
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


def _detection_percent_bbox(detection: dict[str, Any], image_size: tuple[int, int]) -> list[float] | None:
    bbox = detection.get("bbox_absolute") or detection.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    width, height = image_size
    if not width or not height:
        return None
    values = [float(value) for value in bbox]
    if all(0 <= value <= 100 for value in values) and "bbox_absolute" not in detection:
        return [round(value, 3) for value in values]
    x1, y1, x2, y2 = values
    return [
        round(max(0.0, min(100.0, x1 / width * 100)), 3),
        round(max(0.0, min(100.0, y1 / height * 100)), 3),
        round(max(0.0, min(100.0, x2 / width * 100)), 3),
        round(max(0.0, min(100.0, y2 / height * 100)), 3),
    ]


def _frontend_symbols(detections: list[dict[str, Any]], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    symbols = []
    for detection in detections:
        bbox = _detection_percent_bbox(detection, image_size)
        if bbox is None:
            continue
        item = dict(detection)
        item["bbox"] = bbox
        symbols.append(item)
    return symbols


def _graph_text(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    if not nodes:
        return "No graph nodes were produced."
    node_by_id = {str(node.get("id")): node for node in nodes}
    neighbors: dict[str, list[str]] = {str(node.get("id")): [] for node in nodes}
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        ctype = edge.get("connection_type", "unknown")
        if source in neighbors and target in node_by_id:
            neighbors[source].append(f"Node {target} [{node_by_id[target].get('symbol_type', 'unknown')}] ({ctype})")
        if target in neighbors and source in node_by_id:
            neighbors[target].append(f"Node {source} [{node_by_id[source].get('symbol_type', 'unknown')}] ({ctype})")
    lines = []
    for node in nodes:
        node_id = str(node.get("id"))
        label = node.get("label") or ""
        linked = ", ".join(neighbors.get(node_id, [])) or "none"
        lines.append(f"Node {node_id} [{node.get('symbol_type', 'unknown')}] label={label}: connected to {linked}")
    return "\n".join(lines)


def _timing_lines(timing: dict[str, Any], cache_hit: bool) -> list[str]:
    return [
        f"Cache hit: {'yes' if cache_hit else 'no'}",
        f"Spotting: {timing.get('spotting_seconds', 0)}s",
        f"Graph: {timing.get('graph_seconds', 0)}s",
        f"Reasoning: {timing.get('reasoning_seconds', 0)}s",
        f"Total: {timing.get('total_seconds', 0)}s",
    ]


def _save_pipeline_outputs(
    image_path: Path,
    image: Image.Image,
    result: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    paths: dict[str, Path] = {}
    warnings: list[str] = []
    from src.pipeline_spotting import draw_detections
    from src.agents.connector import _draw_graph_topology
    from src.visualize import make_pipeline_figure
    import networkx as nx

    stem = safe_stem(image_path.stem)

    detections = result.get("detections", [])
    if isinstance(detections, list):
        try:
            paths["spotting_image"] = SPOTTING_OUTPUTS_DIR / f"{stem}.png"
            draw_detections(image, detections, str(paths["spotting_image"]))
            SPOTTING_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            (SPOTTING_OUTPUTS_DIR / f"{stem}.json").write_text(json.dumps(detections, indent=2), encoding="utf-8")
        except Exception as exc:
            warnings.append(f"spotting render failed: {str(exc).splitlines()[0]}")
            paths.pop("spotting_image", None)

    try:
        graph_data = result.get("graph", {})
        graph = nx.Graph()
        for node in graph_data.get("nodes", []):
            node = dict(node)
            node_id = node.pop("id")
            graph.add_node(node_id, **node)
        for edge in graph_data.get("edges", []):
            edge = dict(edge)
            source = edge.pop("source")
            target = edge.pop("target")
            graph.add_edge(source, target, **edge)
        paths["graph_image"] = GRAPH_OUTPUTS_DIR / f"{stem}.png"
        _draw_graph_topology(graph, paths["graph_image"])
        GRAPH_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (GRAPH_OUTPUTS_DIR / f"{stem}.json").write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
    except Exception as exc:
        warnings.append(f"graph render failed: {str(exc).splitlines()[0]}")
        paths.pop("graph_image", None)

    try:
        paths["pipeline_figure"] = OUTPUTS_DIR / "figures" / f"pipeline_{stem}.png"
        make_pipeline_figure(str(image_path), str(paths["pipeline_figure"]))
    except Exception as exc:
        warnings.append(f"pipeline figure failed: {str(exc).splitlines()[0]}")
        paths.pop("pipeline_figure", None)

    rendered = {key: relative_url(path) for key, path in paths.items() if path.exists()}
    return rendered, warnings


@app.get("/")
def demo_home():
    return FileResponse(WEB_DIR / "demo" / "index.html")


@app.get("/api/diagrams/{filename}")
def diagram_file(filename: str):
    try:
        path = (DIAGRAMS_DIR / filename).resolve()
        if DIAGRAMS_DIR.resolve() not in path.parents or not path.exists():
            return JSONResponse({"ok": False, "error": "Diagram not found."}, status_code=200)
        return FileResponse(path)
    except Exception as exc:
        return api_error(exc)


@app.get("/api/health")
def health():
    try:
        model_info = safe_llama_models()
        return {
            "ok": True,
            "api": "ok",
            "server_url": BASE_URL,
            "llama_cpp": model_info,
            "active_model": model_info["active_model"],
            "diagrams_count": len(diagrams()),
            "questions_count": len(read_questions()),
            "cache_files_count": cache_count(),
        }
    except Exception as exc:
        return api_error(exc)


@app.get("/api/diagrams")
def api_diagrams():
    try:
        return {"ok": True, "diagrams": diagrams()}
    except Exception as exc:
        return api_error(exc)


@app.get("/api/questions")
def api_questions():
    try:
        qs = read_questions()
        return {"ok": True, "questions": qs, "count": len(qs)}
    except Exception as exc:
        return api_error(exc)


@app.get("/api/demo/state")
def demo_state():
    try:
        model_info = safe_llama_models()
        telemetry = dict(LAST_TELEMETRY)
        telemetry["active_model"] = model_info["active_model"]
        telemetry["model_server_url"] = BASE_URL
        return {
            "ok": True,
            "diagrams": diagrams(),
            "questions": read_questions(),
            "active_model": model_info["active_model"],
            "server_url": BASE_URL,
            "llama_cpp": model_info,
            "telemetry": telemetry,
            "outputs": output_files(),
        }
    except Exception as exc:
        return api_error(exc)


@app.post("/api/analyze")
async def analyze(request: Request):
    try:
        started = time.perf_counter()
        body = await parse_analysis_request(request)
        question = (body.get("question") or "Describe this engineering diagram in one sentence.").strip()
        compare_value = body.get("compare_baseline", False)
        compare_baseline = compare_value if isinstance(compare_value, bool) else _truthy(compare_value)
        model_info = safe_llama_models()

        if not model_info["reachable"]:
            raise RuntimeError("llama.cpp is offline. Start llama-server, then run analysis again.")

        _update_telemetry(
            active_model=model_info["active_model"],
            model_server_url=BASE_URL,
            current_agent="Cartographer",
            current_task="Loading diagram and preparing coordinate space",
            current_input_type="full image",
            cache="pending",
            elapsed_seconds=0.0,
            last_warning_error="",
            progress_stage="cartographer",
        )

        analysis_path, image_name, uploaded = analysis_target_from_payload(body)

        with Image.open(analysis_path) as opened:
            image = opened.copy()

        from src.orchestrator import schemasense

        orchestrated = schemasense(str(analysis_path), question, use_cache=True)
        detections = orchestrated.get("detections", [])
        graph_data = orchestrated.get("graph", {})
        symbols = _frontend_symbols(detections if isinstance(detections, list) else [], image.size)
        rendered_outputs, render_warnings = _save_pipeline_outputs(analysis_path, image, orchestrated)

        timing = orchestrated.get("timing", {})
        graph_stats = orchestrated.get("graph_stats", {"nodes": 0, "edges": 0})
        cache_hit = bool(orchestrated.get("cache_hit", False))
        connector_warnings = list(orchestrated.get("connector_warnings", []))
        connector_queries = dict(orchestrated.get("connector_queries", {"attempted": 0, "failed": 0}))
        warning_lines = render_warnings + [f"connector: {item}" for item in connector_warnings]
        total_time = round(time.perf_counter() - started, 2)
        telemetry = _update_telemetry(
            current_agent="Reasoner",
            current_task="Answer complete",
            current_input_type="crop image" if orchestrated.get("used_image_lookup") else "text-only",
            cache="hit" if cache_hit else "miss",
            elapsed_seconds=timing.get("total_seconds", total_time),
            last_warning_error="; ".join(warning_lines) if warning_lines else "",
            progress_stage="answer",
        )
        timeline = [
            {
                "agent": "Cartographer",
                "status": "done",
                "task": "Load image and prepare tile coordinate space.",
                "detail": f"{image_name} loaded at {image.size[0]}x{image.size[1]}.",
            },
            {
                "agent": "Spotter",
                "status": "done",
                "task": "Tile image, detect symbols, refine boxes, and merge detections.",
                "detail": f"{len(symbols)} detections {'loaded from cache' if cache_hit else 'computed'} in {timing.get('spotting_seconds', 0)}s.",
            },
            {
                "agent": "Connector",
                "status": "done",
                "task": "Build node-edge connectivity graph.",
                "detail": f"{graph_stats.get('nodes', 0)} nodes and {graph_stats.get('edges', 0)} edges in {timing.get('graph_seconds', 0)}s.",
            },
            {
                "agent": "Reasoner",
                "status": "done",
                "task": "Answer using graph-first reasoning with visual lookup if needed.",
                "detail": f"Confidence {orchestrated.get('confidence', 0)}; image lookup {'used' if orchestrated.get('used_image_lookup') else 'not used'}.",
            },
        ]
        symbol_lines = [
            f"{index + 1}. {item.get('symbol_type', 'unknown')}"
            + (f" ({item.get('label')})" if item.get("label") else "")
            + f" confidence {item.get('confidence', 0)}"
            for index, item in enumerate(symbols)
        ]
        baseline = None
        if compare_baseline:
            baseline = _baseline_from_schemasense_result(orchestrated)
            if baseline is None:
                from src.baseline import baseline_answer

                baseline = baseline_answer(str(analysis_path), question)
        files = output_files()
        result = {
            "ok": True,
            "answer": orchestrated.get("answer", "unknown"),
            "reasoning": orchestrated.get("reasoning", ""),
            "confidence": float(orchestrated.get("confidence", 0.0)),
            "used_image_lookup": bool(orchestrated.get("used_image_lookup", False)),
            "graph_answer": orchestrated.get("graph_answer"),
            "visual_answer": orchestrated.get("visual_answer"),
            "hybrid_decision": orchestrated.get("hybrid_decision"),
            "diagram": image_name,
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "symbols": symbols,
            "bbox_format": "percent",
            "detections": detections,
            "graph": graph_data,
            "graph_stats": graph_stats,
            "n_detections": len(detections) if isinstance(detections, list) else len(symbols),
            "timing": timing,
            "cache_hit": cache_hit,
            "telemetry": telemetry,
            "baseline": baseline,
            "image_hash": orchestrated.get("image_hash", ""),
            "timeline": timeline,
            "inspector": {
                "active_model": model_info["active_model"],
                "current_agent": "Reasoner",
                "current_task": "Answer complete",
                "input_type": "uploaded full image" if uploaded else "sample full image",
                "symbols_detected": str(len(symbols)),
                "graph_nodes": str(graph_stats.get("nodes", 0)),
                "graph_edges": str(graph_stats.get("edges", 0)),
                "total_time": f"{timing.get('total_seconds', total_time)}s",
                "confidence": str(orchestrated.get("confidence", 0)),
                "used_image_lookup": "yes" if orchestrated.get("used_image_lookup") else "no",
                "cache_hit": "yes" if cache_hit else "no",
                "used_vlm": "yes",
            },
            "results": {
                "overview": (
                    f"Full SchemaSense pipeline completed with {len(symbols)} detections, "
                    f"{graph_stats.get('nodes', 0)} graph nodes, {graph_stats.get('edges', 0)} graph edges. "
                    f"Cache hit: {'yes' if cache_hit else 'no'}."
                ),
                "symbols": symbols if symbols else "No canonical symbols were detected with the current spotter prompt.",
                "graph": _graph_text(graph_data),
                "trace": [f"{step['agent']}: {step['detail']}" for step in timeline],
                "reasoning": orchestrated.get("reasoning", ""),
                "timing": _timing_lines(timing, cache_hit),
                "symbol_summary": symbol_lines,
            },
            "outputs": {
                "spotting_image": rendered_outputs.get("spotting_image", ""),
                "graph_image": rendered_outputs.get("graph_image", ""),
                "pipeline_figure": rendered_outputs.get("pipeline_figure", ""),
            },
            "output_files": files,
            "render_warnings": render_warnings,
            "connector_warnings": connector_warnings,
            "connector_queries": connector_queries,
        }
        ANALYSIS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (ANALYSIS_OUTPUTS_DIR / "latest_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        return api_error(exc)


@app.post("/api/baseline")
async def baseline_endpoint(request: Request):
    try:
        started = time.perf_counter()
        body = await parse_analysis_request(request)
        question = (body.get("question") or "Describe this engineering diagram in one sentence.").strip()
        analysis_path, image_name, _ = analysis_target_from_payload(body)
        from src.baseline import baseline_answer

        result = baseline_answer(str(analysis_path), question)
        return {
            "ok": True,
            "diagram": image_name,
            "answer": result.get("answer", "unknown"),
            "elapsed_seconds": result.get("elapsed_seconds", round(time.perf_counter() - started, 3)),
            "method": "Single-shot full-image VLM",
        }
    except Exception as exc:
        return api_error(exc)


@app.post("/api/eval/start")
def start_eval():
    global EVAL_THREAD
    try:
        model_info = safe_llama_models()
        if not model_info["reachable"]:
            raise RuntimeError("llama.cpp is offline. Start llama-server, then run the comparison again.")

        snapshot = _eval_snapshot()
        if snapshot.get("status") in {"running", "stopping"}:
            return {"ok": True, "eval": snapshot}

        questions = read_questions()
        if not questions:
            raise ValueError("No evaluation questions were found in data/questions.json.")

        EVAL_STOP_EVENT.clear()
        _reset_eval_state(len(questions))
        EVAL_THREAD = threading.Thread(target=_run_eval_comparison, name="schemasense-eval", daemon=True)
        EVAL_THREAD.start()
        return {"ok": True, "eval": _eval_snapshot()}
    except Exception as exc:
        return api_error(exc)


@app.post("/api/eval/stop")
def stop_eval():
    try:
        snapshot = _eval_snapshot()
        if snapshot.get("status") == "running":
            EVAL_STOP_EVENT.set()
            _set_eval_state(status="stopping", stop_requested=True, current_stage="Stopping after current question")
        return {"ok": True, "eval": _eval_snapshot()}
    except Exception as exc:
        return api_error(exc)


@app.get("/api/eval/status")
def eval_status():
    try:
        snapshot = _eval_snapshot()
        if snapshot.get("status") in {"running", "stopping"}:
            started_at = float(snapshot.get("started_at") or 0)
            if started_at:
                _set_eval_state(elapsed_seconds=round(time.time() - started_at, 1))
                snapshot = _eval_snapshot()
        return {"ok": True, "eval": snapshot}
    except Exception as exc:
        return api_error(exc)


@app.get("/api/outputs/latest")
def latest_outputs():
    try:
        return {"ok": True, "files": output_files()}
    except Exception as exc:
        return api_error(exc)


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=7860, reload=False)
