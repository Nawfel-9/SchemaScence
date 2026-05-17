# SchemaSense

SchemaSense is a fully local vision-language system for engineering diagram analysis. It uses a local Qwen3-VL model served by `llama.cpp`, a FastAPI backend, vanilla HTML/CSS/JS, pure Python orchestration, and NetworkX graph reasoning.

The goal is not just to answer questions about diagrams, but to expose how the answer was produced: detected symbols, graph nodes and edges, timings, cache status, visual fallback decisions, and baseline comparison.

No cloud APIs, LangChain, LangGraph, Streamlit, or React are required.

## What The App Does

SchemaSense lets you:

- Load a sample engineering diagram or upload an image.
- Ask a natural-language question about the diagram.
- Run a multi-agent local analysis pipeline.
- Inspect detected symbols, graph structure, reasoning, and timing.
- Compare SchemaSense against a single-shot full-image baseline.
- Run the full benchmark comparison from the web UI.

The app has two main tabs:

- **Analyze**: run one diagram/question through the pipeline.
- **Comparison**: run the full evaluation set comparing SchemaSense and the baseline, with progress and stop controls.

## Architecture

```text
Browser UI
  |
  | FastAPI endpoints
  v
SchemaSense Orchestrator
  |
  +-- Cartographer   -> tiles the diagram and preserves coordinates
  +-- Symbol Spotter -> asks Qwen3-VL for visible symbols and boxes
  +-- Connector      -> builds a NetworkX component graph
  +-- Reasoner       -> answers from graph evidence
  +-- Visual Verifier -> full-image local VLM fallback for robustness
```

All VLM calls go to a local `llama-server` instance that exposes an OpenAI-compatible API at:

```text
http://127.0.0.1:8080/v1
```

## Pipeline In Plain English

1. **Cartographer**

   Large diagrams are split into overlapping tiles. This helps the model see small symbols and labels. Tile-relative boxes are later converted back into full-image coordinates.

2. **Symbol Spotter**

   Each tile is sent to Qwen3-VL with a prompt asking for engineering symbols, labels, bounding boxes, and confidence scores. The code normalizes model output and uses image geometry to refine some boxes.

3. **Connector**

   Detected symbols become graph nodes. For nearby components, the system crops a local region and asks the model whether the components are directly connected by a pipe, wire, or line. The result is stored as a NetworkX graph.

4. **Reasoner**

   The graph is converted into compact text. The Reasoner tries to answer from graph evidence first, and can request visual lookup when the graph is insufficient.

5. **Visual Verifier**

   Because full-image Qwen3-VL is strong at reading labels and global layout, SchemaSense also runs a local full-image verifier. This prevents graph errors from destroying otherwise easy answers. The graph remains available for interpretability, while the visual verifier improves final-answer robustness.

## Repository Layout

```text
src/
  api_server.py          FastAPI app and demo endpoints
  orchestrator.py        Main SchemaSense pipeline
  baseline.py            Single-shot full-image baseline
  pipeline_spotting.py   Tiling + symbol spotting pipeline
  agents/
    cartographer.py      Tiling and coordinate projection
    spotter.py           VLM symbol detection and bbox normalization
    connector.py         NetworkX graph construction
    reasoner.py          Graph-first reasoning with visual lookup
  vlm.py                 OpenAI-compatible llama.cpp client

web/demo/
  index.html             Vanilla HTML UI
  styles.css             Demo styling
  app.js                 Frontend logic

data/
  diagrams/              Sample diagrams
  questions.json         Evaluation questions
  cache/                 Local generated cache, not for packaging

outputs/
  results_table.md       Markdown evaluation summary
  analysis/              Latest single-run analysis JSON
  eval/                  Full evaluation JSON outputs
  graphs/                Rendered graph outputs
  spotting/              Rendered detection outputs
  figures/               Report/demo figures

tests/                   Unit tests
eval.py                  Full benchmark runner
```

## Requirements

- Python 3.11 or newer
- A local `llama.cpp` build with `llama-server`
- Qwen3-VL GGUF model file
- Qwen3-VL multimodal projector (`mmproj`) file

Python packages are listed in `requirements.txt`.

## Install The App

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional environment variables can be placed in `.env`:

```text
LLAMA_CPP_BASE_URL=http://127.0.0.1:8080/v1
VLM_MODEL=local-vlm
VLM_TIMEOUT_SECONDS=75
```

## Install llama.cpp

Official llama.cpp docs:

- Build guide: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- Server guide: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

### Option A: Install With winget

On Windows, if available:

```powershell
winget install llama.cpp
```

Then confirm:

```powershell
llama-server --help
```

### Option B: Build From Source

Clone and build:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release -t llama-server
```

The server binary is usually under:

```text
build/bin/llama-server
```

On Windows CMake generators, it may be under a config-specific folder such as:

```text
build/bin/Release/llama-server.exe
```

### GPU Build Notes

For NVIDIA CUDA builds, llama.cpp currently uses CMake options such as:

```powershell
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -t llama-server
```

GPU setup depends on your driver, CUDA toolkit, and hardware. Follow the official llama.cpp build guide for the exact backend you want.

## Download / Place The Model

Place the model files here:

```text
models/qwen3-vl-8b-q8/
  Qwen3VL-8B-Instruct-Q8_0.gguf
  mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf
```

Model files are large and should not be committed to Git. The repository `.gitignore` excludes `models/`.

## Start The Local VLM Server

Run `llama-server` before starting the app.

Example:

```powershell
llama-server `
  -m models/qwen3-vl-8b-q8/Qwen3VL-8B-Instruct-Q8_0.gguf `
  --mmproj models/qwen3-vl-8b-q8/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf `
  --host 127.0.0.1 `
  --port 8080 `
  --ctx-size 8192 `
  --n-gpu-layers 999
```

If you are running CPU-only, reduce or remove GPU-specific flags:

```powershell
llama-server `
  -m models/qwen3-vl-8b-q8/Qwen3VL-8B-Instruct-Q8_0.gguf `
  --mmproj models/qwen3-vl-8b-q8/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf `
  --host 127.0.0.1 `
  --port 8080 `
  --ctx-size 8192
```

## Start SchemaSense

In a second terminal:

```powershell
python src/api_server.py
```

Open:

```text
http://127.0.0.1:7860/
```

## Using The Web App

### Analyze Tab

1. Select a sample diagram or upload an image.
2. Enter a question.
3. Optionally enable baseline comparison.
4. Click **Analyze**.
5. Review the answer, confidence, timing, graph stats, detections, and generated artifacts.

Supported uploads are image files only:

```text
PNG, JPG, JPEG, WEBP, BMP, TIFF
```

### Comparison Tab

1. Open the **Comparison** tab.
2. Click **Run comparison**.
3. Watch progress over the full question set.
4. Click **Stop** if you want to halt after the current question finishes.
5. Review SchemaSense vs baseline accuracy, average timing, recent rows, and per-type breakdown.

The comparison writes:

```text
outputs/eval/results.json
outputs/eval/partial.json
outputs/results_table.md
```

## Running Evaluation From CLI

```powershell
python eval.py
```

This runs all questions in `data/questions.json`, compares SchemaSense with the baseline, grades answers, and writes output files under `outputs/`.

## Running Tests

```powershell
python -m unittest discover -v
```

## Outputs And Cache

Generated outputs are organized by type:

```text
outputs/analysis/      latest analysis JSON
outputs/eval/          benchmark JSON outputs
outputs/graphs/        graph JSON and rendered graph images
outputs/spotting/      detection JSON and rendered overlays
outputs/figures/       report/demo figures
outputs/results_table.md
```

Local caches live in:

```text
data/cache/
outputs/cache/
```

These are useful during development but should be cleaned before packaging unless you intentionally want to ship cached demo artifacts.

## Packaging Notes

Before pushing or submitting:

- Do not include `models/`.
- Do not include `__pycache__/`.
- Do not include local cache files unless explicitly needed.
- Keep sample diagrams and selected report/demo artifacts.
- Keep `outputs/results_table.md` if it reflects the final evaluation.

## Troubleshooting

### The app says llama.cpp is offline

Make sure `llama-server` is running on:

```text
http://127.0.0.1:8080/v1
```

Check:

```powershell
curl http://127.0.0.1:8080/v1/models
```

### The model is slow

Try:

- Enabling GPU acceleration in llama.cpp.
- Reducing context size.
- Reusing cached outputs.
- Running fewer evaluation questions during development.

### Uploaded image fails

Only image uploads are supported. Convert PDFs to PNG or JPG before uploading.

### Evaluation takes a long time

The full comparison runs many VLM calls. Use the web UI stop button to stop after the current question.

## Design Philosophy

SchemaSense is intentionally simple:

- one local VLM server
- plain Python orchestration
- explicit JSON artifacts
- FastAPI backend
- vanilla frontend
- NetworkX graph reasoning

The system is now hybrid: graph-first for interpretability, full-image verification for robustness. This makes it easier to defend in a project demo because it exposes intermediate reasoning while avoiding many failures caused by incomplete graph construction.
