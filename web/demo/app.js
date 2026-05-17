const STAGES = ["Cartographer", "Spotter", "Connector", "Reasoner", "Answer"];

const EXAMPLES = [
  "How many pumps are shown?",
  "Which unit process follows the pressure tank?",
  "What component receives the controller output signal?",
  "Is the schematic title missing?",
  "What pressure action value is written beside the relief symbol?",
];

const state = {
  diagrams: [],
  questions: [],
  uploadedFile: null,
  uploadedUrl: "",
  running: false,
  timer: null,
  evalTimer: null,
  startedAt: 0,
  placeholderTimer: null,
  latest: null,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || "Request failed.");
  return data;
}

function selectedDiagram() {
  return state.diagrams.find((item) => item.name === $("sampleSelect").value) || state.diagrams[0];
}

function setActiveView(name) {
  const analyze = name === "analyze";
  $("analyzeView").classList.toggle("hidden", !analyze);
  $("compareView").classList.toggle("hidden", analyze);
  $("analyzeTab").classList.toggle("active", analyze);
  $("compareTab").classList.toggle("active", !analyze);
  moveTabIndicator(analyze ? $("analyzeTab") : $("compareTab"));
}

function moveTabIndicator(activeTab = document.querySelector(".tab-button.active")) {
  const indicator = $("tabIndicator");
  const tabs = document.querySelector(".view-tabs");
  if (!indicator || !tabs || !activeTab) return;
  const tabBox = activeTab.getBoundingClientRect();
  const tabsBox = tabs.getBoundingClientRect();
  indicator.style.width = `${tabBox.width}px`;
  indicator.style.transform = `translateX(${tabBox.left - tabsBox.left}px)`;
}

/* ─────────── status line / header pills ─────────── */
function setStatusLine(data = {}) {
  const model = data.active_model || data.telemetry?.active_model || "local-vlm";
  const server = data.server_url || data.telemetry?.model_server_url || "127.0.0.1";
  $("statusModel").textContent = model;
  $("statusServer").textContent = server.replace(/^https?:\/\//, "");

  let cacheText = "cache —";
  let cacheClass = "pill";
  if (data.cache_hit === true) { cacheText = "cache hit"; cacheClass = "pill success"; }
  else if (data.cache_hit === false) { cacheText = "cache miss"; cacheClass = "pill warn"; }
  else if (data.llama_cpp && !data.llama_cpp.reachable) { cacheText = "model offline"; cacheClass = "pill error"; }
  $("statusCache").textContent = cacheText;
  $("statusCache").className = cacheClass;
}

/* ─────────── sample preview ─────────── */
function renderSamples() {
  $("sampleSelect").innerHTML = state.diagrams
    .map((item) => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`)
    .join("");
  renderPreview();
}

function renderPreview() {
  const sample = selectedDiagram();
  const title = state.uploadedFile ? state.uploadedFile.name : sample?.name || "No diagram selected";
  $("diagramTitle").textContent = title;
  $("inputTypeBadge").textContent = state.uploadedFile ? "uploaded" : "sample";
  if (state.uploadedUrl) {
    $("diagramPreview").src = state.uploadedUrl;
  } else if (sample) {
    $("diagramPreview").src = sample.url;
  }
  prefillQuestionForSample();
}

function prefillQuestionForSample() {
  if (state.uploadedFile) return;
  const sample = selectedDiagram();
  const match = state.questions.find((q) => q.diagram === sample?.name);
  if (match && !$("questionInput").value.trim()) {
    $("questionInput").value = match.question;
  }
}

/* ─────────── progress timeline ─────────── */
function renderTimeline(active = "idle") {
  const activeIndex = STAGES.findIndex((item) => item.toLowerCase() === String(active).toLowerCase());
  $("progressTimeline").innerHTML = STAGES.map((name, index) => {
    const className = activeIndex >= 0 && index < activeIndex ? "done" : index === activeIndex ? "active" : "";
    return `<li class="${className}">${escapeHtml(name)}</li>`;
  }).join("");
}

/* ─────────── answer + meta ─────────── */
function renderAnswer(data = {}) {
  const hasAnswer = Boolean(data.answer);
  const answerEl = $("answerText");
  answerEl.textContent = data.answer || "Run SchemaSense to produce a graph-first answer.";
  answerEl.classList.toggle("muted", !hasAnswer);

  $("answerCorner").textContent = hasAnswer ? "complete" : "awaiting";

  const reasoningEl = $("reasoningText");
  if (data.reasoning) {
    reasoningEl.textContent = data.reasoning;
    reasoningEl.classList.remove("hidden");
  } else {
    reasoningEl.classList.add("hidden");
  }

  const meta = $("answerMeta");
  if (!hasAnswer) {
    meta.classList.add("hidden");
    return;
  }
  meta.classList.remove("hidden");
  const stats = data.graph_stats || {};
  const timing = data.timing || {};
  $("metaConfidence").textContent = (data.confidence != null ? Number(data.confidence).toFixed(2) : "—");
  $("metaTime").textContent = `${timing.total_seconds ?? 0}s`;
  $("metaDetections").textContent = data.n_detections ?? (data.symbols?.length ?? 0);
  $("metaGraph").textContent = `${stats.nodes ?? 0} / ${stats.edges ?? 0}`;
  $("metaLookup").textContent = data.used_image_lookup ? "yes" : "no";
}

/* ─────────── pipeline (collapsed until data exists) ─────────── */
function renderPipeline(data = {}) {
  const pipeline = $("pipeline");
  const outputs = data.outputs || {};
  const spotting = outputs.spotting_image || "";
  const graph = outputs.graph_image || "";
  if (!spotting && !graph) {
    pipeline.classList.add("hidden");
    return;
  }
  pipeline.classList.remove("hidden");
  if (spotting) $("spottingImage").src = spotting; else $("spottingImage").removeAttribute("src");
  if (graph) $("graphImage").src = graph; else $("graphImage").removeAttribute("src");
}

/* ─────────── trace dl ─────────── */
function renderTrace(data = {}) {
  const timing = data.timing || {};
  $("traceReasoning").textContent = data.reasoning || "—";
  $("traceSpotting").textContent = `${timing.spotting_seconds ?? 0}s`;
  $("traceGraph").textContent = `${timing.graph_seconds ?? 0}s`;
  $("traceReasoner").textContent = `${timing.reasoning_seconds ?? 0}s`;
  $("traceTotal").textContent = `${timing.total_seconds ?? 0}s`;
  $("traceCache").textContent = data.cache_hit === true ? "hit" : data.cache_hit === false ? "miss" : "—";
  $("traceLookup").textContent = data.used_image_lookup ? "yes" : "no";
  const queries = data.connector_queries || {};
  const failed = queries.failed || 0;
  const attempted = queries.attempted || 0;
  $("traceConnector").textContent = attempted ? `${attempted} queries · ${failed} failed` : "—";
}

/* ─────────── telemetry dl ─────────── */
function renderTelemetry(telemetry = {}) {
  const rows = [
    ["active model", telemetry.active_model || "local-vlm"],
    ["current agent", telemetry.current_agent || "Idle"],
    ["current task", telemetry.current_task || "Awaiting analysis"],
    ["input type", telemetry.current_input_type || "text-only"],
    ["progress stage", telemetry.progress_stage || "idle"],
    ["elapsed", `${telemetry.elapsed_seconds ?? 0}s`],
    ["cache", telemetry.cache || "unknown"],
    ["warnings/errors", telemetry.last_warning_error || "none"],
  ];
  $("telemetryDl").innerHTML = rows
    .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd class="mono">${escapeHtml(v)}</dd>`)
    .join("");
}

/* ─────────── warnings notice ─────────── */
function renderWarnings(data = {}) {
  const card = $("warningsCard");
  const renderW = Array.isArray(data.render_warnings) ? data.render_warnings : [];
  const connW = Array.isArray(data.connector_warnings) ? data.connector_warnings : [];
  const items = [
    ...renderW.map((w) => `render: ${w}`),
    ...connW.map((w) => `connector: ${w}`),
  ];
  if (!items.length) {
    card.classList.add("hidden");
    card.innerHTML = "";
    return;
  }
  card.classList.remove("hidden");
  card.innerHTML = `<strong>Warnings during this run.</strong><ul>${items.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`;
}

/* ─────────── baseline ─────────── */
function renderBaseline(data = {}) {
  const card = $("baselineDetails");
  const baseline = data.baseline;
  if (!baseline) {
    card.classList.add("hidden");
    card.removeAttribute("open");
    return;
  }
  card.classList.remove("hidden");
  $("baselineTable").innerHTML = `
    <table class="table-academic">
      <thead><tr><th>Field</th><th>SchemaSense</th><th>Baseline</th></tr></thead>
      <tbody>
        <tr><td>Answer</td><td>${escapeHtml(data.answer || "")}</td><td>${escapeHtml(baseline.answer || "—")}</td></tr>
        <tr><td>Total time</td><td>${escapeHtml(String(data.timing?.total_seconds ?? "—"))}s</td><td>${escapeHtml(String(baseline.elapsed_seconds ?? "—"))}s</td></tr>
        <tr><td>Method</td><td>Graph-first multi-agent</td><td>Single-shot full-image</td></tr>
      </tbody>
    </table>
  `;
}

/* ─────────── full comparison ─────────── */
function setEvalPolling(active) {
  if (!active) {
    clearInterval(state.evalTimer);
    state.evalTimer = null;
    return;
  }
  if (state.evalTimer) return;
  state.evalTimer = setInterval(fetchEvalStatus, 1400);
}

async function startEvaluation() {
  $("evalErrorCard").classList.add("hidden");
  setActiveView("compare");
  setEvalButtons(true);
  $("topProgress").classList.add("active");
  try {
    const data = await getJson("/api/eval/start", { method: "POST" });
    renderEvaluation(data.eval || {});
    setEvalPolling(true);
  } catch (error) {
    $("evalErrorCard").textContent = String(error.message || error);
    $("evalErrorCard").classList.remove("hidden");
    setEvalButtons(false);
    $("topProgress").classList.remove("active");
  }
}

async function stopEvaluation() {
  $("stopEvalButton").disabled = true;
  try {
    const data = await getJson("/api/eval/stop", { method: "POST" });
    renderEvaluation(data.eval || {});
  } catch (error) {
    $("evalErrorCard").textContent = String(error.message || error);
    $("evalErrorCard").classList.remove("hidden");
  }
}

async function fetchEvalStatus() {
  try {
    const data = await getJson("/api/eval/status");
    renderEvaluation(data.eval || {});
  } catch (error) {
    $("evalErrorCard").textContent = String(error.message || error);
    $("evalErrorCard").classList.remove("hidden");
    setEvalPolling(false);
    setEvalButtons(false);
    $("topProgress").classList.remove("active");
  }
}

function setEvalButtons(running) {
  $("startEvalButton").disabled = running;
  $("stopEvalButton").disabled = !running;
}

function renderEvaluation(job = {}) {
  const status = job.status || "idle";
  const running = status === "running" || status === "stopping";
  const summary = job.summary || {};
  const completed = Number(job.completed || summary.completed || 0);
  const total = Number(job.total || summary.total || 0);
  const progress = total ? Math.min(100, Math.round((completed / total) * 100)) : 0;

  $("evalStatusBadge").textContent = status;
  $("evalElapsed").textContent = `elapsed ${job.elapsed_seconds ?? 0}s`;
  $("evalMeterBar").style.width = `${progress}%`;
  $("evalProgressText").textContent = `${completed} of ${total} complete`;
  $("evalCurrentTitle").textContent = job.current_id
    ? `${job.current_id} · ${job.current_diagram || ""} · ${job.current_stage || status}`
    : status === "idle" ? "No comparison running" : job.current_stage || status;
  $("evalCurrentQuestion").textContent = job.current_question || "Run the benchmark to compare SchemaSense against the single-shot baseline.";

  $("evalSchemaScore").textContent = `${summary.schemasense_correct || 0} / ${completed || total || 0}`;
  $("evalBaselineScore").textContent = `${summary.baseline_correct || 0} / ${completed || total || 0}`;
  $("evalSchemaMeta").textContent = `${Number(summary.schemasense_accuracy || 0).toFixed(1)}% · ${summary.schemasense_avg_seconds || 0}s avg`;
  $("evalBaselineMeta").textContent = `${Number(summary.baseline_accuracy || 0).toFixed(1)}% · ${summary.baseline_avg_seconds || 0}s avg`;
  $("evalLeader").textContent = summary.leader || "Pending";

  renderEvalTypeBreakdown(summary.by_type || [], Boolean(completed));
  renderEvalRows(job.rows || []);
  renderEvalOutputs(job.outputs || {});

  if (job.error) {
    $("evalErrorCard").textContent = job.error;
    $("evalErrorCard").classList.remove("hidden");
  }

  setEvalButtons(running);
  $("topProgress").classList.toggle("active", running || state.running);
  if (running && !state.evalTimer) setEvalPolling(true);
  if (!running) setEvalPolling(false);
}

function renderEvalTypeBreakdown(rows, show) {
  const card = $("evalSummaryCard");
  if (!show) {
    card.classList.add("hidden");
    $("evalTypeBreakdown").innerHTML = "";
    return;
  }
  card.classList.remove("hidden");
  if (!rows.length) {
    $("evalTypeBreakdown").innerHTML = `<div class="muted">No type summary yet.</div>`;
    return;
  }
  $("evalTypeBreakdown").innerHTML = rows.map((row) => `
    <div class="type-row">
      <span>${escapeHtml(row.type)}</span>
      <span class="mono">SchemaSense ${escapeHtml(row.schemasense)}/${escapeHtml(row.total)}</span>
      <span class="mono">Baseline ${escapeHtml(row.baseline)}/${escapeHtml(row.total)}</span>
    </div>
  `).join("");
}

function renderEvalRows(rows) {
  if (!rows.length) {
    $("evalRows").classList.add("muted");
    $("evalRows").textContent = "No rows yet.";
    return;
  }
  $("evalRows").classList.remove("muted");
  $("evalRows").innerHTML = rows.slice().reverse().map((row) => `
    <div class="eval-row">
      <span class="mono">${escapeHtml(row.id || "")} · ${escapeHtml(row.diagram || "")}</span>
      <span>${escapeHtml(row.question || "")}</span>
      <span class="${row.ss_correct ? "ok" : "no"}">SS ${row.ss_correct ? "YES" : "NO"}</span>
      <span class="${row.bl_correct ? "ok" : "no"}">BL ${row.bl_correct ? "YES" : "NO"}</span>
    </div>
  `).join("");
}

function renderEvalOutputs(outputs = {}) {
  const table = outputs.table || "";
  const results = outputs.results || outputs.partial || "";
  const bits = [];
  if (table) bits.push(`<a href="${escapeHtml(table)}" target="_blank" rel="noreferrer">table</a>`);
  if (results) bits.push(`<a href="${escapeHtml(results)}" target="_blank" rel="noreferrer">json</a>`);
  $("evalOutputLink").innerHTML = bits.length ? bits.join(" · ") : "outputs";
}

/* ─────────── running state ─────────── */
function setRunning(running) {
  state.running = running;
  $("analyzeButton").disabled = running;
  $("diagramUpload").disabled = running;
  $("sampleSelect").disabled = running;
  $("topProgress").classList.toggle("active", running);
  $("analyzeButton").querySelector(".btn-label").textContent = running ? "Analyzing…" : "Analyze";
  if (!running) {
    clearInterval(state.timer);
    state.timer = null;
    return;
  }
  state.startedAt = performance.now();
  $("controlTitle").textContent = "Running pipeline…";
  renderTimeline(STAGES[0]);
  state.timer = setInterval(() => {
    const elapsed = ((performance.now() - state.startedAt) / 1000).toFixed(1);
    const stage = STAGES[Math.min(STAGES.length - 2, Math.floor(elapsed / 1.6))];
    $("controlTitle").textContent = `${stage}: working…`;
    $("statusAside").textContent = `elapsed ${elapsed}s · cache pending`;
    renderTimeline(stage);
  }, 480);
}

/* ─────────── analyze ─────────── */
async function analyze() {
  if (state.running) return;
  $("errorCard").classList.add("hidden");
  $("warningsCard").classList.add("hidden");
  setRunning(true);

  try {
    const form = new FormData();
    if (state.uploadedFile) {
      form.append("diagram", state.uploadedFile);
    } else {
      form.append("sample_name", selectedDiagram()?.name || "");
    }
    form.append("question", $("questionInput").value.trim() || EXAMPLES[0]);
    form.append("compare_baseline", $("baselineToggle").checked ? "true" : "false");

    const data = await getJson("/api/analyze", { method: "POST", body: form });
    state.latest = data;

    setStatusLine(data);
    $("controlTitle").textContent = "Answer ready";
    $("statusAside").textContent = `elapsed ${data.timing?.total_seconds ?? 0}s · ${data.cache_hit ? "cache hit" : "cache miss"}`;
    renderTimeline("Answer");
    renderAnswer(data);
    renderPipeline(data);
    renderTrace(data);
    renderTelemetry(data.telemetry || {});
    renderBaseline(data);
    renderWarnings(data);
  } catch (error) {
    $("errorCard").textContent = String(error.message || error);
    $("errorCard").classList.remove("hidden");
    $("controlTitle").textContent = "Run failed";
    $("statusAside").textContent = "see error above";
    renderTimeline("idle");
  } finally {
    setRunning(false);
  }
}

/* ─────────── events ─────────── */
function wireEvents() {
  $("analyzeTab").addEventListener("click", () => setActiveView("analyze"));
  $("compareTab").addEventListener("click", () => setActiveView("compare"));
  window.addEventListener("resize", () => moveTabIndicator());
  $("analyzeButton").addEventListener("click", analyze);
  $("startEvalButton").addEventListener("click", startEvaluation);
  $("stopEvalButton").addEventListener("click", stopEvaluation);
  $("sampleSelect").addEventListener("change", () => {
    state.uploadedFile = null;
    if (state.uploadedUrl) URL.revokeObjectURL(state.uploadedUrl);
    state.uploadedUrl = "";
    $("uploadFilename").textContent = "";
    $("questionInput").value = "";
    renderPreview();
  });
  $("diagramUpload").addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    if (state.uploadedUrl) URL.revokeObjectURL(state.uploadedUrl);
    state.uploadedFile = file;
    state.uploadedUrl = file.type.startsWith("image/") ? URL.createObjectURL(file) : "";
    $("uploadFilename").textContent = file.name;
    renderPreview();
  });
  $("questionInput").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      analyze();
    }
  });
}

/* ─────────── placeholder rotation ─────────── */
function startPlaceholderRotation() {
  const input = $("questionInput");
  if (!input) return;
  let index = Math.floor(Math.random() * EXAMPLES.length);
  input.placeholder = EXAMPLES[index];
  state.placeholderTimer = setInterval(() => {
    if (document.activeElement === input || input.value.trim()) return;
    index = (index + 1) % EXAMPLES.length;
    input.placeholder = EXAMPLES[index];
  }, 6000);
  input.addEventListener("focus", () => {
    clearInterval(state.placeholderTimer);
    state.placeholderTimer = null;
  });
}

async function init() {
  wireEvents();
  startPlaceholderRotation();
  renderTimeline("idle");
  requestAnimationFrame(() => moveTabIndicator());
  try {
    const data = await getJson("/api/demo/state");
    state.diagrams = data.diagrams || [];
    state.questions = data.questions || [];
    renderSamples();
    setStatusLine(data);
    fetchEvalStatus();
  } catch (error) {
    $("errorCard").textContent = String(error.message || error);
    $("errorCard").classList.remove("hidden");
  }
}

init();
