/**
 * GTAE-IDS Network Intrusion Detection System - Dynamic Snapshot Client Logic.
 * Supports selecting, loading, and evaluating any real temporal graph snapshot
 * from LSPR23 through the clean GTAE PyTorch pipeline.
 */

let state = {
  graphData: null,
  gtaeData: null,
  currentSnapshotIndex: 46,
  ocsvmData: null,
  currentOCSVMSplit: "test",
  iforestData: null,
  currentIForestSplit: "test",
  hbosData: null,
  currentHBOSSplit: "test",
  inneData: null,
  currentINNESplit: "test",
};

document.addEventListener("DOMContentLoaded", () => {
  setupEventHandlers();
  const initialIndex = getSelectedSnapshotIndex();
  processSnapshot(initialIndex);
});

function getSelectedSnapshotIndex() {
  const select = document.getElementById("snapshot-select");
  if (!select) return 46;
  const val = parseInt(select.value, 10);
  return isNaN(val) ? 46 : val;
}

function setupEventHandlers() {
  const btnProcess = document.getElementById("btn-process-snapshot");
  const btnLoad = document.getElementById("btn-load-graph");
  const btnRun = document.getElementById("btn-run-gtae");
  const select = document.getElementById("snapshot-select");

  if (btnProcess) {
    btnProcess.addEventListener("click", () => {
      const idx = getSelectedSnapshotIndex();
      processSnapshot(idx);
    });
  }

  if (btnLoad) {
    btnLoad.addEventListener("click", () => {
      const idx = getSelectedSnapshotIndex();
      loadGraph(idx);
    });
  }

  if (btnRun) {
    btnRun.addEventListener("click", () => {
      const idx = getSelectedSnapshotIndex();
      runGTAE(idx);
    });
  }

  if (select) {
    select.addEventListener("change", () => {
      const idx = getSelectedSnapshotIndex();
      resetSnapshotView(idx);
      loadGraph(idx);
    });
  }
}

function clearCanvas(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function resetSnapshotView(idx) {
  state.graphData = null;
  state.gtaeData = null;
  state.ocsvmData = null;
  state.iforestData = null;
  state.hbosData = null;
  state.inneData = null;

  resetPipelineSteps();

  const snapIdxEl = document.getElementById("val-snap-idx");
  if (snapIdxEl) snapIdxEl.textContent = idx;
  const winIdEl = document.getElementById("val-win-id");
  if (winIdEl) winIdEl.textContent = "Loading...";
  const timeRangeEl = document.getElementById("val-time-range");
  if (timeRangeEl) timeRangeEl.textContent = "Loading...";

  ["val-num-nodes", "val-num-edges", "val-node-shape", "val-edge-shape", "val-benign-count", "val-attack-count"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  const btnRun = document.getElementById("btn-run-gtae");
  if (btnRun) btnRun.disabled = true;

  const provBadge = document.getElementById("badge-ocsvm-provenance");
  if (provBadge) provBadge.textContent = `Snapshot ${idx} (Awaiting GTAE execution...)`;
  const snapSummaryIdx = document.getElementById("summary-snap-idx");
  if (snapSummaryIdx) snapSummaryIdx.textContent = idx;

  ["summary-total-flows", "summary-benign-flows", "summary-attack-flows", "summary-detected-flows", "summary-tp", "summary-fp"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  ["val-ocsvm-prec", "val-ocsvm-rec", "val-ocsvm-f1", "val-ocsvm-fpr"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  const iforestProvBadge = document.getElementById("badge-iforest-provenance");
  if (iforestProvBadge) iforestProvBadge.textContent = `Snapshot ${idx} (Awaiting GTAE execution...)`;
  const iforestSnapSummaryIdx = document.getElementById("summary-iforest-snap-idx");
  if (iforestSnapSummaryIdx) iforestSnapSummaryIdx.textContent = idx;

  ["summary-iforest-total-flows", "summary-iforest-benign-flows", "summary-iforest-attack-flows", "summary-iforest-detected-flows", "summary-iforest-tp", "summary-iforest-fp"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  ["val-iforest-prec", "val-iforest-rec", "val-iforest-f1", "val-iforest-fpr"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  const inneProvBadge = document.getElementById("badge-inne-provenance");
  if (inneProvBadge) inneProvBadge.textContent = `Snapshot ${idx} (Awaiting GTAE execution...)`;
  const inneSnapSummaryIdx = document.getElementById("summary-inne-snap-idx");
  if (inneSnapSummaryIdx) inneSnapSummaryIdx.textContent = idx;

  ["summary-inne-total-flows", "summary-inne-benign-flows", "summary-inne-attack-flows", "summary-inne-detected-flows", "summary-inne-tp", "summary-inne-fp"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  ["val-inne-prec", "val-inne-rec", "val-inne-f1", "val-inne-fpr"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  const hbosProvBadge = document.getElementById("badge-hbos-provenance");
  if (hbosProvBadge) hbosProvBadge.textContent = `Snapshot ${idx} (Awaiting GTAE execution...)`;
  const hbosSnapSummaryIdx = document.getElementById("summary-hbos-snap-idx");
  if (hbosSnapSummaryIdx) hbosSnapSummaryIdx.textContent = idx;

  ["summary-hbos-total-flows", "summary-hbos-benign-flows", "summary-hbos-attack-flows", "summary-hbos-detected-flows", "summary-hbos-tp", "summary-hbos-fp"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  ["val-hbos-prec", "val-hbos-rec", "val-hbos-f1", "val-hbos-fpr"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "-";
  });

  ["graph-canvas", "latent-canvas", "recon-canvas-orig", "recon-canvas-pred", "hist-canvas", "features-canvas", "ocsvm-boundary-canvas", "iforest-pca-canvas", "hbos-hist-canvas", "inne-pca-canvas"].forEach(clearCanvas);

  const tooltip = document.getElementById("boundary-tooltip");
  if (tooltip) tooltip.style.display = "none";
  const iforestTooltip = document.getElementById("iforest-tooltip");
  if (iforestTooltip) iforestTooltip.style.display = "none";
  const hbosTooltip = document.getElementById("hbos-tooltip");
  if (hbosTooltip) hbosTooltip.style.display = "none";
  const inneTooltip = document.getElementById("inne-tooltip");
  if (inneTooltip) inneTooltip.style.display = "none";


}

function updateStatus(message, type = "ready") {
  const statusText = document.getElementById("status-text");
  if (statusText) {
    statusText.textContent = message;
    if (type === "error") statusText.style.color = "#dc2626";
    else if (type === "running") statusText.style.color = "#2563eb";
    else statusText.style.color = "#111827";
  }
}

function setPipelineStep(stepId, status) {
  const step = document.getElementById(stepId);
  if (!step) return;
  step.classList.remove("active", "completed");
  if (status) step.classList.add(status);
}

function resetPipelineSteps() {
  ["step-lspr23", "step-graph", "step-gtae", "step-latent", "step-recon", "step-error", "step-features", "step-ocsvm", "step-iforest", "step-hbos", "step-inne"].forEach((s) => setPipelineStep(s, null));
}

async function processSnapshot(snapshotIndex) {
  const btnProcess = document.getElementById("btn-process-snapshot");
  const btnLoad = document.getElementById("btn-load-graph");
  const btnRun = document.getElementById("btn-run-gtae");

  if (btnProcess) btnProcess.disabled = true;
  if (btnLoad) btnLoad.disabled = true;
  if (btnRun) btnRun.disabled = true;

  try {
    state.currentSnapshotIndex = snapshotIndex;
    updateStatus(`Processing Snapshot ${snapshotIndex}...`, "running");

    resetPipelineSteps();
    setPipelineStep("step-lspr23", "active");
    setPipelineStep("step-graph", "active");

    const response = await fetch(`/api/process_snapshot?index=${snapshotIndex}`);
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || `Server returned HTTP ${response.status}`);
    }

    const res = await response.json();
    if (res.status !== "success") throw new Error(res.message || "Failed processing snapshot");

    state.graphData = res.graph;
    state.gtaeData = res.gtae;

    updateGraphMetadata(res.graph);
    renderGraph(res.graph);
    setPipelineStep("step-lspr23", "completed");
    setPipelineStep("step-graph", "completed");

    setPipelineStep("step-gtae", "active");
    document.getElementById("val-device").textContent = res.gtae.device;
    document.getElementById("val-gpu").textContent = res.gtae.gpu_name;
    document.getElementById("val-params").textContent = Number(res.gtae.param_count).toLocaleString();
    setPipelineStep("step-gtae", "completed");

    setPipelineStep("step-latent", "active");
    renderLatentHeatmap(res.gtae.latent_representation.data);
    setPipelineStep("step-latent", "completed");

    setPipelineStep("step-recon", "active");
    renderReconstructionHeatmaps(res.gtae.reconstruction.original, res.gtae.reconstruction.reconstructed);
    setPipelineStep("step-recon", "completed");

    setPipelineStep("step-error", "active");
    document.getElementById("val-mean-err").textContent = res.gtae.reconstruction_error.mean.toFixed(6);
    document.getElementById("val-median-err").textContent = res.gtae.reconstruction_error.median.toFixed(6);
    document.getElementById("val-max-err").textContent = res.gtae.reconstruction_error.max.toFixed(6);
    renderHistogram(res.gtae.reconstruction_error.histogram, res.gtae.reconstruction_error.mean, res.gtae.reconstruction_error.median);
    setPipelineStep("step-error", "completed");

    setPipelineStep("step-features", "active");
    render113DFeatures(res.gtae.feature_vector_113d.full_113);
    setPipelineStep("step-features", "completed");

    if (res.ocsvm) {
      setPipelineStep("step-ocsvm", "active");
      renderOCSVMSnapshot(res.ocsvm);
      setPipelineStep("step-ocsvm", "completed");
    }

    if (res.iforest) {
      setPipelineStep("step-iforest", "active");
      renderIForestSnapshot(res.iforest);
      setPipelineStep("step-iforest", "completed");
    }

    if (res.hbos) {
      setPipelineStep("step-hbos", "active");
      renderHBOSSnapshot(res.hbos);
      setPipelineStep("step-hbos", "completed");
    }

    if (res.inne) {
      setPipelineStep("step-inne", "active");
      renderINNESnapshot(res.inne);
      setPipelineStep("step-inne", "completed");
    }

    if (btnRun) btnRun.disabled = false;
    updateStatus(`Snapshot ${snapshotIndex} processed successfully: GTAE 113-D features, OCSVM, IForest, HBOS & INNE ready.`, "ready");
  } catch (err) {
    console.error(`Error processing snapshot ${snapshotIndex}:`, err);
    updateStatus(`Error: ${err.message}`, "error");
  } finally {
    if (btnProcess) btnProcess.disabled = false;
    if (btnLoad) btnLoad.disabled = false;
  }
}

async function loadGraph(snapshotIndex) {
  const btnLoad = document.getElementById("btn-load-graph");
  if (btnLoad) btnLoad.disabled = true;

  try {
    state.currentSnapshotIndex = snapshotIndex;
    updateStatus(`Loading Graph for Snapshot ${snapshotIndex}...`, "running");
    setPipelineStep("step-lspr23", "active");
    setPipelineStep("step-graph", "active");

    const response = await fetch(`/api/graph?index=${snapshotIndex}`);
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || `Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    if (data.status !== "success") throw new Error(data.message || "Failed loading graph");

    state.graphData = data;
    updateGraphMetadata(data);
    renderGraph(data);

    setPipelineStep("step-lspr23", "completed");
    setPipelineStep("step-graph", "completed");

    const btnRun = document.getElementById("btn-run-gtae");
    if (btnRun) btnRun.disabled = false;

    updateStatus(`Snapshot ${snapshotIndex} graph loaded (Window ${data.window_id}: ${data.num_nodes} hosts, ${data.num_edges} flows). Ready to run GTAE.`, "ready");
    return data;
  } catch (err) {
    console.error("Error loading graph:", err);
    updateStatus(`Error loading graph: ${err.message}`, "error");
  } finally {
    if (btnLoad) btnLoad.disabled = false;
  }
}

async function runGTAE(snapshotIndex) {
  const btnRun = document.getElementById("btn-run-gtae");
  if (btnRun) btnRun.disabled = true;

  try {
    updateStatus(`Executing GTAE forward pass on Snapshot ${snapshotIndex}...`, "running");
    setPipelineStep("step-gtae", "active");

    const response = await fetch(`/api/run_gtae?index=${snapshotIndex}`, { method: "POST" });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || `Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    if (data.status !== "success") throw new Error(data.message || "Failed running GTAE");

    state.gtaeData = data;

    setPipelineStep("step-gtae", "completed");
    document.getElementById("val-device").textContent = data.device;
    document.getElementById("val-gpu").textContent = data.gpu_name;
    document.getElementById("val-params").textContent = Number(data.param_count).toLocaleString();

    setPipelineStep("step-latent", "active");
    renderLatentHeatmap(data.latent_representation.data);
    setPipelineStep("step-latent", "completed");

    setPipelineStep("step-recon", "active");
    renderReconstructionHeatmaps(data.reconstruction.original, data.reconstruction.reconstructed);
    setPipelineStep("step-recon", "completed");

    setPipelineStep("step-error", "active");
    document.getElementById("val-mean-err").textContent = data.reconstruction_error.mean.toFixed(6);
    document.getElementById("val-median-err").textContent = data.reconstruction_error.median.toFixed(6);
    document.getElementById("val-max-err").textContent = data.reconstruction_error.max.toFixed(6);
    renderHistogram(data.reconstruction_error.histogram, data.reconstruction_error.mean, data.reconstruction_error.median);
    setPipelineStep("step-error", "completed");

    setPipelineStep("step-features", "active");
    render113DFeatures(data.feature_vector_113d.full_113);
    setPipelineStep("step-features", "completed");

    if (data.ocsvm) {
      setPipelineStep("step-ocsvm", "active");
      renderOCSVMSnapshot(data.ocsvm);
      setPipelineStep("step-ocsvm", "completed");
    }

    if (data.iforest) {
      setPipelineStep("step-iforest", "active");
      renderIForestSnapshot(data.iforest);
      setPipelineStep("step-iforest", "completed");
    }

    if (data.hbos) {
      setPipelineStep("step-hbos", "active");
      renderHBOSSnapshot(data.hbos);
      setPipelineStep("step-hbos", "completed");
    }

    if (data.inne) {
      setPipelineStep("step-inne", "active");
      renderINNESnapshot(data.inne);
      setPipelineStep("step-inne", "completed");
    }

    updateStatus(`Snapshot ${snapshotIndex} GTAE forward pass & detectors completed successfully on ${data.gpu_name}.`, "ready");
    return data;
  } catch (err) {
    console.error("Error running GTAE:", err);
    updateStatus(`Error executing GTAE: ${err.message}`, "error");
  } finally {
    if (btnRun) btnRun.disabled = false;
  }
}


function updateGraphMetadata(data) {
  document.getElementById("val-snap-idx").textContent = data.snapshot_index;
  document.getElementById("val-win-id").textContent = data.window_id;
  document.getElementById("val-time-range").textContent = data.time_range;
  document.getElementById("val-num-nodes").textContent = data.num_nodes;
  document.getElementById("val-num-edges").textContent = data.num_edges;
  document.getElementById("val-node-shape").textContent = `[${data.node_feature_shape.join(", ")}]`;
  document.getElementById("val-edge-shape").textContent = `[${data.edge_feature_shape.join(", ")}]`;
  document.getElementById("val-benign-count").textContent = `${data.benign_count} flows`;
  document.getElementById("val-attack-count").textContent = `${data.attack_count} flows`;
}

// ----------------------------------------------------------------------------
// GRAPH VISUALIZATION (HTML5 Canvas on White Background)
// ----------------------------------------------------------------------------
function renderGraph(graphData) {
  const canvas = document.getElementById("graph-canvas");
  const ctx = canvas.getContext("2d");
  const tooltip = document.getElementById("graph-tooltip");

  const width = canvas.width;
  const height = canvas.height;
  const padding = 40;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  // Normalize coordinates from NetworkX layout [-1, 1] to canvas space
  const nodes = graphData.nodes;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  nodes.forEach((n) => {
    if (n.x < minX) minX = n.x;
    if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.y > maxY) maxY = n.y;
  });

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const nodeMap = new Map();
  nodes.forEach((n) => {
    const cx = padding + ((n.x - minX) / rangeX) * (width - 2 * padding);
    const cy = padding + ((n.y - minY) / rangeY) * (height - 2 * padding);
    nodeMap.set(n.id, { ...n, cx, cy });
  });

  // 1. Draw directed edges
  const edges = graphData.edges;
  edges.forEach((edge) => {
    const src = nodeMap.get(edge.source);
    const dst = nodeMap.get(edge.target);
    if (!src || !dst) return;

    ctx.beginPath();
    ctx.moveTo(src.cx, src.cy);

    // Slight curvature
    const midX = (src.cx + dst.cx) / 2;
    const midY = (src.cy + dst.cy) / 2;
    const dx = dst.cx - src.cx;
    const dy = dst.cy - src.cy;
    const normalX = -dy * 0.14;
    const normalY = dx * 0.14;
    const cpX = midX + normalX;
    const cpY = midY + normalY;

    ctx.quadraticCurveTo(cpX, cpY, dst.cx, dst.cy);

    if (edge.label === 1) {
      // Malicious flow: Red
      ctx.strokeStyle = "rgba(220, 38, 38, 0.85)";
      ctx.lineWidth = 1.8;
    } else {
      // Benign flow: Blue
      ctx.strokeStyle = "rgba(37, 99, 235, 0.45)";
      ctx.lineWidth = 1.0;
    }
    ctx.stroke();

    // Arrowhead near destination
    drawArrowhead(ctx, cpX, cpY, dst.cx, dst.cy, edge.label === 1 ? "#dc2626" : "#2563eb");
  });

  // 2. Draw nodes
  nodeMap.forEach((node) => {
    const radius = Math.max(4.5, Math.min(13, 5 + (node.degree || 1) * 0.7));
    ctx.beginPath();
    ctx.arc(node.cx, node.cy, radius, 0, 2 * Math.PI);
    ctx.fillStyle = "#2563eb";
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "#1e40af";
    ctx.stroke();
  });

  // 3. Label key nodes (highest degree hosts)
  const sortedNodes = [...nodeMap.values()].sort((a, b) => b.degree - a.degree).slice(0, 6);
  ctx.font = "bold 9px monospace";
  sortedNodes.forEach((node) => {
    const text = `IP: ${node.ip}`;
    const textWidth = ctx.measureText(text).width;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(node.cx - textWidth / 2 - 4, node.cy - 16, textWidth + 8, 13);
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 0.8;
    ctx.strokeRect(node.cx - textWidth / 2 - 4, node.cy - 16, textWidth + 8, 13);

    ctx.fillStyle = "#111827";
    ctx.textAlign = "center";
    ctx.fillText(text, node.cx, node.cy - 6);
  });

  // Hover interaction
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    let hoveredNode = null;
    for (const node of nodeMap.values()) {
      const dist = Math.hypot(node.cx - mx, node.cy - my);
      if (dist <= 14) {
        hoveredNode = node;
        break;
      }
    }

    if (hoveredNode) {
      tooltip.style.display = "block";
      tooltip.style.left = `${e.clientX + 10}px`;
      tooltip.style.top = `${e.clientY + 10}px`;
      tooltip.innerHTML = `<strong>Host Node #${hoveredNode.id}</strong><br>IP: ${hoveredNode.ip}<br>Degree: ${hoveredNode.degree} flows`;
    } else {
      tooltip.style.display = "none";
    }
  };

  canvas.onmouseleave = () => {
    tooltip.style.display = "none";
  };
}

function drawArrowhead(ctx, fromX, fromY, toX, toY, color) {
  const headlen = 6;
  const angle = Math.atan2(toY - fromY, toX - fromX);
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headlen * Math.cos(angle - Math.PI / 7), toY - headlen * Math.sin(angle - Math.PI / 7));
  ctx.lineTo(toX - headlen * Math.cos(angle + Math.PI / 7), toY - headlen * Math.sin(angle + Math.PI / 7));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

// ----------------------------------------------------------------------------
// 32-D LATENT HEATMAP
// ----------------------------------------------------------------------------
function renderLatentHeatmap(latent2D) {
  const canvas = document.getElementById("latent-canvas");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const numRows = latent2D.length; // 25 flows
  const numCols = latent2D[0].length; // 32 latent dimensions

  let minVal = Infinity, maxVal = -Infinity;
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      const v = latent2D[r][c];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }

  const padLeft = 32;
  const padBottom = 22;
  const cellW = (width - padLeft - 6) / numCols;
  const cellH = (height - padBottom - 6) / numRows;

  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      const v = latent2D[r][c];
      const norm = (v - minVal) / (maxVal - minVal || 1);
      ctx.fillStyle = viridisColor(norm);
      ctx.fillRect(padLeft + c * cellW, 4 + r * cellH, cellW - 0.4, cellH - 0.4);
    }
  }

  // Axes labels
  ctx.fillStyle = "#4b5563";
  ctx.font = "8px monospace";
  ctx.textAlign = "center";
  for (let c = 0; c < numCols; c += 4) {
    ctx.fillText(`z${c}`, padLeft + (c + 0.5) * cellW, height - 6);
  }

  ctx.textAlign = "right";
  for (let r = 0; r < numRows; r += 5) {
    ctx.fillText(`f${r}`, padLeft - 4, 10 + (r + 0.5) * cellH);
  }
}

// ----------------------------------------------------------------------------
// RECONSTRUCTION COMPARISON
// ----------------------------------------------------------------------------
function renderReconstructionHeatmaps(orig2D, pred2D) {
  renderMatrixOnCanvas("orig-recon-canvas", orig2D);
  renderMatrixOnCanvas("pred-recon-canvas", pred2D);
}

function renderMatrixOnCanvas(canvasId, matrix) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const numRows = matrix.length; // 20
  const numCols = matrix[0].length; // 81

  let minVal = -1.0, maxVal = 10.0;
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      const v = matrix[r][c];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }

  const padLeft = 20;
  const padBottom = 18;
  const cellW = (width - padLeft - 4) / numCols;
  const cellH = (height - padBottom - 4) / numRows;

  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      const v = matrix[r][c];
      const norm = Math.max(0, Math.min(1, (v - minVal) / (maxVal - minVal || 1)));
      ctx.fillStyle = plasmaColor(norm);
      ctx.fillRect(padLeft + c * cellW, 4 + r * cellH, cellW, cellH);
    }
  }

  // Axes ticks
  ctx.fillStyle = "#4b5563";
  ctx.font = "8px monospace";
  ctx.textAlign = "center";
  ctx.fillText("0", padLeft, height - 5);
  ctx.fillText("40", padLeft + (width - padLeft) / 2, height - 5);
  ctx.fillText("80", width - 8, height - 5);

  ctx.textAlign = "right";
  ctx.fillText("0", padLeft - 3, 12);
  ctx.fillText(`${numRows - 1}`, padLeft - 3, height - padBottom);
}

// ----------------------------------------------------------------------------
// RECONSTRUCTION ERROR HISTOGRAM
// ----------------------------------------------------------------------------
function renderHistogram(histData, meanVal, medianVal) {
  const canvas = document.getElementById("hist-canvas");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const counts = histData.counts;
  const binEdges = histData.bin_edges;
  const maxCount = Math.max(...counts, 1);
  const minEdge = binEdges[0];
  const maxEdge = binEdges[binEdges.length - 1];

  const padLeft = 36;
  const padBottom = 22;
  const padTop = 10;
  const chartW = width - padLeft - 12;
  const chartH = height - padBottom - padTop;

  const barW = chartW / counts.length;

  // Draw histogram bars
  for (let i = 0; i < counts.length; i++) {
    const barH = (counts[i] / maxCount) * chartH;
    const x = padLeft + i * barW;
    const y = padTop + chartH - barH;

    ctx.fillStyle = "rgba(37, 99, 235, 0.65)";
    ctx.fillRect(x, y, barW - 1, barH);
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 0.8;
    ctx.strokeRect(x, y, barW - 1, barH);
  }

  // Mean line (Red)
  const meanX = padLeft + ((meanVal - minEdge) / (maxEdge - minEdge || 1)) * chartW;
  ctx.beginPath();
  ctx.setLineDash([4, 3]);
  ctx.moveTo(meanX, padTop);
  ctx.lineTo(meanX, padTop + chartH);
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 1.8;
  ctx.stroke();

  // Median line (Green)
  const medianX = padLeft + ((medianVal - minEdge) / (maxEdge - minEdge || 1)) * chartW;
  ctx.beginPath();
  ctx.setLineDash([2, 3]);
  ctx.moveTo(medianX, padTop);
  ctx.lineTo(medianX, padTop + chartH);
  ctx.strokeStyle = "#16a34a";
  ctx.lineWidth = 1.8;
  ctx.stroke();
  ctx.setLineDash([]);

  // Legend annotations inside canvas
  ctx.fillStyle = "#dc2626";
  ctx.font = "bold 9px Arial";
  ctx.textAlign = "right";
  ctx.fillText(`Mean: ${meanVal.toFixed(4)}`, width - 12, padTop + 14);

  ctx.fillStyle = "#16a34a";
  ctx.fillText(`Median: ${medianVal.toFixed(4)}`, width - 12, padTop + 26);

  // Axes ticks
  ctx.fillStyle = "#4b5563";
  ctx.font = "8px monospace";
  ctx.textAlign = "center";
  ctx.fillText(minEdge.toFixed(2), padLeft, height - 6);
  ctx.fillText(((minEdge + maxEdge) / 2).toFixed(2), padLeft + chartW / 2, height - 6);
  ctx.fillText(maxEdge.toFixed(2), padLeft + chartW, height - 6);

  ctx.textAlign = "right";
  ctx.fillText(maxCount, padLeft - 4, padTop + 8);
  ctx.fillText("0", padLeft - 4, padTop + chartH);
}

// ----------------------------------------------------------------------------
// 113-D FEATURE VECTOR VISUALIZATION
// ----------------------------------------------------------------------------
function render113DFeatures(full113) {
  const canvas = document.getElementById("feature-canvas");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const numDims = full113.length; // 113
  let minVal = 0, maxVal = -Infinity;
  full113.forEach((v) => {
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  });

  const padLeft = 36;
  const padBottom = 22;
  const padTop = 10;
  const chartW = width - padLeft - 12;
  const chartH = height - padBottom - padTop;
  const barW = chartW / numDims;

  const range = maxVal - minVal || 1;
  const zeroY = padTop + chartH - ((0 - minVal) / range) * chartH;

  // Draw baseline
  ctx.strokeStyle = "#d1d5db";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, zeroY);
  ctx.lineTo(padLeft + chartW, zeroY);
  ctx.stroke();

  // Draw 113 bars
  for (let i = 0; i < numDims; i++) {
    const val = full113[i];
    const barH = Math.abs(val / range) * chartH;
    const x = padLeft + i * barW;
    const y = val >= 0 ? zeroY - barH : zeroY;

    if (i < 32) {
      // Latent dimensions (1-32): Blue
      ctx.fillStyle = "#2563eb";
      ctx.strokeStyle = "#1d4ed8";
    } else {
      // Reconstruction residual (33-113): Orange
      ctx.fillStyle = "#d97706";
      ctx.strokeStyle = "#b45309";
    }

    ctx.fillRect(x, y, Math.max(1, barW - 0.4), Math.max(1, barH));
  }

  // Divider line at index 32
  const divX = padLeft + 32 * barW;
  ctx.beginPath();
  ctx.setLineDash([3, 3]);
  ctx.moveTo(divX, padTop);
  ctx.lineTo(divX, padTop + chartH);
  ctx.strokeStyle = "#374151";
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.setLineDash([]);

  // Ticks
  ctx.fillStyle = "#4b5563";
  ctx.font = "8px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Dim 1", padLeft + 6, height - 6);
  ctx.fillText("Dim 32", divX, height - 6);
  ctx.fillText("Dim 113", padLeft + chartW - 14, height - 6);

  ctx.textAlign = "right";
  ctx.fillText(maxVal.toFixed(1), padLeft - 4, padTop + 8);
  ctx.fillText("0.0", padLeft - 4, zeroY + 3);
}

// ----------------------------------------------------------------------------
// COLORMAPS
// ----------------------------------------------------------------------------
function viridisColor(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(84 + t * (37 - 84));
  return `rgb(${r}, ${g}, ${b})`;
}

function plasmaColor(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(13 + t * (240 - 13));
  const g = Math.round(8 + t * (249 - 8));
  const b = Math.round(135 + t * (33 - 135));
  return `rgb(${r}, ${g}, ${b})`;
}


function renderOCSVMSnapshot(ocsvm) {
  if (!ocsvm) return;
  state.ocsvmData = ocsvm;

  const provBadge = document.getElementById("badge-ocsvm-provenance");
  if (provBadge) {
    provBadge.textContent = `Snapshot ${ocsvm.snapshot_index} (Window ${ocsvm.window_id} • ${ocsvm.split_name})`;
  }

  const threshBadge = document.getElementById("badge-ocsvm-thresh");
  if (threshBadge) {
    threshBadge.textContent = `τ* = ${ocsvm.frozen_threshold.toFixed(4)} (99th pct benign)`;
  }

  const snapSummaryIdx = document.getElementById("summary-snap-idx");
  if (snapSummaryIdx) snapSummaryIdx.textContent = ocsvm.snapshot_index;

  const totalFlows = document.getElementById("summary-total-flows");
  if (totalFlows) totalFlows.textContent = ocsvm.counts.total_flows.toLocaleString();

  const benignFlows = document.getElementById("summary-benign-flows");
  if (benignFlows) benignFlows.textContent = ocsvm.counts.benign_total.toLocaleString();

  const attackFlows = document.getElementById("summary-attack-flows");
  if (attackFlows) attackFlows.textContent = ocsvm.counts.attack_total.toLocaleString();

  const detectedFlows = document.getElementById("summary-detected-flows");
  if (detectedFlows) detectedFlows.textContent = ocsvm.counts.detected_anomalies.toLocaleString();

  const tpEl = document.getElementById("summary-tp");
  if (tpEl) tpEl.textContent = ocsvm.counts.true_positives.toLocaleString();

  const fpEl = document.getElementById("summary-fp");
  if (fpEl) fpEl.textContent = ocsvm.counts.false_positives.toLocaleString();

  const precEl = document.getElementById("val-ocsvm-prec");
  if (precEl) precEl.textContent = (ocsvm.metrics.precision * 100).toFixed(2) + "%";

  const recEl = document.getElementById("val-ocsvm-rec");
  if (recEl) recEl.textContent = (ocsvm.metrics.recall * 100).toFixed(2) + "%";

  const f1El = document.getElementById("val-ocsvm-f1");
  if (f1El) f1El.textContent = ocsvm.metrics.f1_score.toFixed(4);

  const fprEl = document.getElementById("val-ocsvm-fpr");
  if (fprEl) fprEl.textContent = (ocsvm.metrics.false_positive_rate * 100).toFixed(2) + "%";

  // 4. One-Class SVM Decision Boundary (2D PCA Projection)
  renderOCSVMDecisionBoundary("ocsvm-boundary-canvas", ocsvm.decision_boundary);

  // 5. Flow detections table
  renderOCSVMSnapshotFlowsTable("tbody-detected-flows", ocsvm.flows_table);
}

// ----------------------------------------------------------------------------
// ONE LARGE GRAPH: ONE-CLASS SVM DECISION BOUNDARY (2D PCA PROJECTION)
// ----------------------------------------------------------------------------
let boundaryHoverHandler = null;

function setupBoundaryTooltip(canvas, boundaryData, toCanvasX, toCanvasY) {
  const tooltip = document.getElementById("boundary-tooltip");
  if (!tooltip || !boundaryData) return;

  if (boundaryHoverHandler) {
    canvas.removeEventListener("mousemove", boundaryHoverHandler);
    canvas.removeEventListener("mouseleave", boundaryHoverHandler._leave);
  }

  const points = boundaryData.flow_points || [];

  const handleMouseMove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    // Find nearest flow point
    let nearest = null;
    let minDist = 14;

    for (let pt of points) {
      const cx = toCanvasX(pt.pc1);
      const cy = toCanvasY(pt.pc2);
      const dist = Math.hypot(mouseX - cx, mouseY - cy);
      if (dist < minDist) {
        minDist = dist;
        nearest = pt;
      }
    }

    if (nearest) {
      const isTP = nearest.predicted === 1 && nearest.ground_truth === 1;
      const isFP = nearest.predicted === 1 && nearest.ground_truth === 0;
      const isFN = nearest.predicted === 0 && nearest.ground_truth === 1;

      let statusBadge = isTP ? "True Positive" : (isFP ? "False Alarm" : (isFN ? "Missed Attack" : "Benign Inlier"));
      let badgeColor = isTP ? "#16a34a" : (isFP ? "#dc2626" : (isFN ? "#d97706" : "#2563eb"));

      tooltip.style.display = "block";
      tooltip.style.left = `${(e.clientX - rect.left) + 12}px`;
      tooltip.style.top = `${(e.clientY - rect.top) - 10}px`;
      tooltip.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 2px;">Flow #${nearest.edge_idx}</div>
        <div style="font-size: 10.5px; color: #475569;">${nearest.src} &rarr; ${nearest.dst}</div>
        <div style="font-size: 10.5px; font-family: monospace; margin: 2px 0;">PC1: ${nearest.pc1.toFixed(3)}, PC2: ${nearest.pc2.toFixed(3)}</div>
        <div style="font-size: 10.5px;">Anomaly Score: <strong>${nearest.anomaly_score.toFixed(4)}</strong> <span style="color:#64748b;">(τ*=${boundaryData.frozen_threshold.toFixed(4)})</span></div>
        <div style="font-size: 10.5px; margin-top: 2px;">Ground Truth: <strong>${nearest.ground_truth === 1 ? 'Malicious' : 'Benign'}</strong> | Pred: <strong style="color: ${badgeColor};">${statusBadge}</strong></div>
      `;
    } else {
      tooltip.style.display = "none";
    }
  };

  const handleMouseLeave = () => {
    tooltip.style.display = "none";
  };

  handleMouseMove._leave = handleMouseLeave;
  boundaryHoverHandler = handleMouseMove;
  canvas.addEventListener("mousemove", handleMouseMove);
  canvas.addEventListener("mouseleave", handleMouseLeave);
}

function renderOCSVMDecisionBoundary(canvasId, boundaryData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !boundaryData) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  // Update variance badge if element exists
  const badgeVar = document.getElementById("badge-pca-variance");
  if (badgeVar && boundaryData.pca_variance) {
    badgeVar.textContent = `PC1: ${boundaryData.pca_variance[0]}% | PC2: ${boundaryData.pca_variance[1]}% Var`;
  }

  const [xMin, xMax] = boundaryData.x_range;
  const [yMin, yMax] = boundaryData.y_range;
  const rangeX = xMax - xMin || 1;
  const rangeY = yMax - yMin || 1;

  const padLeft = 60;
  const padRight = 30;
  const padTop = 32;
  const padBottom = 42;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const toCanvasX = (x) => padLeft + ((x - xMin) / rangeX) * chartW;
  const toCanvasY = (y) => padTop + chartH - ((y - yMin) / rangeY) * chartH;

  // 1. Shaded Decision Regions from 2D meshgrid
  const nx = boundaryData.grid_nx;
  const ny = boundaryData.grid_ny;
  const mask = boundaryData.region_mask;

  if (mask && mask.length > 0) {
    const cellPixelW = chartW / nx;
    const cellPixelH = chartH / ny;

    for (let i = 0; i < ny; i++) {
      for (let j = 0; j < nx; j++) {
        const isAnomaly = mask[i][j] === 1;
        const cx = padLeft + j * cellPixelW;
        const cy = padTop + chartH - (i + 1) * cellPixelH;

        if (isAnomaly) {
          // Anomaly rejection region (s > tau*): subtle rose tint
          ctx.fillStyle = "rgba(254, 226, 226, 0.70)";
        } else {
          // Normal acceptance region (s <= tau*): subtle sky blue tint
          ctx.fillStyle = "rgba(224, 242, 254, 0.70)";
        }
        ctx.fillRect(cx - 0.5, cy - 0.5, cellPixelW + 1, cellPixelH + 1);
      }
    }
  }

  // 2. Grid lines & Box border
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, padTop, chartW, chartH);

  // Horizontal grid lines & Y labels
  for (let i = 0; i <= 4; i++) {
    const yVal = yMin + (rangeY / 4) * i;
    const cy = toCanvasY(yVal);
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, cy);
    ctx.lineTo(padLeft + chartW, cy);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "right";
    ctx.fillText(yVal.toFixed(1), padLeft - 6, cy + 3);
  }

  // Vertical grid lines & X labels
  for (let j = 0; j <= 5; j++) {
    const xVal = xMin + (rangeX / 5) * j;
    const cx = toCanvasX(xVal);
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, padTop);
    ctx.lineTo(cx, padTop + chartH);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "center";
    ctx.fillText(xVal.toFixed(1), cx, padTop + chartH + 15);
  }

  // Axis Labels
  ctx.fillStyle = "#334155";
  ctx.font = "bold 10.5px sans-serif";
  ctx.textAlign = "center";
  const varX = boundaryData.pca_variance ? boundaryData.pca_variance[0] : 20.87;
  const varY = boundaryData.pca_variance ? boundaryData.pca_variance[1] : 8.97;
  ctx.fillText(`Principal Component 1 (${varX}% Explained Variance)`, padLeft + chartW / 2, height - 10);

  ctx.save();
  ctx.translate(16, padTop + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`Principal Component 2 (${varY}% Explained Variance)`, 0, 0);
  ctx.restore();

  // 3. Operational Decision Boundary Contour Lines (where s(x) = tau*)
  const segments = boundaryData.boundary_segments || [];
  if (segments.length > 0) {
    ctx.save();
    ctx.strokeStyle = "#dc2626";
    ctx.lineWidth = 2.4;
    ctx.setLineDash([6, 4]);

    segments.forEach(([p1, p2]) => {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(p1[0]), toCanvasY(p1[1]));
      ctx.lineTo(toCanvasX(p2[0]), toCanvasY(p2[1]));
      ctx.stroke();
    });
    ctx.restore();
  }

  // Boundary watermark tag inside chart
  ctx.fillStyle = "rgba(220, 38, 38, 0.9)";
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "right";
  ctx.fillText(`Decision Boundary: τ* = ${boundaryData.frozen_threshold.toFixed(4)}`, padLeft + chartW - 10, padTop + 18);

  // 4. Plot Actual Snapshot Flows
  const points = boundaryData.flow_points || [];
  // Sort so attacks and predicted anomalies draw on top of normal inliers
  const sortedPoints = [...points].sort((a, b) => {
    return (a.predicted * 2 + a.ground_truth) - (b.predicted * 2 + b.ground_truth);
  });

  sortedPoints.forEach((pt) => {
    const cx = toCanvasX(pt.pc1);
    const cy = toCanvasY(pt.pc2);

    if (cx < padLeft || cx > padLeft + chartW || cy < padTop || cy > padTop + chartH) {
      return;
    }

    const isAttack = pt.ground_truth === 1;
    const isAnomalyPred = pt.predicted === 1;
    const radius = isAttack ? 4.5 : 4.0;

    // Draw prediction halo ring if predicted anomalous
    if (isAnomalyPred) {
      ctx.strokeStyle = "#f59e0b"; // Golden amber ring
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw flow center dot
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    if (isAttack) {
      ctx.fillStyle = "#dc2626"; // Crimson
      ctx.strokeStyle = "#7f1d1d";
    } else {
      ctx.fillStyle = "#2563eb"; // Royal blue
      ctx.strokeStyle = "#1e3a8a";
    }
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  });

  // Attach interactive hover listener
  setupBoundaryTooltip(canvas, boundaryData, toCanvasX, toCanvasY);
}

function renderOCSVMSnapshotFlowsTable(tbodyId, flowsList) {
  return;

  if (flowsList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">No flows found for this snapshot.</td></tr>`;
    return;
  }

  const rowsHtml = flowsList.map((f) => {
    return `
      <tr>
        <td style="font-weight: 600;">#${f.rank}</td>
        <td style="font-family: monospace;">${f.edge_idx}</td>
        <td>${f.src}</td>
        <td>${f.dst}</td>
        <td style="font-size: 11px;">${f.timestamp}</td>
        <td style="font-family: monospace;">${f.recon_error.toFixed(4)}</td>
        <td style="font-weight: 700; font-family: monospace; color: ${f.predicted === 1 ? '#dc2626' : '#2563eb'};">${f.anomaly_score.toFixed(4)}</td>
        <td>${f.ground_truth_str}</td>
        <td><span class="${f.badge_class}">${f.status_badge}</span></td>
      </tr>
    `;
  }).join("");

  tbody.innerHTML = rowsHtml;
}


// ============================================================================
// PHASE 4B: ISOLATION FOREST (IFOREST) CLIENT-SIDE DASHBOARD ENGINE
// ============================================================================

/**
 * Render real-time Isolation Forest anomaly detection results for the currently selected snapshot.
 */
function renderIForestSnapshot(iforest) {
  if (!iforest) return;
  state.iforestData = iforest;

  // 1. Provenance & threshold badges
  const provBadge = document.getElementById("badge-iforest-provenance");
  if (provBadge) {
    provBadge.textContent = `Snapshot ${iforest.snapshot_index} (Window ${iforest.window_id} • ${iforest.split_name})`;
  }

  const threshBadge = document.getElementById("badge-iforest-thresh");
  if (threshBadge) {
    threshBadge.textContent = `τ* = ${iforest.frozen_threshold.toFixed(4)} (99th pct benign)`;
  }

  // 2. Summary banner
  const snapSummaryIdx = document.getElementById("summary-iforest-snap-idx");
  if (snapSummaryIdx) snapSummaryIdx.textContent = iforest.snapshot_index;

  const totalFlows = document.getElementById("summary-iforest-total-flows");
  if (totalFlows) totalFlows.textContent = iforest.counts.total_flows.toLocaleString();

  const benignFlows = document.getElementById("summary-iforest-benign-flows");
  if (benignFlows) benignFlows.textContent = iforest.counts.benign_total.toLocaleString();

  const attackFlows = document.getElementById("summary-iforest-attack-flows");
  if (attackFlows) attackFlows.textContent = iforest.counts.attack_total.toLocaleString();

  const detectedFlows = document.getElementById("summary-iforest-detected-flows");
  if (detectedFlows) detectedFlows.textContent = iforest.counts.detected_anomalies.toLocaleString();

  const tpEl = document.getElementById("summary-iforest-tp");
  if (tpEl) tpEl.textContent = iforest.counts.true_positives.toLocaleString();

  const fpEl = document.getElementById("summary-iforest-fp");
  if (fpEl) fpEl.textContent = iforest.counts.false_positives.toLocaleString();

  // 3. Four Important Metrics
  const precEl = document.getElementById("val-iforest-prec");
  if (precEl) {
    precEl.textContent = (iforest.metrics.precision * 100).toFixed(2) + "%";
  }

  const recEl = document.getElementById("val-iforest-rec");
  if (recEl) {
    recEl.textContent = (iforest.metrics.recall * 100).toFixed(2) + "%";
  }

  const f1El = document.getElementById("val-iforest-f1");
  if (f1El) {
    f1El.textContent = iforest.metrics.f1_score.toFixed(4);
  }

  const fprEl = document.getElementById("val-iforest-fpr");
  if (fprEl) {
    fprEl.textContent = (iforest.metrics.false_positive_rate * 100).toFixed(2) + "%";
  }

  // 4. One Primary Graph: Isolation Forest Anomaly Visualization (2D PCA Projection)
  renderIForestPCAScatter("iforest-pca-canvas", iforest.pca_visualization);

  // 5. Flow detections table
  renderIForestSnapshotFlowsTable("tbody-iforest-flows", iforest.flows_table);
}

let iforestHoverHandler = null;

function setupIForestTooltip(canvas, pcaData, toCanvasX, toCanvasY) {
  const tooltip = document.getElementById("iforest-tooltip");
  if (!tooltip || !pcaData) return;

  if (iforestHoverHandler) {
    canvas.removeEventListener("mousemove", iforestHoverHandler);
    canvas.removeEventListener("mouseleave", iforestHoverHandler._leave);
  }

  const points = pcaData.flow_points || [];

  const handleMouseMove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    let nearest = null;
    let minDist = 14;

    for (let pt of points) {
      const cx = toCanvasX(pt.pc1);
      const cy = toCanvasY(pt.pc2);
      const dist = Math.hypot(mouseX - cx, mouseY - cy);
      if (dist < minDist) {
        minDist = dist;
        nearest = pt;
      }
    }

    if (nearest) {
      const isTP = nearest.predicted === 1 && nearest.ground_truth === 1;
      const isFP = nearest.predicted === 1 && nearest.ground_truth === 0;
      const isFN = nearest.predicted === 0 && nearest.ground_truth === 1;

      let statusBadge = isTP ? "True Positive" : (isFP ? "False Alarm" : (isFN ? "Missed Attack" : "Benign Inlier"));
      let badgeColor = isTP ? "#16a34a" : (isFP ? "#dc2626" : (isFN ? "#d97706" : "#2563eb"));

      tooltip.style.display = "block";
      tooltip.style.left = `${(e.clientX - rect.left) + 12}px`;
      tooltip.style.top = `${(e.clientY - rect.top) - 10}px`;
      tooltip.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 2px;">Flow #${nearest.edge_idx}</div>
        <div style="font-size: 10.5px; color: #475569;">${nearest.src} &rarr; ${nearest.dst}</div>
        <div style="font-size: 10.5px; font-family: monospace; margin: 2px 0;">PC1: ${nearest.pc1.toFixed(3)}, PC2: ${nearest.pc2.toFixed(3)}</div>
        <div style="font-size: 10.5px;">IForest Score: <strong>${nearest.anomaly_score.toFixed(4)}</strong> <span style="color:#64748b;">(τ*=${pcaData.frozen_threshold.toFixed(4)})</span></div>
        <div style="font-size: 10.5px; margin-top: 2px;">Ground Truth: <strong>${nearest.ground_truth === 1 ? 'Malicious' : 'Benign'}</strong> | Pred: <strong style="color: ${badgeColor};">${statusBadge}</strong></div>
      `;
    } else {
      tooltip.style.display = "none";
    }
  };

  const handleMouseLeave = () => {
    tooltip.style.display = "none";
  };

  handleMouseMove._leave = handleMouseLeave;
  iforestHoverHandler = handleMouseMove;
  canvas.addEventListener("mousemove", handleMouseMove);
  canvas.addEventListener("mouseleave", handleMouseLeave);
}

function renderIForestPCAScatter(canvasId, pcaData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !pcaData) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  // Update variance badge
  const badgeVar = document.getElementById("badge-iforest-pca-variance");
  if (badgeVar && pcaData.pca_variance) {
    badgeVar.textContent = `PC1: ${pcaData.pca_variance[0]}% | PC2: ${pcaData.pca_variance[1]}% Var`;
  }

  const [xMin, xMax] = pcaData.x_range;
  const [yMin, yMax] = pcaData.y_range;
  const rangeX = xMax - xMin || 1;
  const rangeY = yMax - yMin || 1;

  const padLeft = 60;
  const padRight = 30;
  const padTop = 32;
  const padBottom = 42;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const toCanvasX = (x) => padLeft + ((x - xMin) / rangeX) * chartW;
  const toCanvasY = (y) => padTop + chartH - ((y - yMin) / rangeY) * chartH;

  // Background subtle canvas fill
  ctx.fillStyle = "#fafbfc";
  ctx.fillRect(padLeft, padTop, chartW, chartH);

  // Box border
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, padTop, chartW, chartH);

  // Horizontal grid lines & Y labels
  for (let i = 0; i <= 4; i++) {
    const yVal = yMin + (rangeY / 4) * i;
    const cy = toCanvasY(yVal);
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, cy);
    ctx.lineTo(padLeft + chartW, cy);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "right";
    ctx.fillText(yVal.toFixed(1), padLeft - 6, cy + 3);
  }

  // Vertical grid lines & X labels
  for (let j = 0; j <= 5; j++) {
    const xVal = xMin + (rangeX / 5) * j;
    const cx = toCanvasX(xVal);
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, padTop);
    ctx.lineTo(cx, padTop + chartH);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "center";
    ctx.fillText(xVal.toFixed(1), cx, padTop + chartH + 15);
  }

  // Axis Labels
  ctx.fillStyle = "#334155";
  ctx.font = "bold 10.5px sans-serif";
  ctx.textAlign = "center";
  const varX = pcaData.pca_variance ? pcaData.pca_variance[0] : 21.67;
  const varY = pcaData.pca_variance ? pcaData.pca_variance[1] : 12.54;
  ctx.fillText(`Principal Component 1 (${varX}% Explained Variance)`, padLeft + chartW / 2, height - 10);

  ctx.save();
  ctx.translate(16, padTop + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`Principal Component 2 (${varY}% Explained Variance)`, 0, 0);
  ctx.restore();

  // Watermark tag inside chart
  ctx.fillStyle = "#065f46";
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "right";
  ctx.fillText(`Isolation Forest (100 Trees): τ* = ${pcaData.frozen_threshold.toFixed(4)}`, padLeft + chartW - 10, padTop + 18);

  // Plot Flow Points
  const points = pcaData.flow_points || [];
  // Sort so attacks and predicted anomalies draw on top
  const sortedPoints = [...points].sort((a, b) => {
    return (a.predicted * 2 + a.ground_truth) - (b.predicted * 2 + b.ground_truth);
  });

  sortedPoints.forEach((pt) => {
    const cx = toCanvasX(pt.pc1);
    const cy = toCanvasY(pt.pc2);

    if (cx < padLeft || cx > padLeft + chartW || cy < padTop || cy > padTop + chartH) {
      return;
    }

    const isAttack = pt.ground_truth === 1;
    const isAnomalyPred = pt.predicted === 1;
    const radius = isAttack ? 4.5 : 4.0;

    // Draw prediction halo ring if predicted anomalous
    if (isAnomalyPred) {
      ctx.strokeStyle = "#f59e0b"; // Golden amber ring
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw flow center dot
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    if (isAttack) {
      ctx.fillStyle = "#dc2626"; // Crimson
      ctx.strokeStyle = "#7f1d1d";
    } else {
      ctx.fillStyle = "#2563eb"; // Royal blue
      ctx.strokeStyle = "#1e3a8a";
    }
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  });

  // Attach interactive hover listener
  setupIForestTooltip(canvas, pcaData, toCanvasX, toCanvasY);
}

function renderIForestSnapshotFlowsTable(tbodyId, flowsList) {
  return;

  if (flowsList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">No flows found for this snapshot.</td></tr>`;
    return;
  }

  const rowsHtml = flowsList.map((f) => {
    return `
      <tr>
        <td style="font-weight: 600;">#${f.rank}</td>
        <td style="font-family: monospace;">${f.edge_idx}</td>
        <td>${f.src}</td>
        <td>${f.dst}</td>
        <td style="font-size: 11px;">${f.timestamp}</td>
        <td style="font-family: monospace;">${f.recon_error.toFixed(4)}</td>
        <td style="font-weight: 700; font-family: monospace; color: ${f.predicted === 1 ? '#dc2626' : '#2563eb'};">${f.anomaly_score.toFixed(4)}</td>
        <td>${f.ground_truth_str}</td>
        <td><span class="${f.badge_class}">${f.status_badge}</span></td>
      </tr>
    `;
  }).join("");

  tbody.innerHTML = rowsHtml;
}



// ============================================================================
// PHASE 4C: HISTOGRAM-BASED OUTLIER SCORE (HBOS) CLIENT-SIDE DASHBOARD ENGINE
// ============================================================================

function renderHBOSSnapshot(hbos) {
  if (!hbos) return;
  state.hbosData = hbos;

  const provBadge = document.getElementById("badge-hbos-provenance");
  if (provBadge) {
    provBadge.textContent = `Snapshot ${hbos.snapshot_index} (Window ${hbos.window_id} • ${hbos.split_name})`;
  }

  const threshBadge = document.getElementById("badge-hbos-thresh");
  if (threshBadge) {
    threshBadge.textContent = `τ* = ${hbos.frozen_threshold.toFixed(4)} (99th pct benign)`;
  }

  const snapSummaryIdx = document.getElementById("summary-hbos-snap-idx");
  if (snapSummaryIdx) snapSummaryIdx.textContent = hbos.snapshot_index;

  const totalFlows = document.getElementById("summary-hbos-total-flows");
  if (totalFlows) totalFlows.textContent = hbos.counts.total_flows.toLocaleString();

  const benignFlows = document.getElementById("summary-hbos-benign-flows");
  if (benignFlows) benignFlows.textContent = hbos.counts.benign_total.toLocaleString();

  const attackFlows = document.getElementById("summary-hbos-attack-flows");
  if (attackFlows) attackFlows.textContent = hbos.counts.attack_total.toLocaleString();

  const detectedFlows = document.getElementById("summary-hbos-detected-flows");
  if (detectedFlows) detectedFlows.textContent = hbos.counts.detected_anomalies.toLocaleString();

  const tpEl = document.getElementById("summary-hbos-tp");
  if (tpEl) tpEl.textContent = hbos.counts.true_positives.toLocaleString();

  const fpEl = document.getElementById("summary-hbos-fp");
  if (fpEl) fpEl.textContent = hbos.counts.false_positives.toLocaleString();

  const precEl = document.getElementById("val-hbos-prec");
  if (precEl) precEl.textContent = (hbos.metrics.precision * 100).toFixed(2) + "%";

  const recEl = document.getElementById("val-hbos-rec");
  if (recEl) recEl.textContent = (hbos.metrics.recall * 100).toFixed(2) + "%";

  const f1El = document.getElementById("val-hbos-f1");
  if (f1El) f1El.textContent = hbos.metrics.f1_score.toFixed(4);

  const fprEl = document.getElementById("val-hbos-fpr");
  if (fprEl) fprEl.textContent = (hbos.metrics.false_positive_rate * 100).toFixed(2) + "%";

  // Render Anomaly-Score Histogram
  renderHBOSScoreHistogram("hbos-hist-canvas", hbos.score_histogram);

  renderHBOSSnapshotFlowsTable("tbody-hbos-flows", hbos.flows_table);
}

let hbosHistHoverHandler = null;

function setupHBOSHistTooltip(canvas, histData, toCanvasX, chartW, chartH, padLeft, padTop) {
  const tooltip = document.getElementById("hbos-tooltip");
  if (!tooltip || !histData || !histData.bins) return;

  if (hbosHistHoverHandler) {
    canvas.removeEventListener("mousemove", hbosHistHoverHandler);
    canvas.removeEventListener("mouseleave", hbosHistHoverHandler._leave);
  }

  const bins = histData.bins;
  const thresh = histData.frozen_threshold;

  const handleMouseMove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    if (mouseX < padLeft || mouseX > padLeft + chartW || mouseY < padTop || mouseY > padTop + chartH) {
      tooltip.style.display = "none";
      return;
    }

    let foundBin = null;
    for (let bi = 0; bi < bins.length; bi++) {
      const b = bins[bi];
      const x0 = toCanvasX(b.range[0]);
      const x1 = toCanvasX(b.range[1]);
      if (mouseX >= x0 && mouseX <= x1) {
        foundBin = b;
        break;
      }
    }

    if (foundBin && foundBin.total_count > 0) {
      const isAnomalyBin = foundBin.is_anomaly_region || foundBin.range[1] >= thresh;
      const statusText = isAnomalyBin ? "Predicted Anomaly (s ≥ τ*)" : "Normal / Inlier (s < τ*)";
      const statusColor = isAnomalyBin ? "#d97706" : "#2563eb";

      tooltip.style.display = "block";
      tooltip.style.left = `${(e.clientX - rect.left) + 12}px`;
      tooltip.style.top = `${(e.clientY - rect.top) - 10}px`;
      tooltip.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 2px;">HBOS Score Range: [${foundBin.range[0].toFixed(2)} &ndash; ${foundBin.range[1].toFixed(2)}]</div>
        <div style="font-size: 10.5px; color: #2563eb; margin: 1px 0;">&bull; Benign Flows: <strong>${foundBin.benign_count}</strong></div>
        <div style="font-size: 10.5px; color: #dc2626; margin: 1px 0;">&bull; Attack Flows: <strong>${foundBin.attack_count}</strong></div>
        <div style="font-size: 10.5px; margin-top: 2px;">Total in Bin: <strong>${foundBin.total_count}</strong></div>
        <div style="font-size: 10.5px; margin-top: 3px; border-top: 1px solid #e2e8f0; padding-top: 2px;">
          Region: <strong style="color: ${statusColor};">${statusText}</strong>
        </div>
      `;
    } else {
      tooltip.style.display = "none";
    }
  };

  const handleMouseLeave = () => {
    tooltip.style.display = "none";
  };

  handleMouseMove._leave = handleMouseLeave;
  hbosHistHoverHandler = handleMouseMove;
  canvas.addEventListener("mousemove", handleMouseMove);
  canvas.addEventListener("mouseleave", handleMouseLeave);
}

function renderHBOSScoreHistogram(canvasId, histData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !histData || !histData.bins) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const padLeft = 60;
  const padRight = 40;
  const padTop = 38;
  const padBottom = 48;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const bins = histData.bins;
  const numBins = bins.length;
  const threshVal = histData.frozen_threshold;

  const minX = histData.hist_min !== undefined ? histData.hist_min : histData.bin_edges[0];
  const maxX = histData.hist_max !== undefined ? histData.hist_max : histData.bin_edges[histData.bin_edges.length - 1];
  const rangeX = maxX - minX || 1;

  const maxCount = Math.max(histData.max_bin_count || 1, 1);

  const toCanvasX = (score) => padLeft + ((score - minX) / rangeX) * chartW;

  ctx.fillStyle = "#fafbfc";
  ctx.fillRect(padLeft, padTop, chartW, chartH);

  // Anomaly Region Shading (s >= threshVal)
  const threshX = toCanvasX(threshVal);
  if (threshX < padLeft + chartW) {
    const anomRegionX = Math.max(padLeft, threshX);
    const anomRegionW = (padLeft + chartW) - anomRegionX;
    ctx.fillStyle = "rgba(245, 158, 11, 0.08)";
    ctx.fillRect(anomRegionX, padTop, anomRegionW, chartH);

    ctx.fillStyle = "#b45309";
    ctx.font = "bold 9.5px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("PREDICTED ANOMALY REGION (s ≥ τ*)", padLeft + chartW - 10, padTop + 16);
  }

  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, padTop, chartW, chartH);

  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const countVal = Math.round((maxCount / yTicks) * i);
    const cy = padTop + chartH - (i / yTicks) * chartH;

    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, cy);
    ctx.lineTo(padLeft + chartW, cy);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "right";
    ctx.fillText(countVal.toString(), padLeft - 8, cy + 3.5);
  }

  // Draw Stacked Histogram Bars (Benign Blue + Malicious Crimson Red)
  bins.forEach((b) => {
    const x0 = toCanvasX(b.range[0]);
    const x1 = toCanvasX(b.range[1]);
    const barWidth = Math.max(1, (x1 - x0) - 2);

    const benignH = (b.benign_count / maxCount) * chartH;
    const attackH = (b.attack_count / maxCount) * chartH;
    const totalH = benignH + attackH;

    const barX = x0 + 1;
    const barBaseY = padTop + chartH;

    if (benignH > 0) {
      ctx.fillStyle = "#2563eb";
      ctx.fillRect(barX, barBaseY - benignH, barWidth, benignH);
    }

    if (attackH > 0) {
      ctx.fillStyle = "#dc2626";
      ctx.fillRect(barX, barBaseY - benignH - attackH, barWidth, attackH);
    }

    if (b.is_anomaly_region && totalH > 0) {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 1.8;
      ctx.strokeRect(barX, barBaseY - totalH, barWidth, totalH);

      ctx.fillStyle = "#f59e0b";
      ctx.fillRect(barX, barBaseY - totalH - 2, barWidth, 2);
    }
  });

  // Vertical Threshold Line at τ* = 120.188180
  if (threshX >= padLeft && threshX <= padLeft + chartW) {
    ctx.setLineDash([5, 3]);
    ctx.strokeStyle = "#d97706";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(threshX, padTop);
    ctx.lineTo(threshX, padTop + chartH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#d97706";
    ctx.font = "bold 10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`τ* = ${threshVal.toFixed(4)}`, threshX, padTop - 8);

    ctx.beginPath();
    ctx.moveTo(threshX - 4, padTop - 4);
    ctx.lineTo(threshX + 4, padTop - 4);
    ctx.lineTo(threshX, padTop);
    ctx.closePath();
    ctx.fillStyle = "#d97706";
    ctx.fill();
  }

  const numXTicks = 7;
  for (let j = 0; j <= numXTicks; j++) {
    const sVal = minX + (rangeX / numXTicks) * j;
    const cx = toCanvasX(sVal);

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, padTop + chartH);
    ctx.lineTo(cx, padTop + chartH + 4);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "center";
    ctx.fillText(sVal.toFixed(1), cx, padTop + chartH + 16);
  }

  ctx.fillStyle = "#334155";
  ctx.font = "bold 11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("HBOS Anomaly Score (s)", padLeft + chartW / 2, height - 12);

  ctx.save();
  ctx.translate(16, padTop + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Flow Count (Frequency)", 0, 0);
  ctx.restore();

  ctx.fillStyle = "#6d28d9";
  ctx.font = "bold 10.5px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`Total Flows: ${histData.benign_total + histData.attack_total} (${histData.benign_total} Benign, ${histData.attack_total} Malicious)`, padLeft + 10, padTop + 16);

  setupHBOSHistTooltip(canvas, histData, toCanvasX, chartW, chartH, padLeft, padTop);
}

function renderHBOSSnapshotFlowsTable(tbodyId, flowsList) {
  return;

  if (flowsList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">No flows found for this snapshot.</td></tr>`;
    return;
  }

  const rowsHtml = flowsList.map((f) => {
    return `
      <tr>
        <td style="font-weight: 600;">#${f.rank}</td>
        <td style="font-family: monospace;">${f.edge_idx}</td>
        <td>${f.src}</td>
        <td>${f.dst}</td>
        <td style="font-size: 11px;">${f.timestamp}</td>
        <td style="font-family: monospace;">${f.recon_error.toFixed(4)}</td>
        <td style="font-weight: 700; font-family: monospace; color: ${f.predicted === 1 ? '#dc2626' : '#2563eb'};">${f.anomaly_score.toFixed(4)}</td>
        <td>${f.ground_truth_str}</td>
        <td><span class="${f.badge_class}">${f.status_badge}</span></td>
      </tr>
    `;
  }).join("");

  tbody.innerHTML = rowsHtml;
}


// PHASE 4D: ISOLATION FOREST (IFOREST) CLIENT-SIDE DASHBOARD ENGINE
// ============================================================================

/**
 * Render real-time INNE anomaly detection results for the currently selected snapshot.
 */
function renderINNESnapshot(inne) {
  if (!inne) return;
  state.inneData = inne;

  // 1. Provenance & threshold badges
  const provBadge = document.getElementById("badge-inne-provenance");
  if (provBadge) {
    provBadge.textContent = `Snapshot ${inne.snapshot_index} (Window ${inne.window_id} • ${inne.split_name})`;
  }

  const threshBadge = document.getElementById("badge-inne-thresh");
  if (threshBadge) {
    threshBadge.textContent = `τ* = ${inne.frozen_threshold.toFixed(4)} (99th pct benign)`;
  }

  // 2. Summary banner
  const snapSummaryIdx = document.getElementById("summary-inne-snap-idx");
  if (snapSummaryIdx) snapSummaryIdx.textContent = inne.snapshot_index;

  const totalFlows = document.getElementById("summary-inne-total-flows");
  if (totalFlows) totalFlows.textContent = inne.counts.total_flows.toLocaleString();

  const benignFlows = document.getElementById("summary-inne-benign-flows");
  if (benignFlows) benignFlows.textContent = inne.counts.benign_total.toLocaleString();

  const attackFlows = document.getElementById("summary-inne-attack-flows");
  if (attackFlows) attackFlows.textContent = inne.counts.attack_total.toLocaleString();

  const detectedFlows = document.getElementById("summary-inne-detected-flows");
  if (detectedFlows) detectedFlows.textContent = inne.counts.detected_anomalies.toLocaleString();

  const tpEl = document.getElementById("summary-inne-tp");
  if (tpEl) tpEl.textContent = inne.counts.true_positives.toLocaleString();

  const fpEl = document.getElementById("summary-inne-fp");
  if (fpEl) fpEl.textContent = inne.counts.false_positives.toLocaleString();

  // 3. Four Important Metrics
  const precEl = document.getElementById("val-inne-prec");
  if (precEl) {
    precEl.textContent = (inne.metrics.precision * 100).toFixed(2) + "%";
  }

  const recEl = document.getElementById("val-inne-rec");
  if (recEl) {
    recEl.textContent = (inne.metrics.recall * 100).toFixed(2) + "%";
  }

  const f1El = document.getElementById("val-inne-f1");
  if (f1El) {
    f1El.textContent = inne.metrics.f1_score.toFixed(4);
  }

  const fprEl = document.getElementById("val-inne-fpr");
  if (fprEl) {
    fprEl.textContent = (inne.metrics.false_positive_rate * 100).toFixed(2) + "%";
  }

  // 4. One Primary Graph: INNE Anomaly Visualization (2D PCA Projection)
  renderINNEPCAScatter("inne-pca-canvas", inne.pca_visualization);

  // 5. Flow detections table

  renderINNESnapshotFlowsTable("tbody-inne-flows", inne.flows_table);
}

let inneHoverHandler = null;

function setupINNETooltip(canvas, pcaData, toCanvasX, toCanvasY) {
  const tooltip = document.getElementById("inne-tooltip");
  if (!tooltip || !pcaData) return;

  if (inneHoverHandler) {
    canvas.removeEventListener("mousemove", inneHoverHandler);
    canvas.removeEventListener("mouseleave", inneHoverHandler._leave);
  }

  const points = pcaData.flow_points || [];

  const handleMouseMove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    let nearest = null;
    let minDist = 14;

    for (let pt of points) {
      const cx = toCanvasX(pt.pc1);
      const cy = toCanvasY(pt.pc2);
      const dist = Math.hypot(mouseX - cx, mouseY - cy);
      if (dist < minDist) {
        minDist = dist;
        nearest = pt;
      }
    }

    if (nearest) {
      const isTP = nearest.predicted === 1 && nearest.ground_truth === 1;
      const isFP = nearest.predicted === 1 && nearest.ground_truth === 0;
      const isFN = nearest.predicted === 0 && nearest.ground_truth === 1;

      let statusBadge = isTP ? "True Positive" : (isFP ? "False Alarm" : (isFN ? "Missed Attack" : "Benign Inlier"));
      let badgeColor = isTP ? "#16a34a" : (isFP ? "#dc2626" : (isFN ? "#d97706" : "#2563eb"));

      tooltip.style.display = "block";
      tooltip.style.left = `${(e.clientX - rect.left) + 12}px`;
      tooltip.style.top = `${(e.clientY - rect.top) - 10}px`;
      tooltip.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 2px;">Flow #${nearest.edge_idx}</div>
        <div style="font-size: 10.5px; color: #475569;">${nearest.src} &rarr; ${nearest.dst}</div>
        <div style="font-size: 10.5px; font-family: monospace; margin: 2px 0;">PC1: ${nearest.pc1.toFixed(3)}, PC2: ${nearest.pc2.toFixed(3)}</div>
        <div style="font-size: 10.5px;">INNE Score: <strong>${nearest.anomaly_score.toFixed(4)}</strong> <span style="color:#64748b;">(τ*=${pcaData.frozen_threshold.toFixed(4)})</span></div>
        <div style="font-size: 10.5px; margin-top: 2px;">Ground Truth: <strong>${nearest.ground_truth === 1 ? 'Malicious' : 'Benign'}</strong> | Pred: <strong style="color: ${badgeColor};">${statusBadge}</strong></div>
      `;
    } else {
      tooltip.style.display = "none";
    }
  };

  const handleMouseLeave = () => {
    tooltip.style.display = "none";
  };

  handleMouseMove._leave = handleMouseLeave;
  inneHoverHandler = handleMouseMove;
  canvas.addEventListener("mousemove", handleMouseMove);
  canvas.addEventListener("mouseleave", handleMouseLeave);
}

function renderINNEPCAScatter(canvasId, pcaData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !pcaData) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  // Update variance badge
  const badgeVar = document.getElementById("badge-inne-pca-variance");
  if (badgeVar && pcaData.pca_variance) {
    badgeVar.textContent = `PC1: ${pcaData.pca_variance[0]}% | PC2: ${pcaData.pca_variance[1]}% Var`;
  }

  const [xMin, xMax] = pcaData.x_range;
  const [yMin, yMax] = pcaData.y_range;
  const rangeX = xMax - xMin || 1;
  const rangeY = yMax - yMin || 1;

  const padLeft = 60;
  const padRight = 30;
  const padTop = 32;
  const padBottom = 42;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const toCanvasX = (x) => padLeft + ((x - xMin) / rangeX) * chartW;
  const toCanvasY = (y) => padTop + chartH - ((y - yMin) / rangeY) * chartH;

  // Background subtle canvas fill
  ctx.fillStyle = "#fafbfc";
  ctx.fillRect(padLeft, padTop, chartW, chartH);

  // Box border
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, padTop, chartW, chartH);

  // Horizontal grid lines & Y labels
  for (let i = 0; i <= 4; i++) {
    const yVal = yMin + (rangeY / 4) * i;
    const cy = toCanvasY(yVal);
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, cy);
    ctx.lineTo(padLeft + chartW, cy);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "right";
    ctx.fillText(yVal.toFixed(1), padLeft - 6, cy + 3);
  }

  // Vertical grid lines & X labels
  for (let j = 0; j <= 5; j++) {
    const xVal = xMin + (rangeX / 5) * j;
    const cx = toCanvasX(xVal);
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, padTop);
    ctx.lineTo(cx, padTop + chartH);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "center";
    ctx.fillText(xVal.toFixed(1), cx, padTop + chartH + 15);
  }

  // Axis Labels
  ctx.fillStyle = "#334155";
  ctx.font = "bold 10.5px sans-serif";
  ctx.textAlign = "center";
  const varX = pcaData.pca_variance ? pcaData.pca_variance[0] : 21.67;
  const varY = pcaData.pca_variance ? pcaData.pca_variance[1] : 12.54;
  ctx.fillText(`Principal Component 1 (${varX}% Explained Variance)`, padLeft + chartW / 2, height - 10);

  ctx.save();
  ctx.translate(16, padTop + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`Principal Component 2 (${varY}% Explained Variance)`, 0, 0);
  ctx.restore();

  // Watermark tag inside chart
  ctx.fillStyle = "#065f46";
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "right";
  ctx.fillText(`INNE (100 Trees): τ* = ${pcaData.frozen_threshold.toFixed(4)}`, padLeft + chartW - 10, padTop + 18);

  // Plot Flow Points
  const points = pcaData.flow_points || [];
  // Sort so attacks and predicted anomalies draw on top
  const sortedPoints = [...points].sort((a, b) => {
    return (a.predicted * 2 + a.ground_truth) - (b.predicted * 2 + b.ground_truth);
  });

  sortedPoints.forEach((pt) => {
    const cx = toCanvasX(pt.pc1);
    const cy = toCanvasY(pt.pc2);

    if (cx < padLeft || cx > padLeft + chartW || cy < padTop || cy > padTop + chartH) {
      return;
    }

    const isAttack = pt.ground_truth === 1;
    const isAnomalyPred = pt.predicted === 1;
    const radius = isAttack ? 4.5 : 4.0;

    // Draw prediction halo ring if predicted anomalous
    if (isAnomalyPred) {
      ctx.strokeStyle = "#f59e0b"; // Golden amber ring
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw flow center dot
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    if (isAttack) {
      ctx.fillStyle = "#dc2626"; // Crimson
      ctx.strokeStyle = "#7f1d1d";
    } else {
      ctx.fillStyle = "#2563eb"; // Royal blue
      ctx.strokeStyle = "#1e3a8a";
    }
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  });

  // Attach interactive hover listener
  setupINNETooltip(canvas, pcaData, toCanvasX, toCanvasY);
}

function renderINNESnapshotFlowsTable(tbodyId, flowsList) {
  return;

  if (flowsList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">No flows found for this snapshot.</td></tr>`;
    return;
  }

  const rowsHtml = flowsList.map((f) => {
    return `
      <tr>
        <td style="font-weight: 600;">#${f.rank}</td>
        <td style="font-family: monospace;">${f.edge_idx}</td>
        <td>${f.src}</td>
        <td>${f.dst}</td>
        <td style="font-size: 11px;">${f.timestamp}</td>
        <td style="font-family: monospace;">${f.recon_error.toFixed(4)}</td>
        <td style="font-weight: 700; font-family: monospace; color: ${f.predicted === 1 ? '#dc2626' : '#2563eb'};">${f.anomaly_score.toFixed(4)}</td>
        <td>${f.ground_truth_str}</td>
        <td><span class="${f.badge_class}">${f.status_badge}</span></td>
      </tr>
    `;
  }).join("");

  tbody.innerHTML = rowsHtml;
}



// ============================================================================


let inneHistHoverHandler = null;

function setupINNEHistTooltip(canvas, histData, toCanvasX, chartW, chartH, padLeft, padTop) {
  const tooltip = document.getElementById("inne-hist-tooltip");
  if (!tooltip || !histData || !histData.bins) return;

  if (inneHistHoverHandler) {
    canvas.removeEventListener("mousemove", inneHistHoverHandler);
    canvas.removeEventListener("mouseleave", inneHistHoverHandler._leave);
  }

  const bins = histData.bins;
  const thresh = histData.frozen_threshold;

  const handleMouseMove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    if (mouseX < padLeft || mouseX > padLeft + chartW || mouseY < padTop || mouseY > padTop + chartH) {
      tooltip.style.display = "none";
      return;
    }

    let foundBin = null;
    for (let bi = 0; bi < bins.length; bi++) {
      const b = bins[bi];
      const x0 = toCanvasX(b.range[0]);
      const x1 = toCanvasX(b.range[1]);
      if (mouseX >= x0 && mouseX <= x1) {
        foundBin = b;
        break;
      }
    }

    if (foundBin && foundBin.total_count > 0) {
      const isAnomalyBin = foundBin.is_anomaly_region || foundBin.range[1] >= thresh;
      const statusText = isAnomalyBin ? "Predicted Anomaly (s ≥ τ*)" : "Normal / Inlier (s < τ*)";
      const statusColor = isAnomalyBin ? "#d97706" : "#2563eb";

      tooltip.style.display = "block";
      tooltip.style.left = `${(e.clientX - rect.left) + 12}px`;
      tooltip.style.top = `${(e.clientY - rect.top) - 10}px`;
      tooltip.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 2px;">INNE Score Range: [${foundBin.range[0].toFixed(2)} &ndash; ${foundBin.range[1].toFixed(2)}]</div>
        <div style="font-size: 10.5px; color: #2563eb; margin: 1px 0;">&bull; Benign Flows: <strong>${foundBin.benign_count}</strong></div>
        <div style="font-size: 10.5px; color: #dc2626; margin: 1px 0;">&bull; Attack Flows: <strong>${foundBin.attack_count}</strong></div>
        <div style="font-size: 10.5px; margin-top: 2px;">Total in Bin: <strong>${foundBin.total_count}</strong></div>
        <div style="font-size: 10.5px; margin-top: 3px; border-top: 1px solid #e2e8f0; padding-top: 2px;">
          Region: <strong style="color: ${statusColor};">${statusText}</strong>
        </div>
      `;
    } else {
      tooltip.style.display = "none";
    }
  };

  const handleMouseLeave = () => {
    tooltip.style.display = "none";
  };

  handleMouseMove._leave = handleMouseLeave;
  inneHistHoverHandler = handleMouseMove;
  canvas.addEventListener("mousemove", handleMouseMove);
  canvas.addEventListener("mouseleave", handleMouseLeave);
}

function renderINNEScoreHistogram(canvasId, histData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !histData || !histData.bins) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const padLeft = 60;
  const padRight = 40;
  const padTop = 38;
  const padBottom = 48;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const bins = histData.bins;
  const numBins = bins.length;
  const threshVal = histData.frozen_threshold;

  const minX = histData.hist_min !== undefined ? histData.hist_min : histData.bin_edges[0];
  const maxX = histData.hist_max !== undefined ? histData.hist_max : histData.bin_edges[histData.bin_edges.length - 1];
  const rangeX = maxX - minX || 1;

  const maxCount = Math.max(histData.max_bin_count || 1, 1);

  const toCanvasX = (score) => padLeft + ((score - minX) / rangeX) * chartW;

  ctx.fillStyle = "#fafbfc";
  ctx.fillRect(padLeft, padTop, chartW, chartH);

  // Anomaly Region Shading (s >= threshVal)
  const threshX = toCanvasX(threshVal);
  if (threshX < padLeft + chartW) {
    const anomRegionX = Math.max(padLeft, threshX);
    const anomRegionW = (padLeft + chartW) - anomRegionX;
    ctx.fillStyle = "rgba(245, 158, 11, 0.08)";
    ctx.fillRect(anomRegionX, padTop, anomRegionW, chartH);

    ctx.fillStyle = "#b45309";
    ctx.font = "bold 9.5px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("PREDICTED ANOMALY REGION (s ≥ τ*)", padLeft + chartW - 10, padTop + 16);
  }

  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, padTop, chartW, chartH);

  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const countVal = Math.round((maxCount / yTicks) * i);
    const cy = padTop + chartH - (i / yTicks) * chartH;

    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, cy);
    ctx.lineTo(padLeft + chartW, cy);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "right";
    ctx.fillText(countVal.toString(), padLeft - 8, cy + 3.5);
  }

  // Draw Stacked Histogram Bars (Benign Blue + Malicious Crimson Red)
  bins.forEach((b) => {
    const x0 = toCanvasX(b.range[0]);
    const x1 = toCanvasX(b.range[1]);
    const barWidth = Math.max(1, (x1 - x0) - 2);

    const benignH = (b.benign_count / maxCount) * chartH;
    const attackH = (b.attack_count / maxCount) * chartH;
    const totalH = benignH + attackH;

    const barX = x0 + 1;
    const barBaseY = padTop + chartH;

    if (benignH > 0) {
      ctx.fillStyle = "#2563eb";
      ctx.fillRect(barX, barBaseY - benignH, barWidth, benignH);
    }

    if (attackH > 0) {
      ctx.fillStyle = "#dc2626";
      ctx.fillRect(barX, barBaseY - benignH - attackH, barWidth, attackH);
    }

    if (b.is_anomaly_region && totalH > 0) {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 1.8;
      ctx.strokeRect(barX, barBaseY - totalH, barWidth, totalH);

      ctx.fillStyle = "#f59e0b";
      ctx.fillRect(barX, barBaseY - totalH - 2, barWidth, 2);
    }
  });

  // Vertical Threshold Line at τ* = 120.188180
  if (threshX >= padLeft && threshX <= padLeft + chartW) {
    ctx.setLineDash([5, 3]);
    ctx.strokeStyle = "#d97706";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(threshX, padTop);
    ctx.lineTo(threshX, padTop + chartH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#d97706";
    ctx.font = "bold 10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`τ* = ${threshVal.toFixed(4)}`, threshX, padTop - 8);

    ctx.beginPath();
    ctx.moveTo(threshX - 4, padTop - 4);
    ctx.lineTo(threshX + 4, padTop - 4);
    ctx.lineTo(threshX, padTop);
    ctx.closePath();
    ctx.fillStyle = "#d97706";
    ctx.fill();
  }

  const numXTicks = 7;
  for (let j = 0; j <= numXTicks; j++) {
    const sVal = minX + (rangeX / numXTicks) * j;
    const cx = toCanvasX(sVal);

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, padTop + chartH);
    ctx.lineTo(cx, padTop + chartH + 4);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px monospace";
    ctx.textAlign = "center";
    ctx.fillText(sVal.toFixed(1), cx, padTop + chartH + 16);
  }

  ctx.fillStyle = "#334155";
  ctx.font = "bold 11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("INNE Anomaly Score (s)", padLeft + chartW / 2, height - 12);

  ctx.save();
  ctx.translate(16, padTop + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Flow Count (Frequency)", 0, 0);
  ctx.restore();

  ctx.fillStyle = "#6d28d9";
  ctx.font = "bold 10.5px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`Total Flows: ${histData.benign_total + histData.attack_total} (${histData.benign_total} Benign, ${histData.attack_total} Malicious)`, padLeft + 10, padTop + 16);

  setupINNEHistTooltip(canvas, histData, toCanvasX, chartW, chartH, padLeft, padTop);
}

