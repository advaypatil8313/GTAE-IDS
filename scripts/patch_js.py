import re
import sys

def main():
    file_path = "static/review2_demo.js"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. State
    content = content.replace('  hbosData: null,\n  currentHBOSSplit: "test",\n', '  hbosData: null,\n  currentHBOSSplit: "test",\n  inneData: null,\n  currentINNESplit: "test",\n')

    # 2. Pipeline box in ensureFrontendUIFixes
    content = content.replace('<div class="pipe-box" id="step-hbos">10 HBOS</div>\n      </div>', '<div class="pipe-box" id="step-hbos">10 HBOS</div>\n        <div class="pipe-arrow">&rarr;</div>\n        <div class="pipe-box" id="step-inne">11 INNE</div>\n      </div>')

    # 3. resetPipelineSteps array
    content = content.replace('"step-iforest", "step-hbos"]', '"step-iforest", "step-hbos", "step-inne"]')

    # 4. Clear state in processSnapshot reset (around line 142)
    content = content.replace('  state.hbosData = null;\n', '  state.hbosData = null;\n  state.inneData = null;\n')

    # 5. Clear fields (duplicate iforest logic for inne)
    iforest_clear = """  const iforestProvBadge = document.getElementById("badge-iforest-provenance");
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
  });"""

    inne_clear = iforest_clear.replace('iforest', 'inne').replace('Isolation Forest', 'INNE')
    
    content = content.replace(iforest_clear, iforest_clear + '\n\n' + inne_clear)

    # 6. clearCanvas list
    content = content.replace('"iforest-pca-canvas", "hbos-hist-canvas"]', '"iforest-pca-canvas", "hbos-hist-canvas", "inne-pca-canvas"]')
    
    # 7. tooltip display none
    content = content.replace('if (hbosTooltip) hbosTooltip.style.display = "none";', 'if (hbosTooltip) hbosTooltip.style.display = "none";\n  const inneTooltip = document.getElementById("inne-tooltip");\n  if (inneTooltip) inneTooltip.style.display = "none";')

    # 8. tbody HTML reset
    content = content.replace('const hbosTbody = document.getElementById("tbody-hbos-flows");\n  if (hbosTbody) hbosTbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">Snapshot ${idx} selected. Click "Process Snapshot" to run GTAE and HBOS detection.</td></tr>`;', 'const hbosTbody = document.getElementById("tbody-hbos-flows");\n  if (hbosTbody) hbosTbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">Snapshot ${idx} selected. Click "Process Snapshot" to run GTAE and HBOS detection.</td></tr>`;\n  const inneTbody = document.getElementById("tbody-inne-flows");\n  if (inneTbody) inneTbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 20px; color: var(--text-muted);">Snapshot ${idx} selected. Click "Process Snapshot" to run GTAE and INNE detection.</td></tr>`;')

    # 9. In processSnapshot flow processing
    hbos_process = """    if (res.hbos) {
      setPipelineStep("step-hbos", "active");
      renderHBOSSnapshot(res.hbos);
      setPipelineStep("step-hbos", "completed");
      await stepDelay(90);
    }"""
    inne_process = hbos_process.replace('hbos', 'inne').replace('HBOS', 'INNE')
    content = content.replace(hbos_process, hbos_process + '\n\n' + inne_process)

    # 10. Update status success message
    content = content.replace('OCSVM, IForest & HBOS ready', 'OCSVM, IForest, HBOS & INNE ready')

    # 11. In runGTAE
    hbos_run = """    if (data.hbos) {
      setPipelineStep("step-hbos", "active");
      renderHBOSSnapshot(data.hbos);
      setPipelineStep("step-hbos", "completed");
    }"""
    inne_run = hbos_run.replace('hbos', 'inne').replace('HBOS', 'INNE')
    content = content.replace(hbos_run, hbos_run + '\n\n' + inne_run)

    # 12. Copy the entire IForest section and rename it to INNE (to keep PCA logic)
    # The iforest section starts at `// PHASE 4B: ISOLATION FOREST CLIENT-SIDE DASHBOARD ENGINE`
    # We will append the INNE section at the end of the file.
    
    if_match = re.search(r'// PHASE 4B: ISOLATION FOREST CLIENT-SIDE DASHBOARD ENGINE.*?(?=// PHASE 4C: HISTOGRAM-BASED OUTLIER SCORE)', content, re.DOTALL)
    if if_match:
        if_code = if_match.group(0)
        inne_code = if_code.replace('iforest', 'inne').replace('IForest', 'INNE').replace('Isolation Forest', 'INNE').replace('isolation forest', 'INNE').replace('PHASE 4B:', 'PHASE 4D:')
        content += '\n\n' + inne_code

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    main()
