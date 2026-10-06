import re

def main():
    # 1. Update HTML
    html_path = "templates/review2_demo.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    
    # Remove the histogram canvas
    target_hist_html = """        <div class="graph-wrapper" style="margin-top: 20px;">
          <div class="canvas-box">
            <canvas id="inne-hist-canvas" width="1000" height="400"></canvas>
            <div id="inne-hist-tooltip" class="tooltip"></div>
          </div>
        </div>"""
    if target_hist_html in html:
        html = html.replace(target_hist_html, "")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 2. Update JS
    js_path = "static/review2_demo.js"
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
        
    # a. Add INNE execution to processSnapshot
    target_hbos_process = """    if (res.hbos) {
      setPipelineStep("step-hbos", "active");
      renderHBOSSnapshot(res.hbos);
      setPipelineStep("step-hbos", "completed");
    }"""
    inne_process = """    if (res.inne) {
      setPipelineStep("step-inne", "active");
      renderINNESnapshot(res.inne);
      setPipelineStep("step-inne", "completed");
    }"""
    if target_hbos_process in js and inne_process not in js:
        js = js.replace(target_hbos_process, target_hbos_process + "\n\n" + inne_process)
        
    # b. Remove histogram render call from renderINNESnapshot
    target_render_call = """  if (inne.score_histogram) renderINNEScoreHistogram("inne-hist-canvas", inne.score_histogram);"""
    if target_render_call in js:
        js = js.replace(target_render_call, "")
        
    # c. Remove inne-hist-canvas from clear list
    js = js.replace('"inne-pca-canvas", "inne-hist-canvas"]', '"inne-pca-canvas"]')
    
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js)
        
    print("Done removing histogram and fixing execution in JS.")

if __name__ == "__main__":
    main()
