import re

def main():
    file_path = "static/review2_demo.js"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the block containing HBOS histogram rendering
    # It starts at `let hbosHistHoverHandler = null;` up to `function renderHBOSSnapshotFlowsTable`
    match = re.search(r'let hbosHistHoverHandler = null;.*?(?=function renderHBOSSnapshotFlowsTable)', content, re.DOTALL)
    if not match:
        print("HBOS hist logic not found!")
        return

    hbos_hist_code = match.group(0)
    
    # Replace HBOS to INNE
    inne_hist_code = hbos_hist_code.replace("hbos", "inne")
    inne_hist_code = inne_hist_code.replace("HBOS", "INNE")
    
    # We also need to add a call to renderINNEScoreHistogram inside renderINNESnapshot
    # Let's find `renderINNESnapshot` and append to it
    
    target_call = "renderINNESnapshotFlowsTable(\"tbody-inne-flows\", inne.flows_table);"
    new_call = "if (inne.score_histogram) renderINNEScoreHistogram(\"inne-hist-canvas\", inne.score_histogram);\n  " + target_call
    
    content = content.replace(target_call, new_call)
    
    # Also add clear canvas logic for inne-hist-canvas in ensureFrontendUIFixes / reset function
    content = content.replace('"inne-pca-canvas"]', '"inne-pca-canvas", "inne-hist-canvas"]')
    
    # Append inne_hist_code at the end of the file
    content += "\n\n" + inne_hist_code

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Patched review2_demo.js for INNE histogram.")

if __name__ == "__main__":
    main()
