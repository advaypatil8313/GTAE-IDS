import re
import sys

def main():
    file_path = "static/review2_demo.js"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The iforest section starts at `// PHASE 4B: ISOLATION FOREST (IFOREST) CLIENT-SIDE DASHBOARD ENGINE`
    if_match = re.search(r'// PHASE 4B: ISOLATION FOREST \(IFOREST\) CLIENT-SIDE DASHBOARD ENGINE.*?(?=// PHASE 4C: HISTOGRAM-BASED OUTLIER SCORE)', content, re.DOTALL)
    if if_match:
        if_code = if_match.group(0)
        # Replace occurrences
        inne_code = if_code.replace('iforest', 'inne')
        inne_code = inne_code.replace('IForest', 'INNE')
        inne_code = inne_code.replace('Isolation Forest', 'INNE')
        inne_code = inne_code.replace('isolation forest', 'INNE')
        inne_code = inne_code.replace('PHASE 4B:', 'PHASE 4D:')
        
        # Append to the end of the file
        content += '\n\n' + inne_code
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Successfully appended INNE code.")
    else:
        print("Failed to find the IForest block.")

if __name__ == "__main__":
    main()
