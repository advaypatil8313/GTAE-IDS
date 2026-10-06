import re

def main():
    file_path = "static/review2_demo.js"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find setupINNEHistTooltip block
    match = re.search(r'function setupINNEHistTooltip.*?\{.*?(?=function renderINNEScoreHistogram)', content, re.DOTALL)
    if match:
        block = match.group(0)
        # Inside this block, replace inne-tooltip with inne-hist-tooltip
        new_block = block.replace('"inne-tooltip"', '"inne-hist-tooltip"')
        content = content.replace(block, new_block)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Patched tooltip ID.")
    else:
        print("setupINNEHistTooltip not found")

if __name__ == "__main__":
    main()
