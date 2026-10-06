import urllib.request
import re
import os

BASE = "http://192.168.0.103:8000/processed/"
DEST = "data/processed"

os.makedirs(DEST, exist_ok=True)

html = urllib.request.urlopen(BASE).read().decode()
files = re.findall(r'href="([^"]+)"', html)

for name in files:
    if name.endswith("/") or name == "../":
        continue

    path = os.path.join(DEST, name)

    if os.path.exists(path):
        print("Already exists:", name)
        continue

    print("Downloading:", name)
    data = urllib.request.urlopen(BASE + name).read()

    with open(path, "wb") as f:
        f.write(data)

print("MISSING FILES TRANSFER COMPLETE")