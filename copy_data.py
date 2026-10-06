import urllib.request
import re
import os

BASE = "http://192.168.0.103:8000/"

for folder in ["processed", "raw"]:
    url = BASE + folder + "/"
    html = urllib.request.urlopen(url).read().decode()
    files = re.findall(r'href="([^"]+)"', html)

    os.makedirs("data/" + folder, exist_ok=True)

    for name in files:
        if name.endswith("/") or name == "../":
            continue

        print("Downloading:", folder, name)
        data = urllib.request.urlopen(url + name).read()

        with open(os.path.join("data", folder, name), "wb") as f:
            f.write(data)

print("TRANSFER COMPLETE")