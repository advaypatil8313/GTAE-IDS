import urllib.request, re, os

BASE="http://192.168.0.103:8001/"
DEST="data/processed"

def copy(url, dest):
    html=urllib.request.urlopen(url).read().decode()
    for href in re.findall(r'href="([^"]+)"', html):
        if href in ("../","/"):
            continue
        if href.endswith("/"):
            os.makedirs(os.path.join(dest,href[:-1]),exist_ok=True)
            copy(url+href,os.path.join(dest,href[:-1]))
        else:
            path=os.path.join(dest,href)
            if os.path.exists(path):
                print("Exists:",path)
            else:
                print("COPY:",path)
                with open(path,"wb") as f:
                    f.write(urllib.request.urlopen(url+href).read())

copy(BASE,DEST)
print("DONE")