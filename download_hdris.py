import json
import os
import urllib.request
import yaml

cfg = yaml.safe_load(open("config.yaml"))
os.makedirs(cfg["hdri_dir"], exist_ok=True)
headers = {"User-Agent": "synthetic-object-detection"}

for name in sum(cfg["hdris"].values(), []):
    path = os.path.join(cfg["hdri_dir"], f"{name}.hdr")
    if os.path.exists(path):
        continue
    files = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.polyhaven.com/files/{name}", headers=headers)))
    url = files["hdri"]["2k"]["hdr"]["url"]
    print(f"{name}: {url}")
    urllib.request.urlretrieve(url, path)
