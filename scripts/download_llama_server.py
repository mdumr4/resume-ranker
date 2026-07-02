"""Download and extract llama-server.exe from official GitHub releases."""
import urllib.request
import zipfile
import json
import os

print("Fetching latest llama.cpp release info...")
req = urllib.request.Request("https://api.github.com/repos/ggerganov/llama.cpp/releases/latest")
req.add_header('User-Agent', 'Mozilla/5.0')
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())

download_url = None
for asset in data['assets']:
    if "bin-win-cpu-x64.zip" in asset['name']:
        download_url = asset['browser_download_url']
        break

if not download_url:
    print("Could not find the CPU Windows zip file!")
    exit(1)

zip_path = "llama-server-temp.zip"
print(f"Downloading from {download_url}...")
urllib.request.urlretrieve(download_url, zip_path)

print("Extracting llama-server.exe...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        if "llama-server.exe" in member:
            with zip_ref.open(member) as source, open("llama-server.exe", "wb") as target:
                target.write(source.read())
            print("Extracted llama-server.exe!")
            break

os.remove(zip_path)
print("Done.")
