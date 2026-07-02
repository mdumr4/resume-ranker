"""Download Qwen2.5-1.5B-Instruct GGUF (Q5_K_M) from HuggingFace."""
import urllib.request
import os

url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q5_k_m.gguf"
dest = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf"

if os.path.exists(dest):
    print(f"{dest} already exists, skipping.")
else:
    print(f"Downloading {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Done. Size: {os.path.getsize(dest) / 1024**3:.2f} GB")
