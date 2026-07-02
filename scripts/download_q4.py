import os
import requests

def download_file(url, dest, token):
    print(f"Downloading {dest} from {url}...")
    headers = {"Authorization": f"Bearer {token}"}
    
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Finished downloading {dest}")

def main():
    os.makedirs("models", exist_ok=True)
    token = "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"
    
    downloads = [
        ("https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf", "models/Qwen_Qwen3-1.7B-Q4_K_M.gguf"),
    ]
    
    for url, dest in downloads:
        if not os.path.exists(dest):
            download_file(url, dest, token)
        else:
            print(f"{dest} already exists.")

if __name__ == "__main__":
    main()
