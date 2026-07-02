import argparse
import os
import subprocess
import time
import socket
import sys

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking Pipeline - Offline Reproduction")
    parser.add_argument("--candidates", default="./candidates.jsonl", help="Path to input candidates file")
    parser.add_argument("--out", default="./submission.csv", help="Path to save final ranked CSV")
    args = parser.parse_args()

    # Capture absolute paths before changing directory
    candidates_path = os.path.abspath(args.candidates)
    out_path = os.path.abspath(args.out)

    # Change directory to sandbox where all relative paths resolve
    os.chdir("sandbox")

    # Determine the llama-server binary path based on OS
    is_windows = sys.platform.startswith("win")
    if is_windows:
        llama_server = os.path.join("bin", "llama", "llama-server.exe")
    else:
        # For Linux, check local build bin/llama first, then fallback to global command
        llama_server = os.path.join("bin", "llama", "llama-server")
        if not os.path.exists(llama_server):
            llama_server = "llama-server"

    print("Starting llama-server in the background...")
    llama_cmd = [
        llama_server,
        "-m", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "-c", "16384",
        "-np", "4",
        "--threads", "4",
        "--port", "8080"
    ]
    log_file = open("llama-server.log", "w")
    err_file = open("llama-server-err.log", "w")
    
    try:
        llama_proc = subprocess.Popen(llama_cmd, stdout=log_file, stderr=err_file)
    except Exception as e:
        print(f"Error starting llama-server: {e}. Check path or permissions.")
        log_file.close()
        err_file.close()
        sys.exit(1)

    # Wait for the server to be ready
    ready = False
    print("Waiting for llama-server to be ready...")
    for i in range(30):
        time.sleep(1)
        try:
            s = socket.create_connection(("127.0.0.1", 8080), timeout=1)
            s.close()
            ready = True
            break
        except Exception:
            pass

    if not ready:
        print("Error: llama-server failed to start! Check sandbox/llama-server-err.log")
        llama_proc.terminate()
        llama_proc.wait()
        log_file.close()
        err_file.close()
        sys.exit(1)

    print("Server is ready! Running the pipeline...")
    try:
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        subprocess.run([sys.executable, "-u", "src/optimize_rank.py", "--out", out_path], check=True)
    finally:
        print("Pipeline finished. Terminating llama-server...")
        llama_proc.terminate()
        llama_proc.wait()
        log_file.close()
        err_file.close()
        print("Done.")

if __name__ == "__main__":
    main()
