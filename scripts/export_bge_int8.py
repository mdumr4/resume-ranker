"""Re-export BGE-M3 to ONNX with dynamic batch axes using LEGACY torch exporter."""
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

print("Loading BGE-M3 reranker...")
model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Create dummy input with explicit shapes
dummy = tokenizer(
    ["query"], ["document"], 
    padding="max_length", truncation=True, max_length=32, return_tensors="pt"
)

os.makedirs("models/bge_onnx_v2", exist_ok=True)
onnx_path = "models/bge_onnx_v2/model.onnx"

print("Exporting with LEGACY torch.onnx.export (dynamo=False)...")
with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,  # Force legacy exporter
    )

size_mb = os.path.getsize(onnx_path) / 1024**2
print(f"Exported: {size_mb:.1f} MB")

# Verify with batch_size=1
print("Testing batch_size=1...")
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
t1 = tokenizer(["q1"], ["d1"], padding=True, truncation=True, max_length=512, return_tensors="np")
o1 = sess.run(None, {"input_ids": t1["input_ids"], "attention_mask": t1["attention_mask"]})
print(f"  batch=1 OK, shape={o1[0].shape}, value={o1[0]}")

# Verify with batch_size=4
print("Testing batch_size=4...")
t4 = tokenizer(["q1","q2","q3","q4"], ["d1","d2","d3","d4"], padding=True, truncation=True, max_length=512, return_tensors="np")
o4 = sess.run(None, {"input_ids": t4["input_ids"], "attention_mask": t4["attention_mask"]})
print(f"  batch=4 OK, shape={o4[0].shape}")

# Verify with batch_size=16
print("Testing batch_size=16...")
t16 = tokenizer(["q"]*16, ["d"]*16, padding=True, truncation=True, max_length=512, return_tensors="np")
o16 = sess.run(None, {"input_ids": t16["input_ids"], "attention_mask": t16["attention_mask"]})
print(f"  batch=16 OK, shape={o16[0].shape}")

print(f"\nSUCCESS: Dynamic-batch BGE-M3 ONNX at {onnx_path} ({size_mb:.1f} MB)")
