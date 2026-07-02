"""Quantize BGE-M3 and SPLADE ONNX models to INT8 dynamic quantization."""
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
import os
import shutil

def quantize_model(input_dir, model_name, output_dir):
    """Quantize an ONNX model to INT8 using dynamic quantization."""
    input_model = os.path.join(input_dir, "model.onnx")
    preprocessed = os.path.join(input_dir, "model_preprocessed.onnx")
    output_model = os.path.join(output_dir, "model.onnx")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Quantizing {model_name}")
    print(f"Input: {input_model}")
    input_size = sum(
        os.path.getsize(os.path.join(input_dir, f)) 
        for f in os.listdir(input_dir) 
        if f.startswith("model.onnx")
    )
    print(f"Input size: {input_size / 1024**2:.1f} MB")
    
    # Step 1: Pre-process (fixes shape inference issues that caused previous failures)
    print("Step 1: Pre-processing ONNX graph for quantization...")
    try:
        quant_pre_process(input_model, preprocessed)
        source = preprocessed
        print("  Pre-processing succeeded.")
    except Exception as e:
        print(f"  Pre-processing failed ({e}), trying direct quantization...")
        source = input_model
    
    # Step 2: Dynamic INT8 quantization
    print("Step 2: Applying INT8 dynamic quantization...")
    try:
        quantize_dynamic(
            model_input=source,
            model_output=output_model,
            weight_type=QuantType.QInt8,
        )
    except Exception as e:
        print(f"  ERROR: Quantization failed: {e}")
        # Cleanup
        if os.path.exists(preprocessed):
            os.remove(preprocessed)
        return False
    
    # Step 3: Verify
    output_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.startswith("model")
    )
    print(f"Output size: {output_size / 1024**2:.1f} MB")
    print(f"Compression: {input_size / output_size:.1f}x")
    
    # Verify it loads
    try:
        sess = ort.InferenceSession(output_model, providers=["CPUExecutionProvider"])
        print(f"Verification: model loads OK, inputs: {[i.name for i in sess.get_inputs()]}")
    except Exception as e:
        print(f"  ERROR: Quantized model failed to load: {e}")
        return False
    
    # Cleanup preprocessed file
    if os.path.exists(preprocessed):
        os.remove(preprocessed)
    
    print(f"SUCCESS: {model_name} quantized to INT8.")
    return True

# --- Quantize BGE-M3 ---
bge_ok = quantize_model("models/bge_onnx", "BGE-M3 Reranker", "models/bge_onnx_int8")

# --- Quantize SPLADE ---
splade_ok = quantize_model("models/splade_onnx", "SPLADE-v3", "models/splade_onnx_int8")

print(f"\n{'='*60}")
print(f"BGE-M3: {'OK' if bge_ok else 'FAILED'}")
print(f"SPLADE: {'OK' if splade_ok else 'FAILED'}")
