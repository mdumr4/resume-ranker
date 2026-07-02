import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(model_path, quantized_path):
    print(f"Quantizing {model_path} to {quantized_path}...")
    
    quantize_dynamic(
        model_input=model_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8,
        use_external_data_format=True
    )
    print("Quantization complete!")

if __name__ == "__main__":
    bge_path = "models/bge_onnx_v2/model.onnx"
    quantized_path = "models/bge_onnx_v2_int8/model.onnx"
    os.makedirs("models/bge_onnx_v2_int8", exist_ok=True)
    
    if os.path.exists(bge_path):
        quantize_model(bge_path, quantized_path)
    else:
        print(f"Not found: {bge_path}")
