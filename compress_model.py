import sys
import torch
import nemo.collections.asr as nemo_asr

def compress_model(input_path):
    output_path = input_path.replace(".nemo", "_fp16.nemo")
    
    print(f"🚀 Loading model from: {input_path}")
    # Load model on CPU to save memory during conversion
    model = nemo_asr.models.ASRModel.restore_from(input_path, map_location="cpu")
    
    print("📉 Compressing... (Converting to FP16)")
    # Convert weights to Half Precision (FP16)
    model = model.half()
    
    # Optional: Force strict config cleanup if needed
    if hasattr(model, 'cfg'):
        # Remove optimizer config if present
        if 'optim' in model.cfg:
            del model.cfg.optim

    print(f"BSaving compressed model to: {output_path}")
    model.save_to(output_path)
    print("✅ Done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_model.py <model_filename.nemo>")
    else:
        compress_model(sys.argv[1])