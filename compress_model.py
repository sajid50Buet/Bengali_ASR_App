import torch
import torch.nn.utils.prune as prune
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download
import os

def compress_model():
    """Create multiple compressed versions of the model"""
    
    print("=" * 70)
    print("📦 COMPRESSING PARAKEET BENGALI ASR MODEL")
    print("=" * 70)
    print()
    
    # Download model from HuggingFace
    print("📥 Downloading model from HuggingFace...")
    model_path = hf_hub_download(
        repo_id="Sajid50/parakeet-bengali-asr",
        filename="parakeet-bengali-asr.nemo"
    )
    print(f"✅ Cached at: {model_path}")
    print()
    
    # Get original size
    original_size = os.path.getsize(model_path) / (1024**3)
    print(f"📊 Original size: {original_size:.2f} GB")
    print()
    
    results = []
    
    # ========================================
    # Method 1: FP16 (RECOMMENDED!)
    # ========================================
    print("🔄 Creating FP16 version...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    model_fp16 = model.half()
    
    fp16_path = "parakeet-bengali-asr-fp16.nemo"
    model_fp16.save_to(fp16_path)
    fp16_size = os.path.getsize(fp16_path) / (1024**3)
    
    print(f"✅ FP16: {fp16_size:.2f} GB ({100-fp16_size/original_size*100:.1f}% reduction)")
    results.append(("FP16", fp16_path, fp16_size))
    print()
    
    # ========================================
    # Method 2: INT8 Quantization
    # ========================================
    print("🔄 Creating INT8 version...")
    model_cpu = model.cpu()
    model_int8 = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    int8_path = "parakeet-bengali-asr-int8.nemo"
    model_int8.save_to(int8_path)
    int8_size = os.path.getsize(int8_path) / (1024**3)
    
    print(f"✅ INT8: {int8_size:.2f} GB ({100-int8_size/original_size*100:.1f}% reduction)")
    results.append(("INT8", int8_path, int8_size))
    print()
    
    # ========================================
    # Method 3: Pruning (30%)
    # ========================================
    print("🔄 Creating pruned version (30% pruning)...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, 'weight')
    
    pruned_path = "parakeet-bengali-asr-pruned.nemo"
    model.save_to(pruned_path)
    pruned_size = os.path.getsize(pruned_path) / (1024**3)
    
    print(f"✅ Pruned: {pruned_size:.2f} GB ({100-pruned_size/original_size*100:.1f}% reduction)")
    results.append(("Pruned", pruned_path, pruned_size))
    print()
    
    # ========================================
    # Summary
    # ========================================
    print("=" * 70)
    print("✅ COMPRESSION COMPLETE!")
    print("=" * 70)
    print()
    print(f"📊 Summary:")
    print(f"   Original:  {original_size:.2f} GB (100%)")
    for name, path, size in results:
        reduction = (1 - size/original_size) * 100
        print(f"   {name:8s}:  {size:.2f} GB ({100-reduction:.1f}%) - {reduction:.1f}% smaller")
    
    print()
    print("📁 All models saved in:")
    print(f"   {os.path.abspath('.')}")
    print()
    print("🎯 Recommended: Use FP16 for best balance!")
    print()
    
    # Test inference speed
    print("🧪 Testing inference speed...")
    test_audio = r"F:\ACI_Projects\Bengali_ASR\jahangir-1aI0mOgbLFY-142.wav"
    
    if os.path.exists(test_audio):
        import time
        
        for name, path, _ in results:
            model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(path)
            start = time.time()
            result = model.transcribe([test_audio])
            elapsed = time.time() - start
            print(f"   {name}: {elapsed:.2f}s")
        print()

if __name__ == "__main__":
    compress_model()