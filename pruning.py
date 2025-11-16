import torch
import torch.nn.utils.prune as prune
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download
import os

print("=" * 70)
print("✂️ PRUNING PARAKEET BENGALI ASR MODEL")
print("=" * 70)
print()

# Download model from HuggingFace (or use cached version)
print("📥 Loading model from HuggingFace...")
model_path = hf_hub_download(
    repo_id="Sajid50/parakeet-bengali-asr",
    filename="parakeet-bengali-asr.nemo"
)
print(f"✅ Model path: {model_path}")
print()

# Load model
print("🔄 Loading model...")
model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
print("✅ Model loaded!")
print()

# Get original size
original_size = os.path.getsize(model_path) / (1024**3)
print(f"📊 Original model size: {original_size:.2f} GB")
print()

# Prune 30% of weights (keeps 70%)
print("✂️ Pruning 30% of weights...")
pruned_params = 0
total_params = 0

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, 'weight')  # Make pruning permanent
        pruned_params += torch.nn.utils.parameters_to_vector(module.parameters()).numel()
    total_params += sum(p.numel() for p in module.parameters())

print(f"✅ Pruned {pruned_params:,} parameters")
print()

# Save pruned model to CURRENT DIRECTORY
output_path = "parakeet-bengali-asr-pruned.nemo"
print(f"💾 Saving pruned model to: {output_path}")
model.save_to(output_path)

# Check new size
pruned_size = os.path.getsize(output_path) / (1024**3)
reduction = (1 - pruned_size/original_size) * 100

print()
print("=" * 70)
print("✅ PRUNING COMPLETE!")
print("=" * 70)
print()
print(f"📊 Results:")
print(f"   Original:  {original_size:.2f} GB")
print(f"   Pruned:    {pruned_size:.2f} GB")
print(f"   Reduction: {reduction:.1f}%")
print()
print(f"💾 Saved to: {os.path.abspath(output_path)}")
print()
print("🎯 Next steps:")
print("   1. Test WER on your data")
print("   2. If acceptable, upload to HuggingFace")
print("   3. Compare inference speed")
print()