import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

# 1. Define your paths
INPUT_CKPT = "parakeet-bengali-epoch=10-step=12000-val_wer=0.1156.ckpt"
OUTPUT_NEMO = "parakeet-bengali-asr-fp16.nemo"

print(f"⏳ Loading checkpoint: {INPUT_CKPT}...")

# 2. Load the Lightning Checkpoint
# We load to CPU first to avoid memory spikes, then convert
if torch.cuda.is_available():
    map_location = torch.device('cuda')
else:
    map_location = torch.device('cpu')

try:
    # Try loading directly as a Lightning Checkpoint
    model = EncDecRNNTBPEModel.load_from_checkpoint(INPUT_CKPT, map_location=map_location)
except Exception as e:
    print(f"Standard load failed, trying strict=False. Error: {e}")
    model = EncDecRNNTBPEModel.load_from_checkpoint(INPUT_CKPT, map_location=map_location, strict=False)

# 3. Convert to FP16 (Half Precision)
print("⚙️ Converting model weights to FP16...")
model = model.half()

# 4. Save as .nemo
print(f"💾 Saving to {OUTPUT_NEMO}...")
model.save_to(OUTPUT_NEMO)

print("✅ Conversion Complete!")