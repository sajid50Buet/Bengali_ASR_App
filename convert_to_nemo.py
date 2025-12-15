import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

# --- CONFIGURATION ---
# 1. The path to your .ckpt file (from your screenshot)
CHECKPOINT_PATH = "parakeet-bengali-epoch=10-step=12000-val_wer=0.1156.ckpt"

# 2. The output name for your final model
NEMO_OUTPUT_PATH = "parakeet_bengali_tdt_0.6b.nemo"

# 3. Path to your tokenizer (CRITICAL)
# Point this to the folder containing 'tokenizer.model' and 'vocab.txt'
TOKENIZER_DIR = "tokenizer" 

def convert():
    print(f"🔄 Loading checkpoint: {CHECKPOINT_PATH}...")
    
    # Load the model from the checkpoint
    # map_location="cpu" ensures it works even if you don't have a GPU right now
    model = EncDecRNNTBPEModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH, 
        map_location="cpu"
    )

    # Force the tokenizer path to be correct in the final artifact
    # This prevents "FileNotFound" errors when others download your model
    model.tokenizer_dir = TOKENIZER_DIR 

    print(f"💾 Saving to: {NEMO_OUTPUT_PATH}...")
    model.save_to(NEMO_OUTPUT_PATH)
    
    print("✅ Conversion Complete!")

if __name__ == "__main__":
    convert()