from huggingface_hub import snapshot_download
import os

# --- CONFIGURATION ---
REPO_ID = "Sajid50/parakeet_bengali_asr_v2"
TARGET_DIR = "."  # Downloads into ./tokenizer

print(f"🚀 Downloading 'tokenizer' folder from {REPO_ID}...")

try:
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=TARGET_DIR,
        allow_patterns="tokenizer/*",  # <--- ONLY download the tokenizer folder
        local_dir_use_symlinks=False
    )
    print(f"✅ Success! Tokenizer is located at: {os.path.abspath('tokenizer')}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("   (Ensure you are logged in with 'huggingface-cli login')")