from huggingface_hub import HfApi
import os

# Load from .env manually
def load_env():
    if os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env()

# --- CONFIGURATION ---
REPO_ID = "Sajid50/parakeet_bengali_asr_v2"
LOCAL_MODEL_FILE = "bengali_tdt_val_wer_0.2500.nemo"
REMOTE_FILENAME = "parakeet_bengali_asr_v2.nemo"

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ No HF_TOKEN in .env!")
    exit(1)

api = HfApi(token=HF_TOKEN)

print(f"🚀 Uploading to {REPO_ID}...")
print("=" * 50)

# 1. Upload Model
if os.path.exists(LOCAL_MODEL_FILE):
    size_gb = os.path.getsize(LOCAL_MODEL_FILE) / (1024**3)
    print(f"\n📤 Uploading: {LOCAL_MODEL_FILE} ({size_gb:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=LOCAL_MODEL_FILE,
        path_in_repo=REMOTE_FILENAME,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update model"
    )
    print("   ✅ Model uploaded")
else:
    print(f"⚠️  Model not found: {LOCAL_MODEL_FILE}")

# 2. Upload Tokenizer folder
if os.path.exists("tokenizer"):
    print(f"\n📤 Uploading: tokenizer/...")
    api.upload_folder(
        folder_path="tokenizer",
        path_in_repo="tokenizer",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update tokenizer"
    )
    print("   ✅ Tokenizer uploaded")
else:
    print("⚠️  Tokenizer folder not found")

print("\n" + "=" * 50)
print(f"🎉 Done! https://huggingface.co/{REPO_ID}")