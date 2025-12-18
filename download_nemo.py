from huggingface_hub import hf_hub_download
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
FILENAME = "parakeet_bengali_asr_v2.nemo"
TARGET_DIR = "."

HF_TOKEN = os.getenv("HF_TOKEN")

print(f"🚀 Downloading {FILENAME} from {REPO_ID}...")

try:
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        token=HF_TOKEN
    )
    print(f"✅ Success! File saved at: {file_path}")
    
except Exception as e:
    print(f"❌ Error: {e}")