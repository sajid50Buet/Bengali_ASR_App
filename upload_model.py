from huggingface_hub import HfApi

# --- CONFIGURATION ---
# Your Repo ID (from your screenshot)
REPO_ID = "Sajid50/parakeet-bengali-asr"

# The local file you just created with the conversion script
LOCAL_MODEL_FILE = "parakeet_bengali_tdt_0.6b.nemo"

# The name used on HuggingFace (Must match existing file to overwrite it)
REMOTE_FILENAME = "parakeet-bengali-asr.nemo"

api = HfApi()

print(f"🚀 Connecting to {REPO_ID}...")

# 1. Overwrite the Model File
print(f"☁️  Uploading '{LOCAL_MODEL_FILE}'...")
print(f"    ↳ Will overwrite: '{REMOTE_FILENAME}'")

api.upload_file(
    path_or_fileobj=LOCAL_MODEL_FILE,
    path_in_repo=REMOTE_FILENAME,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Update model with latest checkpoint (Epoch 10)"
)

# 2. Overwrite the Tokenizer
# This uploads your local 'tokenizer/' folder and replaces the remote one
print("☁️  Uploading/Updating tokenizer folder...")
api.upload_folder(
    folder_path="tokenizer",
    path_in_repo="tokenizer",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Update tokenizer"
)

print(f"\n✅ Success! The old model has been replaced.")
print(f"View here: https://huggingface.co/{REPO_ID}")