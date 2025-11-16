import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download

print("=" * 60)
print("🎙️ BENGALI ASR INFERENCE")
print("=" * 60)
print()

# Download and load model
print("📥 Loading model from Hugging Face...")
model_path = hf_hub_download(
    repo_id="Sajid50/parakeet-bengali-asr",
    filename="parakeet-bengali-asr.nemo"
)

asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
print("✅ Model loaded!")
print()

# Audio file
audio_file = r"F:\ACI_Projects\Bengali_ASR\jahangir-1aI0mOgbLFY-142.wav"

print(f"🎵 Transcribing: {audio_file}")
print()

# Transcribe (returns list of strings by default)
transcriptions = asr_model.transcribe([audio_file])

# Get the text (first element)
text = transcriptions[0]

# If you get Hypothesis object, extract text:
if hasattr(text, 'text'):
    text = text.text

print("=" * 60)
print("📝 TRANSCRIPTION:")
print("=" * 60)
print(text)
print()