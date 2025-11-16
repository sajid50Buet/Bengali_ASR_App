import nemo.collections.asr as nemo_asr
import os

print("=" * 60)
print("🎙️ BENGALI ASR - SIMPLE INFERENCE")
print("=" * 60)
print()

# Load model
model_path = "parakeet-bengali-asr-fp16.nemo"

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("\nAvailable models:")
    for f in os.listdir("."):
        if f.endswith(".nemo"):
            print(f"  - {f}")
    exit(1)

print(f"📥 Loading model: {model_path}")
model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
print("✅ Model loaded!")
print()

# Audio file path
audio_file = r"F:\ACI_Projects\Bengali_ASR\jahangir-1aI0mOgbLFY-142.wav"

if not os.path.exists(audio_file):
    print(f"❌ Audio file not found: {audio_file}")
    exit(1)

print(f"🎵 Transcribing: {os.path.basename(audio_file)}")
print()

# ✅ CORRECT WAY - Just pass list, no keyword argument!
transcriptions = model.transcribe([audio_file])

# Extract text from result
if isinstance(transcriptions[0], str):
    text = transcriptions[0]
else:
    # If it returns Hypothesis object
    text = transcriptions[0].text if hasattr(transcriptions[0], 'text') else str(transcriptions[0])

# Print result
print("=" * 60)
print("📝 TRANSCRIPTION:")
print("=" * 60)
print(text)
print()