import os
import time
from datetime import datetime
import tempfile
from pathlib import Path
from fastapi import UploadFile
import librosa
import nemo.collections.asr as nemo_asr
import torch

def log(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

class TranscriptionService:
    """Service for handling Bengali audio transcription"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the transcription model
        
        Args:
            model_path: Path to the .nemo model file
        """
        if model_path is None:
            # Look for .nemo file in current directory
            nemo_files = list(Path('.').glob('*.nemo'))
            if not nemo_files:
                raise FileNotFoundError("No .nemo model file found! Please provide model_path or place a .nemo file in the current directory.")
            model_path = str(nemo_files[0])
        
        log(f"Loading NeMo model: {model_path}")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Using device: {self.device}")
        
        # Load model
        self.model = nemo_asr.models.ASRModel.restore_from(model_path)
        
        # Move model to GPU if available
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        # Set model to eval mode for faster inference
        self.model.eval()
        
        # Disable gradients for faster inference
        torch.set_grad_enabled(False)
        
        log("✅ Model loaded successfully!")
    
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict with 'text' and 'processing_time'
        """
        try:
            log(f"🎤 Starting transcription: {audio_path}")
            start_time = time.time()
            
            # Transcribe - just pass list of audio files
            transcriptions = self.model.transcribe([audio_path])
            
            processing_time = time.time() - start_time
            
            # Extract text - NeMo returns different types depending on model
            if isinstance(transcriptions, list) and len(transcriptions) > 0:
                result = transcriptions[0]
                # Handle Hypothesis object
                if hasattr(result, 'text'):
                    text = result.text
                else:
                    text = str(result)
            else:
                text = str(transcriptions)
            
            log(f"✅ Transcription completed in {processing_time:.2f}s")
            log(f"📝 Result: {text[:100]}..." if len(text) > 100 else f"📝 Result: {text}")
            
            return {
                "text": text,
                "processing_time": processing_time
            }
        except Exception as e:
            log(f"❌ Error during transcription: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")


async def save_uploaded_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file to temporary location and convert to 16kHz mono WAV
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        Path to the saved file
    """
    import soundfile as sf
    import wave
    import subprocess
    
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    timestamp = int(time.time() * 1000)
    temp_input_path = temp_dir / f"{timestamp}_input"
    temp_output_path = temp_dir / f"{timestamp}.wav"
    
    log(f"📥 Saving uploaded file: {upload_file.filename}")
    
    # Read file content
    try:
        content = await upload_file.read()
        log(f"✅ File read: {len(content)} bytes")
    except Exception as e:
        log(f"❌ Error reading file: {str(e)}")
        raise Exception(f"Error reading file: {str(e)}")
    
    # Check if it's WebM (browser recording)
    is_webm = content[:4] == b'\x1a\x45\xdf\xa3'
    
    if is_webm:
        log(f"   Detected WebM format (browser recording)")
        # Save as .webm
        temp_input_path = temp_dir / f"{timestamp}_input.webm"
    else:
        # Save with original-like extension
        temp_input_path = temp_dir / f"{timestamp}_input.wav"
    
    # Save uploaded file
    try:
        with open(temp_input_path, "wb") as f:
            f.write(content)
        log(f"✅ File saved to: {temp_input_path}")
    except Exception as e:
        log(f"❌ Error saving file: {str(e)}")
        raise Exception(f"Error saving file: {str(e)}")
    
    # Small delay to ensure file is released on Windows
    import time as time_module
    time_module.sleep(0.05)
    
    # Try to convert audio
    try:
        log(f"🔄 Converting audio to 16kHz mono WAV...")
        convert_start = time.time()
        
        # Always use FFmpeg - it's faster and more reliable
        log(f"   Using FFmpeg to convert audio...")
        try:
            result = subprocess.run([
                'ffmpeg', '-i', str(temp_input_path),
                '-ar', '16000',  # Sample rate 16kHz
                '-ac', '1',      # Mono
                '-loglevel', 'error',  # Only show errors
                '-y',            # Overwrite output file
                str(temp_output_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            convert_time = time.time() - convert_start
            log(f"✅ Converted using FFmpeg in {convert_time:.2f}s!")
            
        except FileNotFoundError:
            raise Exception(
                "FFmpeg not found! Please make sure FFmpeg is installed and in PATH. "
                "Restart your terminal after installing FFmpeg."
            )
        
        # Clean up input file
        if temp_input_path.exists():
            temp_input_path.unlink()
        
        return str(temp_output_path)
        
    except Exception as e:
        print(f"❌ Error converting audio: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if temp_input_path.exists():
            temp_input_path.unlink()
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise Exception(f"Error converting audio: {str(e)}")


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        duration = librosa.get_duration(path=audio_path)
        log(f"⏱️ Audio duration: {duration:.2f}s")
        return duration
    except Exception as e:
        log(f"⚠️ Error getting audio duration: {e}")
        return 0.0


def cleanup_temp_files(max_age_hours: int = 24):
    """
    Clean up old temporary audio files
    
    Args:
        max_age_hours: Maximum age of files to keep in hours
    """
    temp_dir = Path("temp_audio")
    if not temp_dir.exists():
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in temp_dir.glob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    print(f"Deleted old temp file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")