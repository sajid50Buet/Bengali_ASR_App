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
    
    def split_audio_with_overlap(self, audio, sr, chunk_duration=8.0, overlap=1.0):
        """
        Split long audio into overlapping chunks
        
        Args:
            audio: Audio signal (numpy array)
            sr: Sample rate
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
        
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        stride = chunk_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Skip if chunk is too short (less than 10% of target)
            if len(chunk) < chunk_samples * 0.1:
                break
            
            chunks.append(chunk)
            start += stride
        
        return chunks
    
    def merge_transcriptions_lcs(self, transcriptions: list) -> str:
        """
        Merge overlapping transcriptions using fuzzy matching
        to handle slight variations in overlap regions.
        """
        if not transcriptions:
            return ""
        if len(transcriptions) == 1:
            return transcriptions[0]
        
        def similarity(s1: str, s2: str) -> float:
            """Simple character-level similarity ratio"""
            if not s1 or not s2:
                return 0.0
            matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
            return (2.0 * matches) / (len(s1) + len(s2))
        
        merged = transcriptions[0]
        
        for i in range(1, len(transcriptions)):
            current = transcriptions[i]
            if not current:
                continue
            
            merged_words = merged.split()
            current_words = current.split()
            
            # Search for best fuzzy overlap (check last 15 words of merged vs first 15 of current)
            max_check = min(15, len(merged_words), len(current_words))
            best_overlap = 0
            best_score = 0.0
            
            for overlap_len in range(2, max_check + 1):
                merged_segment = " ".join(merged_words[-overlap_len:])
                current_segment = " ".join(current_words[:overlap_len])
                
                score = similarity(merged_segment, current_segment)
                
                # Accept if similarity > 70%
                if score > 0.70 and score > best_score:
                    best_score = score
                    best_overlap = overlap_len
            
            # Merge: remove overlapping part from current
            if best_overlap > 0:
                merged = merged + " " + " ".join(current_words[best_overlap:])
            else:
                merged = merged + " " + current
        
        return merged.strip()

    def transcribe(self, audio_path: str, max_duration: float = 10.0) -> dict:
        """
        Transcribe audio file to text with automatic chunking for long audio
        
        Args:
            audio_path: Path to the audio file
            max_duration: Maximum duration before chunking (seconds)
            
        Returns:
            dict with 'text' and 'processing_time'
        """
        try:
            log(f"🎤 Starting transcription: {audio_path}")
            start_time = time.time()
            
            # Get audio duration first
            import soundfile as sf
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            
            log(f"⏱️  Audio duration: {duration:.2f}s")
            
            # If audio is short, transcribe directly
            if duration <= max_duration:
                log(f"   Direct transcription (audio ≤ {max_duration}s)")
                transcriptions = self.model.transcribe([audio_path])
                
                # Extract text
                if isinstance(transcriptions, list) and len(transcriptions) > 0:
                    result = transcriptions[0]
                    if hasattr(result, 'text'):
                        text = result.text
                    else:
                        text = str(result)
                else:
                    text = str(transcriptions)
            
            else:
                # Long audio - use chunking
                log(f"   Long audio detected! Using chunking...")
                text = self._transcribe_long_audio(audio_path, duration)
            
            processing_time = time.time() - start_time
            
            log(f"✅ Transcription completed in {processing_time:.2f}s")
            log(f"📝 Result: {text[:100]}..." if len(text) > 100 else f"📝 Result: {text}")
            
            return {
                "text": text,
                "processing_time": processing_time
            }
        except Exception as e:
            log(f"❌ Error during transcription: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _transcribe_long_audio(self, audio_path: str, duration: float) -> str:
        """
        Transcribe long audio by chunking
        
        Args:
            audio_path: Path to audio file
            duration: Audio duration in seconds
        
        Returns:
            Full transcription
        """
        import soundfile as sf
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Split into chunks (8s chunks with 1s overlap)
        chunk_duration = 8.0
        overlap = 1.0
        chunks = self.split_audio_with_overlap(audio, sr, chunk_duration, overlap)
        
        log(f"   Split into {len(chunks)} chunks")
        
        # Create temp directory for chunks
        temp_dir = Path("temp_chunks")
        temp_dir.mkdir(exist_ok=True)
        
        # Transcribe each chunk
        transcriptions = []
        
        for i, chunk in enumerate(chunks):
            log(f"   Transcribing chunk {i+1}/{len(chunks)}...")
            
            # Save chunk temporarily
            chunk_path = temp_dir / f"chunk_{i}_{int(time.time()*1000)}.wav"
            sf.write(str(chunk_path), chunk, sr)
            
            try:
                # Transcribe chunk
                result = self.model.transcribe([str(chunk_path)])
                
                # Extract text
                if isinstance(result, list) and len(result) > 0:
                    chunk_result = result[0]
                    if hasattr(chunk_result, 'text'):
                        chunk_text = chunk_result.text
                    else:
                        chunk_text = str(chunk_result)
                else:
                    chunk_text = str(result)
                
                transcriptions.append(chunk_text)
                log(f"   ✅ Chunk {i+1} transcribed: {chunk_text[:50]}..." if len(chunk_text) > 50 else f"   ✅ Chunk {i+1}: {chunk_text}")
                
            finally:
                # Clean up chunk file
                if chunk_path.exists():
                    chunk_path.unlink()
        
        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except:
            pass
        
        # Merge transcriptions
        full_text = self.merge_transcriptions_lcs(transcriptions)
        log(f"   Merged {len(transcriptions)} chunks into full transcription")
        
        return full_text


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