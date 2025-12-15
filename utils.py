import os
import time
from datetime import datetime
import tempfile
from pathlib import Path
from fastapi import UploadFile
import librosa
import nemo.collections.asr as nemo_asr
import torch
import soundfile as sf
import numpy as np

def log(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


class TranscriptionService:
    """Service for handling Bengali audio transcription"""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            nemo_files = list(Path('.').glob('*.nemo'))
            if not nemo_files:
                raise FileNotFoundError("No .nemo model file found!")
            model_path = str(nemo_files[0])
        
        log(f"Loading NeMo model: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Using device: {self.device}")
        
        self.model = nemo_asr.models.ASRModel.restore_from(model_path)
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        torch.set_grad_enabled(False)
        log("✅ Model loaded successfully!")
    
    def split_audio_with_overlap(self, audio, sr, chunk_duration=8.0, overlap=1.0):
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        stride = chunk_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            if len(chunk) < chunk_samples * 0.1:
                break
            
            chunks.append(chunk)
            start += stride
        
        return chunks
    
    def merge_transcriptions_lcs(self, transcriptions: list) -> str:
        """
        Merge overlapping transcriptions using Longest Common Subsequence (LCS)
        algorithm at word level to remove duplicated text from overlap regions.
        Based on NVIDIA's approach in parakeet_tdt.pdf
        """
        if not transcriptions:
            return ""
        if len(transcriptions) == 1:
            return transcriptions[0]
        
        def lcs_words(seq1: list, seq2: list) -> list:
            """Find Longest Common Subsequence of two word lists"""
            m, n = len(seq1), len(seq2)
            
            # Create DP table
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            # Backtrack to find LCS
            lcs = []
            i, j = m, n
            while i > 0 and j > 0:
                if seq1[i-1] == seq2[j-1]:
                    lcs.append((i-1, j-1, seq1[i-1]))
                    i -= 1
                    j -= 1
                elif dp[i-1][j] > dp[i][j-1]:
                    i -= 1
                else:
                    j -= 1
            
            return list(reversed(lcs))
        
        def find_overlap_point(prev_words: list, curr_words: list, max_overlap: int = 25) -> tuple:
            """
            Find the best overlap point between end of prev and start of curr
            Returns (prev_end_idx, curr_start_idx)
            """
            # Look at last N words of prev and first N words of curr
            check_prev = prev_words[-max_overlap:] if len(prev_words) > max_overlap else prev_words
            check_curr = curr_words[:max_overlap] if len(curr_words) > max_overlap else curr_words
            
            # Find LCS
            lcs = lcs_words(check_prev, check_curr)
            
            if len(lcs) >= 2:  # Need at least 2 matching words to be confident
                # Find the offset for prev_words if we sliced it
                prev_offset = len(prev_words) - len(check_prev)
                
                # Get the last match point
                last_match = lcs[-1]
                prev_idx = prev_offset + last_match[0]  # Index in original prev_words
                curr_idx = last_match[1]  # Index in curr_words
                
                return (prev_idx + 1, curr_idx + 1)  # +1 because we want to exclude the matched word from curr
            
            return (len(prev_words), 0)  # No overlap found, just concatenate
        
        # Merge all transcriptions
        merged_words = transcriptions[0].split()
        
        for i in range(1, len(transcriptions)):
            curr_words = transcriptions[i].split()
            
            if not curr_words:
                continue
            
            if not merged_words:
                merged_words = curr_words
                continue
            
            # Find overlap point
            prev_end, curr_start = find_overlap_point(merged_words, curr_words)
            
            # Merge: keep prev up to overlap, then add curr from overlap
            merged_words = merged_words[:prev_end] + curr_words[curr_start:]
        
        return " ".join(merged_words)

    def transcribe(self, audio_path: str, max_duration: float = 300.0) -> dict:
        try:
            log(f"🎤 Starting transcription: {audio_path}")
            start_time = time.time()
            
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            log(f"⏱️  Audio duration: {duration:.2f}s")
            
            if duration <= max_duration:
                log(f"   Direct transcription (audio ≤ {max_duration}s)")
                transcriptions = self.model.transcribe([audio_path])
                
                if isinstance(transcriptions, list) and len(transcriptions) > 0:
                    result = transcriptions[0]
                    text = result.text if hasattr(result, 'text') else str(result)
                else:
                    text = str(transcriptions)
            else:
                log(f"   Long audio detected! Using chunking...")
                text = self._transcribe_long_audio(audio_path, duration)
            
            processing_time = time.time() - start_time
            log(f"✅ Transcription completed in {processing_time:.2f}s")
            
            return {"text": text, "processing_time": processing_time}
        except Exception as e:
            log(f"❌ Error during transcription: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _transcribe_long_audio(self, audio_path: str, duration: float) -> str:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        chunks = self.split_audio_with_overlap(audio, sr, chunk_duration=300.0, overlap=1.0)
        log(f"   Split into {len(chunks)} chunks")
        
        temp_dir = Path("temp_chunks")
        temp_dir.mkdir(exist_ok=True)
        
        transcriptions = []
        
        for i, chunk in enumerate(chunks):
            log(f"   Transcribing chunk {i+1}/{len(chunks)}...")
            chunk_path = temp_dir / f"chunk_{i}_{int(time.time()*1000)}.wav"
            sf.write(str(chunk_path), chunk, sr)
            
            try:
                result = self.model.transcribe([str(chunk_path)])
                if isinstance(result, list) and len(result) > 0:
                    chunk_result = result[0]
                    chunk_text = chunk_result.text if hasattr(chunk_result, 'text') else str(chunk_result)
                else:
                    chunk_text = str(result)
                transcriptions.append(chunk_text)
            finally:
                if chunk_path.exists():
                    chunk_path.unlink()
        
        try:
            temp_dir.rmdir()
        except:
            pass
        
        return self.merge_transcriptions_lcs(transcriptions)


class DiarizedTranscriptionService:
    """Speaker diarization + transcription with timestamps"""
    
    def __init__(self, asr_model_path: str = None, hf_token: str = None, asr_service: TranscriptionService = None):
        # Use existing ASR service or create new one
        if asr_service:
            self.asr_service = asr_service
        else:
            self.asr_service = TranscriptionService(asr_model_path)
        
        # Initialize Diarization
        log("Loading pyannote diarization model...")
        from pyannote.audio import Pipeline as DiarizationPipeline
        
        # Fix for PyTorch 2.6+ weights_only issue - monkey patch torch.load
        _original_torch_load = torch.load
        
        def _patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
        
        torch.load = _patched_torch_load
        
        try:
            self.diarization = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
        finally:
            torch.load = _original_torch_load  # Restore original
        
        if torch.cuda.is_available():
            self.diarization.to(torch.device("cuda"))
        
        log("✅ Diarization model loaded!")
        
    def transcribe_with_diarization(
        self, 
        audio_path: str, 
        min_segment_duration: float = 0.5,
        merge_same_speaker: bool = True,
        gap_threshold: float = 0.5
    ) -> dict:
        """
        Transcribe audio with speaker labels and timestamps
        
        Args:
            audio_path: Path to audio file
            min_segment_duration: Minimum segment length (seconds)
            merge_same_speaker: Merge adjacent same-speaker segments
            gap_threshold: Max gap to merge same-speaker segments
        
        Returns:
            dict with 'segments', 'full_text', 'processing_time', 'num_speakers'
        """
        log(f"🎤 Starting diarized transcription: {audio_path}")
        start_time = time.time()
        
        # Step 1: Run diarization
        log("🔍 Running speaker diarization...")
        diarization_result = self.diarization(audio_path)
        
        # Step 2: Extract speaker segments
        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if turn.duration < min_segment_duration:
                continue
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.duration
            })
        
        log(f"   Found {len(segments)} speaker segments")
        
        # Step 3: Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Step 4: Transcribe each segment
        temp_dir = Path("temp_diarization")
        temp_dir.mkdir(exist_ok=True)
        
        results = []
        for i, seg in enumerate(segments):
            log(f"   [{i+1}/{len(segments)}] {seg['speaker']} ({seg['start']:.1f}s-{seg['end']:.1f}s)")
            
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            segment_path = temp_dir / f"seg_{i}_{int(time.time()*1000)}.wav"
            sf.write(str(segment_path), segment_audio, sr)
            
            try:
                transcription = self.asr_service.model.transcribe([str(segment_path)])
                if isinstance(transcription, list) and len(transcription) > 0:
                    result = transcription[0]
                    text = result.text if hasattr(result, 'text') else str(result)
                else:
                    text = str(transcription)
                
                results.append({
                    "speaker": seg["speaker"],
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": text.strip()
                })
            finally:
                if segment_path.exists():
                    segment_path.unlink()
        
        try:
            temp_dir.rmdir()
        except:
            pass
        
        # Step 5: Optionally merge same-speaker segments
        if merge_same_speaker:
            results = self._merge_same_speaker(results, gap_threshold)
        
        # Step 6: Format output
        full_text = self._format_output(results)
        processing_time = time.time() - start_time
        
        log(f"✅ Diarized transcription completed in {processing_time:.2f}s")
        
        return {
            "segments": results,
            "full_text": full_text,
            "processing_time": processing_time,
            "num_speakers": len(set(r["speaker"] for r in results))
        }
    
    def _merge_same_speaker(self, segments: list, gap_threshold: float = 0.5) -> list:
        """Merge adjacent segments from same speaker if gap < threshold"""
        if not segments:
            return segments
        
        merged = [segments[0].copy()]
        
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg["start"] - last["end"]
            
            if seg["speaker"] == last["speaker"] and gap < gap_threshold:
                last["end"] = seg["end"]
                last["text"] += " " + seg["text"]
            else:
                merged.append(seg.copy())
        
        return merged
    
    def _format_output(self, results: list) -> str:
        """Format results with speaker labels and timestamps"""
        lines = []
        for r in results:
            timestamp = f"[{r['start']:.1f}s - {r['end']:.1f}s]"
            lines.append(f"{timestamp} {r['speaker']}: {r['text']}")
        return "\n".join(lines)


# ============== UTILITY FUNCTIONS ==============

async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file and convert to 16kHz mono WAV"""
    import subprocess
    
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    temp_output_path = temp_dir / f"{timestamp}.wav"
    
    log(f"📥 Saving uploaded file: {upload_file.filename}")
    
    content = await upload_file.read()
    log(f"✅ File read: {len(content)} bytes")
    
    is_webm = content[:4] == b'\x1a\x45\xdf\xa3'
    ext = ".webm" if is_webm else ".wav"
    temp_input_path = temp_dir / f"{timestamp}_input{ext}"
    
    with open(temp_input_path, "wb") as f:
        f.write(content)
    
    time.sleep(0.05)
    
    try:
        log(f"🔄 Converting audio to 16kHz mono WAV...")
        result = subprocess.run([
            'ffmpeg', '-i', str(temp_input_path),
            '-ar', '16000', '-ac', '1',
            '-loglevel', 'error', '-y',
            str(temp_output_path)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        log(f"✅ Converted successfully!")
    finally:
        if temp_input_path.exists():
            temp_input_path.unlink()
    
    return str(temp_output_path)


def get_audio_duration(audio_path: str) -> float:
    try:
        return librosa.get_duration(path=audio_path)
    except:
        return 0.0


def cleanup_temp_files(max_age_hours: int = 24):
    for temp_dir in ["temp_audio", "temp_chunks", "temp_diarization"]:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            continue
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in temp_path.glob("*"):
            if file_path.is_file():
                if current_time - file_path.stat().st_mtime > max_age_seconds:
                    try:
                        file_path.unlink()
                    except:
                        pass



# ============== REAL-TIME AUDIO ENHANCEMENT ==============

class AudioEnhancer:
    """Real-time noise suppression and voice amplification - Lightweight version"""
    
    def __init__(self):
        log("Loading audio enhancement (lightweight mode)...")
        self.sample_rate = 16000
        self.target_db = -20
        
        # Pre-import noisereduce
        try:
            import noisereduce as nr
            self.nr = nr
            log("✅ Audio enhancer ready (noisereduce)")
        except ImportError:
            log("⚠️ noisereduce not installed. Run: pip install noisereduce")
            self.nr = None
    
    def enhance_chunk(self, audio: np.ndarray, 
                      suppress_noise: bool = True,
                      normalize_volume: bool = True,
                      target_db: float = -20) -> np.ndarray:
        """Enhance audio chunk in real-time."""
        if len(audio) < 512:
            return audio
        
        enhanced = audio.copy()
        
        try:
            if suppress_noise and self.nr is not None:
                enhanced = self._suppress_noise(enhanced)
            
            if normalize_volume:
                enhanced = self._normalize_volume(enhanced, target_db)
        except Exception as e:
            # Fail silently, return original
            pass
        
        return enhanced
    
    def _suppress_noise(self, audio: np.ndarray) -> np.ndarray:
        """Light noise suppression using noisereduce"""
        # Use stationary mode for speed, less aggressive
        return self.nr.reduce_noise(
            y=audio, 
            sr=self.sample_rate,
            stationary=True,       # Faster than non-stationary
            prop_decrease=0.6,     # Less aggressive (was 0.75)
            n_fft=512,
            hop_length=256,        # Larger hop = faster
            n_std_thresh_stationary=1.5
        ).astype(np.float32)
    
    def _normalize_volume(self, audio: np.ndarray, target_db: float = -20) -> np.ndarray:
        """Normalize volume with soft limiting"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio
        
        current_db = 20 * np.log10(rms + 1e-8)
        gain_db = target_db - current_db
        gain_db = np.clip(gain_db, -10, 20)  # More conservative gain limits
        gain = 10 ** (gain_db / 20)
        
        amplified = audio * gain
        
        # Soft clipping
        amplified = np.clip(amplified, -1.0, 1.0)
        
        return amplified.astype(np.float32)
    
# ============== STREAMING ASR SERVICE ==============

class StreamingTranscriptionService:
    """Simple streaming ASR with VAD - Fast version"""
    
    def __init__(self, asr_service: TranscriptionService):
        self.asr_service = asr_service
        self.sample_rate = 16000
        
        log("Loading Silero VAD model...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        log("✅ VAD loaded!")
    
    def transcribe_buffer(self, audio_buffer: np.ndarray) -> str:
        if len(audio_buffer) < 4800:
            return ""
        
        temp_path = Path("temp_streaming") / f"{int(time.time()*1000)}.wav"
        temp_path.parent.mkdir(exist_ok=True)
        
        try:
            sf.write(str(temp_path), audio_buffer, self.sample_rate)
            result = self.asr_service.model.transcribe([str(temp_path)])
            if isinstance(result, list) and result:
                return (result[0].text if hasattr(result[0], 'text') else str(result[0])).strip()
            return ""
        finally:
            temp_path.unlink(missing_ok=True)

def get_gpu_memory_info():
    """Check available GPU memory"""
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total = gpu.total_memory / 1024**3  # GB
        used = torch.cuda.memory_allocated() / 1024**3
        free = total - used
        return {
            "total_gb": round(total, 2),
            "used_gb": round(used, 2),
            "free_gb": round(free, 2),
            "recommended_concurrent": max(1, int(free // 2))  # ~2GB per request
        }
    return {"error": "No GPU available"}