# utils.py
import os
import time
from datetime import datetime
from pathlib import Path
from fastapi import UploadFile
import librosa
import shutil
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
            model_path = "models/bengali_tdt_val_wer_0.2500_compressed.nemo"
            
            if not Path(model_path).exists():
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

    def transcribe(self, audio_path: str, max_chunk_duration: float = 30.0) -> dict:
        """Transcribe audio file with chunking for long audio"""
        try:
            log(f"🎤 Starting transcription: {audio_path}")
            start_time = time.time()
            
            # Get audio duration
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            log(f"⏱️  Audio duration: {duration:.2f}s")
            
            if duration <= max_chunk_duration:
                # Short audio - direct transcription
                log(f"   Direct transcription (audio ≤ {max_chunk_duration}s)")
                text = self._transcribe_single(audio_path)
            else:
                # Long audio - chunk and merge
                log(f"   Long audio detected! Chunking into ~{max_chunk_duration}s segments...")
                text = self._transcribe_chunked(audio_path, max_chunk_duration)
            
            processing_time = time.time() - start_time
            log(f"✅ Transcription completed in {processing_time:.2f}s")
            
            return {"text": text, "processing_time": processing_time}
        except Exception as e:
            log(f"❌ Error during transcription: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _transcribe_single(self, audio_path: str) -> str:
        """Transcribe a single audio file"""
        result = self.model.transcribe([audio_path])
        if isinstance(result, list) and len(result) > 0:
            return result[0].text if hasattr(result[0], 'text') else str(result[0])
        return str(result)
    
    def _transcribe_chunked(self, audio_path: str, chunk_duration: float = 30.0) -> str:
        """Transcribe long audio by chunking with overlap using batch processing"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Chunk parameters
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(2.0 * sr)  # 2 second overlap for context
        stride = chunk_samples - overlap_samples
        
        # Create chunks
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            if len(chunk) > sr * 0.5:  # Only if > 0.5s
                chunks.append(chunk)
            start += stride
        
        log(f"   Split into {len(chunks)} chunks")
        
        temp_dir = Path("temp_chunks")
        temp_dir.mkdir(exist_ok=True)
        
        transcriptions = []
        try:
            # 1. SAVE ALL CHUNKS TO DISK FIRST
            chunk_paths = []
            for i, chunk in enumerate(chunks):
                chunk_path = temp_dir / f"chunk_{i}.wav"
                sf.write(str(chunk_path), chunk, sr)
                chunk_paths.append(str(chunk_path))
            
            # 2. TRANSCRIBE ALL CHUNKS IN ONE SINGLE BATCH
            log(f"   Transcribing {len(chunk_paths)} chunks in batch...")
            # batch_size handles speed, return_hypotheses=False stops the memory leak
            batch_results = self.model.transcribe(
                chunk_paths, 
                batch_size=4, 
                return_hypotheses=False
            )
            
            # 3. EXTRACT TEXT
            for result in batch_results:
                text = result.text if hasattr(result, 'text') else str(result)
                transcriptions.append(text)
                
        finally:
            import shutil
            import gc
            
            # 1. Clean up files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 2. Aggressive VRAM Cleanup
            for var in ['audio', 'chunk', 'batch_results']:
                if var in locals():
                    del locals()[var]
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Merge transcriptions with overlap handling
        return self._merge_transcriptions(transcriptions)
    
    def _merge_transcriptions(self, transcriptions: list) -> str:
        """Merge overlapping transcriptions using LCS algorithm"""
        if not transcriptions:
            return ""
        if len(transcriptions) == 1:
            return transcriptions[0]
        
        def find_overlap(prev: str, curr: str, max_words: int = 20) -> int:
            """Find overlap point between two transcriptions"""
            prev_words = prev.split()
            curr_words = curr.split()
            
            if not prev_words or not curr_words:
                return 0
            
            # Check last N words of prev against first N words of curr
            check_prev = prev_words[-max_words:] if len(prev_words) > max_words else prev_words
            check_curr = curr_words[:max_words] if len(curr_words) > max_words else curr_words
            
            # Find longest matching sequence
            best_overlap = 0
            for i in range(len(check_prev)):
                for j in range(len(check_curr)):
                    match_len = 0
                    while (i + match_len < len(check_prev) and 
                           j + match_len < len(check_curr) and
                           check_prev[i + match_len] == check_curr[j + match_len]):
                        match_len += 1
                    
                    if match_len >= 2 and match_len > best_overlap:
                        best_overlap = j + match_len
            
            return best_overlap
        
        # Merge all transcriptions
        merged = transcriptions[0]
        
        for i in range(1, len(transcriptions)):
            curr = transcriptions[i]
            overlap_start = find_overlap(merged, curr)
            
            if overlap_start > 0:
                # Skip overlapping words in current
                curr_words = curr.split()[overlap_start:]
                merged = merged + " " + " ".join(curr_words)
            else:
                # No overlap found, just concatenate
                merged = merged + " " + curr
        
        return merged.strip()

class DiarizedTranscriptionService:
    """Speaker diarization + transcription with timestamps"""
    
    def __init__(self, hf_token: str, asr_service: TranscriptionService):
        self.asr_service = asr_service
        
        log("Loading pyannote diarization model...")
        from pyannote.audio import Pipeline as DiarizationPipeline
        
        # Fix for PyTorch 2.6+ weights_only issue
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})
        
        try:
            self.diarization = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
        finally:
            torch.load = _original_torch_load
        
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
        """Transcribe audio with speaker labels and timestamps using batch processing"""
        log(f"🎤 Starting diarized transcription: {audio_path}")
        start_time = time.time()
        
        # Run diarization
        log("🔍 Running speaker diarization...")
        diarization_result = self.diarization(audio_path)
        
        # Extract speaker segments
        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if turn.duration >= min_segment_duration:
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
        
        log(f"   Found {len(segments)} speaker segments")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        temp_dir = Path("temp_diarization")
        temp_dir.mkdir(exist_ok=True)
        
        results = []
        try:
            # 1. SAVE ALL CHUNKS TO DISK FIRST
            segment_paths = []
            for i, seg in enumerate(segments):
                log(f"   [{i+1}/{len(segments)}] Preparing {seg['speaker']} ({seg['start']:.1f}s-{seg['end']:.1f}s)")
                segment_audio = audio[int(seg["start"] * sr):int(seg["end"] * sr)]
                segment_path = temp_dir / f"seg_{i}.wav"
                sf.write(str(segment_path), segment_audio, sr)
                segment_paths.append(str(segment_path))
            
            # 2. TRANSCRIBE ALL CHUNKS IN ONE SINGLE BATCH
            log(f"   Transcribing {len(segment_paths)} segments in batch...")
            # batch_size handles speed, return_hypotheses=False stops the memory leak
            transcriptions = self.asr_service.model.transcribe(
                segment_paths, 
                batch_size=4, 
                return_hypotheses=False 
            )
            
            # 3. MATCH TRANSCRIPTIONS BACK TO SPEAKERS
            for i, seg in enumerate(segments):
                trans_obj = transcriptions[i]
                text = trans_obj.text if hasattr(trans_obj, 'text') else str(trans_obj)
                
                results.append({
                    "speaker": seg["speaker"],
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": text.strip()
                })
                
        finally:
            import shutil
            import gc
            
            # 1. Clean up files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 2. Aggressive VRAM Cleanup
            # Explicitly delete heavy local variables if they exist in locals()
            for var in ['audio', 'segment_audio', 'diarization_result', 'transcriptions']:
                if var in locals():
                    del locals()[var]
                    
            # Force Python garbage collector to run
            gc.collect()
            
            # Force PyTorch to release the reserved memory back to NVIDIA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Merge same-speaker segments
        if merge_same_speaker:
            results = self._merge_same_speaker(results, gap_threshold)
        
        processing_time = time.time() - start_time
        log(f"✅ Diarized transcription completed in {processing_time:.2f}s. VRAM cleared.")
        
        return {
            "segments": results,
            "full_text": self._format_output(results),
            "processing_time": processing_time,
            "num_speakers": len(set(r["speaker"] for r in results))
        }
    
    def _merge_same_speaker(self, segments: list, gap_threshold: float) -> list:
        if not segments:
            return segments
        
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            last = merged[-1]
            if seg["speaker"] == last["speaker"] and seg["start"] - last["end"] < gap_threshold:
                last["end"] = seg["end"]
                last["text"] += " " + seg["text"]
            else:
                merged.append(seg.copy())
        return merged
    
    def _format_output(self, results: list) -> str:
        return "\n".join(
            f"[{r['start']:.1f}s - {r['end']:.1f}s] {r['speaker']}: {r['text']}"
            for r in results
        )


class StreamingTranscriptionService:
    """Simple streaming ASR with VAD"""
    
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
            # [FIX] Explicitly disable gradients during inference
            with torch.no_grad():
                result = self.asr_service.model.transcribe([str(temp_path)])
            if isinstance(result, list) and result:
                return (result[0].text if hasattr(result[0], 'text') else str(result[0])).strip()
            return ""
        finally:
            temp_path.unlink(missing_ok=True)


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
    is_webm = content[:4] == b'\x1a\x45\xdf\xa3'
    ext = ".webm" if is_webm else ".wav"
    temp_input_path = temp_dir / f"{timestamp}_input{ext}"
    
    with open(temp_input_path, "wb") as f:
        f.write(content)
    
    try:
        result = subprocess.run([
            'ffmpeg', '-i', str(temp_input_path),
            '-ar', '16000', '-ac', '1',
            '-loglevel', 'error', '-y',
            str(temp_output_path)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
    finally:
        temp_input_path.unlink(missing_ok=True)
    
    return str(temp_output_path)


def get_audio_duration(audio_path: str) -> float:
    try:
        return librosa.get_duration(path=audio_path)
    except:
        return 0.0
    
def is_valid_transcription(text: str) -> bool:
    """Filter out garbage text from hallucinations"""
    if not text:
        return False
    text = text.strip()
    
    # 1. Ignore very short outputs (often just noise interpreted as a letter)
    if len(text) < 2: 
        return False
        
    # 2. Ignore output that is ONLY punctuation
    if all(char in ". ,!?।" for char in text): 
        return False
        
    # 3. (Optional) Ignore common hallucination phrases if you notice them
    # garbage_phrases = ["Thank you", "Subtitles by", "."]
    # if text in garbage_phrases: return False
    
    return True