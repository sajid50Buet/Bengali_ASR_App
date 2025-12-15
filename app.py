from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import os
import numpy as np
import base64
import asyncio


from dotenv import load_dotenv
load_dotenv()

from utils import (
    TranscriptionService, 
    DiarizedTranscriptionService,
    StreamingTranscriptionService,  # ADD THIS
    save_uploaded_file, 
    get_audio_duration, 
    log
)

# Global variables
transcription_service = None
diarized_service = None
streaming_service = None

# Set your HuggingFace token here (or use environment variable)
HF_TOKEN = os.getenv("HF_TOKEN", None)  # or "hf_xxxxx"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_service, diarized_service, streaming_service

    log("🚀 Initializing Bengali ASR Model...")
    transcription_service = TranscriptionService()
    
    # Load Streaming Service (reuses ASR model)
    log("🎙️ Initializing Streaming Service...")
    streaming_service = StreamingTranscriptionService(asr_service=transcription_service)

    
    # Load Diarization (reuse ASR model)
    if HF_TOKEN:
        log("🔊 Initializing Speaker Diarization...")
        try:
            # Pass existing ASR service instead of creating new one
            diarized_service = DiarizedTranscriptionService(
                asr_service=transcription_service,  # Reuse!
                hf_token=HF_TOKEN
            )
        except Exception as e:
            log(f"⚠️ Diarization init failed: {e}. Continuing without it.")
            diarized_service = None
    else:
        log("⚠️ HF_TOKEN not set. Diarization disabled.")
    
    log("✅ Application startup complete!")
    yield
    log("👋 Shutting down...")


app = FastAPI(title="Bengali Audio Transcription API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend HTML page"""
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Simple transcription (no speaker labels)
    Returns: transcription text, audio duration, processing time
    """
    log(f"📨 Received transcription request: {audio.filename}")
    
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    audio_path = None
    try:
        audio_path = await save_uploaded_file(audio)
        duration = get_audio_duration(audio_path)
        result = transcription_service.transcribe(audio_path)
        
        response_data = {
            "transcription": str(result["text"]),
            "audio_duration": float(duration),
            "processing_time": float(result["processing_time"]),
            "status": "success"
        }
        
        log(f"✅ Request completed!")
        return JSONResponse(content=response_data)
    
    except Exception as e:
        log(f"❌ Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass


@app.post("/transcribe_diarize")
async def transcribe_with_diarization(
    audio: UploadFile = File(...),
    min_segment_duration: float = Query(0.5, description="Min segment length (seconds)"),
    merge_same_speaker: bool = Query(True, description="Merge adjacent same-speaker segments"),
    gap_threshold: float = Query(0.5, description="Max gap to merge same-speaker")
):
    """
    Transcription with speaker diarization
    Returns: segments with speaker labels + timestamps
    """
    if diarized_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Diarization not available. Set HF_TOKEN environment variable."
        )
    
    log(f"📨 Received diarized transcription request: {audio.filename}")
    
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    audio_path = None
    try:
        audio_path = await save_uploaded_file(audio)
        duration = get_audio_duration(audio_path)
        
        result = diarized_service.transcribe_with_diarization(
            audio_path,
            min_segment_duration=min_segment_duration,
            merge_same_speaker=merge_same_speaker,
            gap_threshold=gap_threshold
        )
        
        response_data = {
            "segments": result["segments"],
            "full_text": result["full_text"],
            "num_speakers": result["num_speakers"],
            "audio_duration": float(duration),
            "processing_time": float(result["processing_time"]),
            "status": "success"
        }
        
        log(f"✅ Diarized request completed!")
        return JSONResponse(content=response_data)
    
    except Exception as e:
        log(f"❌ Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")
    
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass


@app.get("/health")
async def health_check():
    """Check service status"""
    return {
        "status": "healthy",
        "model_loaded": transcription_service is not None,
        "diarization_available": diarized_service is not None
    }

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time streaming transcription with VAD and audio enhancement.
    """
    await websocket.accept()
    
    # Parse query params manually from scope
    query_string = websocket.scope.get("query_string", b"").decode()
    params = dict(p.split("=") for p in query_string.split("&") if "=" in p)
    noise_suppress = params.get("noise_suppress", "true").lower() == "true"
    normalize = params.get("normalize", "true").lower() == "true"
    target_db = float(params.get("target_db", "-20"))
    
    log(f"🔌 WebSocket connected (noise_suppress={noise_suppress}, normalize={normalize})")
    
    audio_buffer = np.array([], dtype=np.float32)
    pending_audio = np.array([], dtype=np.float32)
    pre_buffer = np.array([], dtype=np.float32)
    enhance_buffer = np.array([], dtype=np.float32)
    silence_frames = 0
    speech_started = False
    
    SILENCE_THRESHOLD = 30
    MIN_SPEECH_DURATION = 0.3
    VAD_CHUNK_SIZE = 512
    PRE_BUFFER_CHUNKS = 20
    ENHANCE_CHUNK_SIZE = 4096
    
    try:
        while True:
            try:
                # Non-blocking receive with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(), 
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
            
            if data.get("type") == "audio":
                audio_bytes = base64.b64decode(data["audio"])
                chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                enhance_buffer = np.concatenate([enhance_buffer, chunk])
                
                # Enhance in batches
                while len(enhance_buffer) >= ENHANCE_CHUNK_SIZE:
                    raw_chunk = enhance_buffer[:ENHANCE_CHUNK_SIZE]
                    enhance_buffer = enhance_buffer[ENHANCE_CHUNK_SIZE:]
                    
                    if noise_suppress or normalize:
                        enhanced_chunk = streaming_service.enhancer.enhance_chunk(
                            raw_chunk,
                            suppress_noise=noise_suppress,
                            normalize_volume=normalize,
                            target_db=target_db
                        )
                    else:
                        enhanced_chunk = raw_chunk
                    
                    pending_audio = np.concatenate([pending_audio, enhanced_chunk])
                
                # VAD processing
                while len(pending_audio) >= VAD_CHUNK_SIZE:
                    vad_chunk = pending_audio[:VAD_CHUNK_SIZE]
                    pending_audio = pending_audio[VAD_CHUNK_SIZE:]
                    
                    import torch
                    chunk_tensor = torch.from_numpy(vad_chunk.copy())
                    speech_prob = streaming_service.vad_model(chunk_tensor, 16000).item()
                    
                    if speech_prob > 0.5:
                        if not speech_started:
                            audio_buffer = np.concatenate([pre_buffer, vad_chunk])
                            pre_buffer = np.array([], dtype=np.float32)
                            speech_started = True
                            log("🎤 Speech started")
                        else:
                            audio_buffer = np.concatenate([audio_buffer, vad_chunk])
                        silence_frames = 0
                        
                    elif speech_started:
                        audio_buffer = np.concatenate([audio_buffer, vad_chunk])
                        silence_frames += 1
                        
                        if silence_frames >= SILENCE_THRESHOLD:
                            if len(audio_buffer) > MIN_SPEECH_DURATION * 16000:
                                log(f"🔄 Transcribing {len(audio_buffer)/16000:.2f}s of audio...")
                                text = streaming_service.transcribe_buffer(audio_buffer)
                                log(f"✅ Transcribed: {text[:50]}...")
                                
                                await websocket.send_json({
                                    "text": text,
                                    "is_final": True,
                                    "duration": round(len(audio_buffer) / 16000, 2)
                                })
                            
                            # Reset for next utterance
                            audio_buffer = np.array([], dtype=np.float32)
                            silence_frames = 0
                            speech_started = False
                            log("🔁 Ready for next utterance")
                    else:
                        pre_buffer = np.concatenate([pre_buffer, vad_chunk])
                        max_pre_buffer = VAD_CHUNK_SIZE * PRE_BUFFER_CHUNKS
                        if len(pre_buffer) > max_pre_buffer:
                            pre_buffer = pre_buffer[-max_pre_buffer:]
                
                if speech_started:
                    await websocket.send_json({
                        "status": "listening",
                        "buffer_duration": round(len(audio_buffer) / 16000, 2)
                    })
                        
            elif data.get("type") == "stop":
                if len(audio_buffer) > MIN_SPEECH_DURATION * 16000:
                    text = streaming_service.transcribe_buffer(audio_buffer)
                    await websocket.send_json({
                        "text": text,
                        "is_final": True,
                        "duration": round(len(audio_buffer) / 16000, 2)
                    })
                break
                
    except WebSocketDisconnect:
        log("🔌 WebSocket disconnected")
    except Exception as e:
        log(f"❌ WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8069)