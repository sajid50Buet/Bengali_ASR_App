# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import os
import numpy as np
import base64

from dotenv import load_dotenv
load_dotenv()

from utils import (
    TranscriptionService, 
    DiarizedTranscriptionService,
    StreamingTranscriptionService,
    save_uploaded_file, 
    get_audio_duration, 
    log
)

# Global variables
transcription_service = None
diarized_service = None
streaming_service = None

HF_TOKEN = os.getenv("HF_TOKEN", None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_service, diarized_service, streaming_service
    
    log("🚀 Loading ASR Model...")
    transcription_service = TranscriptionService()
    
    log("🎙️ Loading Streaming Service...")
    streaming_service = StreamingTranscriptionService(asr_service=transcription_service)
    
    if HF_TOKEN:
        log("🔊 Loading Diarization...")
        try:
            diarized_service = DiarizedTranscriptionService(
                hf_token=HF_TOKEN,
                asr_service=transcription_service
            )
        except Exception as e:
            log(f"⚠️ Diarization failed: {e}")
            diarized_service = None
    
    log("✅ Ready!")
    yield


app = FastAPI(title="Bengali Audio Transcription API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Simple transcription"""
    log(f"📨 Received: {audio.filename}")
    
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    audio_path = None
    try:
        audio_path = await save_uploaded_file(audio)
        duration = get_audio_duration(audio_path)
        result = transcription_service.transcribe(audio_path)
        
        return JSONResponse(content={
            "transcription": result["text"],
            "audio_duration": float(duration),
            "processing_time": float(result["processing_time"]),
            "status": "success"
        })
    except Exception as e:
        log(f"❌ Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/transcribe_diarize")
async def transcribe_with_diarization(
    audio: UploadFile = File(...),
    min_segment_duration: float = Query(0.5),
    merge_same_speaker: bool = Query(True),
    gap_threshold: float = Query(0.5)
):
    """Transcription with speaker diarization"""
    if diarized_service is None:
        raise HTTPException(status_code=503, detail="Diarization not available")
    
    log(f"📨 Diarized request: {audio.filename}")
    
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
        
        return JSONResponse(content={
            "segments": result["segments"],
            "full_text": result["full_text"],
            "num_speakers": result["num_speakers"],
            "audio_duration": float(duration),
            "processing_time": float(result["processing_time"]),
            "status": "success"
        })
    except Exception as e:
        log(f"❌ Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": transcription_service is not None,
        "diarization_available": diarized_service is not None
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Simple real-time streaming - Original fast version.
    """
    await websocket.accept()
    log("🔌 WebSocket connected")
    
    audio_buffer = np.array([], dtype=np.float32)
    pre_buffer = np.array([], dtype=np.float32)
    silence_frames = 0
    speech_started = False
    
    VAD_CHUNK_SIZE = 512
    SILENCE_THRESHOLD = 15        # ~0.5s silence
    PRE_BUFFER_CHUNKS = 25        # ~0.6s lookback
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                chunk = np.frombuffer(base64.b64decode(data["audio"]), dtype=np.float32)
                pre_buffer = np.concatenate([pre_buffer, chunk])
                
                # Process 512-sample chunks for VAD
                while len(pre_buffer) >= VAD_CHUNK_SIZE:
                    vad_chunk = pre_buffer[:VAD_CHUNK_SIZE]
                    pre_buffer = pre_buffer[VAD_CHUNK_SIZE:]
                    
                    # VAD check
                    import torch
                    speech_prob = streaming_service.vad_model(
                        torch.from_numpy(vad_chunk), 16000
                    ).item()
                    
                    if speech_prob > 0.5:
                        silence_frames = 0
                        if not speech_started:
                            speech_started = True
                            log("🎤 Speech started")
                        audio_buffer = np.concatenate([audio_buffer, vad_chunk])
                        
                    elif speech_started:
                        audio_buffer = np.concatenate([audio_buffer, vad_chunk])
                        silence_frames += 1
                        
                        if silence_frames >= SILENCE_THRESHOLD:
                            if len(audio_buffer) > 4800:  # > 300ms
                                text = streaming_service.transcribe_buffer(audio_buffer)
                                if text.strip():
                                    log(f"✅ {text}")
                                    await websocket.send_json({"text": text, "is_final": True})
                            
                            audio_buffer = np.array([], dtype=np.float32)
                            silence_frames = 0
                            speech_started = False
                            
            elif data.get("type") == "stop":
                if len(audio_buffer) > 4800:
                    text = streaming_service.transcribe_buffer(audio_buffer)
                    if text.strip():
                        await websocket.send_json({"text": text, "is_final": True})
                break
                
    except WebSocketDisconnect:
        log("🔌 Disconnected")
    except Exception as e:
        log(f"❌ Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8296)