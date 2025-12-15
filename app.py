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
from api_manager import ASRAPIManager

# Global variables
transcription_service = None
diarized_service = None
streaming_service = None
api_manager = None

# Set your HuggingFace token here (or use environment variable)
HF_TOKEN = os.getenv("HF_TOKEN", None)  # or "hf_xxxxx"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_service, diarized_service, streaming_service, api_manager
    
    log("🚀 Loading ASR Model...")
    transcription_service = TranscriptionService()
    
    log("📡 API Manager...")
    api_manager = ASRAPIManager(transcription_service=transcription_service, max_concurrent=2)
    
    log("🎙️ Streaming Service...")
    streaming_service = StreamingTranscriptionService(asr_service=transcription_service)
    
    if HF_TOKEN:
        log("🔊 Diarization...")
        try:
            diarized_service = DiarizedTranscriptionService(
                asr_service=transcription_service,
                hf_token=HF_TOKEN
            )
        except Exception as e:
            log(f"⚠️ Diarization failed: {e}")
            diarized_service = None
    
    log("✅ Ready!")
    yield

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
    PRE_BUFFER_CHUNKS = 20        # ~0.6s lookback
    
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

# ============== PRODUCTION API ENDPOINTS ==============

@app.post("/api/v1/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    """
    Production API endpoint for transcription.
    Handles queuing and concurrency automatically.
    """
    log(f"📨 API Request: {audio.filename}")
    
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    audio_path = None
    try:
        audio_path = await save_uploaded_file(audio)
        result = await api_manager.transcribe_sync(audio_path)
        
        return JSONResponse(content={
            "transcription": result["text"],
            "processing_time": result["processing_time"],
            "status": "success"
        })
    
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass


@app.post("/api/v1/transcribe/async")
async def api_transcribe_async(audio: UploadFile = File(...)):
    """
    Async API - returns job_id immediately.
    Poll /api/v1/job/{job_id} for result.
    """
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    audio_path = await save_uploaded_file(audio)
    job_id = await api_manager.submit_job(audio_path)
    
    return JSONResponse(content={
        "job_id": job_id,
        "status": "queued",
        "poll_url": f"/api/v1/job/{job_id}"
    })


@app.get("/api/v1/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of async transcription job"""
    job = await api_manager.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at.isoformat()
    }
    
    if job.status == "completed":
        response["result"] = {
            "transcription": job.result["text"],
            "processing_time": job.result["processing_time"]
        }
        response["completed_at"] = job.completed_at.isoformat()
    elif job.status == "failed":
        response["error"] = job.error
    
    return JSONResponse(content=response)


@app.get("/api/v1/stats")
async def get_api_stats():
    """Get API usage statistics"""
    return JSONResponse(content=api_manager.get_stats())


@app.get("/api/v1/health")
async def api_health():
    """Detailed health check for load balancers"""
    stats = api_manager.get_stats()
    
    # Healthy if not overloaded
    is_healthy = stats["active"] < stats["max_concurrent"]
    
    return JSONResponse(
        content={
            "status": "healthy" if is_healthy else "overloaded",
            "model_loaded": transcription_service is not None,
            "active_requests": stats["active"],
            "max_concurrent": stats["max_concurrent"],
            "total_processed": stats["completed"]
        },
        status_code=200 if is_healthy else 503
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8069)