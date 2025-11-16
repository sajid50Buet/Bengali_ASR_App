from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from utils import TranscriptionService, save_uploaded_file, get_audio_duration, log

# Global variable for transcription service
transcription_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global transcription_service
    log("🚀 Initializing Bengali ASR Model...")
    transcription_service = TranscriptionService()
    log("✅ Application startup complete!")
    yield
    # Shutdown: cleanup if needed
    log("👋 Shutting down...")

app = FastAPI(title="Bengali Audio Transcription API", lifespan=lifespan)

app = FastAPI(title="Bengali Audio Transcription API", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend HTML page"""
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe uploaded audio file
    Returns: transcription text, audio duration, and processing time
    """
    log(f"📨 Received transcription request: {audio.filename}")
    
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Supported: .wav, .mp3, .m4a, .flac, .ogg"
        )
    
    audio_path = None
    try:
        # Save uploaded file (will be converted to 16kHz WAV)
        audio_path = await save_uploaded_file(audio)
        
        # Get audio duration
        duration = get_audio_duration(audio_path)
        
        # Transcribe
        result = transcription_service.transcribe(audio_path)
        
        # Ensure all values are JSON-serializable (convert numpy types to Python types)
        response_data = {
            "transcription": str(result["text"]),
            "audio_duration": float(duration),
            "processing_time": float(result["processing_time"]),
            "status": "success"
        }
        
        log(f"✅ Request completed successfully!")
        
        # Clean up immediately before returning
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                log(f"🗑️ Cleaned up temp file")
            except Exception as cleanup_error:
                log(f"⚠️ Could not delete temp file: {cleanup_error}")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        log(f"❌ Request failed: {str(e)}")
        # Clean up on error
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the service is running and model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": transcription_service is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)