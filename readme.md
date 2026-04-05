# Bengali ASR (Automatic Speech Recognition) App

A production-ready FastAPI application for transcribing Bengali audio using NVIDIA NeMo. Supports file upload, real-time streaming via WebSocket, and speaker diarization with timestamps.

## Features

- **Audio Upload** - Transcribe WAV, MP3, M4A, FLAC, OGG, and WebM files
- **Real-time Streaming** - Live transcription over WebSocket with Voice Activity Detection (Silero VAD)
- **Speaker Diarization** - Identify speakers with timestamps using pyannote.audio
- **Long Audio Support** - Automatic chunking with 2s overlap for seamless transcription of long recordings
- **GPU Accelerated** - Optimized for NVIDIA GPUs with CUDA 12.4 and batch processing
- **Modern Web UI** - Responsive interface with drag-and-drop, recording, and streaming tabs
- **Docker Ready** - Full containerization with NVIDIA GPU support and SSL/HTTPS

## Project Structure

```
.
├── app.py                  # FastAPI application, routes, and WebSocket handler
├── utils.py                # Core transcription, diarization, and streaming services
├── api_manager.py          # Async job queue and GPU concurrency manager
├── frontend.html           # Web UI (upload, record, stream tabs)
├── requirements.txt        # Python dependencies
├── frozen_requirements.txt # Pinned dependency versions (reproducible builds)
├── Dockerfile              # Multi-stage Docker build (CUDA 12.4 + Python 3.11)
├── docker-compose.yml      # Docker Compose with GPU and SSL support
├── .env                    # Environment variables (HF_TOKEN)
├── cert.pem / key.pem      # SSL certificates for HTTPS
├── models/
│   └── *.nemo              # NeMo ASR model file
├── tokenizer/
│   ├── vocab.txt           # Tokenizer vocabulary
│   └── tokenizer.model     # SentencePiece tokenizer
├── temp_audio/             # Temporary files for upload processing (auto-cleaned)
└── temp_streaming/         # Temporary files for streaming (auto-cleaned)
```

## Prerequisites

- **Python** 3.8+ (3.11 recommended)
- **NVIDIA GPU** with CUDA support (CPU fallback available but slow)
- **ffmpeg** - Required for audio format conversion
- **libsndfile** - Required for audio I/O
- **HuggingFace Token** - Required only for speaker diarization (pyannote models)

## Setup

### Local Installation

1. **Install system dependencies:**

   ```bash
   sudo apt-get update
   sudo apt-get install -y ffmpeg libsndfile1
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   For reproducible builds with exact versions:

   ```bash
   pip install -r frozen_requirements.txt
   ```

4. **Place your NeMo model:**

   Put your `.nemo` model file in the `models/` directory. The app automatically detects and loads any `.nemo` file found in the project.

5. **Set up environment variables (optional, for diarization):**

   ```bash
   # Create a .env file
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

   Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). You need access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

### Running Locally

```bash
# Start the server (default port 8296)
python app.py

# Or with uvicorn (production port)
uvicorn app:app --host 0.0.0.0 --port 8569

# With auto-reload for development
uvicorn app:app --reload --host 0.0.0.0 --port 8569

# With SSL/HTTPS
uvicorn app:app --host 0.0.0.0 --port 8569 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

Open your browser at `http://localhost:8569` (or `https://` if using SSL).

## Docker Deployment

### Build and Run with Docker Compose (Recommended)

1. **Set your HuggingFace token** in the `.env` file:

   ```
   HF_TOKEN=your_huggingface_token_here
   ```

2. **Build and start:**

   ```bash
   docker-compose up -d --build
   ```

3. **View logs:**

   ```bash
   docker-compose logs -f web
   ```

4. **Stop:**

   ```bash
   docker-compose down
   ```

The app will be available at `https://localhost:8569` (SSL enabled by default in Docker).

### Build and Run with Docker CLI

```bash
# Build the image
docker build -t bengali_asr_app:latest .

# Run with GPU support
docker run -d \
  --gpus all \
  -e HF_TOKEN="your_huggingface_token" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -p 8569:8569 \
  -v $(pwd):/app \
  --memory="16g" \
  --name bengali_asr_container \
  bengali_asr_app:latest
```

### Docker Configuration Details

| Setting | Value |
|---|---|
| Base Image | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` |
| Python | 3.11 |
| Port | 8569 |
| GPU | 1 NVIDIA GPU (all capabilities) |
| Memory Limit | 16 GB |
| Restart Policy | `unless-stopped` |
| Health Check | `GET /health` every 30s, 3 retries |

### Generating SSL Certificates

For self-signed certificates (development/internal use):

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/CN=localhost"
```

## API Endpoints

### `GET /` - Web Interface

Serves the frontend HTML application with tabs for upload, recording, and real-time streaming.

---

### `POST /transcribe` - Transcribe Audio

Transcribes an audio file without speaker identification.

**Request:**

- Content-Type: `multipart/form-data`
- Body: `audio` (file) - Supported formats: WAV, MP3, M4A, FLAC, OGG, WebM

```bash
curl -X POST -F "audio=@recording.wav" http://localhost:8569/transcribe
```

**Response:**

```json
{
    "transcription": "আপনার বাংলা টেক্সট এখানে",
    "audio_duration": 12.5,
    "processing_time": 3.2,
    "status": "success"
}
```

**Audio format notes:** Input audio can be any sample rate and any channel layout (mono, stereo, multi-channel). The app automatically resamples to 16kHz mono internally via librosa and ffmpeg before passing to the model.

**Long audio handling:** Files over 30 seconds are automatically split into 30s chunks with 2s overlap, batch-processed, and seamlessly merged using longest common subsequence matching.

---

### `POST /transcribe_diarize` - Transcribe with Speaker Diarization

Transcribes audio and identifies individual speakers with timestamps. Requires `HF_TOKEN` to be set.

**Request:**

- Content-Type: `multipart/form-data`
- Parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | file | required | Audio file to transcribe |
| `min_segment_duration` | float | 0.5 | Minimum segment length in seconds |
| `merge_same_speaker` | bool | true | Merge consecutive segments from the same speaker |
| `gap_threshold` | float | 0.5 | Max gap (seconds) between segments to merge |

```bash
curl -X POST \
  -F "audio=@meeting.wav" \
  -F "min_segment_duration=0.5" \
  -F "merge_same_speaker=true" \
  -F "gap_threshold=0.5" \
  http://localhost:8569/transcribe_diarize
```

**Response:**

```json
{
    "segments": [
        {
            "speaker": "SPEAKER_0",
            "start": 0.0,
            "end": 5.2,
            "text": "প্রথম বক্তার কথা"
        },
        {
            "speaker": "SPEAKER_1",
            "start": 5.3,
            "end": 10.1,
            "text": "দ্বিতীয় বক্তার কথা"
        }
    ],
    "full_text": "[0.0s - 5.2s] SPEAKER_0: প্রথম বক্তার কথা\n[5.3s - 10.1s] SPEAKER_1: দ্বিতীয় বক্তার কথা",
    "num_speakers": 2,
    "audio_duration": 10.1,
    "processing_time": 4.8,
    "status": "success"
}
```

---

### `WebSocket /ws/stream` - Real-time Streaming Transcription

Live transcription over WebSocket with Voice Activity Detection.

**Connection:**

```
ws://localhost:8569/ws/stream
wss://localhost:8569/ws/stream  (if SSL enabled)
```

**Client sends (audio data):**

```json
{
    "type": "audio",
    "audio": "<base64-encoded float32 PCM data at 16kHz>"
}
```

**Client sends (stop signal):**

```json
{
    "type": "stop"
}
```

**Server responds:**

```json
{
    "text": "স্বীকৃত বাংলা পাঠ্য",
    "is_final": true
}
```

**How it works:**
1. Client streams raw audio as base64-encoded float32 PCM at 16kHz
2. Server runs Silero VAD on 512-sample chunks (~32ms each)
3. Speech is buffered with a 0.6s pre-buffer lookback
4. After 0.5s of silence, the accumulated speech is transcribed and sent back
5. Send `{"type": "stop"}` to end the session

---

### `GET /health` - Health Check

Returns the status of all loaded models.

```bash
curl http://localhost:8569/health
```

**Response:**

```json
{
    "status": "healthy",
    "model_loaded": true,
    "diarization_available": true
}
```

## Web UI Usage

### Upload Tab
1. Click the upload area or drag-and-drop an audio file
2. Preview the audio with the built-in player
3. Toggle **Speaker Diarization** if you need speaker identification
4. Click **Transcribe** and view results

### Record Tab
1. Click the microphone button to start recording from your browser
2. Click stop when finished
3. The recording appears with a playback preview
4. Click **Transcribe** to process

### Streaming Tab
1. Click **Start Listening** to open a WebSocket connection and begin microphone capture
2. Speak naturally - transcriptions appear in real time as you pause
3. Click **Stop** to end the session

## Architecture

### Model Stack

| Component | Model | Purpose |
|---|---|---|
| ASR | NeMo TDT (`.nemo`, WER 0.25) | Bengali speech-to-text |
| VAD | Silero VAD (PyTorch Hub) | Voice activity detection for streaming |
| Diarization | pyannote/speaker-diarization-3.1 | Speaker identification and segmentation |

### Audio Processing Pipeline

1. **Input** - Accept audio in any supported format
2. **Convert** - ffmpeg normalizes to 16kHz mono WAV
3. **Chunk** (if >30s) - Split into 30s chunks with 2s overlap
4. **Transcribe** - Batch process through NeMo ASR (batch size: 4)
5. **Merge** - Recombine chunks using longest common subsequence matching
6. **Cleanup** - Delete temp files, clear CUDA cache, run garbage collection

### Memory Management

The app is designed to handle long audio files without running out of GPU memory:

- Audio chunks are saved to disk before transcription (not held in VRAM)
- Batch size of 4 limits concurrent GPU memory usage
- Temporary files are cleaned up in `finally` blocks
- CUDA cache is explicitly cleared after each batch
- Garbage collection runs after processing

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | For diarization | HuggingFace access token for pyannote models |
| `NVIDIA_VISIBLE_DEVICES` | No | GPU selection (default: `all`) |

### Hardcoded Parameters (in `utils.py`)

| Parameter | Value | Description |
|---|---|---|
| Chunk duration | 30s | Max audio length before chunking |
| Chunk overlap | 2s | Overlap between adjacent chunks |
| Batch size | 4 | Chunks processed per GPU batch |
| Min chunk length | 0.5s | Minimum chunk to process |
| VAD chunk size | 512 samples | ~32ms per VAD analysis frame |
| Silence threshold | 15 frames | ~0.5s of silence to trigger transcription |
| Pre-buffer | 25 chunks | ~0.6s lookback before detected speech |
| Speech probability | 0.5 | VAD threshold for speech detection |

## Troubleshooting

### Model not loading
- Ensure a `.nemo` file exists in the project root or `models/` directory
- Check that you have enough GPU memory (the model requires ~2-4 GB VRAM)

### Diarization not available
- Verify `HF_TOKEN` is set in your `.env` file or environment
- Ensure you have accepted the pyannote model terms on HuggingFace
- Check logs for authentication errors on startup

### Out of memory errors
- The app requires up to 16 GB RAM for large files with diarization
- Reduce input audio length or ensure adequate GPU memory
- Monitor with `nvidia-smi` during processing

### Audio format issues
- Ensure `ffmpeg` is installed and accessible in PATH
- All formats are converted to 16kHz mono WAV internally
- WebM files are detected by magic bytes and handled automatically

### Docker GPU not detected
- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Verify with: `docker run --gpus all nvidia/smi`
- Check that `nvidia-docker2` runtime is configured

## License

MIT License
