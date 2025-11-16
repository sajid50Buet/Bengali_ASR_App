# Bengali Audio Transcription API

A simple FastAPI application for transcribing Bengali audio using AI. Features both audio upload and real-time recording capabilities.

## Features

- 🎤 **Record Audio**: Record Bengali audio directly from your browser
- 📁 **Upload Audio**: Support for WAV, MP3, M4A, FLAC, OGG formats
- ⚡ **Fast Processing**: Model loaded at startup for quick transcriptions
- 📊 **Analytics**: Shows audio duration and processing time
- 🎨 **Modern UI**: Clean, responsive interface with drag-and-drop support

## Project Structure

```
.
├── app.py              # FastAPI endpoints only
├── utils.py            # Transcription logic and helper functions
├── frontend.html       # Frontend UI with upload/recording
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Your NeMo Model

Put your `.nemo` model file in the project directory. The app will automatically detect and load it.

```
.
├── app.py
├── utils.py
├── frontend.html
├── your-model.nemo    ← Your Bengali ASR model here
└── requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the Application

Open your browser and go to:
```
http://localhost:8000
```

## API Endpoints

### `GET /`
Serves the frontend HTML interface

### `POST /transcribe`
Transcribe an audio file

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file

**Response:**
```json
{
    "transcription": "আপনার বাংলা টেক্সট এখানে",
    "audio_duration": 12.5,
    "processing_time": 3.2,
    "status": "success"
}
```

### `GET /health`
Check service health and model status

## Usage

### Upload Method
1. Click the "Upload Audio" tab
2. Click the upload area or drag & drop an audio file
3. Wait for transcription to complete
4. View the results with audio duration and processing time

### Recording Method
1. Click the "Record Audio" tab
2. Click the microphone button to start recording
3. Click the stop button when finished
4. The audio will automatically be transcribed
5. View the results with statistics

## Model Information

The app uses **NVIDIA NeMo** ASR models (`.nemo` format). Place your trained Bengali ASR model in the project directory, and it will be automatically loaded on startup.

You can also specify a custom model path by modifying the `TranscriptionService()` initialization in `app.py`:

```python
transcription_service = TranscriptionService(model_path="path/to/your/model.nemo")
```

## Notes

- The model is loaded during application startup (may take a few moments)
- Temporary audio files are stored in `temp_audio/` directory
- Files are automatically cleaned up after processing
- The app runs on CPU by default (modify `utils.py` to use GPU)

## Requirements

- Python 3.8+
- FastAPI
- NVIDIA NeMo Toolkit
- Librosa
- PyTorch

## License

MIT License