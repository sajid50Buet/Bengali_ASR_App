import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import threading

@dataclass
class TranscriptionJob:
    job_id: str
    audio_path: str
    status: str  # "queued", "processing", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: datetime = None

class ASRAPIManager:
    """
    Manages ASR requests with queuing and concurrency control.
    Handles multiple API clients efficiently.
    """
    
    def __init__(self, transcription_service, max_concurrent: int = 2):
        self.transcription_service = transcription_service
        self.max_concurrent = max_concurrent  # Limit based on GPU memory
        
        # Job management
        self.jobs: Dict[str, TranscriptionJob] = {}
        self.job_queue = asyncio.Queue()
        
        # Thread pool for CPU-bound preprocessing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Semaphore for GPU access
        self.gpu_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Stats
        self.stats = {
            "total_requests": 0,
            "completed": 0,
            "failed": 0,
            "active": 0
        }
        
        self._lock = threading.Lock()
    
    async def submit_job(self, audio_path: str) -> str:
        """Submit a transcription job and return job_id"""
        job_id = str(uuid.uuid4())[:8]
        
        job = TranscriptionJob(
            job_id=job_id,
            audio_path=audio_path,
            status="queued",
            created_at=datetime.now()
        )
        
        self.jobs[job_id] = job
        self.stats["total_requests"] += 1
        
        # Process immediately (with semaphore limiting concurrency)
        asyncio.create_task(self._process_job(job))
        
        return job_id
    
    async def _process_job(self, job: TranscriptionJob):
        """Process a single transcription job"""
        async with self.gpu_semaphore:  # Limits concurrent GPU usage
            try:
                job.status = "processing"
                self.stats["active"] += 1
                
                # Run transcription in thread pool to not block event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    self.transcription_service.transcribe,
                    job.audio_path
                )
                
                job.result = result
                job.status = "completed"
                job.completed_at = datetime.now()
                self.stats["completed"] += 1
                
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                self.stats["failed"] += 1
            finally:
                self.stats["active"] -= 1
    
    async def get_job_status(self, job_id: str) -> Optional[TranscriptionJob]:
        """Get job status by ID"""
        return self.jobs.get(job_id)
    
    async def transcribe_sync(self, audio_path: str, timeout: float = 300) -> Dict[str, Any]:
        """
        Synchronous transcription - waits for result.
        Use this for simple API calls.
        """
        job_id = await self.submit_job(audio_path)
        
        # Poll for completion
        start = datetime.now()
        while True:
            job = self.jobs.get(job_id)
            if job.status == "completed":
                return job.result
            elif job.status == "failed":
                raise Exception(job.error)
            
            if (datetime.now() - start).seconds > timeout:
                raise TimeoutError(f"Job {job_id} timed out")
            
            await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API statistics"""
        return {
            **self.stats,
            "queue_size": len([j for j in self.jobs.values() if j.status == "queued"]),
            "max_concurrent": self.max_concurrent
        }
    
    def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """Remove completed jobs older than max_age"""
        now = datetime.now()
        to_remove = []
        for job_id, job in self.jobs.items():
            if job.completed_at:
                age = (now - job.completed_at).seconds
                if age > max_age_seconds:
                    to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]