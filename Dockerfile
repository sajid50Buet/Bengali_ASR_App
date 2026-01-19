# 1. Base Image: Official NVIDIA CUDA (No Python installed)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Install Python 3.10 and Build Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Link python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# 4. Create and Activate Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 5. Upgrade pip/setuptools (Critical for Cython compatibility)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 6. Install Build Dependencies Manually
# We install a modern Cython here to fix the "METH_METHOD" crash
RUN pip install --no-cache-dir "Cython>=3.0.0" numpy

# 7. Install ctc-segmentation with --no-build-isolation
# This tells pip: "Don't download your own Cython. Use the one I just installed."
RUN CFLAGS="-O0" pip install --no-cache-dir ctc-segmentation --no-build-isolation

# 8. Install PyTorch Trinity (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.4.1+cu124 \
    torchaudio==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 9. Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# 10. Copy Code
COPY . .

EXPOSE 8569

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8569"]