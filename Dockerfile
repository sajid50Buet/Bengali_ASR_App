# 1. Base Image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000

# 2. Install System Dependencies & Python 3.11 PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Link python3 to python3.11
# We force the system to see python3.11 as the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# 4. Create Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 5. Upgrade pip (Critical for 3.11)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 6. Install Frozen Requirements
COPY frozen_requirements.txt .
RUN pip install --no-cache-dir -r frozen_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# 7. Copy Code & Setup Dirs
COPY . .
RUN mkdir -p temp_audio temp_streaming

EXPOSE 8569

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8569"]