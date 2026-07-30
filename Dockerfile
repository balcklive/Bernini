# =============================================================================
# Dockerfile — Bernini Video Diffusion Framework
# =============================================================================
# Build:
#   docker build -t bernini:latest .
#
# Run (Gradio Web UI):
#   docker run --gpus all -p 7860:7860 bernini:latest
#
# Run (single-GPU inference):
#   docker run --gpus all bernini:latest \
#     python infer_single_gpu.py --config ByteDance/Bernini-Diffusers
#
# Run (multi-GPU with Ulysses sequence parallel):
#   docker run --gpus all --shm-size=32g bernini:latest \
#     torchrun --nproc-per-node 8 infer_multi_gpu.py --ulysses 8 \
#       --config ByteDance/Bernini-Diffusers
#
# Run (REST API server on port 8000):
#   docker run --gpus all -p 8000:8000 bernini:latest \
#     python api_server.py --config ByteDance/Bernini-Diffusers
# =============================================================================

# ── Base: CUDA 12.6 (devel for flash-attn compilation) ──────────────────────
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

LABEL org.opencontainers.image.title="Bernini"
LABEL org.opencontainers.image.description="Latent Semantic Planning for Video Diffusion — ByteDance"
LABEL org.opencontainers.image.source="https://github.com/balcklive/Bernini"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Suppress interactive prompts during package installation (e.g. tzdata).
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

SHELL ["/bin/bash", "-c"]

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11 via deadsnakes PPA
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    # Build tools
    git \
    ninja-build \
    # Media / general
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip + uv
RUN python3 -m ensurepip --upgrade && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir uv

# ── Python dependencies ─────────────────────────────────────────────────────
WORKDIR /app

# 1. PyTorch + CUDA 12.6 stack — copied first for Docker layer caching
RUN pip3 install --no-cache-dir \
    torch==2.7.1+cu126 \
    torchvision==0.22.1+cu126 \
    --extra-index-url https://download.pytorch.org/whl/cu126

# 2. Core ML dependencies
RUN pip3 install --no-cache-dir \
    diffusers==0.35.2 \
    accelerate==0.34.2 \
    transformers==4.57.3 \
    safetensors \
    einops \
    numpy \
    Pillow \
    tqdm \
    ftfy \
    datasets==2.21.0 \
    PyYAML \
    librosa \
    av

# 3. Video / image I/O
RUN pip3 install --no-cache-dir \
    decord \
    imageio \
    imageio-ffmpeg

# 4. FlashAttention-2 (compiled from source against local CUDA)
#    Set MAX_JOBS to limit parallel nvcc workers (each uses ~3-5 GB RAM).
#    The pre-built wheel on PyPI is compatible with CUDA 12.6 + torch 2.7.1.
RUN MAX_JOBS=$(nproc) pip3 install --no-cache-dir flash-attn==2.8.3

# 5. Open-VeOmni (required, installed with --no-deps to avoid torch override)
RUN pip3 install --no-cache-dir --no-deps \
    git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.11

# 6. Optional: prompt engineering (OpenAI-compatible endpoint) + Gradio demo + API server
RUN pip3 install --no-cache-dir \
    "openai>=1.0" \
    "gradio==6.15.0" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.34" \
    "python-multipart>=0.0.18" \
    "requests>=2.32"

# ── Copy source code ────────────────────────────────────────────────────────
COPY . .

# Install the project package itself (editable not needed in container)
RUN pip3 install --no-cache-dir -e . --no-deps

# ── Runtime configuration ───────────────────────────────────────────────────
EXPOSE 7860

# Default: launch the Gradio web UI.
# Override CMD to run other scripts (e.g. infer_single_gpu.py).
CMD ["python", "gradio_demo.py", "--config", "ByteDance/Bernini-Diffusers"]
