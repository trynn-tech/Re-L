FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIPENV_VENV_IN_PROJECT=1

# 1. Install Python 3.13
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    build-essential \
    cmake \
    git \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-full \
    python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set 3.13 as the primary Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# 3. Install PIP correctly for Python 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# 4. Install Pipenv
RUN python3.13 -m pip install pipenv

RUN python3.13 -m pip install watchdog

# 2. We point pip to the pre-compiled repository for CUDA 12.2
RUN python3 -m pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

WORKDIR /app

# Install Python dependencies
# We install llama-cpp-python FIRST with CUDA flags to ensure it compiles for the 4060
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python

# Install the rest of the stack
RUN pip install --no-cache-dir \
    python-dotenv \
    watchdog \
    redis \
    faiss-cpu \
    sentence-transformers \
    langchain \
    langchain-community \
    langchain-core

# Command to keep the container alive (NixOS/Systemd will manage the actual start)
CMD ["tail", "-f", "/dev/null"]
