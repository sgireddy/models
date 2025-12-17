FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (Linux wheels)
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      torch \
      transformers \
      ai-edge-torch \
      sentencepiece \
      safetensors \
      accelerate \
      huggingface-hub \
      numpy

# We will mount the repository at /app when running the container
CMD ["bash", "-lc", "python scripts/convert_medgemma.py"]
