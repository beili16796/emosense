FROM python:3.11-slim

WORKDIR /app

# System deps for MNE/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy emokit first and install
COPY emokit_stub/ /app/emokit_stub/
RUN pip install --no-cache-dir -e /app/emokit_stub/

# Copy emosense and install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY emosense/ /app/emosense/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/

# Generate demo checkpoints (random weights; replace with real when available)
RUN python scripts/generate_demo_checkpoints.py

EXPOSE 7860 8000

CMD ["python", "-m", "emosense.app"]
