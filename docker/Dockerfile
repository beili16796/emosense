FROM python:3.11-slim

WORKDIR /app

# System deps for MNE/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy emokit first and install
COPY emokit/ /app/emokit/
RUN pip install --no-cache-dir /app/emokit/

# Copy emosense and install deps
COPY emosense/ /app/emosense/
RUN pip install --no-cache-dir -r /app/emosense/requirements.txt

WORKDIR /app/emosense

# Generate demo checkpoints
RUN python scripts/generate_demo_checkpoints.py

EXPOSE 7860 8000

CMD ["python", "-m", "emosense.app"]
