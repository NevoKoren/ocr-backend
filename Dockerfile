FROM python:3.11-slim

# tesseract-ocr-heb is the Hebrew traineddata — without it lang="heb" fails.
# libgl1 / libglib2.0-0 are what OpenCV links against at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-heb \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Render supplies $PORT at runtime, so bind to it rather than a fixed port.
# One worker, two threads: OCR is CPU-bound and the free instance has 0.5 CPU.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]