FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# התקנת Tesseract ומערכות העזר
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-heb \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# שימוש בפורט הדינמי של Render
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}