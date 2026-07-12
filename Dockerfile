# שימוש בגרסת פייתון קלה
FROM python:3.10-slim

# התקנת Tesseract והחבילה לעברית
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-heb \
    tesseract-ocr-eng \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# הגדרת תיקיית העבודה
WORKDIR /app

# העתקת קובץ הדרישות והתקנת ספריות פייתון
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת שאר קוד השרת
COPY . .

# הפעלת השרת
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]