# שימוש בתמונת לינוקס קלה עם פייתון
FROM python:3.9-slim

# הגדרת משתני סביבה כדי למנוע בקשות אינטראקטיביות בהתקנה
ENV DEBIAN_FRONTEND=noninteractive

# התקנת Tesseract, חבילות שפה (עברית ואנגלית) וספריות תמונה
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-heb \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# הגדרת תיקיית העבודה
WORKDIR /app

# העתקת קובץ הדרישות והתקנת ספריות הפייתון
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת שאר קוד השרת
COPY . .

# פקודת ההרצה של השרת
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]