from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pytesseract
import cv2
import numpy as np

app = FastAPI()

# אישור קבלת בקשות מהאתר שלך (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # בהמשך תוכל לשנות לכתובת ה-Firebase המדויקת שלך
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        # קריאת קובץ התמונה שנשלח מהדפדפן
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # עיבוד תמונה בסיסי (OpenCV) לשיפור איכות הקריאה
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # הפיכה לשחור לבן חד
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # הפעלת Tesseract עם תמיכה בעברית ואנגלית
        # psm 6 אומר למנוע להתייחס לטקסט כבלוק אחיד של נתונים
        custom_config = r'-l heb+eng --psm 6'
        extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

        # חלוקת הטקסט לשורות לטובת הצגה קלה באתר
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]

        return JSONResponse(content={"status": "success", "data": lines})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)