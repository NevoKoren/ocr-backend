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
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. הגדלת התמונה פי 2 (קריטי לצילומי מסך)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. המרה לאפור
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. סף אדפטיבי - עוזר להתעלם מרעשי רקע ולהבליט טקסט
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # 4. שימוש ב-PSM 11 שאומר לטסרקט: "חפש טקסט מפוזר, אל תנסה למצוא פסקאות"
        custom_config = r'-l heb+eng --psm 11'
        extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

        # ניקוי שורות ריקות לחלוטין
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]

        return JSONResponse(content={"status": "success", "data": lines})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)