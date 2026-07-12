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

        # 1. הקטנת התמונה אם היא גדולה מדי (מניעת קריסת שרת מוחלטת)
        max_width = 1200
        height, width = img.shape[:2]
        if width > max_width:
            scaling_factor = max_width / float(width)
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # 2. המרה לשחור לבן
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. טשטוש גאוסיאני (Gaussian Blur) - קריטי! מעלים את הפיקסלים של המסך המצולם
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. חיתוך סף קלאסי (Otsu's) - מנקה את הרקע הלבן והצבעוני ומשאיר טקסט שחור בלבד
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. קריאת הטקסט מחדש כבלוקים מסודרים
        custom_config = r'-l heb+eng --psm 6'
        extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

        # ניקוי שורות ריקות או רעשי רקע קטנים
        lines = [line.strip() for line in extracted_text.split('\n') if len(line.strip()) > 2]

        return JSONResponse(content={"status": "success", "data": lines})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)