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

        # 1. הקטנת התמונה למניעת קריסה (שומר על ביצועים מהירים)
        max_width = 1200
        height, width = img.shape[:2]
        if width > max_width:
            scaling_factor = max_width / float(width)
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # 2. המרה לאפור והגברת קונטרסט (מתיחת היסטוגרמה) - מציל את הטקסט על הרקע הירוק
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # 3. טשטוש גאוסיאני עדין נגד ריצודי מסך המחשב
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. חיתוך סף אדפטיבי - קריטי לטבלאות עם תאים צבעוניים!
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

        # 5. הגדרות Tesseract
        # psm 4 - מניח שהטקסט מסודר כעמודה אחת של טקסט בגדלים משתנים
        custom_config = r'-l heb+eng --psm 4'
        extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

        # ניקוי שורות ריקות או רעשי רקע זעירים
        lines = [line.strip() for line in extracted_text.split('\n') if len(line.strip()) > 2]

        return JSONResponse(content={"status": "success", "data": lines})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)