from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests

app = FastAPI()

# אישור קבלת בקשות מכל מקור (CORS) כולל האתר שלך ב-Firebase
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# מפתח ברירת המחדל לבדיקות. מומלץ להחליף במפתח חינמי אישי מהאתר שלהם
OCR_SPACE_API_KEY = "helloworld" 

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "OCR.space Proxy Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        # קריאת קובץ התמונה שנשלח מהדפדפן
        file_bytes = await file.read()
        
        # הגדרת הפרמטרים עבור ה-API של OCR.space
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",       # חובה: שימוש במנוע המתקדם שתומך בעברית ובטבלאות
            "isTable": "true",      # הפעלת מנוע זיהוי המבנה הטבלאי
            "scale": "true"         # שיפור רזולוציה אוטומטי לצילומי מסך/טלפון
        }
        
        # הכנת הקובץ למשלוח בפורמט Multipart
        files = [
            ('file', (file.filename, file_bytes, file.content_type))
        ]
        
        # ביצוע בקשת ה-POST לשרת ה-OCR המרוחק
        response = requests.post("https://api.ocr.space/parse/image", data=payload, files=files)
        result_json = response.json()
        
        # בדיקה אם השרת המרוחק החזיר שגיאת עיבוד
        if result_json.get("IsErroredOnProcessing"):
            error_msg = result_json.get("ErrorMessage", ["שגיאה לא ידועה בשרת ה-OCR"])[0]
            return JSONResponse(content={"status": "error", "message": error_msg}, status_code=400)
        
        # חילוץ תוצאות הטקסט
        parsed_results = result_json.get("ParsedResults", [])
        if not parsed_results:
            return JSONResponse(content={"status": "success", "data": []})
            
        extracted_text = parsed_results[0].get("ParsedText", "")
        
        # פירוק לשורות וניקוי רווחים מיותרים
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        return JSONResponse(content={"status": "success", "data": lines})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)