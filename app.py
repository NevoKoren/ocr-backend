from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import re

app = FastAPI()

# הגדרת CORS לאישור קבלת בקשות מכל מקור (כולל האתר שלך ב-Firebase)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# מפתח ה-API של OCR.space (מומלץ להחליף במפתח החינמי האישי שלך לקבלת מכסה יציבה)
OCR_SPACE_API_KEY = "helloworld"  

# תבניות Regex מדויקות לזיהוי מספרים ונתונים קבועים
phone_pattern = re.compile(r'\b(05\d[- ]?\d{7}|0[23489][- ]?\d{7})\b')
date_pattern = re.compile(r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b')
time_pattern = re.compile(r'\b(\d{1,2}:\d{2})\b')

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Pipe-Splitting OCR Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        # פרמטרים אופטימליים עבור OCR.space Engine 3
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isTable": "true",  # מאפשר למנוע להחזיר את מבנה הצינורות (|) של הטבלה
            "scale": "true"
        }
        
        files = [('file', (file.filename, file_bytes, file.content_type))]
        
        # שליחת הבקשה לשרת ה-OCR
        response = requests.post("https://api.ocr.space/parse/image", data=payload, files=files)
        result_json = response.json()
        
        # בדיקת שגיאות מצד שרת ה-OCR המרוחק
        if result_json.get("IsErroredOnProcessing"):
            error_msg = result_json.get("ErrorMessage", ["שגיאה בעיבוד ה-OCR"])[0]
            return JSONResponse(content={"status": "error", "message": error_msg}, status_code=400)
        
        parsed_results = result_json.get("ParsedResults", [])
        if not parsed_results:
            return JSONResponse(content={"status": "success", "data": [], "debug": {"raw_ocr_output": "", "detected_name_index": -1, "lines_processing": []}})
            
        extracted_text = parsed_results[0].get("ParsedText", "")
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        structured_data = []
        debug_lines = []
        name_index = -1
        is_header_found = False
        
        for line_idx, line in enumerate(lines):
            # דילוג על שורות המקפים והפרדות של טבלת Markdown (למשל |---|---|)
            if re.match(r'^[\s\|\-]+$', line):
                continue
                
            # פיצול השורה לעמודות על פי התו '|' או טאבים
            cells = [cell.strip() for cell in re.split(r'\||\t', line)]
            
            # שלב א': איתור שורת הכותרות (מתבצע פעם אחת בלבד)
            if not is_header_found:
                for idx, cell in enumerate(cells):
                    if "שם" in cell or "מועמד" in cell:
                        name_index = idx
                        is_header_found = True
                        break
                
                # בדיקת כותרות נוספות לזיהוי שורת הכותרת גם אם עמודת השם לא זוהתה ישירות
                if "טלפון" in line or "תאריך" in line or "אישי" in line or "שירות" in line or "ראיון" in line:
                    is_header_found = True
                
                debug_lines.append({
                    "line_number": line_idx + 1,
                    "action": "header_check",
                    "raw_line": line,
                    "cells_detected": cells,
                    "name_index_found": name_index
                })
                
                if is_header_found:
                    continue
            
            # שלב ב': חילוץ נתונים משורות המידע
            # משיכת השם ישירות מתוך התא המיועד לו לפי האינדקס שנמצא
            raw_name = cells[name_index] if (name_index != -1 and name_index < len(cells)) else ""
            
            # זיהוי שאר הפרמטרים מכלל השורה באמצעות תבניות Regex
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            
            # ניקוי השם מאותיות באנגלית, מספרים ותווים מיוחדים
            clean_name = re.sub(r'[^\u0590-\u05fe\s]', '', raw_name).strip()
            clean_name = " ".join(clean_name.split())
            
            # תיעוד נתוני הדיבאג עבור השורה הנוכחית
            debug_lines.append({
                "line_number": line_idx + 1,
                "action": "data_extraction",
                "raw_line": line,
                "cells_detected": cells,
                "selected_name_cell": raw_name,
                "cleaned_name": clean_name,
                "phone_matched": phone,
                "date_matched": date,
                "time_matched": time
            })
            
            # הוספת הרשומה לרשימה הסופית אם נמצא שם תקין או מספר טלפון
            if len(clean_name) >= 2 or phone:
                structured_data.append({
                    "name": clean_name if clean_name else "-",
                    "phone": phone if phone else "-",
                    "date": date if date else "-",
                    "time": time if time else "-"
                })
        
        # החזרת המידע המובנה יחד עם אובייקט הדיבאג המלא
        return JSONResponse(content={
            "status": "success",
            "data": structured_data,
            "debug": {
                "raw_ocr_output": extracted_text,
                "detected_name_index": name_index,
                "lines_processing": debug_lines
            }
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)