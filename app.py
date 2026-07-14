from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OCR_SPACE_API_KEY = "helloworld"  # החלף במפתח ה-API האישי שלך

# תבניות Regex לזיהוי מספרים ונתונים מובנים
phone_pattern = re.compile(r'(05\d[ \-\.]?\d{7}|0[23489][ \-\.]?\d{7})')
date_pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})')
time_pattern = re.compile(r'(\d{1,2}:\d{2})')
id_pattern = re.compile(r'(\d{9})')

# רשימת מילים שחורות - אם תא מכיל את אחת המילים האלו, הוא בטוח לא עמודת השם!
excluded_keywords = [
    "לוחם", "סדיר", "קבע", "מילואים", "תומך", "לחימה", "חיל", "הים", "אוויר", 
    "יבשה", "סגל", "מיועד", "סטטוס", "דרגה", "תפקיד", "סיווג", "שירות", "מתשחקר", 
    "המתשחקר", "המתתחקר", "מיל", "חובה", "גברים", "נשים", "ימי", "כללי", "מלשב", "מלש\"ב"
]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Bulletproof Column-Filtering OCR Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isTable": "true",  # מחזיר את מבנה הצינורות (|) כדי לשמור על השורות מחוברות
            "scale": "true"
        }
        
        files = [('file', (file.filename, file_bytes, file.content_type))]
        response = requests.post("https://api.ocr.space/parse/image", data=payload, files=files)
        result_json = response.json()
        
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
            # דילוג על שורות המקפים של טבלת ה-Markdown
            if re.match(r'^[\s\|\-]+$', line):
                continue
                
            # פיצול לפי צינורות, תוך שמירה על תאים ריקים כדי למנוע תזוזת אינדקסים!
            cells = [cell.strip() for cell in line.split('|')]
            
            # שלב א': זיהוי עמודת השם לפי שורת הכותרות
            if not is_header_found:
                for idx, cell in enumerate(cells):
                    if any(k in cell for k in ["שם", "מועמד"]):
                        name_index = idx
                        is_header_found = True
                        break
                
                if "טלפון" in line or "תאריך" in line or "אישי" in line:
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
            
            # שלב ב': שליפת השם מהעמודה הספציפית
            raw_name = ""
            if name_index != -1 and name_index < len(cells):
                raw_name = cells[name_index]
            
            # מנגנון הגיבוי והסינון הדו-שלבי:
            # אם התא שנבחר ריק, או שהוא מכיל מספרים, או שהוא קצר מדי - נבצע סריקה חכמה
            clean_check = re.sub(r'[^\u0590-\u05fe]', '', raw_name).strip()
            if not raw_name or len(clean_check) < 2 or any(kw in raw_name for kw in excluded_keywords):
                for cell in cells:
                    cell_clean = re.sub(r'[^\u0590-\u05fe\s]', '', cell).strip()
                    # אם התא מכיל לפחות 2 אותיות בעברית והוא לא מכיל אף מילת תפקיד/יחידה/סטטוס
                    if len(cell_clean) >= 2 and not any(kw in cell_clean for kw in excluded_keywords):
                        raw_name = cell
                        break
            
            # שליפת טלפון, תאריך ושעה מכל השורה
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            id_match = id_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            id_num = id_match.group(1) if id_match else ""
            
            # ניקוי השם הסופי
            clean_name = raw_name
            if phone: clean_name = clean_name.replace(phone, "")
            if date: clean_name = clean_name.replace(date, "")
            if time: clean_name = clean_name.replace(time, "")
            if id_num: clean_name = clean_name.replace(id_num, "")
            
            clean_name = re.sub(r'[^\u0590-\u05fe\s]', '', clean_name).strip()
            clean_name = " ".join(clean_name.split())
            
            debug_lines.append({
                "line_number": line_idx + 1,
                "action": "data_extraction",
                "raw_line": line,
                "cells_detected": cells,
                "selected_name_cell": raw_name,
                "cleaned_name": clean_name,
                "phone_found": phone
            })
            
            # הכנסה רק אם יש שם תקין ונתון מזהה (טלפון או ת"ז) כדי לסנן רעשי ממשק אקסל
            if len(clean_name) >= 2 and (phone or id_num):
                structured_data.append({
                    "name": clean_name,
                    "phone": phone if phone else "-",
                    "date": date if date else "-",
                    "time": time if time else "-"
                })
        
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