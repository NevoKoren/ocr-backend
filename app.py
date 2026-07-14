from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import re
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OCR_SPACE_API_KEY = "helloworld"  # החלף במפתח ה-API האישי שלך

phone_pattern = re.compile(r'(05\d[ \-\.]?\d{7}|0[23489][ \-\.]?\d{7})')
date_pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})')
time_pattern = re.compile(r'(\d{1,2}:\d{2})')
id_pattern = re.compile(r'(\d{9})')

# רשימת הגיבוי: מסננת תפקידים אם ה-OCR מפספס עמודה בגלל לכלוך
excluded_keywords = [
    "לוחם", "סדיר", "קבע", "מילואים", "תומך", "לחימה", "חיל", "הים", "אוויר", 
    "יבשה", "סגל", "מיועד", "סטטוס", "דרגה", "תפקיד", "סיווג", "שירות", "מתשחקר", 
    "המתשחקר", "המתתחקר", "מיל", "חובה", "גברים", "נשים", "ימי", "כללי", "מלשב", "מלש\"ב"
]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Vision-Enhanced Column OCR Server"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        # 1. קריאת התמונה והמרה למערך של OpenCV
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 2. פילטר נטרול צבעים (מעלים פסים ירוקים, סגולים והשתקפויות מסך)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_img = clahe.apply(gray)
        
        # קידוד מחדש לזיכרון כדי לשלוח ל-API
        _, img_encoded = cv2.imencode('.jpg', enhanced_img)
        processed_bytes = img_encoded.tobytes()
        
        # 3. שליחה לפענוח במצב טבלה
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isTable": "true",
            "scale": "true"
        }
        
        files = [('file', (file.filename, processed_bytes, 'image/jpeg'))]
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
            if re.match(r'^[\s\|\-]+$', line):
                continue
                
            cells = [cell.strip() for cell in line.split('|')]
            
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
            
            raw_name = ""
            if name_index != -1 and name_index < len(cells):
                raw_name = cells[name_index]
            
            # סריקת גיבוי חכמה (למקרה שהמנוע פספס עמודה למרות ניקוי התמונה)
            clean_check = re.sub(r'[^\u0590-\u05fe]', '', raw_name).strip()
            if not raw_name or len(clean_check) < 2 or any(kw in raw_name for kw in excluded_keywords):
                for cell in cells:
                    cell_clean = re.sub(r'[^\u0590-\u05fe\s]', '', cell).strip()
                    if len(cell_clean) >= 2 and not any(kw in cell_clean for kw in excluded_keywords):
                        raw_name = cell
                        break
            
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            id_match = id_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            id_num = id_match.group(1) if id_match else ""
            
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