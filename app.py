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

OCR_SPACE_API_KEY = "helloworld"  # זכור להחליף במפתח החינמי שלך

# תבניות Regex
phone_pattern = re.compile(r'\b(05\d[- ]?\d{7}|0[23489][- ]?\d{7})\b')
date_pattern = re.compile(r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b')
time_pattern = re.compile(r'\b(\d{1,2}:\d{2})\b')
id_pattern = re.compile(r'\b\d{9}\b')

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Index-Locked OCR Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isTable": "true",
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
            if re.match(r'^[\s\|\-]+$', line):
                continue
                
            # שינוי קריטי 1: מוחקים תאים ריקים לחלוטין (תאי רפאים) שדוחפים את האינדקסים הצידה
            cells = [cell.strip() for cell in re.split(r'\||\t', line) if len(cell.strip()) > 0]
            
            if not cells:
                continue
            
            # שלב איתור הכותרות נשען עכשיו על מערך נקי ויציב
            if not is_header_found:
                for idx, cell in enumerate(cells):
                    if "שם" in cell or "מועמד" in cell:
                        name_index = idx
                        is_header_found = True
                        break
                
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
            
            # משיכת השם מהתא המדויק והנעול
            raw_name = cells[name_index] if (name_index != -1 and name_index < len(cells)) else ""
            
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            id_match = id_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            id_num = id_match.group(0) if id_match else ""
            
            # שינוי קריטי 2: מנקים את השם עצמו משאריות של נתונים אחרים (אם ה-OCR בטעות איחד תאים)
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
                "cleaned_name": clean_name
            })
            
            # וידוא סופי: השם חייב להיות אמיתי וצריך להיות לפחות נתון מזהה אחד כדי להיכנס לאתר
            if len(clean_name) >= 2 and (phone or date or time or id_num):
                structured_data.append({
                    "name": clean_name if clean_name else "-",
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