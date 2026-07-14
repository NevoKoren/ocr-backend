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

# הורדנו את תגיות גבולות המילה (\b) כדי שהמערכת תתפוס את המספרים גם אם הם נדבקו למילים אחרות
phone_pattern = re.compile(r'(05\d[ \-\.]?\d{7}|0[23489][ \-\.]?\d{7})')
date_pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})')
time_pattern = re.compile(r'(\d{1,2}:\d{2})')
id_pattern = re.compile(r'(\d{9})')

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Horizontal Smart OCR Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isTable": "false",  # <--- שינוי קריטי: מכריח סריקה אופקית כדי לחבר בין טלפון לשם!
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
                
            # פיצול השורה לעמודות לפי קווים, טאבים, או רווחים גדולים של אקסל (2 רווחים ומעלה)
            cells = [cell.strip() for cell in re.split(r'\||\t|\s{2,}', line) if len(cell.strip()) > 0]
            
            if not cells:
                continue
            
            # זיהוי אינדקס עמודת השם
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
            
            # שליפת השם
            raw_name = cells[name_index] if (name_index != -1 and name_index < len(cells)) else ""
            
            # חילוץ הנתונים מהשורה כולה
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            id_match = id_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            id_num = id_match.group(1) if id_match else ""
            
            clean_name = raw_name
            # ניקוי השם אם טלפון או מספר אחר נדבקו אליו
            if phone: clean_name = clean_name.replace(phone, "")
            if date: clean_name = clean_name.replace(date, "")
            if time: clean_name = clean_name.replace(time, "")
            if id_num: clean_name = clean_name.replace(id_num, "")
            
            # השארת אותיות עבריות בלבד
            clean_name = re.sub(r'[^\u0590-\u05fe\s]', '', clean_name).strip()
            clean_name = " ".join(clean_name.split())
            
            debug_lines.append({
                "line_number": line_idx + 1,
                "action": "data_extraction",
                "raw_line": line,
                "cells_detected": cells,
                "raw_name_cell": raw_name,
                "cleaned_name": clean_name,
                "phone_found": phone
            })
            
            # תנאי הכניסה ההכרחי לסינון רעשים של אקסל
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