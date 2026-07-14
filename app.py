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

OCR_SPACE_API_KEY = "helloworld"  # החלף במפתח שלך

phone_pattern = re.compile(r'(05\d[ \-\.]?\d{7}|0[23489][ \-\.]?\d{7})')
date_pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})')
time_pattern = re.compile(r'(\d{1,2}:\d{2})')
id_pattern = re.compile(r'(\d{9})')

# רשימת הגיבוי: מסננת תפקידים בבטחה. 
excluded_keywords = [
    "לוחם", "סדיר", "קבע", "מילואים", "תומך", "לחימה", "חיל", "הים", "אוויר", 
    "יבשה", "סגל", "מיועד", "סטטוס", "דרגה", "תפקיד", "סיווג", "שירות", "מתשחקר", 
    "המתשחקר", "המתתחקר", "מיל", "חובה", "גברים", "נשים", "ימי", "כללי", "מלשב", 'מלש"ב', "ל.ר", "לר"
]

def remove_excluded_words(text):
    """מוחק מילות תפקיד רק אם הן מילים שלמות (כדי לא להרוס שמות כמו 'אלירן')"""
    words = text.split()
    safe_words = [w for w in words if w not in excluded_keywords]
    return " ".join(safe_words)

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Pure Python Column-Shift OCR Server"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        # חזרנו לשליחה הישירה והבטוחה ללא OpenCV שעשה בעיות קידוד
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
            
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            id_match = id_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            id_num = id_match.group(1) if id_match else ""
            
            raw_name = ""
            if name_index != -1 and name_index < len(cells):
                raw_name = cells[name_index]
            
            # שלב 1: מסננים את השם מהעמודה שנבחרה
            candidate_name = remove_excluded_words(raw_name)
            candidate_name = re.sub(r'[^\u0590-\u05fe\s]', '', candidate_name).strip()
            
            # שלב 2 (הסוד לפסים הירוקים): אם העמודה זזה וקיבלנו רק תפקיד (כמו 'ל.ר'), העמודה תישאר ריקה. 
            # אם היא ריקה, נסרוק את כל שאר התאים ונשלוף את השם מהתא שהתפספס!
            if len(candidate_name) < 2:
                for cell in cells:
                    temp_cell = cell
                    if phone: temp_cell = temp_cell.replace(phone, "")
                    if id_num: temp_cell = temp_cell.replace(id_num, "")
                    
                    temp_cell = remove_excluded_words(temp_cell)
                    temp_cell = re.sub(r'[^\u0590-\u05fe\s]', '', temp_cell).strip()
                    temp_cell = " ".join(temp_cell.split())
                    
                    if len(temp_cell) >= 2:
                        candidate_name = temp_cell
                        break
            
            clean_name = candidate_name
            
            debug_lines.append({
                "line_number": line_idx + 1,
                "action": "data_extraction",
                "raw_line": line,
                "cells_detected": cells,
                "cleaned_name": clean_name
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