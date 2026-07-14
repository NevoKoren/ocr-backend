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

phone_pattern = re.compile(r'\b(05\d[ \-\.]?\d{7}|0[23489][ \-\.]?\d{7})\b')
date_pattern = re.compile(r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b')
time_pattern = re.compile(r'\b(\d{1,2}:\d{2})\b')
id_pattern = re.compile(r'\b(\d{9})\b')

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Coordinate-Based OCR Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isOverlayRequired": "true", # מבקש מהמנוע את הקואורדינטות המדויקות של כל מילה
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
            return JSONResponse(content={"status": "success", "data": [], "debug": {"message": "No text found"}})
            
        # חילוץ קואורדינטות המילים
        overlay = parsed_results[0].get("TextOverlay", {})
        lines_data = overlay.get("Lines", [])
        
        words = []
        for line in lines_data:
            for word in line.get("Words", []):
                words.append({
                    "text": word.get("WordText", ""),
                    "top": word.get("Top", 0),
                    "left": word.get("Left", 0),
                    "width": word.get("Width", 0),
                    "height": word.get("Height", 0)
                })
        
        if not words:
            return JSONResponse(content={"status": "success", "data": []})
            
        # מיון כל המילים מלמעלה למטה
        words.sort(key=lambda w: w['top'])
        
        rows = []
        current_row = []
        
        # חישוב גובה ממוצע כדי לדעת מתי מילה עוברת לשורה חדשה
        heights = [w['height'] for w in words]
        median_height = sorted(heights)[len(heights)//2] if heights else 20
        y_tolerance = median_height * 0.6  # סטייה מותרת לאותה השורה
        
        # הרכבה מתמטית של שורות אופקיות
        for w in words:
            if not current_row:
                current_row.append(w)
            else:
                w_center = w['top'] + (w['height'] / 2)
                row_center = sum(x['top'] + (x['height'] / 2) for x in current_row) / len(current_row)
                
                if abs(w_center - row_center) < y_tolerance:
                    current_row.append(w)
                else:
                    rows.append(current_row)
                    current_row = [w]
        if current_row:
            rows.append(current_row)
            
        structured_data = []
        debug_lines = []
        name_index = -1
        is_header_found = False
        
        # ניתוח כל שורה בפני עצמה
        for line_idx, row in enumerate(rows):
            # מיון המילים מימין לשמאל (כמו בעברית)
            row.sort(key=lambda w: w['left'], reverse=True)
            
            cells = []
            current_cell_words = []
            
            # בניית עמודות האקסל על בסיס הרווחים בין המילים בפיקסלים
            for i, word in enumerate(row):
                if not current_cell_words:
                    current_cell_words.append(word)
                else:
                    prev_word = row[i-1]
                    # המרחק מנקודת הסיום של המילה הקודמת (מימין) לנקודת ההתחלה של המילה הנוכחית
                    gap = prev_word['left'] - (word['left'] + word['width'])
                    
                    # אם המרחק גדול מ-25 פיקסלים, זו עמודת אקסל חדשה!
                    gap_threshold = max(word['height'], 25)
                    
                    if gap > gap_threshold:
                        cells.append(" ".join([w['text'] for w in current_cell_words]))
                        current_cell_words = [word]
                    else:
                        current_cell_words.append(word)
            if current_cell_words:
                cells.append(" ".join([w['text'] for w in current_cell_words]))
                
            raw_line_for_regex = " ".join(cells)
            
            # --- הלוגיקה הרגילה שלנו על העמודות המדויקות ---
            if not is_header_found:
                for idx, cell in enumerate(cells):
                    if "שם" in cell or "מועמד" in cell:
                        name_index = idx
                        is_header_found = True
                        break
                
                if "טלפון" in raw_line_for_regex or "תאריך" in raw_line_for_regex or "אישי" in raw_line_for_regex:
                    is_header_found = True
                
                debug_lines.append({
                    "line_number": line_idx + 1,
                    "action": "header_check",
                    "cells_detected": cells,
                    "name_index_found": name_index
                })
                
                if is_header_found:
                    continue
            
            # משיכת השם מהעמודה הספציפית
            raw_name = cells[name_index] if (name_index != -1 and name_index < len(cells)) else ""
            
            phone_match = phone_pattern.search(raw_line_for_regex)
            date_match = date_pattern.search(raw_line_for_regex)
            time_match = time_pattern.search(raw_line_for_regex)
            id_match = id_pattern.search(raw_line_for_regex)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            id_num = id_match.group(1) if id_match else ""
            
            # ניקוי שאריות מספרים מהשם
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
                "cells_detected": cells,
                "cleaned_name": clean_name,
                "phone_found": phone
            })
            
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
                "detected_name_index": name_index,
                "lines_processing": debug_lines
            }
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)