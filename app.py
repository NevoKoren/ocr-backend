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

OCR_SPACE_API_KEY = "helloworld"  # זכור להחליף במפתח החינמי האישי שלך

# תבניות לזיהוי מספרים ונתונים קבועים
phone_pattern = re.compile(r'\b(05\d[- ]?\d{7}|0[23489][- ]?\d{7})\b')
date_pattern = re.compile(r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b')
time_pattern = re.compile(r'\b(\d{1,2}:\d{2})\b')

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Column-Mapping OCR Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        # שימוש ב-isTable=true כדי שהמנוע יחזיר את השורות מופרדות לפי עמודות 
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
            return JSONResponse(content={"status": "success", "data": []})
            
        extracted_text = parsed_results[0].get("ParsedText", "")
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        structured_data = []
        name_index = -1
        is_header_found = False
        
        for line in lines:
            # ה-API מחזיר את הטבלה עם רווחים גדולים (טאבים) בין העמודות, כאן אנחנו מפרקים לתאים
            cells = [cell.strip() for cell in re.split(r'\t|\s{2,}', line)]
            
            # שלב 1: איתור שורת הכותרות (חיפוש פעם אחת בלבד)
            if not is_header_found:
                for idx, cell in enumerate(cells):
                    # אנחנו שומרים את האינדקס המדויק של עמודת השם
                    if "שם" in cell or "מועמד" in cell:
                        name_index = idx
                        is_header_found = True
                        break
                
                # אם מצאנו מילים של כותרת אבל לא את השם, עדיין נסמן שזו שורת כותרת כדי לדלג עליה
                if "טלפון" in line or "תאריך" in line or "אישי" in line or "שירות" in line or "ראיון" in line:
                    is_header_found = True
                    
                if is_header_found:
                    continue  # מדלגים על הכנסת הכותרות לטבלה באתר שלך
            
            # שלב 2: חילוץ נתונים ממוקד
            
            # מושכים את השם אך ורק מתוך העמודה המיועדת לו! אם העמודה לא קיימת, משאירים ריק.
            raw_name = cells[name_index] if (name_index != -1 and name_index < len(cells)) else ""
            
            # חילוץ הטלפון והזמנים מכל מקום בשורה באמצעות התבניות
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            
            # ניקוי השם - מוחק מספרים וסימנים מתוך התא של השם ומשאיר רק אותיות בעברית
            clean_name = re.sub(r'[^\u0590-\u05fe\s]', '', raw_name).strip()
            clean_name = " ".join(clean_name.split()) # סידור רווחים כפולים
            
            # אם השרת נכשל בזיהוי עמודות לגמרי (Fallback נדיר), הוא ינקה את כל המספרים מהשורה
            if name_index == -1 and not is_header_found:
                id_match = re.search(r'\b\d{9}\b', line)
                if phone_match: clean_name = clean_name.replace(phone_match.group(0), "")
                if date_match: clean_name = clean_name.replace(date_match.group(0), "")
                if time_match: clean_name = clean_name.replace(time_match.group(0), "")
                if id_match: clean_name = clean_name.replace(id_match.group(0), "")
                clean_name = re.sub(r'[^\u0590-\u05fe\s]', '', clean_name).strip()
                clean_name = " ".join(clean_name.split())
            
            # מוסיף את הרשומה רק אם באמת נמצא שם עם לפחות 2 אותיות או מספר טלפון
            if len(clean_name) >= 2 or phone:
                structured_data.append({
                    "name": clean_name if clean_name else "-",
                    "phone": phone if phone else "-",
                    "date": date if date else "-",
                    "time": time if time else "-"
                })
        
        return JSONResponse(content={"status": "success", "data": structured_data})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)