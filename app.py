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

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Simple Horizontal OCR Server"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isTable": "false",  # קריאה אופקית פשוטה ואמינה
            "scale": "true"
        }
        
        files = [('file', (file.filename, file_bytes, file.content_type))]
        
        response = requests.post("https://api.ocr.space/parse/image", data=payload, files=files)
        result_json = response.json()
        
        if result_json.get("IsErroredOnProcessing"):
            return JSONResponse(content={"status": "error", "message": "שגיאה ב-OCR"}, status_code=400)
        
        parsed_results = result_json.get("ParsedResults", [])
        if not parsed_results:
            return JSONResponse(content={"status": "success", "data": []})
            
        extracted_text = parsed_results[0].get("ParsedText", "")
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        structured_data = []
        
        for line in lines:
            # דילוג על קווי טבלה שנוצרו במקרה
            if re.match(r'^[\s\|\-]+$', line):
                continue
                
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            id_match = id_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            id_num = id_match.group(1) if id_match else ""
            
            # מנקים את המספרים מתוך השורה
            raw_text = line
            if phone: raw_text = raw_text.replace(phone, "")
            if date: raw_text = raw_text.replace(date, "")
            if time: raw_text = raw_text.replace(time, "")
            if id_num: raw_text = raw_text.replace(id_num, "")
            
            # משאירים רק אותיות עבריות (כאן ישארו גם השם וגם התפקיד)
            raw_hebrew = re.sub(r'[^\u0590-\u05fe\s]', '', raw_text).strip()
            raw_hebrew = " ".join(raw_hebrew.split())
            
            # מכניסים לאתר רק שורות שיש בהן מספרי טלפון או ת"ז, כדי לסנן רעשי כותרות
            if (phone or id_num) and len(raw_hebrew) > 1:
                structured_data.append({
                    "raw_hebrew": raw_hebrew, # נשלח לאתר את הטקסט המלא, האתר יסנן
                    "phone": phone,
                    "date": date,
                    "time": time
                })
        
        return JSONResponse(content={"status": "success", "data": structured_data})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)