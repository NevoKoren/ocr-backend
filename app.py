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

OCR_SPACE_API_KEY = "helloworld"  # שים כאן את מפתח ה-API שלך

# תבניות לזיהוי מספרים (ללא גבולות מילה נוקשים כדי לתפוס הכל)
phone_pattern = re.compile(r'(05\d[ \-\.]?\d{7}|0[23489][ \-\.]?\d{7})')
date_pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})')
time_pattern = re.compile(r'(\d{1,2}:\d{2})')
id_pattern = re.compile(r'(\d{9})')

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Bulletproof XY Coordinate OCR Server"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isOverlayRequired": "true", # הפקודה שמביאה את המיקומים של כל מילה
            "isTable": "false",
            "scale": "true"
        }
        
        files = [('file', (file.filename, file_bytes, file.content_type))]
        response = requests.post("https://api.ocr.space/parse/image", data=payload, files=files)
        result_json = response.json()
        
        if result_json.get("IsErroredOnProcessing"):
            return JSONResponse(content={"status": "error", "message": "שגיאה בעיבוד התמונה"}, status_code=400)
        
        parsed_results = result_json.get("ParsedResults", [])
        if not parsed_results:
            return JSONResponse(content={"status": "success", "data": []})
            
        # שלב 1: איסוף כל המילים והמיקומים שלהן על המסך
        overlay = parsed_results[0].get("TextOverlay", {})
        lines_data = overlay.get("Lines", [])
        
        words = []
        for line in lines_data:
            for word in line.get("Words", []):
                words.append({
                    "text": word.get("WordText", ""),
                    "top": word.get("Top", 0),   # מיקום בציר Y
                    "left": word.get("Left", 0), # מיקום בציר X
                    "height": word.get("Height", 0)
                })
        
        if not words:
            return JSONResponse(content={"status": "success", "data": []})
            
        # שלב 2: הרכבה מתמטית של שורות אופקיות (מחבר את השם והטלפון גם אם הם רחוקים)
        words.sort(key=lambda w: w['top']) # מיון מלמעלה למטה
        
        rows = []
        current_row = []
        avg_height = sum(w['height'] for w in words) / len(words)
        y_tolerance = avg_height * 0.7  # סטייה מותרת לאותה שורה
        
        for w in words:
            if not current_row:
                current_row.append(w)
            else:
                row_top_avg = sum(x['top'] for x in current_row) / len(current_row)
                if abs(w['top'] - row_top_avg) <= y_tolerance:
                    current_row.append(w)
                else:
                    rows.append(current_row)
                    current_row = [w]
        if current_row:
            rows.append(current_row)
            
        structured_data = []
        
        # שלב 3: ניתוח כל שורה שהרכבנו
        for row in rows:
            # מיון המילים מימין לשמאל (כמו שקוראים בעברית)
            row.sort(key=lambda w: w['left'], reverse=True)
            
            # חיבור כל המילים למשפט ארוך אחד מושלם
            full_row_text = " ".join([w['text'] for w in row])
            
            # חילוץ נתונים
            phone_match = phone_pattern.search(full_row_text)
            date_match = date_pattern.search(full_row_text)
            time_match = time_pattern.search(full_row_text)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            
            # ניקוי כל המספרים כדי שיישאר רק הטקסט (השם + התפקיד)
            clean_text = full_row_text
            if phone: clean_text = clean_text.replace(phone, "")
            if date: clean_text = clean_text.replace(date, "")
            if time: clean_text = clean_text.replace(time, "")
            
            raw_hebrew = re.sub(r'[^\u0590-\u05fe\s]', '', clean_text).strip()
            raw_hebrew = " ".join(raw_hebrew.split())
            
            # מסנן אוטומטית שורות זבל: מכניס לאתר רק אם יש מספר טלפון וטקסט אמיתי
            if phone and len(raw_hebrew) >= 2:
                structured_data.append({
                    "raw_hebrew": raw_hebrew, # נשלח לאתר עם התפקיד, האתר יסנן אותו
                    "phone": phone,
                    "date": date,
                    "time": time
                })
        
        return JSONResponse(content={"status": "success", "data": structured_data})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)