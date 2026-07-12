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

OCR_SPACE_API_KEY = "helloworld"  # מומלץ להחליף במפתח החינמי האישי שלך

# תבניות Regex מדויקות לזיהוי נתונים
phone_pattern = re.compile(r'\b(05\d[- ]?\d{7}|0[23489][- ]?\d{7})\b')
date_pattern = re.compile(r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b')
time_pattern = re.compile(r'\b(\d{1,2}:\d{2})\b')
id_pattern = re.compile(r'\b\d{9}\b')

# רשימת מילים מורחבת לסינון תפקידים וסטטוסים כדי שלא יזלגו לשם המועמד
excluded_keywords = [
    "לוחם", "סדיר", "קבע", "מילואים", "תומך", "לחימה", "חייל", 
    "גובניק", "ג'ובניק", "קצין", "מפקד", "שירות", "תפקיד", "סטטוס",
    "מועמד", "מיועד", "מיושב", "סיווג", "שובץ", "מתי", "סוג", "צוות",
    "עובד", "כללי", "בהמתנה", "נפתח", "נסגר", "בטיפול", "חובה"
]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Horizontal Row OCR Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isTable": "false",     # שינוי קריטי: סריקה אופקית שומרת את נתוני השורה יחד!
            "scale": "true"
        }
        
        files = [('file', (file.filename, file_bytes, file.content_type))]
        
        response = requests.post("https://api.ocr.space/parse/image", data=payload, files=files)
        result_json = response.json()
        
        if result_json.get("IsErroredOnProcessing"):
            error_msg = result_json.get("ErrorMessage", ["שגיאה בעיבוד"])[0]
            return JSONResponse(content={"status": "error", "message": error_msg}, status_code=400)
        
        parsed_results = result_json.get("ParsedResults", [])
        if not parsed_results:
            return JSONResponse(content={"status": "success", "data": []})
            
        extracted_text = parsed_results[0].get("ParsedText", "")
        
        # פיצול לפי שורות אופקיות אמיתיות
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        structured_data = []
        
        for line in lines:
            # חילוץ נתונים מתוך השורה המאוחדת
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            id_match = id_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            
            # ניקוי השורה מכל המספרים והתאריכים שנמצאו כדי לבודד את הטקסט
            clean_line = line
            if phone_match: clean_line = clean_line.replace(phone_match.group(0), "")
            if date_match: clean_line = clean_line.replace(date_match.group(0), "")
            if time_match: clean_line = clean_line.replace(time_match.group(0), "")
            if id_match: clean_line = clean_line.replace(id_match.group(0), "")
            
            # הסרת מילות תפקיד וסוג שירות מתוך השורה
            for word in excluded_keywords:
                clean_line = re.sub(r'\b' + re.escape(word) + r'\b', '', clean_line)
                clean_line = re.sub(r'\b[בכלהמ]?' + re.escape(word) + r'\b', '', clean_line)
            
            # השארת אותיות בעברית בלבד לטובת השם
            clean_line = re.sub(r'[^\u0590-\u05fe\s]', '', clean_line)
            
            name_words = clean_line.split()
            name = " ".join(name_words).strip()
            
            # מניעת הוספת שורות שהן רק כותרות אקסל או ריקות לחלוטין אחרי הניקוי
            if name in ["שם", "שם מלא", "מספר טלפון", "תאריך", "שעה", "תז", "מספר אישי"] or len(name) < 2:
                # אם שאר השדות קיימים נשמור אותם, אם הכל ריק נדלג
                if not (phone or date or time):
                    continue
            
            # הוספת הרשומה המאוחדת שבה הכל מחובר יחד
            structured_data.append({
                "name": name if name else "-",
                "phone": phone if phone else "-",
                "date": date if date else "-",
                "time": time if time else "-"
            })
        
        return JSONResponse(content={"status": "success", "data": structured_data})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)