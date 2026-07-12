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

# תבניות Regex לזיהוי נתונים
phone_pattern = re.compile(r'\b(05\d[- ]?\d{7}|0[23489][- ]?\d{7})\b')
date_pattern = re.compile(r'\b(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\b')
time_pattern = re.compile(r'\b(\d{1,2}:\d{2})\b')
id_pattern = re.compile(r'\b\d{9}\b')

# רשימת מילים לסינון מוחלט (תפקידים, סטטוסים וסוגי שירות) כדי שלא ייכנסו לשם המועמד
excluded_keywords = [
    "לוחם", "סדיר", "קבע", "מילואים", "תומך", "לחימה", 
    "גובניק", "ג'ובניק", "קצין", "מפקד", "חייל", "שירות", "תפקיד",
    "ג'וב", "מועמד", "מיועד", "מיושב", "סיווג"
]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Advanced Structured OCR Server is running"}

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
            return JSONResponse(content={"status": "success", "data": []})
            
        extracted_text = parsed_results[0].get("ParsedText", "")
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        structured_data = []
        
        for line in lines:
            phone_match = phone_pattern.search(line)
            date_match = date_pattern.search(line)
            time_match = time_pattern.search(line)
            id_match = id_pattern.search(line)
            
            phone = phone_match.group(1) if phone_match else ""
            date = date_match.group(1) if date_match else ""
            time = time_match.group(1) if time_match else ""
            
            # ניקוי השורה מנתונים מבוססי תבנית
            clean_line = line
            if phone_match: clean_line = clean_line.replace(phone_match.group(0), "")
            if date_match: clean_line = clean_line.replace(date_match.group(0), "")
            if time_match: clean_line = clean_line.replace(time_match.group(0), "")
            if id_match: clean_line = clean_line.replace(id_match.group(0), "")
            
            # הסרת מילים המשתייכות לתפקיד או סוג שירות מתוך השורה
            for word in excluded_keywords:
                clean_line = re.sub(r'\b' + re.escape(word) + r'\b', '', clean_line)
                # תמיכה גם בהסרה אם המילה צמודה לאות יחס (כמו "בלוחם", "כסדיר")
                clean_line = re.sub(r'\b[בכלהמ]?' + re.escape(word) + r'\b', '', clean_line)
            
            # השארת אותיות בעברית ורווחים בלבד לטובת בידוד השם הנקי
            clean_line = re.sub(r'[^\u0590-\u05fe\s]', '', clean_line)
            
            name_words = clean_line.split()
            name = " ".join(name_words)
            
            # דילוג על שורות כותרת פוטנציאליות של אקסל
            if name in ["שם", "מספר טלפון", "תאריך", "שעה", "תז", "מספר אישי", "תפקיד", "סוג שירות"]:
                continue
                
            if name or phone or date or time:
                structured_data.append({
                    "name": name.strip(),
                    "phone": phone.strip(),
                    "date": date.strip(),
                    "time": time.strip()
                })
        
        return JSONResponse(content={"status": "success", "data": structured_data})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)