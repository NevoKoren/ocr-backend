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

OCR_SPACE_API_KEY = "helloworld"  # החלף במפתח ה-API שלך

phone_pattern = re.compile(r'(05\d[ \-\.]?\d{7}|0[23489][ \-\.]?\d{7})')
date_pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})')
time_pattern = re.compile(r'(\d{1,2}:\d{2})')
id_pattern = re.compile(r'(\d{9})')

# מילות סינון למקרה של סטיות קלות בפיקסלים
excluded_keywords = [
    "לוחם", "סדיר", "קבע", "מילואים", "תומך", "לחימה", "חיל", "הים", "אוויר", 
    "יבשה", "סגל", "מיועד", "סטטוס", "דרגה", "תפקיד", "סיווג", "שירות", "מתשחקר", 
    "המתשחקר", "המתתחקר", "מיל", "חובה", "גברים", "נשים", "ימי", "כללי", "מלשב", 'מלש"ב', "ל.ר", "לר"
]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Spatial Anchor OCR Server is running"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "OCREngine": "3",
            "isOverlayRequired": "true",  # קריטי: מביא קואורדינטות X,Y לכל מילה
            "isTable": "false",           # לא צריכים את המנוע שלו, אנחנו בונים טבלה מתמטית
            "scale": "true"
        }
        
        files = [('file', (file.filename, file_bytes, file.content_type))]
        response = requests.post("https://api.ocr.space/parse/image", data=payload, files=files)
        result_json = response.json()
        
        if result_json.get("IsErroredOnProcessing"):
            return JSONResponse(content={"status": "error", "message": "שגיאה בעיבוד ה-OCR"}, status_code=400)
        
        parsed_results = result_json.get("ParsedResults", [])
        if not parsed_results:
            return JSONResponse(content={"status": "success", "data": []})
            
        overlay = parsed_results[0].get("TextOverlay", {})
        lines_data = overlay.get("Lines", [])
        
        # 1. איסוף כל המילים ומיקומן הפיזי
        words = []
        for line in lines_data:
            for word in line.get("Words", []):
                left = word.get("Left", 0)
                width = word.get("Width", 0)
                words.append({
                    "text": word.get("WordText", ""),
                    "top": word.get("Top", 0),
                    "left": left,
                    "width": width,
                    "height": word.get("Height", 0),
                    "right": left + width,
                    "center_x": left + (width / 2),
                    "center_y": word.get("Top", 0) + (word.get("Height", 0) / 2)
                })
        
        if not words:
            return JSONResponse(content={"status": "success", "data": []})
            
        # 2. מציאת ה"עוגן" של עמודת השם (היכן היא ממוקמת על ציר ה-X)
        name_col_center_x = None
        for w in words:
            if "שם" in w['text'] or "מועמד" in w['text'] or "פרטי" in w['text']:
                name_col_center_x = w['center_x']
                break
                
        # 3. קיבוץ המילים לשורות אופקיות לפי ציר Y
        words.sort(key=lambda w: w['center_y'])
        rows = []
        current_row = []
        # חישוב גובה ממוצע כדי לדעת מה הסטייה המותרת לשורה
        median_height = sorted([w['height'] for w in words])[len(words)//2] if words else 20
        y_tolerance = median_height * 0.7  
        
        for w in words:
            if not current_row:
                current_row.append(w)
            else:
                row_center_y = sum(x['center_y'] for x in current_row) / len(current_row)
                if abs(w['center_y'] - row_center_y) <= y_tolerance:
                    current_row.append(w)
                else:
                    rows.append(current_row)
                    current_row = [w]
        if current_row:
            rows.append(current_row)
            
        structured_data = []
        
        # 4. ניתוח חכם של כל שורה
        for row in rows:
            # מיון השורה מימין לשמאל (כמו בעברית)
            row.sort(key=lambda w: w['left'], reverse=True)
            
            # חיבור השורה לטקסט מלא כדי לבדוק אם יש כאן "עוגן" (טלפון)
            row_full_text = " ".join([w['text'] for w in row])
            phone_match = phone_pattern.search(row_full_text)
            
            # אם אין טלפון בשורה הזו בגובה הזה - מדלגים. זה מסנן את כל ה"רעש".
            if not phone_match:
                continue
                
            phone = phone_match.group(1)
            date = date_pattern.search(row_full_text).group(1) if date_pattern.search(row_full_text) else ""
            time = time_pattern.search(row_full_text).group(1) if time_pattern.search(row_full_text) else ""
            
            # 5. חיתוך השורה לתאים לפי המרחק (רווחים פיזיים גדולים בין מילים)
            cells = []
            current_cell = []
            for i, word in enumerate(row):
                if not current_cell:
                    current_cell.append(word)
                else:
                    prev_word = row[i-1]
                    # המרחק בין המילה הימנית (הקודמת) למילה השמאלית (הנוכחית)
                    gap = prev_word['left'] - word['right']
                    gap_threshold = max(word['height'] * 1.2, 20)
                    
                    if gap > gap_threshold: # רווח גדול = עמודה חדשה
                        cells.append(current_cell)
                        current_cell = [word]
                    else:
                        current_cell.append(word)
            if current_cell:
                cells.append(current_cell)
                
            # 6. מציאת התא ששייך לעמודת השם!
            best_name_text = ""
            if name_col_center_x is not None:
                # מחפשים את התא שהמרכז שלו הכי קרוב למרכז של כותרת "שם מלא"
                min_dist = float('inf')
                for cell in cells:
                    cell_center_x = sum(w['center_x'] for w in cell) / len(cell)
                    dist = abs(cell_center_x - name_col_center_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_name_text = " ".join([w['text'] for w in cell])
            else:
                # גיבוי: אם לא מצאנו כותרת, ניקח את התא הראשון מימין שיש בו אותיות בעברית
                for cell in cells:
                    text = " ".join([w['text'] for w in cell])
                    if len(re.sub(r'[^\u0590-\u05fe]', '', text)) >= 2:
                        best_name_text = text
                        break
            
            # ניקוי סופי של התא שנבחר
            best_name_text = best_name_text.replace(phone, "").strip()
            words_in_name = best_name_text.split()
            safe_words = [w for w in words_in_name if w not in excluded_keywords]
            clean_name = " ".join(safe_words)
            clean_name = re.sub(r'[^\u0590-\u05fe\s]', '', clean_name).strip()
            
            if len(clean_name) >= 2:
                structured_data.append({
                    "name": clean_name,
                    "phone": phone,
                    "date": date,
                    "time": time
                })
        
        return JSONResponse(content={"status": "success", "data": structured_data})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)