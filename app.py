from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

phone_pattern = re.compile(r'(05\d[ \-\.]?\d{7}|0[23489][ \-\.]?\d{7})')
date_pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})')
time_pattern = re.compile(r'(\d{1,2}:\d{2})')

excluded_keywords = [
    "לוחם", "סדיר", "קבע", "מילואים", "תומך", "לחימה", "חיל", "הים", "אוויר", 
    "יבשה", "סגל", "מיועד", "סטטוס", "דרגה", "תפקיד", "סיווג", "שירות", "מתשחקר", 
    "המתשחקר", "המתתחקר", "מיל", "חובה", "גברים", "נשים", "ימי", "כללי", "מלשב", 'מלש"ב', "ל.ר", "לר"
]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Tesseract Server with Morphological Line Removal"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(content={"status": "error", "message": "קובץ התמונה פגום או לא קריא"}, status_code=400)
            
        # ==========================================================
        # הגישה הנכונה: מחיקת הטבלה כדי ש-Tesseract יקרא רק טקסט
        # ==========================================================
        
        # 1. המרה לאפור והגדלה לתפיסת פרטים
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # 2. טשטוש קל כדי להעלים את הפיקסלים של צילום המסך (Moiré)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. בינאריזציה הפוכה: הטקסט והקווים הופכים ללבן (255) והרקע לשחור (0)
        # זה חובה כדי שנוכל לבצע פעולות מורפולוגיות כדי למצוא קווים
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
        
        # 4. מחיקת קווים אנכיים (קירות העמודות של אקסל)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts_v, _ = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_v:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), 5) # צובע את הקו בשחור כדי להעלים אותו
            
        # 5. מחיקת קווים אופקיים (רצפת השורות של אקסל)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts_h, _ = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_h:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), 5)
            
        # 6. היפוך חזרה: הטקסט נשאר שחור טהור על רקע לבן טהור (ללא טבלאות!)
        clean_img = 255 - thresh
        
        # טשטוש מדיאני קל כדי להחליק את השוליים של האותיות אחרי המחיקות
        clean_img = cv2.medianBlur(clean_img, 3)

        # ==========================================================
        
        # הרצת Tesseract. 
        # משתמשים ב-PSM 6 שאומר ל-Tesseract: "זה בלוק טקסט אחיד, תקרא אותו משמאל לימין"
        try:
            d = pytesseract.image_to_data(clean_img, lang='heb+eng', config='--psm 6', output_type=Output.DICT)
        except Exception as tesseract_err:
            return JSONResponse(content={"status": "error", "message": f"שגיאת מנוע OCR מקומי: {str(tesseract_err)}"}, status_code=500)
        
        words = []
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 15:  # סינון מילים ברמת ודאות נמוכה (רעשים קטנים שנשארו)
                text = d['text'][i].strip()
                # מסנן גם תווים בודדים כמו נקודות או פסיקים שהמנוע ממציא מהלכלוך
                if text and len(text) > 1 or text.isdigit():
                    left = d['left'][i]
                    top = d['top'][i]
                    width = d['width'][i]
                    height = d['height'][i]
                    words.append({
                        "text": text,
                        "top": top,
                        "left": left,
                        "width": width,
                        "height": height,
                        "right": left + width,
                        "center_x": left + (width / 2),
                        "center_y": top + (height / 2)
                    })
        
        if not words:
            return JSONResponse(content={
                "status": "success", 
                "data": [], 
                "debug": {"total_words_found": 0, "message": "לא זוהה טקסט לאחר ניקוי הקווים"}
            })
            
        name_col_center_x = None
        for w in words:
            if "שם" in w['text'] or "מועמד" in w['text'] or "פרטי" in w['text']:
                name_col_center_x = w['center_x']
                break
                
        words.sort(key=lambda w: w['center_y'])
        rows = []
        current_row = []
        heights = [w['height'] for w in words]
        median_height = sorted(heights)[len(heights)//2] if heights else 20
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
        
        for row in rows:
            row.sort(key=lambda w: w['left'], reverse=True)
            row_full_text = " ".join([w['text'] for w in row])
            phone_match = phone_pattern.search(row_full_text)
            
            if not phone_match:
                continue
                
            phone = phone_match.group(1)
            date = date_pattern.search(row_full_text).group(1) if date_pattern.search(row_full_text) else ""
            time = time_pattern.search(row_full_text).group(1) if time_pattern.search(row_full_text) else ""
            
            cells = []
            current_cell = []
            for i, word in enumerate(row):
                if not current_cell:
                    current_cell.append(word)
                else:
                    prev_word = row[i-1]
                    gap = prev_word['left'] - word['right']
                    gap_threshold = max(word['height'] * 1.5, 25) # הגדלתי את מרווח התאים כי אין יותר קווי טבלה
                    
                    if gap > gap_threshold: 
                        cells.append(current_cell)
                        current_cell = [word]
                    else:
                        current_cell.append(word)
            if current_cell:
                cells.append(current_cell)
                
            best_name_text = ""
            if name_col_center_x is not None:
                min_dist = float('inf')
                for cell in cells:
                    cell_center_x = sum(w['center_x'] for w in cell) / len(cell)
                    dist = abs(cell_center_x - name_col_center_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_name_text = " ".join([w['text'] for w in cell])
            else:
                for cell in cells:
                    text = " ".join([w['text'] for w in cell])
                    if len(re.sub(r'[^\u0590-\u05fe]', '', text)) >= 2:
                        best_name_text = text
                        break
            
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
        
        return JSONResponse(content={
            "status": "success",
            "data": structured_data,
            "debug": {
                "total_words_found": len(words),
                "name_col_center_x": name_col_center_x,
                "all_words": [w['text'] for w in words]
            }
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)