from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Simple Raw Tesseract OCR Server"}

@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        # 1. טעינת התמונה מתוך הקובץ שהועלה
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes))
        
        # 2. חילוץ הטקסט (הוספתי את השפות עברית ואנגלית כדי שלא יוציא רק ג'יבריש)
        extracted_text = pytesseract.image_to_string(image, lang='heb+eng')
        
        # 3. הדפסה לשרת והחזרת הטקסט הגולמי לאתר
        print("=== Extracted Text ===")
        print(extracted_text)
        print("======================")
        
        return JSONResponse(content={
            "status": "success",
            "raw_text": extracted_text
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)