"""
מחלץ אנשי קשר — שרת
Extracts (Hebrew name, Israeli mobile) pairs from a phone photo of a
spreadsheet displayed on a computer screen.

Pairing strategy
----------------
Columns are unreliable: the name and the phone sit on the same visual ROW,
but not necessarily in adjacent columns, and Tesseract often splits a table
into several blocks. So we ignore Tesseract's own block/line numbering and
regroup every recognised word by its vertical position on the page. Words
whose vertical centres fall within a fraction of the median word height are
one row. Inside a row we look for an Israeli mobile number and take the
Hebrew words as the name. That survives extra columns, empty cells and
column order changes for free.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("extractor")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Comma-separated list, e.g. "https://my-app.web.app,https://my-app.firebaseapp.com"
# "*" is fine while developing; tighten it once the site is live.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MIN_WORK_WIDTH = 1400      # upscale smaller images; small text OCRs badly
MAX_WORK_WIDTH = 2600      # and downscale huge ones, for speed
OCR_LANGS = "heb+eng"

HEBREW = re.compile(r"[\u05D0-\u05EA]")
NON_NAME = re.compile(r"[^\u05D0-\u05EA\u0027\u0022\s\-׳״]")

# Words that are almost certainly a column header, not a person
HEADER_WORDS = {"שם", "שמות", "טלפון", "טלפונים", "נייד", "פלאפון",
                "מספר", "משפחה", "פרטי", "איש", "קשר", "כתובת", "מייל"}

app = FastAPI(title="Hebrew contact extractor", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Phone handling
# --------------------------------------------------------------------------

def normalize_phone(raw: str) -> str | None:
    """Return a clean 05XXXXXXXX string, or None if it isn't an Israeli mobile."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("972"):
        digits = "0" + digits[3:]
    elif digits.startswith("00972"):
        digits = "0" + digits[5:]
    if len(digits) == 9 and digits.startswith("5"):
        digits = "0" + digits          # the leading zero was cropped or lost
    return digits if re.fullmatch(r"05\d{8}", digits) else None


def phone_from_candidate(text: str) -> str | None:
    """
    Try a digit string as-is and reversed.

    Tesseract renders digits inside right-to-left text in logical order, but a
    photo of a screen can still hand back a mirrored run when the surrounding
    cell is detected as RTL. Testing the reverse costs nothing and recovers
    numbers that would otherwise be discarded.
    """
    return normalize_phone(text) or normalize_phone(text[::-1])


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------

def load_image(data: bytes) -> np.ndarray:
    """Decode upload to a BGR array, honouring EXIF rotation from the phone."""
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Could not read that image: {exc}") from exc
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def resize_to_work(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w < MIN_WORK_WIDTH:
        scale = MIN_WORK_WIDTH / w
    elif w > MAX_WORK_WIDTH:
        scale = MAX_WORK_WIDTH / w
    else:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten small rotations. Big angles are left alone — a bad guess
    hurts more than a slight tilt."""
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is None:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.4 or abs(angle) > 8:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def variants(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Build a few cleaned-up versions of the photo.

    A picture of a screen brings problems a scan doesn't: moiré interference
    from the pixel grid, uneven backlight, and glare. medianBlur kills the
    moiré speckle, CLAHE evens out the backlight, and the unsharp mask puts
    the edges back. We OCR more than one version because which is best
    depends on the screen and the lighting — and then keep the best result.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)
    gray = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp = cv2.addWeighted(clahe, 1.6, cv2.GaussianBlur(clahe, (0, 0), 3), -0.6, 0)

    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 12)
    return [("sharp", sharp), ("otsu", otsu), ("adaptive", adaptive)]


# --------------------------------------------------------------------------
# OCR + row assembly
# --------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    x: int
    y_centre: float
    height: int
    conf: float
    width: int = 0

    @property
    def right(self) -> int:
        return self.x + self.width


@dataclass
class Contact:
    name: str
    phone: str
    confidence: int
    row_text: str = field(default="")


def ocr_words(image: np.ndarray, psm: int) -> list[Word]:
    data = pytesseract.image_to_data(
        image, lang=OCR_LANGS,
        config=f"--oem 1 --psm {psm} -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DICT,
    )
    words: list[Word] = []
    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        h = int(data["height"][i])
        words.append(Word(text, int(data["left"][i]),
                          int(data["top"][i]) + h / 2, h, conf,
                          int(data["width"][i])))
    return words


def group_rows(words: list[Word]) -> list[list[Word]]:
    """Cluster words into visual rows by vertical centre."""
    if not words:
        return []
    tolerance = float(np.median([w.height for w in words])) * 0.6
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.y_centre):
        if rows and abs(w.y_centre - rows[-1][0].y_centre) <= tolerance:
            rows[-1].append(w)
        else:
            rows.append([w])
    # right-to-left reading order inside each row
    for row in rows:
        row.sort(key=lambda w: -w.x)
    return rows


def contact_from_row(row: list[Word]) -> Contact | None:
    phone, phone_conf, used = None, 0.0, set()

    # A number may arrive whole ("0501234567") or split across neighbouring
    # tokens ("050" "-" "1234567"), so also try joining runs of digit tokens.
    digit_idx = [i for i, w in enumerate(row) if sum(c.isdigit() for c in w.text) >= 2]

    for i in digit_idx:
        found = phone_from_candidate(row[i].text)
        if found:
            phone, phone_conf, used = found, row[i].conf, {i}
            break

    if not phone:
        for start in range(len(digit_idx)):
            for end in range(start + 1, min(start + 4, len(digit_idx)) + 1):
                idx = digit_idx[start:end]
                if idx != list(range(idx[0], idx[-1] + 1)):
                    continue  # not adjacent tokens
                joined = "".join(row[i].text for i in idx)
                found = phone_from_candidate(joined) or phone_from_candidate(joined[::-1])
                if found:
                    phone = found
                    used = set(idx)
                    phone_conf = float(np.mean([row[i].conf for i in idx]))
                    break
            if phone:
                break

    if not phone:
        return None

    # The name: Hebrew words in this row that aren't part of the number.
    candidates: list[tuple[Word, str]] = []
    for i, w in enumerate(row):
        if i in used or not HEBREW.search(w.text):
            continue
        cleaned = NON_NAME.sub("", w.text).strip()
        if len(cleaned) < 2 or cleaned in HEADER_WORDS:
            continue
        candidates.append((w, cleaned))

    if not candidates:
        return None

    # A row can hold more than one Hebrew column ("דוד כהן" | "חבר מועדון").
    # Words inside one cell sit a space apart; a column boundary is a much
    # wider gap. Split on that gap and treat each group as one cell.
    gap_limit = float(np.median([w.height for w, _ in candidates])) * 1.6
    cells: list[list[tuple[Word, str]]] = [[candidates[0]]]
    for prev, current in zip(candidates, candidates[1:]):
        # row is in right-to-left order, so the previous word is to the right
        if prev[0].x - current[0].right > gap_limit:
            cells.append([current])
        else:
            cells[-1].append(current)

    # Pick the cell with the most Hebrew text; ties go to the rightmost, which
    # is where the name column normally sits in a Hebrew sheet.
    chosen = max(cells, key=lambda cell: (sum(len(t) for _, t in cell),
                                          max(w.right for w, _ in cell)))

    parts = [t for _, t in chosen][:4]
    confs = [w.conf for w, _ in chosen]

    name = " ".join(parts)
    confidence = int(round(np.mean(confs + [phone_conf])))
    if len(cells) > 1:
        # more than one Hebrew column means we had to guess — say so, and the
        # site will flag the row for a human to glance at
        confidence = min(confidence, 68)
    return Contact(name=name, phone=phone, confidence=confidence,
                   row_text=" ".join(w.text for w in row))


def extract_contacts(img: np.ndarray) -> tuple[list[Contact], str]:
    """OCR several preprocessing/segmentation combinations, keep the best."""
    best: list[Contact] = []
    best_label = "none"

    for label, prepared in variants(img):
        for psm in (6, 4):
            try:
                rows = group_rows(ocr_words(prepared, psm))
            except pytesseract.TesseractError as exc:
                log.warning("tesseract failed on %s/psm%s: %s", label, psm, exc)
                continue

            found = [c for c in (contact_from_row(r) for r in rows) if c]

            # de-duplicate on the phone number, keeping the surest reading
            by_phone: dict[str, Contact] = {}
            for c in found:
                if c.phone not in by_phone or c.confidence > by_phone[c.phone].confidence:
                    by_phone[c.phone] = c
            found = list(by_phone.values())

            score = (len(found), sum(c.confidence for c in found))
            best_score = (len(best), sum(c.confidence for c in best))
            if score > best_score:
                best, best_label = found, f"{label}/psm{psm}"

            # a clean, high-confidence read — no need to try the rest
            if len(best) >= 5 and all(c.confidence >= 80 for c in best):
                return best, best_label

    return best, best_label


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    try:
        langs = pytesseract.get_languages(config="")
    except Exception:
        langs = []
    return {
        "status": "ok",
        "tesseract": str(pytesseract.get_tesseract_version()),
        "hebrew_installed": "heb" in langs,
    }


@app.post("/extract")
async def extract(image: UploadFile = File(...)) -> dict:
    data = await image.read()
    if not data:
        raise HTTPException(400, "No image was uploaded.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That image is larger than 12 MB.")

    img = resize_to_work(load_image(data))
    contacts, method = extract_contacts(img)
    log.info("extracted %d contacts using %s", len(contacts), method)

    return {
        "contacts": [
            {"name": c.name, "phone": c.phone, "confidence": c.confidence}
            for c in contacts
        ],
        "count": len(contacts),
        "method": method,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))