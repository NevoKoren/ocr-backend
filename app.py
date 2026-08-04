"""
מחלץ אנשי קשר — שרת
Extracts (Hebrew name, Israeli mobile) pairs from a phone photo of a
spreadsheet displayed on a computer screen.

Pipeline
--------
Reading the whole photo at once fails: the table is a fraction of the frame,
so the row text lands around 12px tall and Tesseract needs roughly 30px. And
a page-wide read has to guess at a multi-column layout it can't see.

So this works in three stages instead, each measured against real photos:

  1. LOCATE  — cheap pass over a downscaled copy, digits only, to find where
               the phone column sits and where the data rows are.
  2. PHONES  — re-read that one column, cropped from the full-resolution
               original and enlarged. On a test sheet this went from 15 of 20
               rows to 19 of 20, and corrected a digit the wide pass got wrong.
  3. NAMES   — read only the area to the right of the phone column, cropped
               to the rows found in stage 1.

Then the two are paired by vertical position.

Two findings drove the shape of this, both contrary to the obvious approach:

  * Plain grayscale beats denoise + CLAHE + unsharp masking. At this text size
    the filtering erased thin strokes: 25 phones found versus 7 on the same
    photo. There is no preprocessing chain here on purpose.
  * PSM 11 (sparse text) beats PSM 6 (uniform block) roughly fourfold. A
    spreadsheet is not a block of prose, and telling Tesseract it is makes it
    force text into a layout that isn't there.

Why the name is taken as the rightmost Hebrew cell in a row: these sheets are
right-to-left, so the name column sits at or near column A, and every other
Hebrew column (role, unit, rank, status, notes) falls to its left. Picking the
longest or most confident Hebrew cell instead reliably picks a role like
"לוחם ימי גברים" over the actual name.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from dataclasses import dataclass

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

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_SOURCE_WIDTH = 4000      # cap the original, for memory
LOCATE_WIDTH = 1800          # the cheap first pass runs here
TARGET_TEXT_PX = 30          # Tesseract's comfort zone for letter height
MAX_CROP_PIXELS = 3_000_000  # ceiling on any single enlarged crop

TIME_BUDGET_SECONDS = float(os.getenv("TIME_BUDGET_SECONDS", "55"))

HEBREW = re.compile(r"[\u05D0-\u05EA]")
NON_NAME = re.compile(r"[^\u05D0-\u05EA\u0027\u0022\s\-׳״]")
DIGITS_ONLY = "0123456789-"

# Column headings and stock cell values that are never a person's name.
NOT_A_NAME = {
    "שם", "שמות", "מלא", "טלפון", "נייד", "פלאפון", "מספר", "אישי", "תז",
    "משפחה", "פרטי", "איש", "קשר", "כתובת", "מייל", "תפקיד", "רמה", "סוג",
    "גיל", "דרגה", "סמכות", "יחידה", "הערות", "סטטוס", "שעת", "זימון",
    "מתחקר", "המתחקר", "שבוע", "בדיקה", "התייצבות", "במגן",
}

app = FastAPI(title="Hebrew contact extractor", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@dataclass
class Word:
    text: str
    x: int
    y: float          # vertical centre
    height: int
    width: int
    conf: float

    @property
    def right(self) -> int:
        return self.x + self.width


@dataclass
class Contact:
    name: str
    phone: str
    confidence: int


# --------------------------------------------------------------------------
# Phone parsing
# --------------------------------------------------------------------------

def normalize_phone(raw: str) -> str | None:
    """Return a clean 05XXXXXXXX string, or None if it isn't an Israeli mobile."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("00972"):
        digits = "0" + digits[5:]
    elif digits.startswith("972"):
        digits = "0" + digits[3:]
    if len(digits) == 9 and digits.startswith("5"):
        digits = "0" + digits           # leading zero cropped or dropped
    return digits if re.fullmatch(r"05\d{8}", digits) else None


# --------------------------------------------------------------------------
# OCR helpers
# --------------------------------------------------------------------------

def read_words(image: np.ndarray, psm: int, lang: str,
               whitelist: str | None = None) -> list[Word]:
    config = f"--oem 1 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(image, lang=lang, config=config,
                                     output_type=pytesseract.Output.DICT)
    words: list[Word] = []
    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        h = int(data["height"][i])
        words.append(Word(text, int(data["left"][i]),
                          int(data["top"][i]) + h / 2, h,
                          int(data["width"][i]), conf))
    return words


def enlarge(crop: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    """Scale a crop toward Tesseract's comfortable text size, within a pixel
    ceiling so a big photo can't blow the container's memory."""
    h, w = crop.shape[:2]
    if h * w * scale * scale > MAX_CROP_PIXELS:
        scale = max(1.0, (MAX_CROP_PIXELS / (h * w)) ** 0.5)
    if scale <= 1.02:
        return crop, 1.0
    return cv2.resize(crop, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC), scale


def group_rows(words: list[Word], tolerance: float) -> list[list[Word]]:
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.y):
        if rows and abs(w.y - rows[-1][0].y) <= tolerance:
            rows[-1].append(w)
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: -w.x)      # right-to-left reading order
    return rows


# --------------------------------------------------------------------------
# Stage 1 — locate the phone column
# --------------------------------------------------------------------------

def locate_phones(gray: np.ndarray) -> list[Word]:
    """Cheap wide pass, digits only. We don't trust the numbers it returns —
    only where on the page they are."""
    found = []
    for word in read_words(gray, psm=11, lang="eng", whitelist=DIGITS_ONLY):
        if normalize_phone(word.text):
            found.append(word)
    return found


def phone_band(hits: list[Word], page_width: int) -> tuple[int, int]:
    """
    Horizontal extent of the phone column, taken from the numbers themselves
    plus a small margin.

    The margin matters more than it looks. Too wide and the strip swallows the
    table's black column rule, which reads as a giant vertical stroke and
    wrecks sparse-text detection — on a test sheet, widening the margin from
    0.35 to 0.6 of a cell width dropped the result from 19 rows to 3.
    """
    lefts = np.array([w.x for w in hits])
    rights = np.array([w.right for w in hits])
    margin = float(np.median(rights - lefts)) * 0.35
    return (max(2, int(lefts.min() - margin)),
            min(page_width - 2, int(rights.max() + margin)))


# --------------------------------------------------------------------------
# Stage 2 — re-read the phone column properly
# --------------------------------------------------------------------------

def read_phone_column(source: np.ndarray, band: tuple[int, int],
                      rows_top: int, rows_bottom: int,
                      scale: float) -> list[tuple[float, str, float]]:
    """Returns (y in source coordinates, phone, confidence)."""
    x0, x1 = band
    crop = source[rows_top:rows_bottom, x0:x1]
    if crop.size == 0:
        return []
    enlarged, applied = enlarge(crop, scale)

    out: list[tuple[float, str, float]] = []
    for word in read_words(enlarged, psm=11, lang="eng", whitelist=DIGITS_ONLY):
        phone = normalize_phone(word.text)
        if phone:
            out.append((rows_top + word.y / applied, phone, word.conf))
    return out


# --------------------------------------------------------------------------
# Stage 3 — read the names
# --------------------------------------------------------------------------

def read_name_area(source: np.ndarray, left: int, rows_top: int,
                   rows_bottom: int, scale: float) -> tuple[list[Word], float, int]:
    """Everything to the right of the phone column, limited to the data rows.
    Returns words in source coordinates."""
    crop = source[rows_top:rows_bottom, left:]
    if crop.size == 0:
        return [], 1.0, left
    enlarged, applied = enlarge(crop, scale)

    best: list[Word] = []
    for psm in (11, 6):
        try:
            words = read_words(enlarged, psm=psm, lang="heb+eng")
        except pytesseract.TesseractError as exc:
            # Missing Hebrew traineddata, most likely. Phones are still good,
            # so return what we have rather than failing the whole request.
            log.error("Hebrew OCR failed (psm %d): %s", psm, exc)
            return [], 1.0, left
        hebrew = [w for w in words if HEBREW.search(w.text)]
        if len(hebrew) > len(best):
            best = hebrew
        if len(best) >= 8:
            break

    for w in best:
        w.x = int(left + w.x / applied)
        w.width = int(w.width / applied)
        w.y = rows_top + w.y / applied
        w.height = int(w.height / applied)
    return best, applied, left


def name_from_row(row: list[Word]) -> tuple[str, float] | None:
    """The rightmost Hebrew cell in the row. Words a space apart are one cell;
    a wide gap means a new column."""
    cells: list[list[Word]] = []
    usable = []
    for w in row:
        cleaned = NON_NAME.sub("", w.text).strip()
        if len(cleaned) < 2 or cleaned in NOT_A_NAME:
            continue
        usable.append((w, cleaned))

    if not usable:
        return None

    gap_limit = float(np.median([w.height for w, _ in usable])) * 1.6
    cells = [[usable[0]]]
    for prev, current in zip(usable, usable[1:]):
        if prev[0].x - current[0].right > gap_limit:
            cells.append([current])
        else:
            cells[-1].append(current)

    rightmost = cells[0]                       # row is already right-to-left
    name = " ".join(text for _, text in rightmost[:4])
    conf = float(np.mean([w.conf for w, _ in rightmost]))
    return name, conf


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def load_image(data: bytes) -> np.ndarray:
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
    except Exception as exc:
        raise HTTPException(400, f"Could not read that image: {exc}") from exc
    gray = np.array(pil)
    if gray.shape[1] > MAX_SOURCE_WIDTH:
        f = MAX_SOURCE_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return gray


def extract_contacts(source: np.ndarray) -> tuple[list[Contact], dict]:
    started = time.monotonic()
    H, W = source.shape
    info: dict = {"width": W, "height": H}

    # --- stage 1 -----------------------------------------------------------
    f = min(1.0, LOCATE_WIDTH / W)
    locate_img = (source if f == 1.0 else
                  cv2.resize(source, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    hits = locate_phones(locate_img)
    info["located"] = len(hits)
    log.info("stage 1 located %d phone-shaped cells in %.1fs", len(hits),
             time.monotonic() - started)

    if len(hits) < 2:
        info["reason"] = "no phone column found"
        return [], info

    for w in hits:                                   # back to source coordinates
        w.x, w.width = int(w.x / f), int(w.width / f)
        w.y, w.height = w.y / f, int(w.height / f)

    ys = sorted(w.y for w in hits)
    row_height = float(np.median(np.diff(ys))) if len(ys) > 2 else hits[0].height * 2
    row_height = max(row_height, hits[0].height * 1.2)
    # Pad generously. The wide pass routinely misses the first and last rows,
    # and if the crop stops at the last row it found, the high-resolution pass
    # can never recover them. Rows above the table have no phone number in them,
    # so over-reaching costs nothing.
    pad = row_height * 3
    rows_top = max(0, int(min(ys) - pad))
    rows_bottom = min(H, int(max(ys) + pad))
    scale = float(np.clip(TARGET_TEXT_PX / max(6.0, row_height * 0.5), 1.0, 3.0))
    band = phone_band(hits, W)
    info.update({"row_height": round(row_height, 1), "scale": round(scale, 2),
                 "band": list(band)})

    # --- stage 2 -----------------------------------------------------------
    # Neither pass dominates: on one test photo the wide pass found 25 rows and
    # the strip 8; on another the strip found 19 and the wide 15. So keep both
    # and let them vote per row. The strip gets a small bonus because when the
    # two disagree it has the resolution advantage — it corrected a wrong final
    # digit the wide pass produced.
    t = time.monotonic()
    strip = read_phone_column(source, band, rows_top, rows_bottom, scale)
    log.info("stage 2 read %d phones from the column strip in %.1fs",
             len(strip), time.monotonic() - t)

    candidates = [(w.y, normalize_phone(w.text) or "", w.conf) for w in hits]
    candidates += [(y, phone, conf + 8) for y, phone, conf in strip]

    by_row: dict[int, tuple[float, str, float]] = {}
    for y, phone, conf in candidates:
        if not phone:
            continue
        key = round(y / (row_height * 0.5))
        if key not in by_row or conf > by_row[key][2]:
            by_row[key] = (y, phone, conf)
    phones = sorted(by_row.values())
    info["rows_matched"] = len(phones)

    if not phones:
        info["reason"] = "phone column unreadable"
        return [], info

    # --- stage 3 -----------------------------------------------------------
    # The name sits at the far right of a right-to-left sheet, so try a narrow
    # right-hand slice first — it's a third of the pixels and excludes most of
    # the other Hebrew columns. Widen to everything right of the phone column
    # only if that comes back empty.
    narrow_left = int(W - 0.45 * (W - band[1]))
    names: list[Word] = []
    for attempt, left in enumerate((narrow_left, band[1])):
        if time.monotonic() - started > TIME_BUDGET_SECONDS:
            log.warning("skipped name pass — out of time budget")
            break
        t = time.monotonic()
        names, _, _ = read_name_area(source, left, rows_top, rows_bottom, scale)
        log.info("stage 3 (attempt %d, x>=%d) read %d Hebrew words in %.1fs",
                 attempt + 1, left, len(names), time.monotonic() - t)
        if names:
            break

    name_rows = group_rows(names, tolerance=row_height * 0.45)

    # --- pair by vertical position ----------------------------------------
    contacts: dict[str, Contact] = {}
    for y, phone, phone_conf in sorted(phones, key=lambda p: p[0]):
        best_row, best_gap = None, row_height * 0.6
        for row in name_rows:
            gap = abs(row[0].y - y)
            if gap < best_gap:
                best_row, best_gap = row, gap

        name, name_conf = ("", 40.0)
        if best_row:
            got = name_from_row(best_row)
            if got:
                name, name_conf = got

        confidence = int(round((phone_conf + name_conf) / 2)) if name else 45
        existing = contacts.get(phone)
        if existing is None or confidence > existing.confidence:
            contacts[phone] = Contact(name=name, phone=phone, confidence=confidence)

    info["seconds"] = round(time.monotonic() - started, 1)
    return list(contacts.values()), info


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

    started = time.monotonic()
    contacts, info = extract_contacts(load_image(data))
    seconds = round(time.monotonic() - started, 1)
    log.info("extracted %d contacts in %.1fs %s", len(contacts), seconds, info)

    return {
        "contacts": [
            {"name": c.name, "phone": c.phone, "confidence": c.confidence}
            for c in contacts
        ],
        "count": len(contacts),
        "seconds": seconds,
        "debug": info,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))