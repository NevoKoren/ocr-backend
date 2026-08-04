"""
מחלץ אנשי קשר — שרת
Extracts (Hebrew name, Israeli mobile) pairs from a phone photo of a
spreadsheet displayed on a computer screen.

Pipeline
--------
Reading the whole photo at once fails: the table is a fraction of the frame,
so the row text lands around 12px tall and Tesseract needs roughly 30px. And
a page-wide read has to guess at a multi-column layout it can't see.

Both columns therefore get the same treatment — locate, then crop and enlarge
that one column and read it alone:

  1. LOCATE PHONES — cheap pass over a downscaled copy, digits only, to find
                     where the phone column sits and where the data rows are.
  2. READ PHONES   — re-read that column, cropped from the full-resolution
                     original and enlarged. On a test sheet this went from 15
                     of 20 rows to 19 of 20 and corrected a wrong digit.
  3. LOCATE NAMES  — find the "שם מלא" heading and take its column band. If the
                     heading can't be read, fall back to the band occupied by
                     the rightmost Hebrew cell of each data row.
  4. READ NAMES    — read that band alone, cropped and enlarged.

Then the two are paired by vertical position.

Findings that drove this, all contrary to the obvious approach:

  * Plain grayscale beats denoise + CLAHE + unsharp masking. At this text size
    the filtering erased thin strokes: 25 phones found versus 7 on the same
    photo. There is no preprocessing chain here on purpose.
  * PSM 11 (sparse text) beats PSM 6 (uniform block) roughly fourfold when
    scanning a whole page — a spreadsheet is not a block of prose. But once a
    single column has been cropped out, it IS a uniform block, and PSM 6 wins.
  * An isolated column strip vastly outperforms reading a multi-column region,
    which is why stages 3 and 4 mirror stages 1 and 2 rather than reading the
    whole right-hand side of the sheet at once.

Why the name column sits at the right: these sheets are right-to-left, so
"שם מלא" is column A or B and every other Hebrew column (role, unit, rank,
status, notes) falls to its left. Picking the longest or most confident Hebrew
cell instead reliably picks a role like "לוחם ימי גברים" over the actual name.
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("extractor")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_SOURCE_WIDTH = 4000       # cap the original, for memory
LOCATE_WIDTH = 1800           # the cheap first pass runs here
TARGET_DIGIT_PX = 30          # Tesseract's comfort zone for digit height
TARGET_NAME_PX = 48           # measured: 44-56 clearly beats 36 on faint text
MAX_CROP_PIXELS = 3_000_000   # ceiling on any single enlarged crop

TIME_BUDGET_SECONDS = float(os.getenv("TIME_BUDGET_SECONDS", "55"))

NAME_LANG = os.getenv("NAME_LANG", "heb")
DIGITS_ONLY = "0123456789-"

# Overridable so the column-finding logic can be exercised without Hebrew
# traineddata installed.
LETTERS = re.compile(os.getenv("LETTER_CLASS", r"[\u05D0-\u05EA]"))
NON_NAME = re.compile(os.getenv("NON_NAME_CLASS",
                                r"[^\u05D0-\u05EA\u0027\u0022\s\-׳״]"))
NAME_HEADER = re.compile(os.getenv("NAME_HEADER", r"מלא|^שם$"))

# Column headings and stock cell values that are never a person's name.
NOT_A_NAME = {
    "שם", "שמות", "מלא", "טלפון", "נייד", "פלאפון", "מספר", "אישי", "תז",
    "משפחה", "פרטי", "איש", "קשר", "כתובת", "מייל", "תפקיד", "רמה", "סוג",
    "גיל", "דרגה", "סמכות", "יחידה", "הערות", "סטטוס", "שעת", "זימון",
    "מתחקר", "המתחקר", "שבוע", "בדיקה", "התייצבות", "במגן",
}

app = FastAPI(title="Hebrew contact extractor", version="3.0")
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


def split_cells(row: list[Word]) -> list[list[Word]]:
    """Split a row into cells. Words a space apart belong together; a gap of
    more than about 1.6 line-heights means a new column."""
    if not row:
        return []
    gap_limit = float(np.median([w.height for w in row])) * 1.6
    cells = [[row[0]]]
    for prev, current in zip(row, row[1:]):
        if prev.x - current.right > gap_limit:
            cells.append([current])
        else:
            cells[-1].append(current)
    return cells


def clean_name(cell: list[Word]) -> tuple[str, float] | None:
    parts, confs = [], []
    for w in cell:
        cleaned = NON_NAME.sub("", w.text).strip()
        if len(cleaned) < 2 or cleaned in NOT_A_NAME:
            continue
        parts.append(cleaned)
        confs.append(w.conf)
    if not parts:
        return None
    return " ".join(parts[:4]), float(np.mean(confs))


# --------------------------------------------------------------------------
# Stage 1 — locate the phone column
# --------------------------------------------------------------------------

def locate_phones(gray: np.ndarray) -> list[Word]:
    """Cheap wide pass, digits only. We don't trust the numbers it returns —
    only where on the page they are."""
    return [w for w in read_words(gray, psm=11, lang="eng", whitelist=DIGITS_ONLY)
            if normalize_phone(w.text)]


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
                      top: int, bottom: int,
                      scale: float) -> list[tuple[float, str, float]]:
    """Returns (y in source coordinates, phone, confidence)."""
    x0, x1 = band
    crop = source[top:bottom, x0:x1]
    if crop.size == 0:
        return []
    enlarged, applied = enlarge(crop, scale)

    out: list[tuple[float, str, float]] = []
    for word in read_words(enlarged, psm=11, lang="eng", whitelist=DIGITS_ONLY):
        phone = normalize_phone(word.text)
        if phone:
            out.append((top + word.y / applied, phone, word.conf))
    return out


# --------------------------------------------------------------------------
# Stage 3 — find the name column
# --------------------------------------------------------------------------

def survey_names(source: np.ndarray, left: int, top: int, bottom: int,
                 scale: float) -> list[Word]:
    """Wide-ish pass over the right-hand side, including the heading row, to
    work out where the name column is. Sparse mode, because this region still
    holds several columns."""
    crop = source[top:bottom, left:]
    if crop.size == 0:
        return []
    enlarged, applied = enlarge(crop, scale)
    try:
        words = read_words(enlarged, psm=11, lang=NAME_LANG)
    except pytesseract.TesseractError as exc:
        log.error("name survey failed: %s", exc)
        return []

    for w in words:
        w.x = int(left + w.x / applied)
        w.width = int(w.width / applied)
        w.y = top + w.y / applied
        w.height = int(w.height / applied)
    return [w for w in words if LETTERS.search(w.text)]


def band_from_header(words: list[Word], row_height: int,
                     page_width: int) -> tuple[int, int] | None:
    """Find the "שם מלא" heading and return its column's horizontal extent.

    This is the most reliable signal available: the sheet labels its own
    columns, so we read the label instead of inferring the layout.
    """
    for word in words:
        if not NAME_HEADER.search(word.text):
            continue
        siblings = [w for w in words if abs(w.y - word.y) <= row_height * 0.45]
        for cell in split_cells(sorted(siblings, key=lambda w: -w.x)):
            if word not in cell:
                continue
            left = min(w.x for w in cell)
            right = max(w.right for w in cell)
            pad = (right - left) * 0.25
            band = (max(2, int(left - pad)), min(page_width - 2, int(right + pad)))
            if 0.03 * page_width < band[1] - band[0] < 0.45 * page_width:
                return band
    return None


def widen_to_content(anchor: tuple[int, int], words: list[Word],
                     row_height: float, page_width: int) -> tuple[int, int]:
    """
    Grow a band derived from a heading until it covers the cells beneath it.

    A heading label is usually narrower than its column — "שם מלא" is shorter
    than most full names — so cropping to the label alone slices the ends off
    long entries. The heading tells us WHICH column; the rows tell us how wide
    it really is. Percentiles rather than extremes, so one row that reads badly
    can't stretch the band into the neighbouring column.
    """
    lefts, rights = [], []
    for row in group_rows(words, tolerance=row_height * 0.45):
        for cell in split_cells(row):
            left = min(w.x for w in cell)
            right = max(w.right for w in cell)
            if right > anchor[0] and left < anchor[1]:      # overlaps the heading
                lefts.append(left)
                rights.append(right)
                break

    if len(lefts) < 3:
        return anchor

    left = float(np.percentile(lefts, 5))
    right = float(np.percentile(rights, 95))
    pad = (right - left) * 0.08
    return (max(2, int(min(left - pad, anchor[0]))),
            min(page_width - 2, int(max(right + pad, anchor[1]))))


def detect_columns(words: list[Word], page_width: int,
                   row_height: float) -> list[tuple[int, int]]:
    """
    Find the sheet's column boundaries from every word at once.

    Splitting each row on its own was the bug behind names like "לוחם ימי":
    one row's word gaps are noisy, and a single row where two columns happen to
    sit close together merges them permanently. Columns are a property of the
    whole sheet, so project every word onto the x axis and look for the vertical
    corridors that no word crosses. Those corridors are the column rules.
    """
    if not words:
        return []
    occupied = np.zeros(page_width + 2, dtype=bool)
    for w in words:
        occupied[max(0, w.x):min(page_width, w.right) + 1] = True

    # a corridor narrower than this is just the space between two words
    min_corridor = max(6, int(row_height * 0.45))

    columns: list[tuple[int, int]] = []
    start = None
    gap = 0
    for x in range(page_width + 2):
        if occupied[x]:
            if start is None:
                start = x
            elif gap and gap < min_corridor:
                pass                       # word spacing: stay in this column
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_corridor:
                    columns.append((start, x - gap))
                    start, gap = None, 0
    if start is not None:
        columns.append((start, page_width))

    return [c for c in columns if c[1] - c[0] >= row_height * 0.4]


def cells_in_column(words: list[Word], column: tuple[int, int],
                    row_height: float) -> dict[int, tuple[float, str, float]]:
    """Everything inside one column, grouped into rows."""
    x0, x1 = column
    inside = [w for w in words if w.x >= x0 - 2 and w.right <= x1 + 2]
    out: dict[int, tuple[float, str, float]] = {}
    for row in group_rows(inside, tolerance=row_height * 0.45):
        got = clean_name(row)
        if got:
            out[round(row[0].y / (row_height * 0.5))] = (row[0].y, got[0], got[1])
    return out


def choose_name_column(words: list[Word], page_width: int, row_height: float,
                       header_anchor: tuple[int, int] | None
                       ) -> tuple[int, int] | None:
    """
    Pick the column holding the names.

    If the "שם מלא" heading was read, use whichever column contains it — the
    sheet labelling itself is the best evidence available. Otherwise take the
    rightmost column that actually holds name-shaped text in most rows, because
    these sheets are right-to-left and every other Hebrew column (role, unit,
    rank, status) sits to the left of the name.
    """
    columns = detect_columns(words, page_width, row_height)
    if not columns:
        return None

    if header_anchor:
        centre = (header_anchor[0] + header_anchor[1]) / 2
        for column in columns:
            if column[0] <= centre <= column[1]:
                return column

    best: tuple[int, int] | None = None
    for column in sorted(columns, key=lambda c: -c[1]):     # rightmost first
        filled = cells_in_column(words, column, row_height)
        if len(filled) >= 3:
            best = column
            break
    return best


def band_from_rows(words: list[Word], row_height: float,
                   page_width: int) -> tuple[int, int] | None:
    """Fallback when the heading can't be read: the rightmost cell of each data
    row. Medians, not extremes, so one row that reads badly can't stretch the
    band across neighbouring columns."""
    lefts, rights = [], []
    for row in group_rows(words, tolerance=row_height * 0.45):
        cells = split_cells(row)
        if not cells:
            continue
        rightmost = cells[0]
        if clean_name(rightmost):
            lefts.append(min(w.x for w in rightmost))
            rights.append(max(w.right for w in rightmost))

    if len(lefts) < 3:
        return None
    left, right = float(np.median(lefts)), float(np.median(rights))
    pad = (right - left) * 0.3
    return (max(2, int(left - pad)), min(page_width - 2, int(right + pad)))


# --------------------------------------------------------------------------
# Stage 4 — read the name column on its own
# --------------------------------------------------------------------------

def read_name_column(source: np.ndarray, band: tuple[int, int],
                     top: int, bottom: int,
                     scale: float) -> list[Word]:
    """
    One column, cropped and enlarged. Now that it really is a single block of
    text, PSM 6 applies — and it beats sparse mode here.

    Measured on a degraded fixture at 10px letter height (about what a distant
    phone photo gives): plain grayscale scored 20/25 and a mild unsharp mask
    22/25, while CLAHE dropped to 14 and Otsu to 15. So only those two variants
    are tried, and the stronger filtering that helps scanned documents is
    deliberately absent.
    """
    x0, x1 = band
    crop = source[top:bottom, x0:x1]
    if crop.size == 0:
        return []
    enlarged, applied = enlarge(crop, scale)
    sharpened = cv2.addWeighted(enlarged, 1.5,
                                cv2.GaussianBlur(enlarged, (0, 0), 1.2), -0.5, 0)

    best: list[Word] = []
    best_score = (-1, 0.0)
    for image, psm in ((enlarged, 6), (sharpened, 6), (enlarged, 11)):
        try:
            words = read_words(image, psm=psm, lang=NAME_LANG)
        except pytesseract.TesseractError as exc:
            log.error("name column read failed (psm %d): %s", psm, exc)
            return []
        keep = [w for w in words if LETTERS.search(w.text)]
        score = (len(keep), float(np.mean([w.conf for w in keep])) if keep else 0.0)
        if score > best_score:
            best, best_score = keep, score

    for w in best:
        w.x = int(x0 + w.x / applied)
        w.width = int(w.width / applied)
        w.y = top + w.y / applied
        w.height = int(w.height / applied)
    return best


def names_by_row(words: list[Word], row_height: float,
                 bonus: float = 0.0) -> dict[int, tuple[float, str, float]]:
    out: dict[int, tuple[float, str, float]] = {}
    for row in group_rows(words, tolerance=row_height * 0.45):
        cells = split_cells(row)
        if not cells:
            continue
        got = clean_name(cells[0])          # rightmost cell in the row
        if not got:
            continue
        name, conf = got
        key = round(row[0].y / (row_height * 0.5))
        if key not in out or conf + bonus > out[key][2]:
            out[key] = (row[0].y, name, conf + bonus)
    return out


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


def extract_contacts(source: np.ndarray,
                     manual_name: tuple[int, int] | None = None,
                     manual_phone: tuple[int, int] | None = None,
                     ) -> tuple[list[Contact], dict]:
    started = time.monotonic()
    H, W = source.shape
    info: dict = {"width": W, "height": H}

    # --- stage 1: locate the phone column ---------------------------------
    f = min(1.0, LOCATE_WIDTH / W)
    locate_img = (source if f == 1.0 else
                  cv2.resize(source, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    hits = locate_phones(locate_img)
    for w in hits:                                   # back to source coordinates
        w.x, w.width = int(w.x / f), int(w.width / f)
        w.y, w.height = w.y / f, int(w.height / f)

    if manual_phone:
        # keep only what falls in the column the user marked
        hits = [w for w in hits
                if manual_phone[0] - 10 <= (w.x + w.right) / 2 <= manual_phone[1] + 10]
        info["phone_band_source"] = "manual"

    info["located"] = len(hits)
    log.info("stage 1 located %d phone-shaped cells in %.1fs", len(hits),
             time.monotonic() - started)

    if len(hits) < 2 and manual_phone:
        # The wide pass found nothing usable, but we know where to look, so
        # bootstrap the row geometry from the marked column alone.
        boot = read_phone_column(source, manual_phone, 0, H, 2.0)
        spacing = float(np.median(np.diff(sorted(y for y, _, _ in boot)))) if len(boot) > 2 else 20.0
        hits = [Word(phone, manual_phone[0], y, max(8, int(spacing * 0.5)),
                     manual_phone[1] - manual_phone[0], conf)
                for y, phone, conf in boot]
        info["located"] = len(hits)
        info["phone_band_source"] = "manual-bootstrap"

    if len(hits) < 2:
        info["reason"] = "no phone column found"
        return [], info

    ys = sorted(w.y for w in hits)
    row_height = float(np.median(np.diff(ys))) if len(ys) > 2 else hits[0].height * 2
    row_height = max(row_height, hits[0].height * 1.2)

    # Pad generously. The wide pass routinely misses the first and last rows,
    # and if the crop stops at the last row it found, the high-resolution pass
    # can never recover them.
    pad = row_height * 3
    rows_top = max(0, int(min(ys) - pad))
    rows_bottom = min(H, int(max(ys) + pad))
    digit_scale = float(np.clip(TARGET_DIGIT_PX / max(6.0, row_height * 0.5), 1.0, 3.0))
    name_scale = float(np.clip(TARGET_NAME_PX / max(6.0, row_height * 0.5), 1.0, 4.0))
    band = manual_phone or phone_band(hits, W)
    letter_px = int(np.median([w.height for w in hits]))
    info.update({"row_height": round(row_height, 1), "phone_band": list(band),
                 "letter_px": letter_px})
    if letter_px < 15:
        # below roughly this size Hebrew stops being recoverable, while the
        # digit whitelist keeps phones working - worth saying so explicitly
        info["warning"] = "text_too_small_for_names"
        log.warning("letters are only %dpx tall - Hebrew names will be unreliable", letter_px)

    # --- stage 2: read the phone column -----------------------------------
    # Neither pass dominates: on one test photo the wide pass found 25 rows and
    # the strip 8; on another the strip found 19 and the wide 15. So keep both
    # and let them vote per row. The strip gets a small bonus because when the
    # two disagree it has the resolution advantage.
    t = time.monotonic()
    strip = read_phone_column(source, band, rows_top, rows_bottom, digit_scale)
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

    # --- stage 3: locate the name column ----------------------------------
    names: dict[int, tuple[float, str, float]] = {}
    name_band = None

    if time.monotonic() - started < TIME_BUDGET_SECONDS:
        t = time.monotonic()
        # start above the data rows so the heading row is inside the crop
        survey_top = max(0, int(rows_top - row_height * 2))
        survey = survey_names(source, 0, survey_top, rows_bottom, digit_scale)
        survey = [w for w in survey if w.x >= band[1] - row_height]

        anchor = manual_name or band_from_header(survey, row_height, W)
        if manual_name:
            info["band_source"] = "manual"
        elif anchor:
            info["band_source"] = "header"

        # A hand-drawn band is only an anchor: it snaps to whichever detected
        # column it lands in, so the stroke doesn't have to be accurate.
        column = choose_name_column(survey, W, row_height, anchor)
        if column:
            pad = int(row_height * 0.5)
            name_band = (max(2, column[0] - pad), min(W - 2, column[1] + pad))
            info.setdefault("band_source", "columns")
        elif anchor:
            name_band = widen_to_content(anchor, survey, row_height, W)
        else:
            info["band_source"] = "none"

        log.info("stage 3 surveyed %d words, %d columns, name band %s (%s) in %.1fs",
                 len(survey), len(detect_columns(survey, W, row_height)), name_band,
                 info.get("band_source"), time.monotonic() - t)

        if name_band:
            names = cells_in_column(survey, name_band, row_height)
        else:
            names = names_by_row(survey, row_height)

    # --- stage 4: read the name column on its own -------------------------
    if name_band and time.monotonic() - started < TIME_BUDGET_SECONDS:
        t = time.monotonic()
        column = read_name_column(source, name_band, rows_top, rows_bottom, name_scale)
        log.info("stage 4 read %d words from the name column in %.1fs",
                 len(column), time.monotonic() - t)
        for key, value in names_by_row(column, row_height, bonus=8).items():
            if key not in names or value[2] > names[key][2]:
                names[key] = value

    info["names_found"] = len(names)
    info["name_band"] = list(name_band) if name_band else None

    # --- pair by vertical position ----------------------------------------
    name_rows = sorted(names.values())
    contacts: dict[str, Contact] = {}
    for y, phone, phone_conf in phones:
        name, name_conf = "", 40.0
        best_gap = row_height * 0.6
        for ny, candidate, conf in name_rows:
            gap = abs(ny - y)
            if gap < best_gap:
                name, name_conf, best_gap = candidate, conf, gap

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


def parse_band(raw: str | None, width: int) -> tuple[int, int] | None:
    """A band arrives as two fractions of the image width, "0.61,0.78", so it
    survives the client resizing the photo before upload."""
    if not raw:
        return None
    try:
        a, b = (float(v) for v in raw.split(",")[:2])
    except ValueError:
        return None
    x0, x1 = sorted((a, b))
    if not (0.0 <= x0 < x1 <= 1.0) or x1 - x0 < 0.005:
        return None
    return max(2, int(x0 * width)), min(width - 2, int(x1 * width))


@app.post("/extract")
async def extract(image: UploadFile = File(...),
                  name_band: str | None = Form(None),
                  phone_band: str | None = Form(None)) -> dict:
    data = await image.read()
    if not data:
        raise HTTPException(400, "No image was uploaded.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That image is larger than 12 MB.")

    started = time.monotonic()
    source = load_image(data)
    contacts, info = extract_contacts(
        source,
        manual_name=parse_band(name_band, source.shape[1]),
        manual_phone=parse_band(phone_band, source.shape[1]),
    )
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