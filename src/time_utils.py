import re
from datetime import datetime, timedelta
import dateparser
from dateutil.relativedelta import relativedelta

MONTH_DAYS = 30
YEAR_DAYS = 365

NUMBER_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}

_NUM = r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)"
_IMMEDIATE_WORDS = {"immediately", "right away", "asap", "as soon as possible", "now", "today"}

def _words_to_digits(text: str) -> str:
    out = text
    for w, d in NUMBER_MAP.items():
        out = re.sub(rf"\b{w}\b", str(d), out)
    return out

def normalize_time_text(text: str) -> str:
    if not text:
        return text
    low = text.lower().strip()
    low = re.sub(r"^[\s:;,\-–—]+", "", low)
    low = re.sub(r"[\s:;,\.!\?]+$", "", low)
    low = low.replace("follow-up", "follow up").replace("followup", "follow up")
    low = re.sub(r"\bfollow\s+up\b", "follow up", low)
    low = re.sub(r"\bmos?\b", "month", low)
    low = re.sub(r"\bmo\b", "month", low)
    low = re.sub(r"\bmths?\b", "month", low)
    low = re.sub(r"\bwks?\b", "week", low)
    low = re.sub(r"\bhrs?\b", "hour", low)
    low = re.sub(r"\byrs?\b", "year", low)
    low = re.sub(r"\byr\b", "year", low)
    low = low.replace("-", " ")
    low = _words_to_digits(low)
    low = re.sub(r"\s+time$", "", low)

    m = re.fullmatch(rf"({_NUM})\s+(day|week|month|year)s?\s+follow\s+up", low)
    if m: return f"in {m.group(1)} {m.group(2)}s"

    m = re.fullmatch(rf"({_NUM})\s+(day|week|month|year)s?", low)
    if m: return f"in {m.group(1)} {m.group(2)}s"

    m = re.fullmatch(rf"within\s+({_NUM})\s+(day|week|month|year)s?", low)
    if m: return f"in {m.group(1)} {m.group(2)}s"

    m = re.fullmatch(rf"over\s+the\s+next\s+({_NUM})\s+(day|week|month|year)s?", low)
    if m: return f"in {m.group(1)} {m.group(2)}s"

    m = re.fullmatch(rf"in\s+(about|approximately)\s+({_NUM})\s+(day|week|month|year)s?", low)
    if m: return f"in {m.group(2)} {m.group(3)}s"

    m = re.fullmatch(rf"(about|approximately)\s+({_NUM})\s+(day|week|month|year)s?", low)
    if m: return f"in {m.group(2)} {m.group(3)}s"

    low = re.sub(rf"\b({_NUM})\s+(day|week|month|year)\b", r"\1 \2s", low)

    if re.match(rf"^({_NUM})\s+(days|weeks|months|years)\b", low) and not low.startswith("in "):
        low = "in " + low

    return low.strip()

def time_text_to_days_offset(period_text: str):
    if not period_text:
        return None
    t = normalize_time_text(period_text).casefold().strip()

    if any(w in t for w in _IMMEDIATE_WORDS): return 0
    if re.search(r"\btomorrow\b", t): return 1
    if re.search(r"\bday after tomorrow\b", t): return 2

    t = re.sub(r"\ban?\b", "1", t)
    t = re.sub(r"\bwks\b", "weeks", t)
    t = re.sub(r"\bwk\b", "week", t)
    t = re.sub(r"\bmos\b", "months", t)
    t = re.sub(r"\bmo\b", "month", t)
    t = re.sub(r"\byrs\b", "years", t)
    t = re.sub(r"\byr\b", "year", t)
    t = re.sub(r"\bds\b", "days", t)
    t = re.sub(r"\bd\b", "day", t)

    m = re.search(r"(\d+)\s*(day|days|week|weeks|month|months|year|years)\b", t)
    if not m:
        m = re.search(r"(?:in|within|after|for)\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)\b", t)
    
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit.startswith("day"): return n
        if unit.startswith("week"): return 7 * n
        if unit.startswith("month"): return MONTH_DAYS * n
        if unit.startswith("year"): return YEAR_DAYS * n

    m = re.search(r"\bx\s*(\d+)\s*(w|wk|week|m|mo|month|d|day|y|yr|year)\b", t)
    if m:
        n, u = int(m.group(1)), m.group(2)
        if u in {"d", "day"}: return n
        if u in {"w", "wk", "week"}: return 7 * n
        if u in {"m", "mo", "month"}: return MONTH_DAYS * n
        if u in {"y", "yr", "year"}: return YEAR_DAYS * n

    if re.search(r"\bnext\s+week\b", t): return 7
    if re.search(r"\bnext\s+month\b", t): return MONTH_DAYS
    if re.search(r"\bnext\s+year\b", t): return YEAR_DAYS

    return None

def _duration_from_text(text: str):
    if not text:
        return None
    low = _words_to_digits(text.lower().strip())
    m = re.search(rf"\b(?:in|within|over the next)\s+({_NUM})\s+(day|week|month|year)s?\b", low)
    if not m:
        m = re.search(rf"\b({_NUM})\s+(day|week|month|year)s?\b", low)
    if m:
        return int(m.group(1)), m.group(2)
    return None

def parse_period_date(period_text: str, visit_date: str) -> str | None:
    if not period_text or not visit_date:
        return None
    try:
        base = datetime.strptime(visit_date, "%Y-%m-%d")
    except Exception:
        return None

    days = time_text_to_days_offset(period_text)
    if days is not None:
        return (base + timedelta(days=int(days))).strftime("%Y-%m-%d")

    settings = {"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base, "DATE_ORDER": "DMY"}
    
    dt = dateparser.parse(period_text, settings=settings)
    if dt: return dt.strftime("%Y-%m-%d")

    norm = normalize_time_text(period_text)
    if norm and norm != period_text:
        dt = dateparser.parse(norm, settings=settings)
        if dt: return dt.strftime("%Y-%m-%d")

    dur = _duration_from_text(norm or period_text)
    if dur:
        n, unit = dur
        if unit == "day": out = base + timedelta(days=n)
        elif unit == "week": out = base + timedelta(weeks=n)
        elif unit == "month": out = base + relativedelta(months=n)
        elif unit == "year": out = base + relativedelta(years=n)
        else: return None
        return out.strftime("%Y-%m-%d")
    return None