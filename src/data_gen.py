import asyncio
import json
import os
import random
import re
from datetime import datetime
from typing import Any, Optional, Tuple, List
import pandas as pd
import dateparser
from faker import Faker
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
import nest_asyncio

# =========================
# CONFIGURATION
# =========================
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOTAL_SAMPLES = int(os.getenv("TOTAL_SAMPLES", "2000"))
HARD_TEST = bool(int(os.getenv("HARD_TEST", "1")))  
HARD_TOTAL_SAMPLES = int(os.getenv("HARD_TOTAL_SAMPLES", "200"))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "450"))

TEMP_LOW, TEMP_HIGH = 0.45, 0.65
ACTION_COUNT_PROBS = {0: 0.25, 1: 0.50, 2: 0.25}
MAX_REGEN_ATTEMPTS = 8

if HARD_TEST:
    TOTAL_SAMPLES = HARD_TOTAL_SAMPLES
    ACTION_COUNT_PROBS = {0: 0.10, 1: 0.35, 2: 0.55}
    MAX_REGEN_ATTEMPTS = 12
    MAX_TOKENS = 520
    TEMP_HIGH = 0.75

SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# Require API key from environment
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("You must set the OPENAI_API_KEY environment variable before running this script.")

client = AsyncOpenAI()
nest_asyncio.apply()

# --- NOTE: For brevity in this instruction, paste all your dictionaries here ---
# (TARGET_SPECIALTIES, SPECIALTY_MAPPING, TOPICS, STYLE_FEATURES_DICT, PLAN_HEADERS)
NO_HEADER_TOKEN = "__NO_HEADER__"

TARGET_SPECIALTIES = [
    "Orthopedic",
    "Cardiovascular / Pulmonary",
    "Gastroenterology",
    "Neurology",
    "General Medicine",
]

SPECIALTY_MAPPING = {
    "Orthopedic": [
        "MRI", "Physical Therapy", "CT Scan", "X-Ray", "Orthopedic Consult", "Joint Injection"
    ],
    "Cardiovascular / Pulmonary": [
        "Echocardiogram", "Stress Test", "Holter Monitor", "Pulmonary Function Test",
        "Cardiac MRI", "Cardiology Consult"
    ],
    "Gastroenterology": [
        "Endoscopy", "Colonoscopy", "Stool Antigen Test", "Abdominal Ultrasound",
        "GI Consult", "Breath Test"
    ],
    "Neurology": [
        "CT Scan", "EEG", "MRI Brain", "EMG", "Neurology Consult", "Sleep Study"
    ],
    "General Medicine": [
        "Blood Test", "X-Ray", "Vaccination", "Annual Physical", "Urinalysis", "Lipid Panel"
    ]
}

TOPICS = {
    "Orthopedic": [
        "ACL Tear", "Rotator Cuff Injury", "Lumbar Herniated Disc", "Ankle Sprain",
        "Carpal Tunnel Syndrome", "Hip Osteoarthritis", "Meniscus Tear", "Tennis Elbow",
        "Plantar Fasciitis", "Distal Radius Fracture",
        "Sciatica", "Scoliosis", "Bunion (Hallux Valgus)", "Patellar Tendonitis",
        "Shoulder Dislocation"
    ],
    "Cardiovascular / Pulmonary": [
        "Atrial Fibrillation", "COPD Exacerbation", "Acute Bronchitis", "Hypertension",
        "Mitral Valve Prolapse", "Pneumonia", "Congestive Heart Failure", "Deep Vein Thrombosis",
        "Asthma Attack", "Pericarditis",
        "Pulmonary Embolism", "Coronary Artery Disease", "Aortic Stenosis", "Bradycardia",
        "Syncope/Fainting"
    ],
    "Gastroenterology": [
        "GERD", "Irritable Bowel Syndrome", "Crohn's Disease", "Ulcerative Colitis",
        "Gallstones", "Celiac Disease", "Peptic Ulcer", "Diverticulitis",
        "Hemorrhoids", "Liver Cirrhosis",
        "Barrett's Esophagus", "Acute Gastritis", "Hiatal Hernia", "Chronic Pancreatitis",
        "Hepatitis C"
    ],
    "Neurology": [
        "Migraine with Aura", "Epilepsy/Seizure", "Multiple Sclerosis", "Parkinson's Disease",
        "Ischemic Stroke", "Carpal Tunnel (Neuro view)", "Vertigo/BPPV", "Alzheimer's/Dementia",
        "Bell's Palsy", "Neuropathy",
        "Myasthenia Gravis", "Trigeminal Neuralgia", "Huntington's Disease", "Guillain-Barre Syndrome",
        "Restless Leg Syndrome"
    ],
    "General Medicine": [
        "Type 2 Diabetes", "Seasonal Influenza", "Hypothyroidism", "Urinary Tract Infection",
        "Anemia", "Vitamin D Deficiency", "Hypertension", "Annual Physical",
        "Lyme Disease", "Gout",
        "Hyperlipidemia (High Cholesterol)", "Fibromyalgia", "Chronic Fatigue Syndrome", "Infectious Mononucleosis",
        "Osteoporosis"
    ]
}

STYLE_FEATURES_DICT = {
    "Narrative Voice": [
        "First Person ('I examined the patient', 'I recommend')",
        "Third Person ('Patient presents with', 'It is recommended')"
    ],
    "Temporal Precision": [
        "Vague ('History of surgery years ago', 'pain for a while')",
        "Precise ('Surgery on 12/05/2020', 'pain started at 2 PM')"
    ],
    "Certainty": [
        "Definitive ('Patient has pneumonia')",
        "Hedging/Uncertain ('Findings suggestive of possible pneumonia', 'cannot rule out')"
    ],
    "Formality": [
        "Formal/Academic (Complete sentences, proper grammar)",
        "Casual/Direct (Conversational, simple sentence structures)",
        "Standard Clinical (Professional but concise)"
    ],
    "Detail Level": [
        "Highly Detailed/Verbose (Explains rationale, describes scene)",
        "Abbreviated/Telegraphic (Notes style, fragments)",
        "Standard (Balanced)"
    ],
    "Tone": [
        "Polite/Empathetic ('Patient is a pleasant 45yo...')",
        "Clinical/Detached (Just the facts)",
        "Direct/Urgent ('Patient in distress, immediate action required')"
    ],
    "Terminology": [
        "Simple Language (Patient-friendly terms like 'heart attack')",
        "Heavy Medical Jargon (terms like 'myocardial infarction')",
        "Mixed (Standard EHR style)"
    ],
    "Structure": [
        "Standard SOAP Headers (Subjective: ... Objective: ...)",
        "Minimal Headers (HPI: ... PE: ... Imp: ...)",
        "Run-on Paragraph (No clear section breaks, one block of text)",
        "Bullet Points (Heavy use of lists for symptoms/plan)"
    ],
    "Imperfections": [
        "Perfect Grammar (textbook quality)",
        "Slight Shorthand (Standard abbreviations like 'pt', 'yo', 'hx')",
        "Heavy Shorthand (Aggressive abbrv: 'c/o CP, SOB, N/V, rec MRI')",
        "Dictation Style (Occasional missing punctuation, run-on sentences)",
        "Minor Typos (Simulated fast typing errors like 'pateint' or 'swelng')"
    ],
    "Clinician Persona": [
        "Defensive Medicine (Over-explaining rationale to justify decisions)",
        "Action-Oriented (Brief history, very detailed plan)",
        "Burnout/Hasty (Minimum viable documentation, very brief)",
        "Educator (Explaining the 'why' behind the diagnosis)"
    ]
}
PLAN_HEADERS = ["PLAN:", "RECOMMENDATIONS:", "ASSESSMENT & PLAN:", "INSTRUCTIONS:"]
NEG_PLAN_HEADERS = ["PLAN:", "RECOMMENDATIONS:", "INSTRUCTIONS:", "ASSESSMENT & PLAN:", None]
NEG_PLAN_WEIGHTS = [0.08, 0.07, 0.07, 0.08, 0.70]
# (Paste the time phrase logic: NUM_WORD, TIME_TEMPLATE_WEIGHTS, normalize_time_phrase, sample_period_text)

# =========================
# 1) TIME PHRASES
# =========================

NUM_WORD = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen"
}

TIME_TEMPLATE_WEIGHTS = [
    ("in {n} {unit}", 1.8),
    ("return in {n} {unit}", 1.2),
    ("follow up in {n} {unit}", 1.2),
    ("within {n} {unit}", 2.0),
    ("over the next {n} {unit}", 2.0),
    ("in about {n} {unit}", 1.8),
    ("in approximately {n} {unit}", 1.6),
    ("{n}-{unit_singular} follow-up", 2.0),
    ("{n} {unit_abbrev}", 1.6),
    ("in {n} {unit} time", 1.0),
]


# Extra shorthand/variant time expressions for the HARD test set.
HARD_TEST_TIME_TEMPLATES = [
    ("{n}wk f/u", 1.2),
    ("{n} wks f/u", 1.0),
    ("{n}{unit_abbrev} f/u", 1.0),
    ("f/u in {n} {unit}", 1.1),
    ("in {n}{unit_abbrev}", 1.0),          # e.g., in 6wks / in 3mos
    ("in ~{n} {unit}", 0.9),               # e.g., in ~8 weeks
    ("in approx. {n} {unit}", 0.9),
    ("in about {n}{unit_abbrev}", 0.8),
    ("in {n}-{m} weeks", 0.9),             # range
    ("in {n} to {m} weeks", 0.8),          # range
]
if HARD_TEST:
    TIME_TEMPLATE_WEIGHTS = TIME_TEMPLATE_WEIGHTS + HARD_TEST_TIME_TEMPLATES
_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen)"

def normalize_time_phrase(s: str) -> str:
    if not s:
        return s
    raw = s.strip()
    low = raw.lower().strip()

    m = re.fullmatch(rf"({_NUM})\s+(wk|wks)", low)
    if m: return f"in {m.group(1)} weeks"
    m = re.fullmatch(rf"({_NUM})\s+(mo|mos)", low)
    if m: return f"in {m.group(1)} months"

    m = re.fullmatch(rf"({_NUM})-(day|week|month)\s+follow[- ]?up", low)
    if m:
        n, unit = m.group(1), m.group(2)
        return f"in {n} {unit}s"

    m = re.fullmatch(rf"in\s+({_NUM})\s+(day|days|week|weeks|month|months)\s+time", low)
    if m: return f"in {m.group(1)} {m.group(2)}"

    m = re.fullmatch(rf"within\s+({_NUM})\s+(day|days|week|weeks|month|months)", low)
    if m: return f"in {m.group(1)} {m.group(2)}"

    m = re.fullmatch(rf"over the next\s+({_NUM})\s+(day|days|week|weeks|month|months)", low)
    if m: return f"in {m.group(1)} {m.group(2)}"

    m = re.fullmatch(rf"in\s+(about|approximately)\s+({_NUM})\s+(day|days|week|weeks|month|months)", low)
    if m: return f"in {m.group(2)} {m.group(3)}"

    return raw

def parse_relative_date(period_text: str, visit_date_str: str) -> str | None:
    if not period_text or not visit_date_str:
        return None
    try:
        base = datetime.strptime(visit_date_str, "%Y-%m-%d")
    except Exception:
        return None

    settings = {"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base, "DATE_ORDER": "DMY"}

    dt = dateparser.parse(period_text, settings=settings)
    if dt:
        return dt.strftime("%Y-%m-%d")

    norm = normalize_time_phrase(period_text)
    if norm != period_text:
        dt2 = dateparser.parse(norm, settings=settings)
        if dt2:
            return dt2.strftime("%Y-%m-%d")

    return None

_last_templates: list[str] = []

def sample_period_text() -> str:
    unit_type = random.choice(["days", "weeks", "months"])
    if unit_type == "days":
        n = random.randint(1, 14)
        unit_singular, unit_abbrev = "day", "days"
    elif unit_type == "weeks":
        n = random.randint(1, 12)
        unit_singular, unit_abbrev = "week", "wks"
    else: # months
        n = random.randint(1, 12)
        unit_singular, unit_abbrev = "month", "mos"

    unit = unit_singular if n == 1 else unit_singular + "s"
    n_str = NUM_WORD[n] if (random.random() < 0.35 and n in NUM_WORD) else str(n)

    templates, weights = zip(*TIME_TEMPLATE_WEIGHTS)
    weights = list(weights)

    for i, t in enumerate(templates):
        if t in _last_templates:
            weights[i] *= 0.45

    template = random.choices(list(templates), weights=weights, k=1)[0]
    _last_templates.append(template)
    if len(_last_templates) > 6:
        _last_templates.pop(0)

    # Prepare format arguments dynamically
    format_kwargs = {
        'n': n_str,
        'unit': unit,
        'unit_singular': unit_singular,
        'unit_abbrev': unit_abbrev
    }

    if '{m}' in template:
        # If a range template is chosen, ensure unit_type is 'weeks' and n is appropriate.
        # This will prevent the ValueError when n=14.
        unit_type = "weeks"
        n = random.randint(1, 12) # Max n for weeks is 12
        unit_singular, unit_abbrev = "week", "wks"
        unit = unit_singular if n == 1 else unit_singular + "s"
        n_str = NUM_WORD[n] if (random.random() < 0.35 and n in NUM_WORD) else str(n)

        # Update format_kwargs for the potentially re-generated n and unit
        format_kwargs['n'] = n_str
        format_kwargs['unit'] = unit
        format_kwargs['unit_singular'] = unit_singular
        format_kwargs['unit_abbrev'] = unit_abbrev

        # Generate a second number for range-based templates
        # m_val will now be safe as n max is 12 here
        m_val = random.randint(n + 1, min(n + 5, 14))
        m_str = NUM_WORD[m_val] if (random.random() < 0.35 and m_val in NUM_WORD) else str(m_val)
        format_kwargs['m'] = m_str

    return template.format(**format_kwargs)

def sample_period_text_with_date(visit_date: str, tries: int = 12) -> Tuple[str, str]:
    for _ in range(tries):
        pt = sample_period_text()
        pd = parse_relative_date(pt, visit_date)
        if pd is not None:
            return pt, pd
    fallback_pt = "in 7 days"
    fallback_pd = parse_relative_date(fallback_pt, visit_date)
    return fallback_pt, fallback_pd

# =========================
# 2) STYLE + PLAN VARIANT
# =========================

PLAN_VARIANTS = ["A", "B", "C", "D", "E", "F"]
PLAN_VARIANT_WEIGHTS = [1.2, 1.2, 1.0, 1.0, 1.0, 1.0]  # A/B slightly more common

def sample_style_features(min_k=3, max_k=5) -> list[dict[str, str]]:
    keys = list(STYLE_FEATURES_DICT.keys())
    k = random.randint(min_k, max_k)
    chosen = random.sample(keys, k=k)
    return [{"key": key, "value": random.choice(STYLE_FEATURES_DICT[key])} for key in chosen]

# (Paste the skeleton and prompt generation: make_skeleton, build_prompt)

# =========================
# 3) SKELETON
# =========================

def weighted_choice(probs: dict[int, float]) -> int:
    r = random.random()
    cum = 0.0
    for k in sorted(probs.keys()):
        cum += probs[k]
        if r <= cum:
            return k
    return 1

def choose_negative_header():
    return random.choices(NEG_PLAN_HEADERS, weights=NEG_PLAN_WEIGHTS, k=1)[0]

def reset_action_spans(sk: dict[str, Any]) -> None:
    for a in sk.get("actions_gt", []):
        a["time_char_start"] = None
        a["time_char_end"] = None
        a["action_char_start"] = None
        a["action_char_end"] = None

def make_skeleton() -> dict[str, Any]:
    specialty = random.choice(TARGET_SPECIALTIES)
    topic = random.choice(TOPICS[specialty])
    visit_date_obj = fake.date_between(start_date="-2y", end_date="today")
    visit_date = visit_date_obj.strftime("%Y-%m-%d")

    num_actions = weighted_choice(ACTION_COUNT_PROBS)
    plan_header = choose_negative_header() if num_actions == 0 else random.choice(PLAN_HEADERS)
    style_features = sample_style_features()
    plan_variant = random.choices(PLAN_VARIANTS, weights=PLAN_VARIANT_WEIGHTS, k=1)[0]

    actions_gt = []
    if num_actions > 0:
        chosen_actions = random.sample(SPECIALTY_MAPPING[specialty], k=num_actions)

        used_period_texts = set()
        for act in chosen_actions:
            pt, pd = sample_period_text_with_date(visit_date)

            if num_actions == 2:
                tries = 0
                while pt in used_period_texts and tries < 10:
                    pt, pd = sample_period_text_with_date(visit_date)
                    tries += 1

            used_period_texts.add(pt)

            actions_gt.append({
                "action": act,
                "period_text": pt,
                "period_date": pd,
                "time_char_start": None,
                "time_char_end": None,
                "action_char_start": None,
                "action_char_end": None,
            })

    return {
        "specialty": specialty,
        "topic": topic,
        "visit_date": visit_date,
        "plan_header": plan_header,
        "style_features": style_features,
        "plan_variant": plan_variant,
        "num_actions": num_actions,
        "actions_gt": actions_gt,
    }

# =========================
# 4) PROMPT
# =========================

POSITIVE_PLAN_HINT = """
PLAN SECTION (REALISTIC + DIVERSE, BUT CONTROLLED)

You are given a list of SCHEDULED follow-up items (closed-set actions + their time phrases).
Your job: write a natural plan section that includes these items, with realistic wording/format.

HARD RULES:
1) The plan MUST contain EXACTLY {num_actions} scheduled follow-up items from the CLOSED LIST below.
2) Do NOT introduce any additional scheduled tests/imaging/labs/referrals/consults/therapy/procedures beyond the CLOSED LIST.
3) Each scheduled ACTION STRING must appear EXACTLY ONCE in the plan section (case-insensitive ok; spelling must match).
4) Each scheduled TIME PHRASE must appear EXACTLY ONCE in the plan section (copy exactly; no paraphrase; no truncation).
5) Action and time do NOT need to be adjacent. Time may appear before action or after action.
6) Do NOT repeat any scheduled action or scheduled time elsewhere in the note.

REALISM / HARDENING:
- Include 1–2 extra NON-scheduled time expressions somewhere in the note.
  These MUST be history/duration/past timing only (e.g., "for 3 days", "2 weeks ago", "since last month").
  Do NOT write extra future follow-up scheduling like "follow up in X", "return in X", "recheck in X" beyond the CLOSED LIST.

FORMAT VARIANTS:
You MUST follow the chosen variant letter: {variant}

A) Inline per item:
   - ACTION — TIME
B) Time-first per item:
   - TIME: ACTION
C) Mixed sentence:
   "Plan: TIME, ACTION; TIME2, ACTION2."
D) Parentheses:
   - ACTION (TIME)
E) Colon grouping:
   "Diagnostics: ACTION (TIME); ACTION2 (TIME2)."
F) Split association:
   - ACTION.
     TIME.

CLOSED LIST (must include each ACTION and each TIME exactly once; order is free):
{closed_list}

SELF-CHECK (silently):
- Exactly {num_actions} scheduled items.
- Each scheduled action appears exactly once in plan.
- Each scheduled time phrase appears exactly once in plan.
- No extra scheduled follow-up phrases beyond the list.
""".strip()


HARD_EXTRA_HINT = """
HARDENING (HARD TEST MODE)
In addition to the scheduled items above, add the following distractors to make extraction harder:
- Add 2-4 extra time expressions elsewhere in the note (NOT scheduled follow-ups).
  Mix:
  * at least 1 past time (e.g., "2 weeks ago", "for 3 days")
  * at least 1 future-looking non-scheduling time (e.g., "if not improved in 48 hours, go to ER")
  * at least 1 shorthand/abbrev time (e.g., "four mos", "8wk")
- Mention 1-2 other CLOSED-SET actions in a clearly past/negated context outside the plan
  (examples: "MRI last year was normal", "CT scan previously negative", "no need for colonoscopy today").
  Do NOT schedule these distractor actions.
- At least ONE scheduled item must have the time phrase and action separated by a sentence (not adjacent).
""".strip()

NEGATIVE_PLAN_HINT = """
IMPORTANT: NO SCHEDULED ACTIONS

HARD RULES:
- Do NOT schedule any future tests/imaging/labs/referrals/consults/therapy/procedures.
- Do NOT use future follow-up timing language (forbidden examples: "in 2 weeks", "return in", "recheck", "follow up in",
  "next week", "within 3 months", "over the next", "schedule", "order", "plan for").
- You MAY mention past timing in history (e.g., "symptoms started 3 days ago", "MRI last month") but it must clearly be past.

STRUCTURE:
- Write the note naturally.
- You may include a plan header or omit it (depending on skeleton).
- End with general advice/return precautions with NO future timing.
""".strip()

NO_SIGNATURE_RULE = """
FORMAT RESTRICTIONS:
- No greeting/letter style.
- No signature block, no clinician contact details.
""".strip()

def build_prompt(sk: dict[str, Any]) -> str:
    style_lines = "\n".join([f"- {s['key']}: {s['value']}" for s in sk["style_features"]])

    if sk["num_actions"] == 0:
        if sk.get("plan_header") is None:
            plan_instruction = f"""
{NEGATIVE_PLAN_HINT}

STRUCTURE:
- Do NOT add a plan header.
- Finish with 1–2 short general advice sentences.
""".strip()
        else:
            plan_instruction = f"""
{NEGATIVE_PLAN_HINT}

PLAN SECTION:
- Include the header line exactly: {sk["plan_header"]}
- After that header, write only general advice/return precautions (no future timing).
""".strip()
    else:
        closed_list = "\n".join([
            f'- ACTION: "{a["action"]}" | TIME: "{a["period_text"]}"'
            for a in sk["actions_gt"]
        ])
        plan_instruction = f"""
You MUST include a plan section near the end.

PLAN HEADER:
- Use this exact header line: {sk["plan_header"]}

{(POSITIVE_PLAN_HINT + ("\n\n" + HARD_EXTRA_HINT if HARD_TEST else "")).format(
    num_actions=sk["num_actions"],
    closed_list=closed_list,
    variant=sk["plan_variant"],
)}
""".strip()

    # Define length_words here
    length_words = "160–280 words" # Default value, can be made dynamic if needed

    prompt = f"""
You are an expert clinician writing a realistic outpatient note.

REQUIRED: Include this exact line ONE TIME at the beginning:
Date of Visit: {sk['visit_date']}

Context:
- Specialty: {sk['specialty']}
- Topic/Condition: {sk['topic']}

Style features (use naturally; do NOT list them explicitly):
{style_lines}

Length: {length_words}. If you need to shorten, shorten the narrative—not the plan lines.

{plan_instruction}

{NO_SIGNATURE_RULE}

Return plain text only (no Markdown).
""".strip()

    return prompt
# (Paste the API call logic: generate_note_once)

# =========================
# 5) API CALL
# =========================

def sample_temperature() -> float:
    return random.uniform(TEMP_LOW, TEMP_HIGH)

async def generate_note_once(sem: asyncio.Semaphore, sk: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    prompt = build_prompt(sk)
    last_err = None

    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Write realistic outpatient clinical notes. Follow constraints while keeping natural style."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=sample_temperature(),
                    max_tokens=MAX_TOKENS,
                )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("empty_response")
            return text, None
        except Exception as e:
            last_err = repr(e)
            await asyncio.sleep((2 ** attempt) * 0.6 + random.random() * 0.2)

    return None, last_err
# (Paste the span validation logic: assign_spans_in_note, get_all_known_actions)

# =========================
# 6) MATCHING HELPERS (fix overlaps like MRI vs MRI Brain)
# =========================

def _flex_pat_from_literal(lit: str, *, ignore_case: bool, word_boundary: bool) -> re.Pattern:
    """
    Turns a literal string into a regex that allows flexible whitespace.
    Optionally adds word boundaries (non-alnum boundaries) around the whole phrase.
    """
    escaped = re.escape(lit)
    escaped = escaped.replace(r"\\ ", r"\\s+")
    flags = re.IGNORECASE if ignore_case else 0

    if word_boundary:
        # non-alphanumeric boundaries (so "MRI" doesn't match inside "XMRIY")
        pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    else:
        pattern = escaped

    return re.compile(pattern, flags=flags)

def find_all_spans(pat: re.Pattern, text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in pat.finditer(text)]

def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])

def has_exact_visit_date_line(note: str, visit_date: str) -> bool:
    """
    Must appear exactly once and near the beginning.
    """
    if not note:
        return False
    required = f"Date of Visit: {visit_date}"
    if note.count(required) != 1:
        return False
    # also require it within first ~200 chars (top of note)
    return note.find(required) >= 0 and note.find(required) < 200

# =========================
# 7) VALIDATE + ASSIGN SPANS
# =========================

# Extra "future follow-up" phrases we DO NOT want beyond allowed list
BANNED_FOLLOWUP_STYLE_RE = re.compile(
    r"\\b(follow[- ]?up in|return in|recheck in|come back in|see you in|schedule (an|a)?|"
    r"book (an|a)?|set up (an|a)?|call (the )?office in)\\b",
    flags=re.IGNORECASE,
)

def get_all_known_actions() -> List[str]:
    all_known = []
    for acts in SPECIALTY_MAPPING.values():
        all_known.extend(acts)
    # longest first helps overlap handling conceptually
    all_known = sorted(set(all_known), key=lambda x: -len(x))
    return all_known

ALL_KNOWN_ACTIONS = get_all_known_actions()

def assign_spans_in_note(sk: dict[str, Any], note: str) -> Tuple[bool, dict[str, Any]]:
    """
    Option C (realistic/harder):
    - allow extra time expressions anywhere (including plan), BUT
    - forbid extra scheduled-followup phrasing beyond the closed list.
    - forbid extra CLOSED-SET action names in the plan section beyond allowed actions.
    - require each allowed action and each allowed period_text appear EXACTLY ONCE in plan section.
    """
    reset_action_spans(sk)

    span_info = {
        "plan_header_start_char": None,
        "plan_header_end_char": None,
        "plan_section_start_char": None,
        "plan_section_end_char": None,
        "span_error": "",
    }

    plan_header = sk.get("plan_header", None)

    # ---- Plan header / plan section bounds
    if plan_header is None:
        # No plan section exists
        if sk["num_actions"] > 0:
            span_info["span_error"] = "positive_but_no_plan_header"
            return False, span_info
        return True, span_info

    # Require header exactly once for clean spans
    if note.count(plan_header) != 1:
        span_info["span_error"] = "plan_header_not_exactly_once"
        return False, span_info

    hs = note.find(plan_header)
    he = hs + len(plan_header)
    span_info["plan_header_start_char"] = hs
    span_info["plan_header_end_char"] = he
    span_info["plan_section_start_char"] = he
    span_info["plan_section_end_char"] = len(note)

    plan_text = note[he:]

    # ---- Negatives: after header, only general advice, no future timing & no scheduling phrases
    if sk["num_actions"] == 0:
        # For negative-with-header: forbid future scheduling language in plan section
        if BANNED_FOLLOWUP_STYLE_RE.search(plan_text):
            span_info["span_error"] = "neg_plan_contains_future_followup_language"
            return False, span_info
        # Also forbid explicit "in X weeks/months" etc. (future-looking) after header
        future_like = re.search(r"\\b(in|within|over the next)\\s+\\w+\\s+(day|days|week|weeks|month|months)\\b", plan_text, re.I)
        if future_like:
            span_info["span_error"] = "neg_plan_contains_future_time_phrase"
            return False, span_info
        return True, span_info

    # ---- Positives: enforce allowed actions/times exactly once in plan section
    allowed_actions = [a["action"] for a in sk["actions_gt"]]
    allowed_times = [a["period_text"] for a in sk["actions_gt"]]

    # 1) Each allowed time phrase appears exactly once in plan_text AND exactly once in whole note
    allowed_time_spans_in_plan = []
    for t in allowed_times:
        tp = _flex_pat_from_literal(t, ignore_case=True, word_boundary=False)
        spans_plan = find_all_spans(tp, plan_text)
        spans_note = find_all_spans(tp, note)
        if len(spans_plan) != 1:
            span_info["span_error"] = f"allowed_time_not_exactly_once_in_plan::{t}"
            return False, span_info
        if len(spans_note) != 1:
            span_info["span_error"] = f"allowed_time_repeated_in_note::{t}"
            return False, span_info
        allowed_time_spans_in_plan.append(spans_plan[0])

    # 2) Each allowed action appears exactly once in plan_text
    allowed_action_spans_in_plan = {}
    for act in allowed_actions:
        ap = _flex_pat_from_literal(act, ignore_case=True, word_boundary=True)
        spans_plan = find_all_spans(ap, plan_text)
        if len(spans_plan) != 1:
            span_info["span_error"] = f"allowed_action_not_exactly_once_in_plan::{act}"
            return False, span_info
        allowed_action_spans_in_plan[act] = spans_plan[0]

    # 3) Forbid extra closed-set actions in plan section (handle overlaps via span overlap)
    # Build coverage of allowed action spans
    allowed_spans_list = list(allowed_action_spans_in_plan.values())

    extra_closed = False # Initialize extra_closed here
    for other_act in ALL_KNOWN_ACTIONS:
        if other_act in allowed_actions: # Corrected: check against allowed_actions
            continue
        op = _flex_pat_from_literal(other_act, ignore_case=True, word_boundary=True)
        spans_other = find_all_spans(op, plan_text)
        for sp in spans_other:
            # If this match is entirely inside an allowed action span, it's not "extra"
            if any(spans_overlap(sp, allowed_sp) for allowed_sp in allowed_spans_list):
                continue
            if sp:
                extra_closed = True
                break
        if extra_closed:
            break

    # If after checking all other_acts, an extra_closed was found, then it's an error
    if extra_closed:
        # The traceback indicated the error was at `if extra_closed: break` outside the inner loop.
        # The original code's intent was to return False, span_info if any extra_closed was found.
        # This block was missing after the loops finished.
        span_info["span_error"] = f"extra_closed_set_action_in_plan::{other_act}"
        return False, span_info

    # 4) Forbid extra future follow-up scheduling language beyond allowed list
    # Scrub allowed time strings then check banned followup phrases.
    scrubbed = plan_text
    for t in allowed_times:
        tp = _flex_pat_from_literal(t, ignore_case=True, word_boundary=False)
        scrubbed = tp.sub("<<<ALLOWED_TIME>>>", scrubbed)

    if BANNED_FOLLOWUP_STYLE_RE.search(scrubbed):
        span_info["span_error"] = "extra_future_followup_language_in_plan"
        return False, span_info

    # ---- Assign spans back to GT (global char indices)
    base = he
    for a in sk["actions_gt"]:
        act = a["action"]
        t = a["period_text"]

        as0, ae0 = allowed_action_spans_in_plan[act]
        tp = _flex_pat_from_literal(t, ignore_case=True, word_boundary=False)
        ts0, te0 = find_all_spans(tp, plan_text)[0]

        a["action_char_start"] = base + as0
        a["action_char_end"] = base + ae0
        a["time_char_start"] = base + ts0
        a["time_char_end"] = base + te0

    return True, span_info

# =========================
# 8) AUDIT (optional, now consistent w/ overlap)
# =========================

def _get_plan_section_text(row: dict) -> str:
    note = row.get("note_text", "") or ""
    s = row.get("plan_section_start_char", None)
    e = row.get("plan_section_end_char", None)
    if isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= len(note):
        return note[s:e]
    return ""  # no plan section

def audit_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    """
    Adds audit flags (informational). We do NOT enforce "extra_time_in_plan near 0"
    because Option C intentionally allows distractor times.
    """
    audit_extra_closed_action = []
    audit_extra_future_followup_language = []

    for _, row in df_batch.iterrows():
        plan_text = _get_plan_section_text(row.to_dict())
        num_actions = int(row.get("num_actions", 0))

        try:
            actions_gt = json.loads(row.get("actions_gt", "[]") or "[]")
        except Exception:
            actions_gt = []

        allowed_actions = [a.get("action", "") for a in actions_gt if a.get("action")]
        allowed_times = [a.get("period_text", "") for a in actions_gt if a.get("period_text")]

        # Scrub allowed times then check banned followup language
        scrubbed = plan_text
        for t in allowed_times:
            tp = _flex_pat_from_literal(t, ignore_case=True, word_boundary=False)
            scrubbed = tp.sub("<<<ALLOWED_TIME>>>", scrubbed)

        has_extra_future_followup = bool(BANNED_FOLLOWUP_STYLE_RE.search(scrubbed)) if num_actions > 0 else False

        # Extra closed-set actions in plan (overlap-aware)
        allowed_spans = []
        for act in allowed_actions:
            ap = _flex_pat_from_literal(act, ignore_case=True, word_boundary=True)
            spans = find_all_spans(ap, plan_text)
            allowed_spans.extend(spans)

        extra_closed = False
        for other_act in ALL_KNOWN_ACTIONS:
            if other_act in allowed_actions: # Corrected: check against allowed_actions
                continue
            op = _flex_pat_from_literal(other_act, ignore_case=True, word_boundary=True)
            spans_other = find_all_spans(op, plan_text)
            for sp in spans_other:
                # If this match is entirely inside an allowed action span, it's not "extra"
                if any(spans_overlap(sp, asp) for asp in allowed_spans):
                    continue
                if sp:
                    extra_closed = True
                    break
            if extra_closed:
                break

        # For negatives: we mainly care about future follow-up language; closed-set actions in plan are suspicious too
        if num_actions == 0 and plan_text:
            if BANNED_FOLLOWUP_STYLE_RE.search(plan_text):
                has_extra_future_followup = True

        audit_extra_closed_action.append(extra_closed)
        audit_extra_future_followup_language.append(has_extra_future_followup)

    df_batch = df_batch.copy()
    df_batch["audit_extra_closed_set_action_in_plan"] = audit_extra_closed_action
    df_batch["audit_extra_future_followup_language_in_plan"] = audit_extra_future_followup_language
    return df_batch


async def main():
    print(f"Starting Generation. Target Samples: {TOTAL_SAMPLES}")
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    rows = []
    pbar = tqdm(total=TOTAL_SAMPLES, desc="Generating notes")

    pbar.set_description("Generated notes")

    async def one_sample(i: int):
        sk = make_skeleton()

        last_err = None
        last_note = None
        last_span_info = None

        for _ in range(MAX_REGEN_ATTEMPTS):
            note, err = await generate_note_once(sem, sk)
            last_note, last_err = note, err

            if not note:
                continue

            if not has_exact_visit_date_line(note, sk["visit_date"]):
                last_span_info = {
                    "plan_header_start_char": None,
                    "plan_header_end_char": None,
                    "plan_section_start_char": None,
                    "plan_section_end_char": None,
                    "span_error": "missing_or_duplicate_visit_date_line",
                }
                continue

            ok, span_info = assign_spans_in_note(sk, note)
            last_span_info = span_info
            if ok:
                break

        span_info = last_span_info or {
            "plan_header_start_char": None,
            "plan_header_end_char": None,
            "plan_section_start_char": None,
            "plan_section_end_char": None,
            "span_error": "no_note_or_no_spaninfo",
        }

        plan_header_is_none = (sk["plan_header"] is None)
        plan_header_value = NO_HEADER_TOKEN if plan_header_is_none else sk["plan_header"]

        return {
            "note_text": last_note if last_note else "",
            "api_error": last_err if last_err else "",
            "specialty": sk["specialty"],
            "topic": sk["topic"],
            "visit_date": sk["visit_date"],

            "plan_variant": sk["plan_variant"],

            "plan_header_is_none": plan_header_is_none,
            "plan_header_value": plan_header_value,

            "num_actions": sk["num_actions"],

            # spans (None -> empty cell in CSV)
            "plan_header_start_char": span_info["plan_header_start_char"],
            "plan_header_end_char": span_info["plan_header_end_char"],
            "plan_section_start_char": span_info["plan_section_start_char"],
            "plan_section_end_char": span_info["plan_section_end_char"],
            "span_error": span_info.get("span_error", ""),

            "actions_gt": json.dumps(sk["actions_gt"], ensure_ascii=False),
            "style_features": json.dumps(sk["style_features"], ensure_ascii=False),
        }

    batch_size = CONCURRENCY_LIMIT * 5
    created = 0

    while created < TOTAL_SAMPLES:
        n = min(batch_size, TOTAL_SAMPLES - created)
        batch_rows = await asyncio.gather(*[one_sample(created + j) for j in range(n)])
        df_batch = pd.DataFrame(batch_rows)

        df_batch = audit_batch(df_batch)

        rows.extend(df_batch.to_dict(orient="records"))
        created += n
        pbar.update(n)

        ok_span_rate = (df_batch["span_error"] == "").mean()
        api_empty_rate = (df_batch["note_text"] == "").mean()
        extra_action_rate = df_batch["audit_extra_closed_set_action_in_plan"].mean()
        extra_followup_rate = df_batch["audit_extra_future_followup_language_in_plan"].mean()

        print(
            f"\n[Batch stats] span_ok_rate={ok_span_rate:.2f} api_empty={api_empty_rate:.2f} "
            f"extra_closed_action_in_plan={extra_action_rate:.2f} "
            f"extra_future_followup_language_in_plan={extra_followup_rate:.2f}"
        )

    pbar.close()

    df = pd.DataFrame(rows)
    out_path = f"synthetic_clinical_notes_{'hard_' if HARD_TEST else ''}{TOTAL_SAMPLES}.csv"
    df.to_csv(out_path, index=False)
    print("\n✅ Saved:", out_path)

    print("\nSpan quality summary:")
    print("span_ok_rate:", (df["span_error"] == "").mean())
    print("span_error top:", df["span_error"].value_counts().head(10))

    print("\nAudit summary:")
    print("extra_closed_action_in_plan:", df["audit_extra_closed_set_action_in_plan"].mean())
    print("extra_future_followup_language_in_plan:", df["audit_extra_future_followup_language_in_plan"].mean())

    print("\nnum_actions counts:")
    print(df["num_actions"].value_counts(dropna=False))

    print("\nCSV reload tip:")
    print("pd.read_csv(..., keep_default_na=False)")


    print("\n✅ Dataset generation complete.")

if __name__ == "__main__":
    # To run this, you will type:
    # export OPENAI_API_KEY="your-key-here"
    # python -m src.data_gen
    asyncio.run(main())
