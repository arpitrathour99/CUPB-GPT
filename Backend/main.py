from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import logging
import os
import re
from dotenv import load_dotenv
load_dotenv()

# Faculty / staff contact lookup (Data/contacts.json). Works whether the app is started as
# `uvicorn main:app` (from Backend/) or `uvicorn Backend.main:app` (from the project root).
try:
    from contact_search import ContactSearch
except ImportError:
    try:
        from Backend.contact_search import ContactSearch
    except ImportError:
        ContactSearch = None

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

FRONTEND_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Frontend", "dist")
)

ASSETS_DIR = os.path.join(FRONTEND_DIST, "assets")

if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Unchanged model. Only the token limit below was raised (see ask_groq).
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_REPLY_TOKENS = 1024


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Data",
    "campus.json"
)

SYNONYMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Data",
    "Synonyms.json"
)

CONTACTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Data",
    "contacts.json"
)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    campus = json.load(f)

try:
    with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
        _raw_synonyms = json.load(f)
except FileNotFoundError:
    _raw_synonyms = {}

_raw_synonyms.pop("_comment", None)

# Contact search is optional: if the module or contacts.json is missing or broken,
# the app still starts and everything else works exactly as before.
contact_search = None
if ContactSearch is not None:
    try:
        contact_search = ContactSearch(CONTACTS_PATH, SYNONYMS_PATH)
        if not contact_search.available:
            contact_search = None
    except Exception as exc:
        logging.getLogger("cupb.contacts").warning("contact search disabled: %s", type(exc).__name__)
        contact_search = None


class Question(BaseModel):
    question: str


# ===========================================================================
# PART 1: TEXT HELPERS
# ===========================================================================
def _normalize(text):
    """
    Treat hyphens, slashes, ampersands, parentheses, etc. as spaces so
    that "AI-ML", "AI & ML", "AI/ML", and "AI ML" all tokenize into the
    identical ["ai", "ml"]. Applied to both the record text (at build
    time) and the incoming query, so matching stays consistent.
    """
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower())


# Words ignored when comparing phrases, so "computer science and technology"
# (alias) still matches "Computer Science & Technology" (room name, where
# "&" has become a space).
PHRASE_FILLER = {"and", "of", "the"}


def _padded(text):
    """
    Normalised text with single spaces and a space at both ends, e.g.
    " computer science technology ". A phrase such as " computer science "
    can then be checked with a plain `in`, and it only matches whole words.
    """
    tokens = [t for t in _normalize(text).split() if t not in PHRASE_FILLER]
    return " " + " ".join(tokens) + " "


def _clean(term):
    return " ".join(_normalize(term).split())


# ---------------------------------------------------------------------------
# Bidirectional synonym lookup: any alias OR the canonical term maps to the
# FULL set of terms that mean the same thing. Terms are stored normalised so
# "ai-ml" and "ai ml" are the same key.
# ---------------------------------------------------------------------------
def _build_synonym_lookup(raw):
    lookup = {}
    for canonical, aliases in raw.items():
        group = {_clean(a) for a in aliases} | {_clean(canonical)}
        group.discard("")
        for term in group:
            lookup.setdefault(term, set()).update(group)
    return lookup


SYNONYM_LOOKUP = _build_synonym_lookup(_raw_synonyms)


# ---------------------------------------------------------------------------
# Build a flat list of searchable "records" ONCE at startup (rooms and
# offices only, so key names like "rooms" can't cause false matches).
# ---------------------------------------------------------------------------
def _build_records():
    records = []

    for r in campus.get("rooms", []):
        text = "room {} {} floor {}".format(
            r.get("room", ""),
            r.get("name", ""),
            r.get("floor", "")
        )
        records.append({"type": "room", "data": r, "text": _normalize(text), "padded": _padded(text)})

    for o in campus.get("offices", []):
        text = "{} {}".format(
            o.get("name", ""),
            o.get("floor", "")
        )
        records.append({"type": "office", "data": o, "text": _normalize(text), "padded": _padded(text)})

    return records


RECORDS = _build_records()

STOPWORDS = {
    "where", "is", "are", "was", "were", "the", "a", "an", "of", "in",
    "at", "on", "to", "for", "and", "or", "with", "me", "tell", "please",
    "do", "does", "can", "could", "you", "i", "find", "give", "show",
    "what", "which", "how", "near", "about", "this", "that", "these",
    "those", "kya", "hai", "kahan", "ka", "ki", "ke", "mein", "batao",
    # Structural words baked into almost every record's search text.
    "room", "rooms", "block", "floor", "floors", "department",
    "departments", "facility", "located", "location",
    # Generic type-descriptors that appear inside a huge fraction of names.
    "lab", "labs", "laboratory", "laboratories", "class", "classroom",
    "classrooms", "office", "offices",
    # Greetings and conversational glue (NEW). Needed now that a greeting can
    # arrive together with a campus question, e.g. "Hi, where is CSE?".
    "hi", "hello", "hey", "hii", "namaste", "thanks", "thank", "then",
    "it", "also", "pls", "btao", "bataiye", "kidhar", "kaha", "kahaan",
    "hain", "mujhe", "aur"
}


# ===========================================================================
# PART 2: IS THIS A CAMPUS QUESTION?
# ===========================================================================
# Words that suggest the user is asking about a place on campus. If none are
# present the message is treated as normal conversation and goes straight to
# Groq with NO campus records. If one is present we search the campus data
# and give the results (or "nothing found") to Groq as evidence.
# A false alarm is harmless: the system prompt tells the model to ignore the
# records when the message is not really about the campus.

CAMPUS_CUES = {
    "where", "kahan", "kaha", "kahaan", "kidhar", "kidar", "kithe",
    "located", "location", "room", "rooms", "rm", "kamra", "kamre",
    "floor", "floors", "lab", "labs", "laboratory", "laboratories",
    "classroom", "classrooms", "department", "departments", "dept",
    "office", "offices", "block", "building", "library", "centre",
    "center", "campus", "cupb", "university",
    "bpharma", "pharmacy", "pharma", "pharmaceutics",
    "pharmacology", "bachelor",
}



def is_campus_question(question):
    return any(w in CAMPUS_CUES for w in _normalize(question).split())


# ===========================================================================
# PART 3: CAMPUS SEARCH
# ===========================================================================
def _extract_room_numbers(room_field):
    """
    Room labels in the data aren't always single numbers -- e.g.
    "111–112" or "310, 312 & 313". Pull out every individual number
    so lookups match any of them, not just an exact full-string equal.
    """
    return re.findall(r"\d+", str(room_field))


def _find_by_room_number(room_no):
    matches = [
        rec["data"] for rec in RECORDS
        if rec["type"] == "room" and room_no in _extract_room_numbers(rec["data"].get("room", ""))
    ]
    return matches


def _group_phrases(group):
    phrases = []
    for term in group:
        p = _padded(term)
        if p.strip():
            phrases.append(p)
    return phrases


def _query_units(question):
    """
    Turn the question into "search units". A unit is a list of alternative
    phrases that all mean the same thing.

      "CSE"              -> one unit: computer science / cse / cs / comp sci ...
      "computer centre"  -> one unit: computer centre / computer center / cc
      "astrophysics"     -> one unit: just that word (no synonym known)

    KEY DIFFERENCE from the old code: synonyms are matched as WHOLE PHRASES.
    The old code split "computer science and technology" into separate words
    and matched "science" and "technology" on their own, which dragged in
    Environmental Science & Technology, Food Science & Technology Lab, etc.
    """
    words = [w for w in _normalize(question).split() if len(w) >= 2 and w not in STOPWORDS]

    units = []
    seen = set()
    i = 0
    while i < len(words):
        group = None
        step = 1

        if i + 1 < len(words):
            pair = words[i] + " " + words[i + 1]
            if pair in SYNONYM_LOOKUP:
                group, step = SYNONYM_LOOKUP[pair], 2

        if group is None and words[i] in SYNONYM_LOOKUP:
            group = SYNONYM_LOOKUP[words[i]]

        phrases = _group_phrases(group) if group is not None else [" " + words[i] + " "]

        key = tuple(sorted(phrases))
        if key not in seen:
            seen.add(key)
            units.append(phrases)

        i += step

    return units


MAX_RECORDS = 10


def search_campus(question):
    """
    Returns (records, note).
      records: list of campus entries (dicts) that match the question
      note:    short text explaining a miss or a limit, or None
    """
    q = _normalize(question)

    # --- Direct room-number lookup -----------------------------------
    room_numbers = list(dict.fromkeys(re.findall(r'\b(\d{3})\b', q)))[:5]
    if room_numbers:
        found = []
        missing = []
        for room_no in room_numbers:
            matches = _find_by_room_number(room_no)
            if not matches:
                missing.append(room_no)
            for m in matches:
                if m not in found:
                    found.append(m)

        note = None
        if missing:
            note = "No record for room {} exists in the campus data.".format(", ".join(missing))
        return found, note

    # --- Keyword search over flat leaf records only -------------------
    units = _query_units(question)
    if not units:
        return [], "The message does not name a specific department, lab, room or office to search for."

    scored = []
    for rec in RECORDS:
        hits = sum(1 for unit in units if any(p in rec["padded"] for p in unit))
        if hits > 0:
            scored.append((hits, rec["data"]))

    if not scored:
        return [], "No matching records were found in the campus data."

    # Keep only the records that match the MOST units. For a one-unit query
    # (the usual case) every match is kept; for longer queries this prefers
    # records that match more of what the user typed. Data order is kept.
    best = max(hits for hits, _ in scored)
    top = [data for hits, data in scored if hits == best]

    note = None
    if len(top) > MAX_RECORDS:
        note = "{} records matched; only the first {} are shown.".format(len(top), MAX_RECORDS)
        top = top[:MAX_RECORDS]

    return top, note


def find_relevant_data(question):
    """Search result as text, ready to place in the prompt."""
    records, note = search_campus(question)
    parts = []

    if records:
        dumped = json.dumps(records, ensure_ascii=False, indent=2)
        # Safety cap by record COUNT, so a JSON object is never cut in half.
        if len(dumped) > 6000:
            records = records[:6]
            dumped = json.dumps(records, ensure_ascii=False, indent=2)
            note = (note + " " if note else "") + "Only the first 6 records are shown."
        parts.append(dumped)

    if note:
        parts.append("NOTE: " + note)

    return "\n\n".join(parts)


# ===========================================================================
# PART 4: TALKING TO GROQ
# ===========================================================================
SYSTEM_PROMPT = """You are CUPB-GPT, a friendly AI assistant for Central University of Punjab (CUPB).

You do two kinds of things.

1. GENERAL CONVERSATION: greetings, small talk, thanks, questions about yourself, and general knowledge or explanations (for example "Explain what an AI chatbot is"). Answer these naturally and helpfully using your own knowledge.

2. CAMPUS QUESTIONS: where a department, room, lab, floor or office is. For these you may use ONLY the entries under "CAMPUS RECORDS" in the user's message.

Rules for campus questions:
- Every room number, floor, department and office you mention must appear in the CAMPUS RECORDS. Never guess or invent one.
- The records come from a keyword search and may include loosely related entries. Use only the entries that really answer the question. If the user asked for something specific (for example a department that is not in the records), say it is not available in the campus data. You may mention the closest records.
- If the CAMPUS RECORDS contain only a NOTE saying nothing was found, say clearly that the campus data does not contain this information. Do not guess a location.
- The records contain only room number, name and floor (offices have no room number). If the user asks for anything else (staff, courses, timings, contacts), say the campus data does not include it, then share the location details you do have.
- If the message is not really about the CUPB campus, ignore the records and answer normally.
- If a message asks about a campus place but there is no CAMPUS RECORDS section at all, say you could not find it in the campus data and suggest asking like "Where is the CSE department?".

Style:
- Reply in the language the user used: English, Hindi, or Hinglish (Hindi in English letters).
- If the message has a greeting and a question, greet briefly, then answer the question.
- Keep answers short and friendly. Use plain text only: no markdown headings or bold. Simple "- " bullet lines are fine.
- When listing campus entries, give room number, name and floor, for example "- Room 215: Department of Computer Science & Technology (Floor 2)". For an office with no room number, give the name and floor.
- Answer a location question with just the matching entries. Do not add a closing sentence that explains, restates or justifies the result (for example, never write "That is the location for the X department according to the campus data"). Only mention a limitation when the records really do not contain what was asked.
- If asked who you are: you are CUPB-GPT, an AI assistant for Central University of Punjab that can chat, explain topics, and help find departments, rooms, labs and offices on campus."""

EMPTY_REPLY = "Sorry, I could not generate a reply just now. Please try again."


def build_messages(question, records_text=None):
    """
    records_text is None  -> normal conversation (no campus records sent).
    records_text is a str -> campus question; the text is the search result
                             (records and/or a NOTE saying nothing matched).
    """
    if records_text is None:
        user_content = question
    else:
        user_content = "CAMPUS RECORDS:\n\n{}\n\nUSER MESSAGE:\n\n{}".format(records_text, question)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def ask_groq(question, records_text=None):
    # Campus answers use temperature 0 (stick to the records). Normal
    # conversation is allowed to sound a little more natural.
    temperature = 0 if records_text is not None else 0.6

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=temperature,
        messages=build_messages(question, records_text),
        max_tokens=MAX_REPLY_TOKENS
    )

    answer = (response.choices[0].message.content or "").strip()
    return answer or EMPTY_REPLY


# ===========================================================================
# PART 5: ROUTES
# ===========================================================================
@app.get("/")
def home():
    index_file = os.path.join(FRONTEND_DIST, "index.html")

    if os.path.isfile(index_file):
        return FileResponse(index_file)

    return {"message": "CUPB-GPT Backend is working"}


@app.post("/ask")
def ask(data: Question):

    question = data.question.strip()

    if not question:
        return {"answer": "Please type your question."}

    # Contact questions (professor / HOD / staff phone, email, department) are answered
    # directly from Data/contacts.json and never sent to Groq, so numbers and email
    # addresses cannot be invented. Anything else returns None and continues below.
    if contact_search is not None:
        contact_result = contact_search.handle_question(question)
        if contact_result is not None:
            return {"answer": contact_result["answer"]}

    # Campus-looking questions are searched and answered from the records.
    # Everything else (greetings, "who are you", general knowledge) goes
    # straight to Groq. Either way Groq is called, so "no search results"
    # no longer blocks a normal conversation.
    records_text = find_relevant_data(question) if is_campus_question(question) else None

    return {
        "answer": ask_groq(question, records_text)
    }

@app.get("/{full_path:path}")
def frontend_routes(full_path: str):
    index_file = os.path.join(FRONTEND_DIST, "index.html")

    if os.path.isfile(index_file):
        return FileResponse(index_file)

    return {"detail": "Not Found"}