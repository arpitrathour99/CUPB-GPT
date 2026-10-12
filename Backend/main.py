from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import os
import re

app = FastAPI()
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

# LM Studio
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

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

with open(DATA_PATH, "r", encoding="utf-8") as f:
    campus = json.load(f)

try:
    with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
        _raw_synonyms = json.load(f)
except FileNotFoundError:
    _raw_synonyms = {}

_raw_synonyms.pop("_comment", None)


# ---------------------------------------------------------------------------
# Build a bidirectional lookup: any alias OR the canonical term itself maps
# to the FULL set of terms that mean the same thing (canonical + all its
# aliases). This lets a search for "CSE" also search for "computer science",
# and a search for "computer science" also pick up "cse", "cs", etc.
# ---------------------------------------------------------------------------
def _build_synonym_lookup(raw):
    lookup = {}
    for canonical, aliases in raw.items():
        group = set(a.lower() for a in aliases) | {canonical.lower()}
        for term in group:
            lookup.setdefault(term, set()).update(group)
    return lookup


SYNONYM_LOOKUP = _build_synonym_lookup(_raw_synonyms)


def _normalize(text):
    """
    Treat hyphens, slashes, ampersands, parentheses, etc. as spaces so
    that "AI-ML", "AI & ML", "AI/ML", and "AI ML" all tokenize into the
    identical ["ai", "ml"] -- instead of "ai-ml" being one glued-together
    token that only matches an exact hyphenated alias. Applied uniformly
    to both the record text (at build time) and the incoming query, so
    matching stays consistent no matter how either one is punctuated.
    """
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower())


def _expand_with_synonyms(words):
    """
    Given a list of query words/phrases, return the original words plus
    any synonym terms they map to. Checks both single words and adjacent
    word-pairs so multi-word aliases like "ai ml" or "comp sci" still match.

    Alias phrases are run through _normalize() before splitting -- an
    alias like "ai & ml" or "b.ed" needs its punctuation turned into
    spaces first, or "&"/"b.ed" end up as their own garbled search
    tokens. Stopwords are filtered back out afterwards too: an alias
    like "computer science and technology" would otherwise reintroduce
    "and" as a live search term, and "and" appears in enough unrelated
    room names to add noise to every single query.
    """
    expanded = set(words)

    for w in words:
        if w in SYNONYM_LOOKUP:
            for term in SYNONYM_LOOKUP[w]:
                expanded.update(_normalize(term).split())

    for i in range(len(words) - 1):
        pair = f"{words[i]} {words[i+1]}"
        if pair in SYNONYM_LOOKUP:
            for term in SYNONYM_LOOKUP[pair]:
                expanded.update(_normalize(term).split())

    return [w for w in expanded if w and w not in STOPWORDS]


class Question(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Build a flat list of searchable "records" ONCE at startup.
#
# The old code recursively scored every nested dict in the JSON tree,
# INCLUDING the top-level `campus` dict itself. That meant the entire
# dataset could show up as a single "match" (since it obviously contains
# every keyword somewhere), get ranked alongside real matches, and then
# get sliced by the hard [:5000] cutoff -- silently truncating away the
# real answer. Scoring only these flat, leaf-level records prevents that
# entirely, and also keeps key names (like "departments") out of the
# searchable text so they can't cause false-positive matches.
# ---------------------------------------------------------------------------
def _build_records():
    records = []

    for r in campus.get("rooms", []):
        text = "room {} {} floor {}".format(
            r.get("room", ""),
            r.get("name", ""),
            r.get("floor", "")
        )
        records.append({"type": "room", "data": r, "text": _normalize(text)})

    for o in campus.get("offices", []):
        text = "{} {}".format(
            o.get("name", ""),
            o.get("floor", "")
        )
        records.append({"type": "office", "data": o, "text": _normalize(text)})

    return records


RECORDS = _build_records()

STOPWORDS = {
    "where", "is", "are", "was", "were", "the", "a", "an", "of", "in",
    "at", "on", "to", "for", "and", "or", "with", "me", "tell", "please",
    "do", "does", "can", "could", "you", "i", "find", "give", "show",
    "what", "which", "how", "near", "about", "this", "that", "these",
    "those", "kya", "hai", "kahan", "ka", "ki", "ke", "mein", "batao",
    # Structural words baked into almost every record's search text
    # (e.g. "room 101 ..." / "floor 1 block A ..."). If left in, a query
    # like "AI ML room" scores every single room equally on "room" alone
    # and drowns out the actual signal ("ai", "ml"). The digit itself is
    # still handled separately by the room-number regex branch above.
    "room", "rooms", "block", "floor", "floors", "department",
    "departments", "facility", "located", "location",
    # Generic type-descriptors that appear inside a huge fraction of
    # room names ("... Lab", "... Classroom", "... Laboratory"). Same
    # problem as "room"/"floor" above: leaving these in makes queries
    # like "AI ML lab" score every lab in the building equally on "lab"
    # alone, burying the actual "ai"/"ml" signal under noise.
    "lab", "labs", "laboratory", "laboratories", "class", "classroom",
    "classrooms", "office", "offices"
}


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


def find_relevant_data(question):
    q = _normalize(question)

    # --- Direct room-number lookup -----------------------------------
    match = re.search(r'\b(?:room|rm)?\s*(\d{3})\b', q)
    if match:
        room_no = match.group(1)
        results = _find_by_room_number(room_no)

        if results:
            return json.dumps(results, ensure_ascii=False, indent=2)

        return f"Room {room_no} is not available in the campus data."

    # --- Keyword search over flat leaf records only -------------------
    # Minimum length is 2, not 3 -- short abbreviations like "ai", "ml",
    # "cs" are legitimate query terms once synonym expansion is applied,
    # and cutting them at length 3 silently dropped them from the query.
    raw_words = [w.strip(".,?!") for w in q.split()]
    words = [w for w in raw_words if len(w) >= 2 and w not in STOPWORDS]

    # If every word happened to be a stopword, relax the stopword filter
    # (but keep the length filter) so the query isn't dropped entirely.
    if not words:
        words = [w for w in raw_words if len(w) >= 2]

    if not words:
        return "No matching campus information found."

    words = _expand_with_synonyms(words)

    # Word-boundary matching, not plain substring counting. This matters
    # once synonym expansion introduces short tokens like "ai" or "ml" --
    # a plain `in` / `.count()` substring check would match "ai" inside
    # unrelated text like "PSA, AIS", giving false positives.
    patterns = [re.compile(r'\b' + re.escape(w) + r'\b') for w in words if w]

    scored = []
    for rec in RECORDS:
        score = sum(len(p.findall(rec["text"])) for p in patterns)
        if score > 0:
            scored.append((score, rec["data"]))

    # Stable sort keeps original data order for ties, so e.g. all the
    # Chemistry rooms stay together in the results instead of one being
    # pushed out by an unrelated high-scoring item.
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [item[1] for item in scored[:10]]

    if not selected:
        return "No matching campus information found."

    dumped = json.dumps(selected, ensure_ascii=False, indent=2)

    # Safety cap, but by record COUNT first (above) rather than a hard
    # character slice, so we don't cut a JSON object in half. This is
    # just a final guard in case of unusually long department name lists.
    if len(dumped) > 6000:
        selected = selected[:6]
        dumped = json.dumps(selected, ensure_ascii=False, indent=2)

    return dumped


@app.get("/")
def home():
    index_file = os.path.join(FRONTEND_DIST, "index.html")

    if os.path.isfile(index_file):
        return FileResponse(index_file)

    return {"message": "CUPB-GPT Backend is working"}


@app.post("/ask")
def ask(data: Question):

    question = data.question

    relevant_data = find_relevant_data(question)

    # If the search genuinely found nothing, say so directly and skip the
    # LLM call entirely. This is the fix for the "CSE department is not
    # listed" problem: the search WAS finding the right rooms, but a
    # small local model would sometimes second-guess correctly-retrieved
    # data and claim it wasn't there. Removing the LLM from this path
    # removes its ability to make that mistake, and is faster too.
    if relevant_data.startswith("No matching campus information") or \
       relevant_data.startswith("Room ") and relevant_data.endswith("campus data."):
        return {"answer": relevant_data}

    response = client.chat.completions.create(
        model="qwen/qwen3-vl-8b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are CUPB-GPT for Central University of Punjab.

A search system has ALREADY found the campus entries below that match the user's
question -- including resolving any abbreviation, acronym, or synonym they used
(e.g. "CSE" -> "Computer Science", "phy" -> "Physics", "AI-ML" -> "Artificial
Intelligence"). Your only job is to present these already-matched entries clearly.

Rules:
- The entries below ARE the answer. Do not judge whether they are relevant, and
  do not say the information is unavailable -- if entries are listed, they exist.
- List every entry supplied -- do not omit or filter any of them.
- Do not invent room numbers or departments that are not present in the entries.
- Format as bullet points: room number + name (+ floor if useful).
- Understand Hindi, Hinglish and English, and reply in the language the user used.
- Keep it short and direct. No preamble like "Based on the data...".
- Only if the entries truly contain nothing relevant to the question, say so --
  but this should be rare, since the search has already filtered for relevance."""
            },
            {
                "role": "user",
                "content": f"""MATCHED CAMPUS ENTRIES:

{relevant_data}

USER'S QUESTION:

{question}"""
            }
        ],
        max_tokens=400
    )

    return {
        "answer": response.choices[0].message.content
    }

@app.get("/{full_path:path}")
def frontend_routes(full_path: str):
    index_file = os.path.join(FRONTEND_DIST, "index.html")

    if os.path.isfile(index_file):
        return FileResponse(index_file)

    return {"detail": "Not Found"}    