"""Structured faculty / staff contact lookup for CUPB-GPT.

All contact facts come from Data/contacts.json (extracted from the official telephone
directory dated 02 June 2025). Answers are built in plain Python -- the LLM is never asked
to produce or "remember" a phone number or e-mail address.

Public API:
    cs = ContactSearch(contacts_path, synonyms_path)
    result = cs.handle_question("Dr. Meenakshi ka contact detail do")
    # -> {"answer": str, "kind": str, "matches": int}   or   None if this is not a contact question
"""
import difflib
import json
import logging
import os
import re
from collections import OrderedDict, defaultdict

log = logging.getLogger("cupb.contacts")

NOT_LISTED = "Not listed in the directory"
MAX_LIST = 30
MAX_CANDIDATES = 8

CONTACT_KW = {
    "contact", "contacts", "number", "numbers", "mobile", "mobiles", "phone", "phones", "landline",
    "email", "emails", "mail", "mails", "extension", "ext", "detail", "details", "call", "telephone",
    "tel", "sampark",
}
ROLE_KW = {
    "professor", "professors", "prof", "profs", "hod", "hods", "dean", "deans", "faculty",
    "faculties", "registrar", "warden", "wardens", "director", "librarian", "teacher", "teachers",
    "chancellor", "vc", "pvc", "controller",
}
# Definition / general-knowledge phrasing: without a contact word these go to the LLM, not the directory.
GENERAL_RE = re.compile(
    r"\b(explain|define|definition|difference|differences|meaning|why|biography|salary|"
    r"qualification|qualifications|eligibility|become|duties|responsibilities|responsibility|"
    r"how to|how do|how can|role of)\b|^\s*(what is|what are|whats) (a|an) |^\s*what (does|do) (a|an|the) ")
LOCATION_KW = {"where", "kahan", "kaha", "kidhar", "located", "location", "floor", "room", "rooms",
               "block", "building", "baithte", "baithta", "baithti"}
STOP = {
    # English
    "a", "an", "the", "of", "in", "at", "on", "to", "for", "and", "or", "with", "me", "my", "is", "are",
    "was", "were", "be", "do", "does", "did", "can", "could", "you", "i", "please", "pls", "tell", "show",
    "give", "find", "get", "what", "whats", "which", "who", "whom", "whose", "how", "all", "any", "available",
    "about", "this", "that", "his", "her", "their", "s", "list", "lists", "belong", "belongs", "works", "work",
    "under", "from", "by", "name", "names", "full", "complete", "info", "information", "need", "want", "know",
    "there", "its", "it", "department", "departments", "dept", "depts", "school", "schools",
    "official", "listed", "directory", "faculty", "faculties", "teacher", "teachers", "staff", "member", "members",
    # Hindi / Hinglish
    "ka", "ki", "ke", "ko", "kya", "hai", "hain", "ho", "kaun", "kon", "kaunsa", "kaunse", "konsa", "kis",
    "kiska", "kiski", "kiske", "batao", "bata", "batana", "bataiye", "btao", "do", "de", "dijiye", "dedo",
    "dikhao", "dikha", "mujhe", "muje", "mereko", "mein", "me", "se", "wali", "wale", "wala", "ji", "sir",
    "madam", "mam", "maam", "aur", "tatha", "sabhi", "saare", "pe", "par", "hi", "bhi", "yeh", "ye", "woh",
    "vo", "chahiye", "chahie", "wo", "koi", "liye", "lie",
}
TITLE_TOKENS = {"dr", "prof", "mr", "ms", "mrs", "er", "professor"}
TITLE_CANON = {"dr": "Dr.", "prof": "Prof.", "professor": "Prof.", "mr": "Mr.", "ms": "Ms.", "mrs": "Mrs.", "er": "Er."}
FIELD_WORDS = {
    "email": {"email", "emails", "mail", "mails", "gmail", "id"},
    "mobile": {"mobile", "mobiles", "phone", "phones", "cell", "whatsapp", "number", "numbers", "contact"},
    "landline": {"landline", "telephone", "tel"},
    "extension": {"extension", "ext", "extn"},
}
PHONE_FIELDS = {"mobile", "landline", "extension"}


# --------------------------------------------------------------------------- helpers
def _norm(text):
    text = (text or "").lower().replace("&", " and ").replace("’", "'")
    text = re.sub(r"'s\b", "", text).replace("'", "")
    return re.sub(r"[^a-z0-9\s]", " ", text)


def _tokens(text):
    return _norm(text).split()


def _digits(s):
    return re.sub(r"\D", "", s or "")


def _phrase_in(phrase, padded_q):
    return f" {phrase} " in padded_q


def _remove_phrase(tokens, phrase_tokens):
    out, i, n = [], 0, len(phrase_tokens)
    while i < len(tokens):
        if tokens[i:i + n] == phrase_tokens:
            i += n
        else:
            out.append(tokens[i])
            i += 1
    return out


def _strip_unit_prefix(name):
    n = " ".join(_tokens(name))
    for p in ("department of ", "school of ", "centre for ", "center for "):
        if n.startswith(p):
            return n[len(p):]
    return n


def _join(values):
    return ", ".join(values) if values else NOT_LISTED


# --------------------------------------------------------------------------- main class
class ContactSearch:
    def __init__(self, contacts_path, synonyms_path=None):
        self.available = False
        self.source_date = None
        self.records = []
        self._load(contacts_path)
        self.synonym_groups = self._load_synonyms(synonyms_path)
        if self.available:
            self._index()

    # ---- loading -------------------------------------------------------------
    def _load(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            recs = data.get("contacts", [])
            if not isinstance(recs, list):
                raise ValueError("'contacts' must be a list")
            self.records = [r for r in recs if isinstance(r, dict)]
            self.source_date = data.get("source_date")
            self.available = bool(self.records)
            if not self.available:
                log.warning("contacts.json loaded but contains no records")
        except FileNotFoundError:
            log.warning("contacts.json not found at %s - contact search disabled", path)
        except (ValueError, OSError) as exc:  # JSONDecodeError is a ValueError
            log.warning("contacts.json could not be read (%s) - contact search disabled", type(exc).__name__)

    @staticmethod
    def _load_synonyms(path):
        groups = []
        if not path:
            return groups
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return groups
        raw.pop("_comment", None)
        for canonical, aliases in raw.items():
            if isinstance(aliases, list):
                groups.append((" ".join(_tokens(canonical)), [" ".join(_tokens(a)) for a in aliases]))
        return groups

    # ---- indexing ----------------------------------------------------------
    def _index(self):
        self.entities = OrderedDict()  # entity key -> list of role records
        for r in self.records:
            key = r.get("person_id") or r.get("id") or str(id(r))
            self.entities.setdefault(key, []).append(r)
        self.ent_tokens = {}
        for key, roles in self.entities.items():
            name = roles[0].get("name_without_title") or roles[0].get("name") or ""
            self.ent_tokens[key] = _tokens(name)
        # vocabulary of words that appear in designation/section/department/school text
        self.vocab = set()
        for r in self.records:
            self.vocab.update(_tokens(" ".join(filter(None, [r.get("designation"), r.get("section"),
                                                             r.get("department"), r.get("school")]))))
        self.units = self._build_units()

    def _build_units(self):
        """Departments and schools with their alias phrases (own name = priority 2, synonym = 1)."""
        names = OrderedDict()
        for r in self.records:
            for fld in ("department", "school"):
                if r.get(fld):
                    names[r[fld]] = fld
        units = []
        for full, kind in names.items():
            phrase = _strip_unit_prefix(full)
            aliases = {phrase: 2}
            for canon, alist in self.synonym_groups:
                members = {canon, *alist}
                linked = (phrase in members) or (len(canon.split()) >= 2 and _phrase_in(canon, f" {phrase} "))
                if linked:
                    for m in members:
                        if m and m not in aliases and not (len(m) <= 2 and m in STOP):
                            aliases[m] = 1
            units.append({"full": full, "kind": kind, "phrase": phrase, "aliases": aliases})
        return units

    # ---- question parsing ---------------------------------------------------
    def _match_units(self, padded_q):
        hits = []
        for u in self.units:
            best = None
            for alias, prio in u["aliases"].items():
                if _phrase_in(alias, padded_q):
                    cand = (prio, len(alias))
                    if best is None or cand > best[0]:
                        best = (cand, alias)
            if best:
                hits.append((best[0], best[1], u))
        if not hits:
            return [], None
        top_prio = max(h[0][0] for h in hits)
        hits = [h for h in hits if h[0][0] == top_prio]
        top_len = max(h[0][1] for h in hits)
        hits = [h for h in hits if h[0][1] == top_len]
        return [h[2] for h in hits], hits[0][1]

    @staticmethod
    def _role_in_unit(r, unit):
        return r.get(unit["kind"]) == unit["full"]

    @staticmethod
    def _rank_ok(designation, rank):
        d = (designation or "").lower()
        if rank == "assistant":
            return "assistant professor" in d
        if rank == "associate":
            return "associate professor" in d
        if rank == "professor":
            return bool(re.match(r"\s*professor\b", d)) and "of practice" not in d
        if rank == "faculty":
            return "professor" in d or "faculty" in d
        return True

    @staticmethod
    def _is_hod(designation):
        return "hod" in re.sub(r"[^a-z]", "", (designation or "").lower())

    def _field_request(self, tokens, raw_q):
        wanted = set()
        tset = set(tokens)
        for fld, words in FIELD_WORDS.items():
            if tset & (words - {"contact", "number", "numbers"}):
                wanted.add(fld)
        if tset & {"phone", "phones", "telephone", "tel"}:
            wanted.update(PHONE_FIELDS)
        if tset & {"number", "numbers"} and not wanted:
            wanted.update(PHONE_FIELDS)
        if re.search(r"\b(which|kaun\w*|kon\w*|what)\b.*\bdepartment\b|\bdepartment\b.*\b(belong|kaun\w*|konsa|kaunsa)\b|\bbelong", raw_q.lower()):
            wanted.add("department")
        if tset & {"designation", "post", "position", "role"}:
            wanted.add("designation")
        # "contact", "details", "all information" => full card (no restriction)
        if tset & {"details", "detail", "contact", "contacts", "info", "information"} and not (wanted & PHONE_FIELDS | wanted & {"email"}):
            return None
        return wanted or None

    # ---- public entry point ------------------------------------------------------
    def handle_question(self, question):
        if not self.available or not question or not question.strip():
            return None
        try:
            return self._handle(question)
        except Exception as exc:  # never let a contact-search bug break /ask
            log.exception("contact search failed: %s", type(exc).__name__)
            return None

    def _handle(self, question):
        q_norm = _norm(question)
        q_norm = re.sub(r"\bhead of (the )?department\b|\bdepartment head\b|\bh o d\b", " hod ", q_norm)
        q_norm = re.sub(r"\bpro vice chancellor\b", " pvc ", q_norm)
        q_norm = re.sub(r"\bvice chancellor\b", " vc ", q_norm)
        tokens = q_norm.split()
        tset = set(tokens)
        has_contact = bool(tset & CONTACT_KW)
        has_loc = bool(tset & LOCATION_KW)

        # reverse lookup by a full phone number
        rev = self._reverse_lookup(question)
        if rev is not None:
            return rev

        # room numbers belong to the campus flow
        if re.search(r"\b\d{3}\b", q_norm) and not has_contact:
            return None

        # definition / general-knowledge questions go to the LLM unless a contact word is present
        if not has_contact and GENERAL_RE.search(q_norm):
            return None

        # ranks
        rank = None
        padded = f" {' '.join(tokens)} "
        if re.search(r"\bassistant professors?\b", padded):
            rank = "assistant"
            tokens = _remove_phrase(_remove_phrase(tokens, ["assistant", "professor"]), ["assistant", "professors"])
        elif re.search(r"\bassociate professors?\b", padded):
            rank = "associate"
            tokens = _remove_phrase(_remove_phrase(tokens, ["associate", "professor"]), ["associate", "professors"])
        want_hod = bool(set(tokens) & {"hod", "hods"})
        tokens = [t for t in tokens if t not in ("hod", "hods")]

        # department / school
        padded = f" {' '.join(tokens)} "
        units, alias = self._match_units(padded)
        if alias:
            tokens = _remove_phrase(tokens, alias.split())

        # pro/vice chancellor shortcuts -> phrase tokens used by the generic search
        tokens = [x for t in tokens for x in ({"vc": ["vice", "chancellor"], "pvc": ["pro", "vice", "chancellor"]}.get(t, [t]))]

        title = None
        for t in tokens:
            if t in TITLE_CANON:
                title = TITLE_CANON[t]
                break
        plural_prof = bool(set(tokens) & {"professors", "profs"})
        faculty_words = bool(set(_tokens(question)) & {"faculty", "faculties", "teachers", "teacher"})

        name_tokens = [t for t in tokens if t not in STOP and t not in CONTACT_KW and t not in TITLE_TOKENS
                       and t not in {"professors", "profs", "dean", "deans", "warden", "wardens", "registrar",
                                     "director", "librarian", "designation", "post", "position", "role", "hods",
                                     "email", "emails", "mail", "gmail", "id", "cell", "extn", "whatsapp"}]
        role_tokens = [t for t in tokens if t in {"dean", "deans", "warden", "wardens", "registrar", "director", "librarian"}]

        fields = self._field_request(_tokens(question), question)

        if rank is None and plural_prof and not name_tokens:
            rank = "professor"
        if rank is None and faculty_words and not name_tokens and (units or want_hod):
            rank = "faculty"

        strong = has_contact
        # a rank word ("assistant professor") only counts together with a department / HOD / contact word
        rank_counts = bool(rank) and bool(units or want_hod or has_contact or name_tokens)
        weak_trigger = bool(tset & ROLE_KW) or rank_counts or want_hod
        if rank and not rank_counts and not (tset & ROLE_KW - {"professor", "professors", "prof", "profs"}):
            return None
        if has_loc and not has_contact and not title:
            return None
        if not has_contact and "office" in tset and not title:
            return None  # e.g. "Registrar's Office" -> campus/room flow

        # ---- 1) person-name search ------------------------------------------------
        if name_tokens:
            places_ok = not (role_tokens or rank or want_hod or units)
            res = self._name_search(name_tokens, title, rank, units, fields, strong or bool(title) or weak_trigger,
                                    places_ok, suggest=strong)
            if res is not None:
                return res

        # ---- 2) role / department list or generic text search -----------------------
        if not (strong or weak_trigger or units):
            # no contact-ish signal at all: only treat as contact question if a 2+ word name matches
            if len(name_tokens) >= 2:
                res = self._name_search(name_tokens, title, rank, units, fields, False, not (role_tokens or rank or want_hod or units))
                if res is not None:
                    return res
            return None
        if units and not (strong or weak_trigger):
            return None  # e.g. "Where is the Physics department" -> campus flow
        res = self._list_search(name_tokens, role_tokens, rank, want_hod, units, fields)
        if res is not None:
            return res
        if strong and name_tokens:
            return self._not_found(name_tokens)
        return None

    # ---- name search ----------------------------------------------------------
    @staticmethod
    def _tok_score(qt, nt):
        if qt == nt:
            return 3
        if len(qt) == 1 and nt.startswith(qt):
            return 1
        if len(qt) >= 4 and nt.startswith(qt):
            return 2
        return 0

    def _match_entity(self, qtoks, ntoks):
        """Return (tier, ok): tier 0 identical tokens, 1 all exact (name has extras), 2 prefix/initial match."""
        used, scores = set(), []
        for qt in qtoks:
            best, bi = 0, None
            for i, nt in enumerate(ntoks):
                if i in used:
                    continue
                s = self._tok_score(qt, nt)
                if s > best:
                    best, bi = s, i
            if not best:
                return None
            used.add(bi)
            scores.append(best)
        if all(s == 3 for s in scores):
            return 0 if len(qtoks) == len(ntoks) else 1
        return 2

    def _name_search(self, name_tokens, title, rank, units, fields, allow, places_ok=True, suggest=False):
        matches = defaultdict(list)  # tier -> [entity keys]
        for key, ntoks in self.ent_tokens.items():
            if not ntoks:
                continue
            if not places_ok and self.entities[key][0].get("kind") != "person":
                continue
            tier = self._match_entity(name_tokens, ntoks)
            if tier is not None:
                matches[tier].append(key)
        if not matches:
            return self._suggest_similar(name_tokens) if (suggest and places_ok) else None
        # single-token queries that only prefix-match are too vague to be a name search unless contact-ish
        if not allow and len(name_tokens) < 2:
            return None
        best_tier = min(matches)
        keys = list(matches[best_tier])
        others = [k for t, ks in matches.items() if t != best_tier for k in ks]

        # narrowing: department, rank, title (title/rank are preferences, department is a filter if it helps)
        if units:
            f = [k for k in keys if any(self._role_in_unit(r, u) for r in self.entities[k] for u in units)]
            if f:
                keys = f
        if rank and rank != "faculty":
            f = [k for k in keys if any(self._rank_ok(r.get("designation"), rank) for r in self.entities[k])]
            if f:
                keys = f
        title_dropped = []
        if title:
            f = [k for k in keys if self.entities[k][0].get("title") == title]
            if f:
                title_dropped = [k for k in keys if k not in f]
                keys = f
        if len(keys) == 1:
            ent = keys[0]
            text = self._render_entity(ent, fields)
            similar = [k for k in others if k != ent][:5]
            if title_dropped:
                text += ("\n\nNote: the directory also lists other people with the same name but a different title: "
                         + "; ".join(self._one_line_identity(k) for k in title_dropped[:5])
                         + ". Ask again with their department if you meant one of them.")
            elif similar and best_tier <= 1:
                text += "\n\nOther people with a similar name (full details on request): " + "; ".join(
                    self._one_line_identity(k) for k in similar)
            return self._wrap(text, "person", 1)
        return self._wrap(self._render_ambiguous(name_tokens, keys), "ambiguous", len(keys))

    def _suggest_similar(self, name_tokens):
        words = {t for toks in self.ent_tokens.values() for t in toks if len(t) > 2}
        sims = []
        for qt in name_tokens:
            if len(qt) < 4:
                continue
            sims.extend(difflib.get_close_matches(qt, list(words), n=3, cutoff=0.8))
        if not sims:
            return None
        cands = []
        for key, ntoks in self.ent_tokens.items():
            if any(t in sims for t in ntoks):
                cands.append(key)
        if not cands:
            return None
        text = ("I couldn't find an exact match for \"" + " ".join(name_tokens) + "\" in the directory. "
                "Did you mean one of these? (No contact details are shown for approximate matches - "
                "please repeat the question with the correct name.)\n")
        text += "\n".join(f"{i}. {self._one_line_identity(k)}" for i, k in enumerate(cands[:MAX_CANDIDATES], 1))
        return self._wrap(text, "did_you_mean", len(cands))

    def _not_found(self, name_tokens):
        text = (f"I couldn't find \"{' '.join(name_tokens)}\" in the telephone directory. "
                "I won't guess a contact detail that isn't listed there.")
        return self._wrap(text, "not_found", 0)

    # ---- list / generic search ----------------------------------------------
    def _list_search(self, leftover, role_tokens, rank, want_hod, units, fields):
        recs = list(self.records)
        if units:
            recs = [r for r in recs if any(self._role_in_unit(r, u) for u in units)]
        if want_hod:
            recs = [r for r in recs if self._is_hod(r.get("designation"))]
        if rank:
            recs = [r for r in recs if self._rank_ok(r.get("designation"), rank)]

        sig = [t for t in (leftover + role_tokens) if t in self.vocab]
        place_hit = False
        if sig:
            def hay(r):
                return set(_tokens(" ".join(filter(None, [r.get("designation"), r.get("section"), r.get("department"),
                                                          r.get("school"), r.get("name") if r.get("kind") != "person" else None]))))
            scored = [(sum(1 for t in sig if t in hay(r)), r) for r in recs]
            top = max((s for s, _ in scored), default=0)
            need = max(1, -(-len(sig) * 6 // 10))
            if top < need:
                recs = []
            else:
                recs = [r for s, r in scored if s == top]
                # if some entries carry every word in their *designation*, those are the real answer
                in_desig = [r for r in recs if all(t in set(_tokens(r.get("designation"))) for t in sig)]
                if in_desig:
                    recs = in_desig
                    exact = [r for r in recs if sorted(_tokens(r.get("designation"))) == sorted(sig)]
                    if exact:
                        recs = exact
        elif leftover and not (units or want_hod or rank or role_tokens):
            # leftover words that are not in the vocabulary: maybe a place/unit name (Ambulance, Main Gate...)
            hits = []
            for r in self.records:
                if r.get("kind") == "place_or_unit":
                    nt = set(_tokens(r.get("name")))
                    if set(leftover) & nt:
                        hits.append(r)
            if hits:
                recs, place_hit = hits, True
            else:
                return None
        elif not (units or want_hod or rank or role_tokens):
            return None
        else:
            if role_tokens:
                recs = [r for r in recs if set(role_tokens) & set(_tokens(r.get("designation")))
                        or set(role_tokens) & set(_tokens(r.get("section")))]

        if recs and role_tokens and not leftover:
            exact = [r for r in recs if " ".join(_tokens(r.get("designation"))) == " ".join(role_tokens)]
            if exact:
                recs = exact

        if not recs:
            if units or want_hod or rank:
                return self._wrap("I couldn't find a matching entry in the telephone directory.", "list_empty", 0)
            return None
        if len(recs) == 1 and recs[0].get("kind") == "person":
            return self._wrap(self._render_role_card(recs[0], fields), "person", 1)
        return self._wrap(self._render_list(recs, fields, units, rank, want_hod), "list", len(recs))

    # ---- reverse lookup ---------------------------------------------------------
    def _reverse_lookup(self, question):
        nums = [d for d in (_digits(m) for m in re.findall(r"\d[\d\-\s]{8,}\d", question)) if len(d) >= 10]
        if not nums:
            return None
        hits = []
        for r in self.records:
            vals = [_digits(x) for x in (r.get("mobile") or []) + (r.get("landline") or [])]
            if any(n in vals or n[-10:] in [v[-10:] for v in vals] for n in nums):
                hits.append(r)
        if not hits:
            return self._wrap("That number is not listed in the telephone directory.", "reverse_none", 0)
        text = "That number is listed in the directory for:\n\n" + "\n\n".join(self._render_role_card(r, None) for r in hits[:5])
        return self._wrap(text, "reverse", len(hits))

    # ---- rendering ---------------------------------------------------------------
    @staticmethod
    def _unit_label(r):
        dept, school, section = r.get("department"), r.get("school"), r.get("section")
        if dept and school:
            return f"{dept} ({school})"
        if dept or school:
            return dept or school
        if section:
            return section.split("(")[0].strip() if len(section) > 70 else section
        return None

    @staticmethod
    def _ext_note(landlines):
        return [ll[-3:] for ll in landlines if ll.startswith("0164-2864-")]

    def _role_lines(self, r, fields, with_identity):
        lines = []
        if with_identity:
            lines.append(f"Name: {r.get('name') or NOT_LISTED}")
        if fields is None or "designation" in fields or not with_identity:
            lines.append(f"Designation: {r.get('designation') or NOT_LISTED}")
        if True:  # department is always shown, also for single-field questions
            lines.append(f"Department: {self._unit_label(r) or NOT_LISTED}")
        shared = r.get("shared_cell_values") or {}
        if fields is None or fields & {"mobile"}:
            lines.append("Mobile/Phone: " + _join(r.get("mobile")))
        if fields is None or fields & {"landline"}:
            if r.get("landline"):
                lines.append("Landline: " + ", ".join(r["landline"]))
            elif shared.get("landline"):
                lines.append("Landline: not listed separately for this person; the directory prints "
                             + ", ".join(shared["landline"]) + " in a cell shared with the adjacent row(s)")
            else:
                lines.append("Landline: " + NOT_LISTED)
        if fields is None or fields & {"extension", "landline"}:
            ext = r.get("extension")
            lines.append("Extension: " + (", ".join(ext) + " (last 3 digits of the printed landline)" if ext else NOT_LISTED))
        if fields is None or fields & {"email"}:
            if r.get("email"):
                lines.append("Official Email: " + ", ".join(r["email"]))
            elif r.get("email_raw"):
                lines.append("Official Email: printed in the directory as \"" + "\", \"".join(r["email_raw"])
                             + "\" (malformed in the source - please verify)")
            elif shared.get("email"):
                lines.append("Official Email: not listed separately; the directory prints "
                             + ", ".join(shared["email"]) + " in a cell shared with the adjacent row(s)")
            else:
                lines.append("Official Email: " + NOT_LISTED)
        if fields is None:
            lines.append("Office: " + (r.get("office") or NOT_LISTED))
        return lines

    def _render_role_card(self, r, fields):
        return "\n".join(self._role_lines(r, fields, True))

    def _render_entity(self, key, fields):
        roles = self.entities[key]
        if len(roles) == 1:
            return self._render_role_card(roles[0], fields)
        head = [f"Name: {roles[0].get('name') or NOT_LISTED}"]
        mobiles = list(OrderedDict.fromkeys(m for r in roles for m in (r.get("mobile") or [])))
        if fields is None or (fields & {"mobile"}):
            head.append("Mobile/Phone: " + _join(mobiles))
        blocks = []
        for i, r in enumerate(roles, 1):
            sub = self._role_lines(r, fields, False)
            sub = [ln for ln in sub if not ln.startswith("Mobile/Phone:")]
            blocks.append(f"Role {i}:\n  " + "\n  ".join(sub))
        return "\n".join(head) + f"\n(listed in {len(roles)} places in the directory)\n" + "\n".join(blocks)

    def _one_line_identity(self, key):
        roles = self.entities[key]
        bits = []
        for r in roles[:3]:
            desc = r.get("designation") or "designation not listed"
            unit = self._unit_label(r)
            bits.append(f"{desc}, {unit}" if unit else desc)
        extra = f" (+{len(roles) - 3} more)" if len(roles) > 3 else ""
        return f"{roles[0].get('name')} - " + "; ".join(bits) + extra

    def _render_ambiguous(self, name_tokens, keys):
        shown = keys[:MAX_CANDIDATES]
        text = (f"I found {len(keys)} people in the directory who match \"{' '.join(name_tokens)}\". "
                "Which one do you mean?\n")
        text += "\n".join(f"{i}. {self._one_line_identity(k)}" for i, k in enumerate(shown, 1))
        if len(keys) > len(shown):
            text += f"\n...and {len(keys) - len(shown)} more."
        text += "\n\nPlease ask again with the person's full name or add their department (e.g. \"Sanjeev Kumar Botany contact\")."
        return text

    def _render_list(self, recs, fields, units, rank, want_hod):
        title_bits = []
        if want_hod:
            title_bits.append("Heads of Department")
        elif rank:
            title_bits.append({"assistant": "Assistant Professors", "associate": "Associate Professors",
                               "professor": "Professors", "faculty": "Faculty"}[rank])
        else:
            title_bits.append("Matching entries")
        if units:
            title_bits.append("- " + " / ".join(u["full"] for u in units))
        total = len(recs)
        head = f"{' '.join(title_bits)} ({total} found):"
        lines = [head]
        for i, r in enumerate(recs[:MAX_LIST], 1):
            parts = []
            if r.get("mobile"):
                parts.append("Mobile " + ", ".join(r["mobile"]))
            if r.get("landline"):
                parts.append("Landline " + ", ".join(r["landline"]))
            if r.get("email"):
                parts.append("Email " + ", ".join(r["email"]))
            if fields:
                keep = []
                for p in parts:
                    if (p.startswith("Mobile") and "mobile" in fields) or (p.startswith("Landline") and fields & {"landline", "extension"}) \
                            or (p.startswith("Email") and "email" in fields):
                        keep.append(p)
                parts = keep
            who = r.get("name") or "(no name listed)"
            desc = r.get("designation") or ""
            unit = self._unit_label(r)
            ctx = ", ".join(x for x in (desc, unit if (not units or len(units) > 1 or r.get("department") is None) else None) if x)
            line = f"{i}. {who}" + (f" - {ctx}" if ctx else "")
            if parts:
                line += "\n   " + " | ".join(parts)
            lines.append(line)
        if total > MAX_LIST:
            lines.append(f"...and {total - MAX_LIST} more. Add a department or designation to narrow the list.")
        return "\n".join(lines)

    def _wrap(self, text, kind, n):
        return {"answer": text, "kind": kind, "matches": n}