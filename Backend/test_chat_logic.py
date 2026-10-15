"""
Offline checks for the CUPB-GPT chat logic. Run from the Backend folder:

    python test_chat_logic.py

What this does NOT do: it never calls Groq and never starts a web server.
  - Sections 1-3 run your real search / routing code on your real data.
  - Section 4 swaps in a small recording stand-in for the Groq client so we
    can SEE what would be sent to the model. That is a routing check, not a
    live-model test. For live tests use the curl/Invoke-RestMethod commands.
"""
import os
import sys

# The OpenAI client object is created at import time and needs *some* value.
# This placeholder is never sent anywhere because this script makes no calls.
os.environ.setdefault("GROQ_API_KEY", "offline-test-placeholder")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main  # noqa: E402

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print("PASS  " + name)
    else:
        failed += 1
        print("FAIL  " + name + ("  -> " + str(detail) if detail else ""))


def names(records):
    return [r["name"] for r in records]


# ---------------------------------------------------------------- 1. routing
print("\n== 1. Is it a campus question? ==")
for q in ["Hi", "hello", "how are you", "Thank you", "Who are you?",
          "What can you do?", "Explain what an AI chatbot is.",
          "What is the capital of France?"]:
    check("general -> no campus search: " + q, not main.is_campus_question(q))

for q in ["CSE department kahan hai?", "Where is the Computer Centre?",
          "Room 215 kya hai?", "Hi, where is the CSE department?",
          "Where is the Department of Astrophysics?",
          "Tell me about the CSE department, and then tell me where it is."]:
    check("campus -> search: " + q, main.is_campus_question(q))

# ---------------------------------------------------------------- 2. search
print("\n== 2. Search results ==")
recs, note = main.search_campus("CSE department kahan hai?")
check("CSE: found records", len(recs) > 0, recs)
check("CSE: every record is a Computer Science one",
      all("computer science" in r["name"].lower() for r in recs), names(recs))
check("CSE: no Environmental / Food / Pharma noise",
      not any(w in " ".join(names(recs)).lower()
              for w in ["environmental", "food", "aerobiology", "drug"]), names(recs))
check("CSE: includes rooms 203 and 215",
      {"203", "215"} <= {r.get("room") for r in recs}, [r.get("room") for r in recs])

recs, note = main.search_campus("Where is the Computer Centre?")
check("Computer Centre: exactly the CC office",
      len(recs) == 1 and "Computer Centre" in recs[0]["name"], names(recs))

recs, note = main.search_campus("Room 215 kya hai?")
check("Room 215: exactly room 215", len(recs) == 1 and recs[0]["room"] == "215", recs)

recs, note = main.search_campus("What is in Room 205?")
check("Room 205 (suggestion chip) still works",
      len(recs) >= 1 and all("205" in r["room"] for r in recs), recs)

recs, note = main.search_campus("Room 999 kya hai?")
check("Missing room: no records + note mentions 999", recs == [] and "999" in (note or ""), (recs, note))

recs, note = main.search_campus("Where is the Department of Astrophysics?")
check("Nonexistent department: no records + note", recs == [] and note, (recs, note))

recs, note = main.search_campus("Hi, where is the CSE department?")
cse_recs, _ = main.search_campus("CSE department kahan hai?")
check("Mixed greeting + CSE: same records as plain CSE query", recs == cse_recs, names(recs))

recs, note = main.search_campus("Tell me about the CSE department, and then tell me where it is.")
check("Two-part question: same CSE records", recs == cse_recs, names(recs))

recs, note = main.search_campus("Where is Physics Department")
check("Physics (suggestion chip): results, all mention physics",
      len(recs) > 0 and all("physics" in r["name"].lower() for r in recs), names(recs))

recs, note = main.search_campus("Where is pharmacology classroom?")
check("Pharmacology (suggestion chip): results, all mention pharmacology",
      len(recs) > 0 and all("pharmacology" in r["name"].lower() for r in recs), names(recs))

recs, note = main.search_campus("AI-ML lab kahan hai")
check("AI-ML alias still works (room 113)", any(r.get("room") == "113" for r in recs), names(recs))

# ------------------------------------------- 3. no invented rooms in the data
print("\n== 3. Records handed to the model come only from campus.json ==")
all_rooms = {r["room"] for r in main.campus["rooms"]}
for q in ["CSE department kahan hai?", "Where is Physics Department", "Room 215 kya hai?"]:
    recs, _ = main.search_campus(q)
    check("only real rooms returned: " + q,
          all(("room" not in r) or r["room"] in all_rooms for r in recs))

# --------------------------------- 4. what is sent to the model (stand-in)
print("\n== 4. Routing into the model call (recording stand-in, NOT live Groq) ==")


class _Msg:
    content = "stand-in reply"


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]


class _Completions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp()


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class RecordingClient:
    def __init__(self):
        self.chat = _Chat()


main.client = RecordingClient()
calls = main.client.chat.completions.calls


def ask(text):
    calls.clear()
    out = main.ask(main.Question(question=text))
    return out, (calls[0] if calls else None)


for q in ["Hi", "Thank you", "Who are you?", "What can you do?",
          "Explain what an AI chatbot is.", "What is the capital of France?"]:
    out, call = ask(q)
    user_msg = call["messages"][1]["content"] if call else ""
    check("general reaches the model, no campus block: " + q,
          call is not None and user_msg == q and "CAMPUS RECORDS" not in user_msg)

out, call = ask("CSE department kahan hai?")
user_msg = call["messages"][1]["content"]
check("campus question reaches the model WITH records",
      "CAMPUS RECORDS" in user_msg and "Computer Science" in user_msg)
check("campus question uses temperature 0", call["temperature"] == 0)

out, call = ask("Where is the Department of Astrophysics?")
user_msg = call["messages"][1]["content"]
check("nonexistent department: model still called, told nothing was found",
      "NOTE:" in user_msg and "[" not in user_msg.split("USER MESSAGE")[0])

out, call = ask("Hi, where is the CSE department?")
user_msg = call["messages"][1]["content"]
check("mixed greeting + campus: records AND original text both sent",
      "Computer Science" in user_msg and "Hi, where is the CSE department?" in user_msg)

check("model name unchanged", call["model"] == "openai/gpt-oss-120b")
check("system prompt present", call["messages"][0]["role"] == "system")
check("response shape is exactly {'answer': ...}", out == {"answer": "stand-in reply"}, out)

out, _ = ask("   ")
check("blank question: friendly answer, still {'answer': ...}",
      list(out.keys()) == ["answer"] and len(calls) == 0)

print("\n{} passed, {} failed".format(passed, failed))
sys.exit(1 if failed else 0)