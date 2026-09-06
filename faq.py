"""
Zero-API answer layers.

Anything answered here never reaches Gemini:
  1. smalltalk    - "hi", "thanks", "who are you", "bye"
  2. quick facts  - email / phone / links / location, read straight off profile
  3. faq          - the question is (near) identical to a chunk's own example
                    question, so the chunk text already IS the answer
  4. intents      - common phrasings ("tell me about him", "why hire him")
                    mapped straight onto a chunk
  5. out of scope - retrieval found nothing; a canned redirect beats a guess
"""

import re

from knowledge import kb, normalize, tokenize

# A question this close to a stored example question is answered from the KB
# verbatim instead of being paraphrased by the model.
FAQ_MATCH_THRESHOLD = 0.75

P = kb.profile
LINKS = P["links"]

OUT_OF_SCOPE = (
    f"I don't have that detail in {P['preferred_name']}'s profile. I can help with his "
    "work experience, projects, tech stack, education, certifications and contact "
    f"details - or you can ask him directly at {P['email']}."
)

CAPABILITIES = (
    f"I'm the AI assistant on {P['preferred_name']}'s portfolio. Ask me about his work "
    "experience, projects, tech stack, education, certifications or how to contact him."
)

# ---------------------------------------------------------------------------
# 1. Smalltalk - matched against the whole (normalized) message, so "hi" is free
#    but "hi, what is his tech stack" still goes through retrieval.
# ---------------------------------------------------------------------------
_SMALLTALK = [
    (
        r"^(hi|hey|hello|hii+|yo|hola|namaste|good (morning|afternoon|evening)|"
        r"hi there|hey there|greetings)[!. ]*$",
        f"Hi! I'm {P['preferred_name']}'s portfolio assistant. Ask me about his "
        "experience, projects, skills or how to reach him.",
    ),
    (
        r"^(thanks|thank you|thx|ty|thanks a lot|thank you so much|appreciate it)[!. ]*$",
        "Happy to help! Anything else you'd like to know about Yash?",
    ),
    (
        r"^(bye|goodbye|see ya|see you|cya|good night|gn)[!. ]*$",
        f"Thanks for stopping by. Feel free to reach out to Yash at {P['email']}.",
    ),
    (
        r"^(ok|okay|k|cool|nice|great|awesome|got it|sure|fine|hmm+|alright)[!. ]*$",
        "Anything else you'd like to know about Yash?",
    ),
    (
        r"^(how are you|how r u|whats up|what's up|sup|how do you do)[?! ]*$",
        f"Doing well, thanks! I'm here to answer questions about {P['preferred_name']}. "
        "What would you like to know?",
    ),
    (
        r"^(who are you|what are you|what can you do|what do you do|help|"
        r"how can you help|what can i ask)[?! ]*$",
        CAPABILITIES,
    ),
    (r"^(test|testing|ping)[!. ]*$", "I'm up and running. Ask me anything about Yash."),
]
_SMALLTALK = [(re.compile(p), a) for p, a in _SMALLTALK]

# ---------------------------------------------------------------------------
# 2. Quick facts - one-line answers assembled from `profile`.
#    (pattern, tokens the fact owns, answer). A fact only fires on a short
#    question whose remaining tokens are generic: "his email" is a quick fact,
#    "his celery experience" is not.
# ---------------------------------------------------------------------------
_QUICK_FACTS = [
    (r"\b(e ?mail|mail id|email id)\b",
     {"email", "mail", "id", "address"},
     f"You can email {P['preferred_name']} at {P['email']}."),
    (r"\b(phone|mobile|contact number|whatsapp)\b",
     {"phone", "mobile", "number", "whatsapp", "contact"},
     f"His phone number is {P['phone']}."),
    (r"\bgithub\b", {"github"}, f"His GitHub is {LINKS['github']}."),
    (r"\blinked ?in\b", {"linkedin"}, f"His LinkedIn is {LINKS['linkedin']}."),
    (r"\bleet ?code\b", {"leetcode"}, f"His LeetCode profile is {LINKS['leetcode']}."),
    (r"\bhacker ?rank\b", {"hackerrank"},
     f"His HackerRank profile is {LINKS['hackerrank']}."),
    (r"\bportfolio\b",
     {"portfolio", "link", "url", "site", "website"},
     f"His portfolio is {LINKS['portfolio']} - you're on it right now."),
    (r"\b(where (is|does) he|based|located|which city|his location|where does he live)\b",
     {"where", "based", "located", "location", "city", "live", "lives", "he"},
     f"He is based in {P['location']}."),
    (r"\b(full name|his name|what is he called)\b",
     {"full", "name", "called", "he"},
     f"His full name is {P['full_name']}; he goes by {P['preferred_name']}."),
    (r"\b(languages? (he|does he)? ?speaks?|spoken languages?|which languages)\b",
     {"language", "languages", "speak", "speaks", "spoken", "he"},
     "He speaks " + ", ".join(P["languages_spoken"]) + "."),
    (r"\b(how many years|years of experience|how much experience|how experienced)\b",
     {"how", "many", "much", "years", "experience", "experienced", "he", "have", "does"},
     f"He has {P['years_experience']} years of professional experience - currently "
     f"{P['current_role']}."),
]
_QUICK_FACTS = [(re.compile(p), own, a) for p, own, a in _QUICK_FACTS]

# Words that signal the visitor wants substance, not a profile field.
_NOT_A_QUICK_FACT = {
    "built", "build", "building", "made", "powers", "tech", "stack", "framework",
    "frameworks", "project", "projects", "work", "worked", "working", "use", "used",
    "using", "why", "compare", "summarize", "summary", "explain",
}

# ---------------------------------------------------------------------------
# 4. Intents - phrasings the chunk `q` lists don't cover, mapped onto a chunk.
# ---------------------------------------------------------------------------
_INTENTS = [
    (r"\b(who is|tell me about|about him|about yash|introduce|intro|bio|"
     r"his background|overview|elevator pitch|resume|cv)\b", "summary"),
    (r"\b(why (should|would) (we|i|you) hire|good fit|right fit|is he a good|"
     r"as a candidate|what do you think of him|should we hire)\b", "summary"),
    (r"\b(how (do|can) i (contact|reach|hire)|get in touch|reach out|contact him|"
     r"contact details|contact info)\b", "contact"),
    (r"\b(current (job|role|company|position)|where does he work|working now|"
     r"present employer|who does he work for)\b", "job-smartercodes"),
    (r"\b(tech stack|skill set|skills|technologies|what does he know)\b",
     "skills-engineering"),
    (r"\b(education|qualification|degree|college|university|studied|cgpa)\b",
     "education"),
    (r"\b(certification|certificate|certificates|courses)\b", "certifications"),
]
_INTENTS = [(re.compile(p), cid) for p, cid in _INTENTS]

# ---------------------------------------------------------------------------
# 3. FAQ index built from every chunk's own example questions.
# ---------------------------------------------------------------------------
_FAQ = [
    (tokenize(q), normalize(q), chunk)
    for chunk in kb.chunks
    for q in chunk.get("q", [])
]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def smalltalk(query: str):
    q = normalize(query)
    if not q:
        return CAPABILITIES
    for pattern, answer in _SMALLTALK:
        if pattern.match(q):
            return answer
    return None


def quick_fact(query: str):
    """Fires only on short questions asking for a single profile field."""
    q_norm = normalize(query)
    tokens = tokenize(query)
    if not tokens or len(tokens) > 7:
        return None
    if tokens & _NOT_A_QUICK_FACT:
        return None

    for pattern, own, answer in _QUICK_FACTS:
        if not pattern.search(q_norm):
            continue
        # A topic word the fact doesn't own ("celery", "django") means the
        # visitor wants context, not a profile field.
        if (tokens - own) & kb.specific_tokens:
            continue
        return answer
    return None


def faq(query: str):
    """Return a stored chunk's text when the question is already a known one."""
    q_tokens = tokenize(query)
    q_norm = normalize(query)
    if not q_norm:
        return None

    best, best_score = None, 0.0
    for f_tokens, f_norm, chunk in _FAQ:
        score = 1.0 if q_norm == f_norm else _jaccard(q_tokens, f_tokens)
        if score > best_score:
            best, best_score = chunk, score

    return best["text"] if best_score >= FAQ_MATCH_THRESHOLD else None


def intent(query: str):
    """Common phrasings the `q` lists miss, answered from the mapped chunk."""
    q_norm = normalize(query)
    tokens = tokenize(query)
    if not q_norm:
        return None
    for pattern, chunk_id in _INTENTS:
        match = pattern.search(q_norm)
        if not match:
            continue
        chunk = kb.by_id.get(chunk_id)
        if not chunk:
            continue
        # "tell me about him" is this chunk; "tell me about his celery
        # experience" is not - a topic word left over once the matched phrase
        # and the chunk's own tags are removed means this deserves retrieval.
        leftover = tokens - tokenize(match.group(0)) - kb.tag_tokens[chunk_id]
        if leftover & kb.specific_tokens:
            continue
        return chunk["text"]
    return None
