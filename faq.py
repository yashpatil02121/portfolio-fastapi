"""
Answer layers that run before (or instead of) the model.

  1. smalltalk    - "hi", "thanks", "who are you", "bye"
  2. quick facts  - email / phone / links / location, read straight off profile
  3. overviews    - list questions ("his projects") built from a whole topic
  4. faq          - the question is (near) identical to a chunk's own example
                    question, so the chunk text is the material for the answer
  5. intents      - common phrasings ("tell me about him", "why hire him")
                    mapped straight onto a chunk

Layers 1-3 answer on their own and never call the model: a greeting isn't
worth a token, and an email address must be reproduced exactly, not reworded.
They vary their wording by rotating through hand-written phrasings. Layers 4-5
return source material that main.py hands to the model to phrase.
"""

import random
import re

from knowledge import kb, normalize, tokenize

# A question this close to a stored example question is answered from that
# chunk instead of going through retrieval.
FAQ_MATCH_THRESHOLD = 0.75

P = kb.profile
LINKS = P["links"]

# Only used if the model is unreachable - normally the redirect is generated.
OUT_OF_SCOPE_FALLBACKS = (
    f"That's not something {P['preferred_name']}'s profile covers, but I can tell you "
    "about his roles, projects, tech stack, education or how to reach him.",
    f"I don't have that one. Ask me about his work at Smarter.Codes, his projects or "
    "his tech stack and I'll be more use.",
    f"His profile doesn't go into that. Happy to walk you through his experience, "
    f"projects or skills instead - or you can email him at {P['email']}.",
)

CAPABILITIES = (
    f"I'm the AI assistant on {P['preferred_name']}'s portfolio. Ask me about his work "
    "experience, projects, tech stack, education, certifications or how to contact him.",
    f"I answer questions about {P['preferred_name']} - his roles, the things he's "
    "built, the stack he works in, his education and how to get in touch.",
    "I'm here to talk about Yash's background: where he's worked, what he's built, "
    "what he knows and how to reach him. What would you like to know?",
)

# What the profile can actually talk about - given to the model when it has to
# turn a visitor away, so the redirect names real topics.
COVERAGE = (
    "work experience and current role, individual projects, tech stack and skills, "
    "education, certifications, a published paper, and contact details"
)


def _pick(options):
    """One of several phrasings, so repeat visitors don't see a canned line."""
    return random.choice(options)


# ---------------------------------------------------------------------------
# 1. Smalltalk - matched against the whole (normalized) message, so "hi" is free
#    but "hi, what is his tech stack" still goes through retrieval.
# ---------------------------------------------------------------------------
_SMALLTALK = [
    (
        r"^(hi|hey|hello|hii+|yo|hola|namaste|good (morning|afternoon|evening)|"
        r"hi there|hey there|greetings)[!. ]*$",
        (
            f"Hi! I'm {P['preferred_name']}'s portfolio assistant. Ask me about his "
            "experience, projects, skills or how to reach him.",
            "Hey there! Happy to answer anything about Yash - his work, his projects "
            "or his tech stack.",
            "Hello! What would you like to know about Yash? His roles, projects and "
            "skills are all fair game.",
            "Hi! You can ask me anything about Yash's background - where he's worked, "
            "what he's built, or how to get in touch.",
        ),
    ),
    (
        r"^(thanks|thank you|thx|ty|thanks a lot|thank you so much|appreciate it)[!. ]*$",
        (
            "Happy to help! Anything else you'd like to know about Yash?",
            "Anytime. Ask away if something else comes to mind.",
            "You're welcome - let me know if you want to dig into anything else.",
            "Glad that helped. Anything else about his work or projects?",
        ),
    ),
    (
        r"^(bye|goodbye|see ya|see you|cya|good night|gn)[!. ]*$",
        (
            f"Thanks for stopping by. Feel free to reach out to Yash at {P['email']}.",
            f"Take care! If you'd like to talk to him directly, he's at {P['email']}.",
            f"See you around. His inbox is open at {P['email']}.",
        ),
    ),
    (
        r"^(ok|okay|k|cool|nice|great|awesome|got it|sure|fine|hmm+|alright)[!. ]*$",
        (
            "Anything else you'd like to know about Yash?",
            "Got it. What else can I tell you?",
            "Sure thing - ask me anything else about his work.",
        ),
    ),
    (
        r"^(how are you|how r u|whats up|what's up|sup|how do you do)[?! ]*$",
        (
            f"Doing well, thanks! I'm here to answer questions about "
            f"{P['preferred_name']}. What would you like to know?",
            "All good here. Ask me anything about Yash's work or projects.",
            "Doing fine! Happy to talk you through Yash's background whenever "
            "you're ready.",
        ),
    ),
    (
        r"^(who are you|what are you|what can you do|what do you do|help|"
        r"how can you help|what can i ask)[?! ]*$",
        CAPABILITIES,
    ),
    (
        r"^(test|testing|ping)[!. ]*$",
        (
            "I'm up and running. Ask me anything about Yash.",
            "Working fine! Go ahead and ask about his experience or projects.",
            "All systems go - what would you like to know?",
        ),
    ),
]
_SMALLTALK = [(re.compile(p), a) for p, a in _SMALLTALK]

# ---------------------------------------------------------------------------
# 2. Quick facts - exact values from `profile`, wrapped in rotating phrasings.
#    (pattern, tokens the fact owns, phrasings, strict). A fact only fires on a
#    short question whose remaining tokens are generic: "his email" is a quick
#    fact, "his celery experience" is not. `strict` facts additionally refuse
#    any leftover word at all - "how much experience with React" is a question
#    about React, not about his total years.
# ---------------------------------------------------------------------------
_QUICK_FACTS = [
    (r"\b(e ?mail|mail id|email id)\b",
     {"email", "mail", "id", "address"},
     (f"You can email {P['preferred_name']} at {P['email']}.",
      f"His email is {P['email']} - that's the fastest way to reach him.",
      f"Drop him a line at {P['email']}.",
      f"Reach him by email at {P['email']}."),
     False),
    (r"\b(phone|mobile|contact number|whatsapp)\b",
     {"phone", "mobile", "number", "whatsapp", "contact"},
     (f"His phone number is {P['phone']}.",
      f"You can call him on {P['phone']}.",
      f"He's reachable at {P['phone']}."),
     False),
    (r"\bgithub\b", {"github"},
     (f"His GitHub is {LINKS['github']}.",
      f"You'll find his code at {LINKS['github']}.",
      f"GitHub: {LINKS['github']}."),
     False),
    (r"\blinked ?in\b", {"linkedin"},
     (f"His LinkedIn is {LINKS['linkedin']}.",
      f"You can connect with him on LinkedIn: {LINKS['linkedin']}.",
      f"LinkedIn: {LINKS['linkedin']}."),
     False),
    (r"\bleet ?code\b", {"leetcode"},
     (f"His LeetCode profile is {LINKS['leetcode']}.",
      f"He's on LeetCode at {LINKS['leetcode']}."),
     False),
    (r"\bhacker ?rank\b", {"hackerrank"},
     (f"His HackerRank profile is {LINKS['hackerrank']}.",
      f"He's on HackerRank at {LINKS['hackerrank']}."),
     False),
    (r"\bportfolio\b",
     {"portfolio", "link", "url", "site", "website"},
     (f"His portfolio is {LINKS['portfolio']} - you're on it right now.",
      f"You're already on it: {LINKS['portfolio']}.",
      f"This site is his portfolio - {LINKS['portfolio']}."),
     False),
    (r"\b(where (is|does) he|based|located|which city|his location|where does he live)\b",
     {"where", "based", "located", "location", "city", "live", "lives", "he"},
     (f"He is based in {P['location']}.",
      f"He lives in {P['location']}.",
      f"He's in {P['location']}, and works remotely."),
     True),
    (r"\b(full name|his name|what is he called)\b",
     {"full", "name", "called", "he"},
     (f"His full name is {P['full_name']}; he goes by {P['preferred_name']}.",
      f"{P['full_name']} - most people call him {P['preferred_name']}.",
      f"He's {P['full_name']}, usually {P['preferred_name']}."),
     True),
    (r"\b(languages? (he|does he)? ?speaks?|spoken languages?|which languages)\b",
     {"language", "languages", "speak", "speaks", "spoken", "he"},
     ("He speaks " + ", ".join(P["languages_spoken"]) + ".",
      "He's comfortable in " + ", ".join(P["languages_spoken"]) + ".",
      ", ".join(P["languages_spoken"]) + " - all three."),
     True),
    (r"\b(how many years|years of experience|how much experience|how experienced)\b",
     {"how", "many", "much", "years", "experience", "experienced", "he", "have", "does"},
     (f"He has {P['years_experience']} years of professional experience - currently "
      f"{P['current_role']}.",
      f"{P['years_experience']} years so far. Right now he's {P['current_role']}.",
      f"About {P['years_experience']} years in the industry, currently as "
      f"{P['current_role']}."),
     True),
]
_QUICK_FACTS = [(re.compile(p), own, a, strict) for p, own, a, strict in _QUICK_FACTS]

# Words that signal the visitor wants substance, not a profile field.
_NOT_A_QUICK_FACT = {
    "built", "build", "building", "made", "powers", "tech", "stack", "framework",
    "frameworks", "project", "projects", "work", "worked", "working", "use", "used",
    "using", "why", "compare", "summarize", "summary", "explain",
}

# ---------------------------------------------------------------------------
# 3. Overviews - list questions ("what are his projects?") that no single chunk
#    answers, built from every chunk under the matching topic(s). The bullets
#    stay verbatim (a reworded list is a list that quietly loses entries); the
#    line introducing them rotates.
# ---------------------------------------------------------------------------
_OVERVIEWS = [
    (r"\b(projects?|apps?|applications|things he (has )?built|what has he built|"
     r"portfolio of work|side projects?)\b",
     ["project"],
     ("Yash Patil has worked on {n} projects:",
      "Here are the {n} projects he's worked on:",
      "He's built {n} projects so far:",
      "{n} projects, professional and academic:")),
    (r"\b(work (experience|history)|professional experience|career|"
     r"companies|employers?|jobs?|where has he worked|previous roles?|"
     r"past (experience|roles?)|experience)\b",
     ["experience"],
     ("Yash Patil's work experience ({n} roles):",
      "He's held {n} roles so far:",
      "Here's where he has worked ({n} roles):",
      "His career so far, {n} roles:")),
    (r"\b(achievements?|accomplishments?|awards?)\b",
     ["publication", "certifications", "extracurricular"],
     ("Yash Patil's achievements:",
      "A few things he's proud of:",
      "Here's what stands out:")),
]
_OVERVIEWS = [(re.compile(p), t, leads) for p, t, leads in _OVERVIEWS]

# Words that trigger an overview or merely frame one - never a reason to reject
# it as "too specific".
_AGGREGATE_WORDS = {
    "project", "projects", "experience", "experienced", "work", "worked",
    "working", "career", "job", "jobs", "company", "companies", "employer",
    "employers", "history", "built", "build", "building", "made", "roles",
    "role", "app", "apps", "applications", "achievements", "accomplishments",
    "list", "show", "best", "biggest", "main", "past", "previous", "side",
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
    (r"\b(education|qualifications?|degree|college|university|studied|cgpa|"
     r"academic background)\b", "education"),
    (r"\b(certification|certificate|certificates|courses)\b", "certifications"),
]
_INTENTS = [(re.compile(p), cid) for p, cid in _INTENTS]

# ---------------------------------------------------------------------------
# 5. FAQ index built from every chunk's own example questions.
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
        return _pick(CAPABILITIES)
    for pattern, answers in _SMALLTALK:
        if pattern.match(q):
            return _pick(answers)
    return None


def quick_fact(query: str):
    """Fires only on short questions asking for a single profile field."""
    q_norm = normalize(query)
    tokens = tokenize(query)
    if not tokens or len(tokens) > 7:
        return None
    if tokens & _NOT_A_QUICK_FACT:
        return None

    for pattern, own, answers, strict in _QUICK_FACTS:
        if not pattern.search(q_norm):
            continue
        leftover = tokens - own
        # A topic word the fact doesn't own ("celery", "django") means the
        # visitor wants context, not a profile field.
        if leftover & kb.specific_tokens:
            continue
        if strict and leftover:
            continue
        return _pick(answers)
    return None


def overview(query: str):
    """List answers ("what are his projects?") composed from a whole topic."""
    q_norm = normalize(query)
    tokens = tokenize(query)
    if not q_norm:
        return None

    for pattern, topics, leads in _OVERVIEWS:
        match = pattern.search(q_norm)
        if not match:
            continue
        # "his projects" is a list question; "his frappe projects" is not.
        # Any profile word left over once the matched phrase and the framing
        # words are removed means the visitor is asking about that thing, not
        # asking for the list.
        leftover = tokens - tokenize(match.group(0)) - _AGGREGATE_WORDS
        if leftover & kb.vocabulary:
            continue
        answer = kb.overview(topics, _pick(leads))
        if answer:
            return answer
    return None


def faq(query: str):
    """The chunk whose own example questions match - source for the answer."""
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
    """Common phrasings the `q` lists miss, mapped onto a chunk."""
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
