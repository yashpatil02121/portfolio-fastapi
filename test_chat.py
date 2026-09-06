"""
Tests for the /ask pipeline.

Gemini is stubbed, so these run without an API key and without spending a
single token. Run with `pytest -q` or plain `python test_chat.py`.

What they guard:
  - every realistic portfolio question gets a real answer, not a dead end
    (the bug that made "his projects" unanswerable)
  - answers vary in wording instead of repeating one canned sentence
  - exact values (email, phone, links) are never reworded
  - each question is handled by the cheapest layer that can handle it, and
    the cost of a repeated question is bounded
"""

import re
import sys
import types

# --------------------------------------------------------------------------
# Stub the Gemini SDK before main.py imports it. The stub echoes a counter so
# every call returns different text - that is what lets us tell a fresh
# generation apart from a cached one.
# --------------------------------------------------------------------------
MODEL_PROMPTS = []
MODEL_FAILS = False
MODEL_REPEATS = False
_call_count = 0


class _StubModel:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def generate_content(self, prompt):
        global _call_count
        if MODEL_FAILS:
            raise RuntimeError("quota exhausted")
        MODEL_PROMPTS.append(prompt)
        _call_count += 1
        # The counter is deliberately independent of MODEL_PROMPTS, which tests
        # clear - otherwise the stub would repeat itself and hide real variation.
        text = "stubbed answer" if MODEL_REPEATS else f"stubbed answer #{_call_count}"
        return types.SimpleNamespace(text=text)


_stub = types.ModuleType("google.generativeai")
_stub.configure = lambda **kwargs: None
_stub.GenerativeModel = _StubModel
_google = types.ModuleType("google")
_google.generativeai = _stub
sys.modules.setdefault("google", _google)
sys.modules["google.generativeai"] = _stub

import faq  # noqa: E402
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from knowledge import kb  # noqa: E402

client = TestClient(main.app)

# Layers that answer without ever calling the model.
LOCAL_SOURCES = {"smalltalk", "quick_fact", "overview"}


def ask(question, fresh=True):
    """`fresh` clears the response cache, so tests don't leak into each other."""
    MODEL_PROMPTS.clear()
    if fresh:
        main._cache.clear()
    body = client.post("/ask", json={"query": question}).json()
    assert "response" in body, body
    assert body["response"].strip(), f"empty answer for {question!r}"
    return body


# --------------------------------------------------------------------------
# The regression: list questions must be answered, and answered completely.
# --------------------------------------------------------------------------
PROJECT_QUESTIONS = [
    "projects",
    "what are his projects",
    "what projects has he worked on",
    "tell me about his projects",
    "show me his projects",
    "list his projects",
    "what has he built",
    "his best project",
    "does he have side projects",
    "what apps has he built",
]

EXPERIENCE_QUESTIONS = [
    "experience",
    "work experience",
    "what is his experience",
    "tell me about his experience",
    "his work history",
    "where has he worked",
    "past experience",
    "professional experience",
    "career",
    "which companies has he worked for",
    "what jobs has he had",
]


def test_project_questions_list_every_project():
    expected = len(kb.by_topic["project"])
    for q in PROJECT_QUESTIONS:
        body = ask(q)
        assert body["source"] == "overview", (q, body["source"])
        assert body["response"].count("\n- ") == expected, (q, body["response"])
        assert not MODEL_PROMPTS, f"{q!r} should not cost an API call"


def test_experience_questions_list_every_role():
    expected = len(kb.by_topic["experience"])
    for q in EXPERIENCE_QUESTIONS:
        body = ask(q)
        assert body["source"] == "overview", (q, body["source"])
        assert body["response"].count("\n- ") == expected, (q, body["response"])
        assert "Smarter.Codes" in body["response"], q
        assert "Kulp Labs" in body["response"], q


def test_list_answers_are_well_formed():
    for q in ["what are his projects", "work experience", "achievements"]:
        response = ask(q)["response"]
        for line in response.splitlines()[1:]:
            assert line.startswith("- "), (q, line)
            assert not line.endswith("...."), (q, line)
            assert line.rstrip().endswith((".", "...")), (q, line)


def test_every_question_gets_an_answer():
    """No question anywhere should reach a dead end."""
    everything = PROJECT_QUESTIONS + EXPERIENCE_QUESTIONS + [
        "skills", "what are his skills", "his education", "qualifications",
        "certifications", "publications", "achievements", "his companies",
        "what is his tech stack", "who is Yash Patil", "what is his background",
        "how do I contact him", "what is his current job", "what is his CGPA",
        "did he do any internships", "what is Urban Ride", "tell me about kulp.ai",
        "what databases has he used", "why should we hire him",
        "what is his expected salary", "who won the world cup", "hi", "thanks",
    ]
    for q in everything:
        body = ask(q)
        assert len(body["response"]) > 15, (q, body["response"])


# --------------------------------------------------------------------------
# Wording variation.
# --------------------------------------------------------------------------
def test_local_answers_do_not_repeat_the_same_wording():
    """Greetings and profile fields rotate through hand-written phrasings."""
    for q in ["hi", "thanks", "what is his email", "where is he based",
              "what are his projects"]:
        seen = {ask(q)["response"] for _ in range(40)}
        assert len(seen) > 1, f"{q!r} always answers with the same words"
        assert not MODEL_PROMPTS, f"{q!r} should not cost an API call"


def test_exact_values_are_never_reworded():
    """Phrasing may vary; an email address may not."""
    checks = [
        ("what is his email", kb.profile["email"]),
        ("phone number", kb.profile["phone"]),
        ("github link", kb.profile["links"]["github"]),
        ("linkedin", kb.profile["links"]["linkedin"]),
        ("portfolio link", kb.profile["links"]["portfolio"]),
        ("where is he based", kb.profile["location"]),
    ]
    for q, value in checks:
        for _ in range(20):
            assert value in ask(q)["response"], (q, value)


def test_model_answers_vary_until_the_variant_budget_is_reached():
    q = "has he used redis"
    answers = {ask(q, fresh=(i == 0))["response"]
               for i in range(main.VARIANTS_PER_QUESTION)}
    assert len(answers) == main.VARIANTS_PER_QUESTION, answers


def test_overview_lead_in_varies_but_bullets_do_not():
    bullet_sets = set()
    leads = set()
    for _ in range(30):
        response = ask("what are his projects")["response"]
        head, _, body = response.partition("\n")
        leads.add(head)
        bullet_sets.add(body)
    assert len(leads) > 1, "the introducing line should vary"
    assert len(bullet_sets) == 1, "the list itself must stay stable"


# --------------------------------------------------------------------------
# The canned out-of-scope sentence must not come back.
# --------------------------------------------------------------------------
BANNED = (
    "I don't have that detail in Yash Patil's profile. I can help with his work "
    "experience, projects, tech stack, education, certifications and contact "
    "details - or you can ask him directly at yashspatil202002@gmail.com."
)


def test_the_old_canned_sentence_is_gone():
    assert not hasattr(faq, "OUT_OF_SCOPE"), "the single fixed message is retired"
    for q in ["what is his expected salary", "who won the world cup",
              "what is the weather today", "write me a python script"]:
        body = ask(q)
        assert body["response"] != BANNED, q
        assert body["source"] == "redirect", (q, body["source"])
        assert MODEL_PROMPTS, f"{q!r} should be answered by the model, not a canned line"


def test_uncovered_questions_are_redirected_not_answered():
    """The model gets told not to answer, and is given no material to do so."""
    ask("who won the world cup")
    prompt = MODEL_PROMPTS[0]
    assert "Do not answer the question itself" in prompt
    assert "invent" in prompt
    # No chunk text is shipped for a question nothing covers.
    assert len(prompt) < 900, len(prompt)


def test_redirect_falls_back_to_a_varied_line_when_the_api_is_down():
    global MODEL_FAILS
    MODEL_FAILS = True
    try:
        seen = {ask("who won the world cup")["response"] for _ in range(40)}
        assert seen <= set(faq.OUT_OF_SCOPE_FALLBACKS)
        assert len(seen) > 1, "even the fallback should not be a single fixed line"
    finally:
        MODEL_FAILS = False


# --------------------------------------------------------------------------
# Routing: every question handled by the cheapest layer that can handle it.
# --------------------------------------------------------------------------
ROUTES = [
    # smalltalk - never worth an API call
    ("Hi", "smalltalk"),
    ("hello!", "smalltalk"),
    ("thanks", "smalltalk"),
    ("bye", "smalltalk"),
    ("ok", "smalltalk"),
    ("how are you", "smalltalk"),
    ("who are you", "smalltalk"),
    ("help", "smalltalk"),
    ("", "smalltalk"),
    # single profile fields
    ("what is his email", "quick_fact"),
    ("phone number", "quick_fact"),
    ("github link", "quick_fact"),
    ("linkedin", "quick_fact"),
    ("where is he based", "quick_fact"),
    ("how many years of experience", "quick_fact"),
    ("what languages does he speak", "quick_fact"),
    ("portfolio link", "quick_fact"),
    # list answers
    ("what are his projects", "overview"),
    ("work experience", "overview"),
    ("achievements", "overview"),
    # stored questions - chunk is the material, model supplies the wording
    ("Who is Yash Patil?", "faq"),
    ("What is his tech stack?", "faq"),
    ("Does he know React?", "faq"),
    ("What is Urban Ride?", "faq"),
    ("what certifications does he have", "faq"),
    ("what is his CGPA", "faq"),
    # phrasings the stored questions miss
    ("tell me about him", "intent"),
    ("what is his background", "intent"),
    ("why should we hire him", "intent"),
    ("qualifications", "intent"),
    # genuinely new questions
    ("has he used redis", "retrieval"),
    ("what did he build at Kulp Labs", "retrieval"),
    ("tell me about his celery experience", "retrieval"),
    ("has he worked with payments", "retrieval"),
    # nothing in the profile covers these
    ("what is his expected salary", "redirect"),
    ("what is the weather today", "redirect"),
    ("who won the world cup", "redirect"),
    ("write me a python script to sort a list", "redirect"),
]


def test_routing():
    for question, expected in ROUTES:
        body = ask(question)
        assert body["source"] == expected, (question, body["source"], expected)


def test_specific_questions_are_not_swallowed_by_list_answers():
    """"his celery experience" is a real question, not "his experience"."""
    for q in ["tell me about his celery experience",
              "does he have experience with vue",
              "how much experience does he have with react",
              "his react projects",
              "his frappe projects",
              "what did he build at Kulp Labs"]:
        body = ask(q)
        assert body["source"] == "retrieval", (q, body["source"])


# --------------------------------------------------------------------------
# Cost guards.
# --------------------------------------------------------------------------
def test_local_layers_never_call_the_model():
    for question, expected in ROUTES:
        if expected in LOCAL_SOURCES:
            ask(question)
            assert not MODEL_PROMPTS, f"{question!r} cost an API call"


def test_model_prompt_stays_small():
    """The old prompt shipped the whole resume (~5000 chars) every time."""
    for question, expected in ROUTES:
        if expected in LOCAL_SOURCES:
            continue
        ask(question)
        assert MODEL_PROMPTS, question
        assert len(MODEL_PROMPTS[0]) < 2500, (question, len(MODEL_PROMPTS[0]))


def test_repeat_question_costs_at_most_the_variant_budget():
    q = "has he integrated any payment gateway"
    calls = 0
    ask(q)  # clears the cache
    calls += 1
    for _ in range(20):
        MODEL_PROMPTS.clear()
        client.post("/ask", json={"query": q})
        calls += len(MODEL_PROMPTS)
    assert calls == main.VARIANTS_PER_QUESTION, calls
    entry = main._cache[faq.normalize(q)]
    assert entry["calls"] == main.VARIANTS_PER_QUESTION
    assert len(entry["variants"]) == main.VARIANTS_PER_QUESTION


def test_budget_holds_even_when_the_model_repeats_itself():
    """A low-variety model must not be re-called forever chasing new wordings."""
    global MODEL_REPEATS
    MODEL_REPEATS = True
    try:
        q = "has he deployed anything to production"
        ask(q)
        calls = 1
        for _ in range(15):
            MODEL_PROMPTS.clear()
            client.post("/ask", json={"query": q})
            calls += len(MODEL_PROMPTS)
        assert calls == main.VARIANTS_PER_QUESTION, calls
    finally:
        MODEL_REPEATS = False


def test_cache_is_bounded():
    main._cache.clear()
    for i in range(main.CACHE_SIZE + 25):
        client.post("/ask", json={"query": f"has he used library number {i}"})
    assert len(main._cache) <= main.CACHE_SIZE


def test_model_is_configured_for_cheap_but_varied_output():
    config = main.model.kwargs["generation_config"]
    assert config["max_output_tokens"] <= 300
    assert config["temperature"] >= 0.7, "too cold to reword between visits"
    assert main.model.kwargs["system_instruction"], "static rules belong here"


def test_api_failure_still_answers_from_the_knowledge_base():
    global MODEL_FAILS
    MODEL_FAILS = True
    try:
        body = ask("has he used mariadb")
        assert body["source"] == "retrieval"
        assert "MariaDB" in body["response"], "should fall back to the retrieved text"
        # A dead API key must not be reported as a model-written answer - that
        # is exactly what hid a suspended key during development.
        assert body["phrasing"] == "fallback", body
    finally:
        MODEL_FAILS = False


def test_phrasing_field_reports_where_the_wording_came_from():
    assert ask("hi")["phrasing"] == "local"
    assert ask("what is his email")["phrasing"] == "local"
    assert ask("what are his projects")["phrasing"] == "local"

    q = "has he used celery in production"
    assert ask(q)["phrasing"] == "model"
    for _ in range(main.VARIANTS_PER_QUESTION):
        client.post("/ask", json={"query": q})
    assert ask(q, fresh=False)["phrasing"] == "cache"


def test_edge_case_inputs():
    assert ask("   ")["source"] == "smalltalk"
    assert ask("?" * 50)["source"] in {"smalltalk", "redirect"}
    assert ask("x" * 5000)["source"] == "redirect"


def test_health_and_root():
    assert client.get("/").json()["message"]
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["chunks"] == len(kb.chunks)


# --------------------------------------------------------------------------
# Standalone runner, so the suite works without pytest too.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
