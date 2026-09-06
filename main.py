import random
from collections import OrderedDict

import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import faq
from config import GEMINI_API_KEY, MODEL_NAME
from knowledge import kb

app = FastAPI(title="Portfolio AI Backend")

# CORS (Allowed Hosts)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://portfolio-yash-patil.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Gemini
# ---------------------------
# The instructions are static, so they live in system_instruction instead of
# being re-written into every prompt. Only the retrieved context and the
# question change per request.
SYSTEM_INSTRUCTION = (
    "You are the AI assistant on Yash Patil's portfolio website. Answer only from "
    "the CONTEXT provided in the message. If the context does not contain the "
    "answer, say so and point the visitor at something you can help with. Reply in "
    "2-4 sentences, plain text, third person, warm but not salesy. Vary your "
    "wording naturally - do not reuse a stock opening line. Never invent facts, "
    "dates, numbers or links: copy those from the context exactly."
)

# Nudges that push each retry toward a different phrasing of the same facts.
STYLE_NUDGES = (
    "Answer conversationally.",
    "Keep it crisp and direct.",
    "Lead with the most interesting detail.",
    "Answer as if speaking to a recruiter.",
    "Keep it friendly and plain-spoken.",
)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION,
    generation_config={
        # Warm enough to reword between visits; the facts are pinned by the
        # context and the "copy them exactly" rule above.
        "temperature": 0.85,
        "top_p": 0.95,
        "max_output_tokens": 220,  # hard cap on output billing
    },
)

MAX_QUERY_CHARS = 400
CACHE_SIZE = 256
# How many different phrasings to collect per question before answering purely
# from cache. This is the cost dial: a popular question costs at most this many
# calls for the life of the process, then it is free and still varied.
VARIANTS_PER_QUESTION = 3

# normalized question -> {"variants": [phrasings], "calls": n}. The budget is
# counted in calls, not in distinct phrasings: if the model happens to repeat
# itself, that still has to count, or a popular question never stops costing.
_cache = OrderedDict()


def _cached(key: str):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _remember(key: str, answer: str):
    entry = _cache.setdefault(key, {"variants": [], "calls": 0})
    entry["calls"] += 1
    if answer not in entry["variants"]:
        entry["variants"].append(answer)
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)


def _generate(prompt: str):
    """One model call. Returns None rather than raising - answers still flow."""
    try:
        response = model.generate_content(prompt)
        return (response.text or "").strip() or None
    except Exception:
        return None


def _answer(key: str, prompt: str, fallback: str):
    """
    Model-phrased answer, with the variant cache in front of it.

    Returns (text, phrasing) where phrasing says where the wording came from:
      "model"    - generated just now
      "cache"    - an earlier phrasing of this same question
      "fallback" - the model was unreachable, so the caller's own text is used

    Until VARIANTS_PER_QUESTION phrasings are collected each ask generates a
    fresh one; after that the question is answered from cache, still varying.
    """
    entry = _cached(key)
    if entry and entry["calls"] >= VARIANTS_PER_QUESTION:
        return random.choice(entry["variants"]), "cache"

    answer = _generate(prompt)
    if answer:
        _remember(key, answer)
        return answer, "model"

    # The model is unreachable. Don't count that against the budget - reuse an
    # earlier phrasing if there is one, otherwise the caller's own text.
    variants = entry["variants"] if entry else []
    if variants:
        return random.choice(variants), "cache"
    return fallback, "fallback"


def _context_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT:\n{kb.profile_card}\n{context}\n\n"
        f"QUESTION: {question}\n{random.choice(STYLE_NUDGES)}"
    )


def _redirect_prompt(question: str) -> str:
    return (
        f"CONTEXT:\n{kb.profile_card}\n"
        f"His profile covers: {faq.COVERAGE}. It does not cover anything else.\n\n"
        f"QUESTION: {question}\n"
        "The profile does not answer this. In one or two sentences, say so in your "
        "own words without apologising twice, and suggest one thing about Yash you "
        "could help with instead. Do not answer the question itself or invent facts."
    )


# ---------------------------
# Request Body
# ---------------------------
class Query(BaseModel):
    query: str


# ---------------------------
# Ask Endpoint
# ---------------------------
@app.post("/ask")
async def ask_portfolio(query: Query):
    """
    Cheapest path first. Greetings, profile fields and list answers are served
    locally (and reworded from hand-written phrasings, since an email address
    must not be paraphrased). Everything else is phrased by the model over the
    smallest context that answers it.
    """
    text = (query.query or "").strip()[:MAX_QUERY_CHARS]
    key = faq.normalize(text)

    # 1. Greetings and smalltalk - no model call.
    answer = faq.smalltalk(text)
    if answer:
        return {"response": answer, "source": "smalltalk", "phrasing": "local"}

    # 2. Single-fact lookups (email, links, location) - no model call. These
    #    are exact values; rotating phrasings is safe, rewording them is not.
    answer = faq.quick_fact(text)
    if answer:
        return {"response": answer, "source": "quick_fact", "phrasing": "local"}

    # 3. List questions ("his projects", "work experience") - no model call.
    #    A reworded list is a list that quietly drops entries.
    answer = faq.overview(text)
    if answer:
        return {"response": answer, "source": "overview", "phrasing": "local"}

    # 4. A known question, or a phrasing mapped onto a topic: the chunk is the
    #    material, the model supplies the wording.
    context, source = faq.faq(text), "faq"
    if not context:
        context, source = faq.intent(text), "intent"

    # 5. Otherwise retrieve. 1-2 chunks is all the model ever sees.
    if not context:
        matches = kb.search(text)
        if matches:
            context = "\n".join(chunk["text"] for chunk, _ in matches)
            source = "retrieval"

    if context:
        answer, phrasing = _answer(key, _context_prompt(text, context), context)
        return {"response": answer, "source": source, "phrasing": phrasing}

    # 6. Nothing in the profile covers this. Let the model turn the visitor
    #    away in its own words rather than repeating a canned sentence.
    answer, phrasing = _answer(
        key, _redirect_prompt(text), random.choice(faq.OUT_OF_SCOPE_FALLBACKS)
    )
    return {"response": answer, "source": "redirect", "phrasing": phrasing}


@app.get("/")
def root():
    return {"message": "Portfolio Backend Running!"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(kb.chunks),
        "cached_questions": len(_cache),
        "cached_variants": sum(len(e["variants"]) for e in _cache.values()),
    }
