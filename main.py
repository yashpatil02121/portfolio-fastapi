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
    "answer, say you don't have that detail and suggest contacting Yash. Reply in "
    "2-4 sentences, plain text, third person. Never invent facts, dates or links."
)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION,
    generation_config={
        "temperature": 0.2,      # factual lookup, not creative writing
        "max_output_tokens": 220,  # hard cap on output billing
        "top_p": 0.9,
    },
)

MAX_QUERY_CHARS = 400
CACHE_SIZE = 256
_cache = OrderedDict()  # normalized question -> answer, for repeat visitors


def _cache_get(key: str):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key: str, value: str):
    _cache[key] = value
    _cache.move_to_end(key)
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)


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
    Cheapest path first. Gemini is only called when the question is genuinely
    new and specific enough to need paraphrasing over retrieved context.
    """
    text = (query.query or "").strip()[:MAX_QUERY_CHARS]

    # 1. Greetings and smalltalk - no model call.
    answer = faq.smalltalk(text)
    if answer:
        return {"response": answer, "source": "smalltalk"}

    # 2. Single-fact lookups (email, links, location) - no model call.
    answer = faq.quick_fact(text)
    if answer:
        return {"response": answer, "source": "quick_fact"}

    # 3. Already-answered question - no model call.
    answer = faq.faq(text)
    if answer:
        return {"response": answer, "source": "faq"}

    # 4. Common phrasing mapped onto a known topic - no model call.
    answer = faq.intent(text)
    if answer:
        return {"response": answer, "source": "intent"}

    # 5. Same question as an earlier visitor - no model call.
    key = faq.normalize(text)
    answer = _cache_get(key)
    if answer:
        return {"response": answer, "source": "cache"}

    # 6. Retrieval. Nothing relevant means nothing to pay for.
    matches = kb.search(text)
    if not matches:
        return {"response": faq.OUT_OF_SCOPE, "source": "out_of_scope"}

    # 7. Model call with the smallest useful prompt: identity card + 1-2 chunks.
    context = "\n".join(chunk["text"] for chunk, _ in matches)
    prompt = f"CONTEXT:\n{kb.profile_card}\n{context}\n\nQUESTION: {text}"

    try:
        response = model.generate_content(prompt)
        answer = (response.text or "").strip()
        if not answer:
            return {"response": context, "source": "context_fallback"}
        _cache_put(key, answer)
        return {
            "response": answer,
            "source": "model",
            "chunks": [chunk["id"] for chunk, _ in matches],
        }

    except Exception as e:
        # Never fail the widget over an API hiccup - the retrieved text is
        # already a correct, if unpolished, answer.
        return {"response": context, "source": "context_fallback", "error": str(e)}


@app.get("/")
def root():
    return {"message": "Portfolio Backend Running!"}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(kb.chunks), "cached": len(_cache)}
