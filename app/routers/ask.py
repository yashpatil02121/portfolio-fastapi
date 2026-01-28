from fastapi import APIRouter
from app.models.query import Query
from app.services.resume import RESUME_TEXT
from app.services.prompt_builder import build_prompt
from app.core.prompt_ontology import PROMPT_ONTOLOGY_V1
from app.core.gemini import model
from google.api_core.exceptions import ResourceExhausted

router = APIRouter()


@router.post("/ask")
async def ask_portfolio(query: Query):
    prompt = build_prompt(
        schema=PROMPT_ONTOLOGY_V1,
        resume_text=RESUME_TEXT,
        user_query=query.query
    )

    try:
        response = model.generate_content(prompt)
        return {
            "response": response.text,
            "prompt_version": PROMPT_ONTOLOGY_V1["version"]
        }

    except ResourceExhausted:
        return {
            "response": (
                "The system is temporarily rate-limited due to API usage limits. "
                "Please try again shortly."
            ),
            "prompt_version": PROMPT_ONTOLOGY_V1["version"],
            "error": "RATE_LIMITED"
        }