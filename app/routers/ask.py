from fastapi import APIRouter
from app.models.query import Query
from app.services.resume import RESUME_TEXT
from app.core.gemini import model

router = APIRouter()

@router.post("/ask")
async def ask_portfolio(query: Query):
    prompt = f"""
You are an AI chatbot for Yash's Portfolio.
Answer strictly using the resume data below.

Resume Data:
{RESUME_TEXT}

User Query:
{query.query}

Return a clear, short and accurate answer.
"""
    response = model.generate_content(prompt)
    return {"response": response.text}
