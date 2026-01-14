from fastapi import APIRouter
from app.models.query import Query
from app.services.resume import RESUME_TEXT
from app.core.gemini import model

router = APIRouter()

@router.post("/ask")
async def ask_portfolio(query: Query):
    prompt = f"""
Role:
You are an AI assistant representing Yash's professional portfolio.

Task:
Answer the user's question strictly using the provided resume data.

Context:
Resume Data:
{RESUME_TEXT}

Constraints:
- Do NOT assume information not present in the resume.
- If the answer is not found, say "This information is not available in the resume."
- Keep responses short, clear, and factual.

User Question:
{query.query}

Output Format:
Plain text, concise, professional.
"""
    response = model.generate_content(prompt)
    return {"response": response.text}
