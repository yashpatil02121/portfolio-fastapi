from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from config import GEMINI_API_KEY, MODEL_NAME
import os

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
# Load Gemini
# ---------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ---------------------------
# Load Static Resume Files
# ---------------------------
def load_resume_data():
    folder = "resume"
    combined_text = ""

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                combined_text += f"\n\n### {file.replace('.txt','').upper()} ###\n"
                combined_text += f.read()

    return combined_text.strip()


RESUME_TEXT = load_resume_data()


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
    prompt = f"""
You are an AI chatbot for Yash's Portfolio.
You must answer strictly based on the following resume data.

Resume Data:
{RESUME_TEXT}

User Query:
{query.query}

Return a clear, short and accurate answer.
"""

    try:
        response = model.generate_content(prompt)
        return {"response": response.text}

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "Portfolio Backend Running!"}
