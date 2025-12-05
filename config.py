import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # put your key in .env
MODEL_NAME = "gemini-2.5-flash"               # fast + cheap
