from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.ask import router as ask_router

app = FastAPI(title="Portfolio AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://portfolio-yash-patil.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router)

@app.get("/")
def root():
    return {"message": "Portfolio Backend Running!"}
