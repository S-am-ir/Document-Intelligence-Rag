"""
Document Intelligence RAG — Backend Entry Point
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith tracing (safe — never crashes the app) ────────────────────────
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    try:
        langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
        if not langsmith_key:
            print(
                "[langsmith] LANGCHAIN_TRACING_V2=true but no LANGCHAIN_API_KEY — tracing disabled"
            )
        else:
            project = os.getenv("LANGCHAIN_PROJECT", "document-rag")
            endpoint = os.getenv(
                "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
            )
            print(
                f"[langsmith] Tracing enabled → project={project} endpoint={endpoint}"
            )
            # Set env vars that LangChain auto-reads
            os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
    except Exception as e:
        print(f"[langsmith] Setup failed: {e} — tracing disabled")

from api.routes import router

app = FastAPI(title="Document Intelligence RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Initialize database tables and pre-load ML models."""
    try:
        from memory.database import init_tables

        init_tables()
        print("[startup] Database tables initialized.")
    except Exception as e:
        print(f"[startup] DB init failed: {e} — check DATABASE_URL")

    # Pre-load embedder (nomic-embed-text-v1.5 on GPU)
    try:
        from agent.tools.rag import _get_embeddings

        _get_embeddings()
        print("[startup] Embedder loaded.")
    except Exception as e:
        print(f"[startup] Embedder load failed: {e}")

    # Pre-load reranker (bge-reranker-base on GPU)
    try:
        from agent.tools.reranker import get_cross_encoder

        get_cross_encoder()
        print("[startup] Reranker loaded.")
    except Exception as e:
        print(f"[startup] Reranker load failed: {e}")

    # Pre-load Groq client
    try:
        from agent.tools.doc_router import get_groq_client

        get_groq_client()
        print("[startup] Groq client loaded.")
    except Exception as e:
        print(f"[startup] Groq client load failed: {e}")


@app.get("/")
async def root():
    return {
        "name": "Document Intelligence RAG",
        "status": "running",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True,
    )
