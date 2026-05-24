"""FastAPI application entrypoint.

Run:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import get_settings

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(
    title="Agentic Memory AI",
    version="0.1.0",
    description="Long-term personalised assistant with Reasoning Gatekeeper, "
                "tiered memory, and controlled forgetting.",
)

# Allow Streamlit + local dev to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"name": "agentic-memory-ai", "docs": "/docs"}
