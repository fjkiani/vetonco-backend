"""
VetOnco Backend — FastAPI Application
Standalone canine TCC oncology decision support API.
Includes LangGraph agent endpoints for agentic pipeline + monitoring.
"""
from __future__ import annotations
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.health import router as health_router
from routers.canine import router as canine_router
from routers.agent import router as agent_router

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VetOnco API",
    description=(
        "Canine TCC oncology decision support — "
        "LangGraph agentic pipeline, drug scoring, dosing, recipe cards, test analysis"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

allowed_origins_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:3001",
)
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health_router)
app.include_router(canine_router)
app.include_router(agent_router)

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "VetOnco API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "agents": {
            "tcc_pipeline": "/api/canine/agent/run (SSE)",
            "monitoring": "/api/canine/agent/monitor",
            "agent_health": "/api/canine/agent/health",
        },
    }
