from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.llm_manager import LLMManager
from agent.core import TalentScoutAgent, TalentScoutResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Talent Scout API",
    description="Parse a job description and get a ranked shortlist of matched candidates.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ScoutRequest(BaseModel):
    job_description: str

    model_config = {"json_schema_extra": {
        "example": {
            "job_description": (
                "Senior Python Engineer\n\n"
                "We're looking for a senior backend engineer with 5+ years Python experience.\n"
                "Must have: Django/FastAPI, PostgreSQL, Docker, AWS, Redis\n"
                "Nice to have: Kubernetes, React\n"
                "Location: Remote\nSalary: $140k-$180k"
            )
        }
    }}


class HealthResponse(BaseModel):
    status: str
    dry_run: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness check."""
    dry_run = os.getenv("LLM_DRY_RUN", "true").lower() == "true"
    return HealthResponse(status="healthy", dry_run=dry_run)


@app.post("/scout", response_model=TalentScoutResult, tags=["Agent"])
async def scout_talent(request: ScoutRequest) -> TalentScoutResult:
    """
    Run the full talent scouting pipeline on a job description.

    Steps:
    1. Parse JD → structured fields
    2. Load 50 mock candidates
    3. Score every candidate (skills, experience, location, culture)
    4. Simulate conversations with top 20
    5. Return top 10 ranked by combined score
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=422, detail="job_description cannot be empty.")

    try:
        async with LLMManager() as llm:
            agent = TalentScoutAgent(llm)
            result = await agent.scout_talent(request.job_description)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Scout pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")


@app.get("/stats", tags=["System"])
async def stats() -> dict:
    """Return LLM provider config (no secrets exposed)."""
    async with LLMManager() as llm:
        import json
        return json.loads(llm.debug_provider_config())


if __name__ == "__main__":
    import sys
    import os
    # Ensure project root is on sys.path so 'agent' package is found
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False, app_dir=project_root)
