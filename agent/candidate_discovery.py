from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    id: str
    name: str
    title: str
    current_company: str
    location: str
    skills: List[str]
    experience_years: int
    linkedin_summary: str
    github_repos: int
    open_to_opportunities: bool
    response_speed: str   # fast | medium | slow
    personality: str      # eager | cautious | passive


class CandidateDiscovery:
    """Loads candidates from the mock JSON file."""

    def __init__(self, data_file: str = "data/candidates.json") -> None:
        self._data_file = Path(data_file)
        if not self._data_file.exists():
            raise FileNotFoundError(
                f"Candidate data file not found: {data_file}. "
                "Run generate_candidates.py first."
            )

    async def discover(self, parsed_jd=None) -> List[Candidate]:
        """Return all candidates from the JSON pool."""
        try:
            raw = self._data_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            candidates = [Candidate(**c) for c in data]
            logger.info("Discovered %d candidates from %s", len(candidates), self._data_file)
            return candidates
        except json.JSONDecodeError as exc:
            logger.error("Malformed candidates JSON: %s", exc)
            raise
        except Exception as exc:
            logger.error("Failed to load candidates: %s", exc)
            raise
