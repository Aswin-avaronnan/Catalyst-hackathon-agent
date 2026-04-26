from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel

from agent.candidate_discovery import CandidateDiscovery
from agent.conversation import ConversationSimulator
from agent.jd_parser import JDParser
from agent.llm_manager import LLMManager
from agent.matcher import Matcher

logger = logging.getLogger(__name__)


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    name: str
    title: str
    current_company: str
    location: str
    match_score: float
    interest_score: float
    combined_score: float
    strengths: List[str]
    gaps: List[str]
    skills_reason: str
    experience_reason: str
    conversation_summary: str
    conversation_turns: int


class TalentScoutResult(BaseModel):
    job_title: str
    experience_level: str
    location_type: str
    required_skills: List[str]
    total_candidates_reviewed: int
    candidates_engaged: int
    shortlist: List[RankedCandidate]
    timestamp: str


class TalentScoutAgent:
    """
    End-to-end talent scouting pipeline:
    Parse JD → Discover candidates → Match → Engage → Rank.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        data_file: str = "data/candidates.json",
    ) -> None:
        self.jd_parser = JDParser(llm_manager)
        self.discovery = CandidateDiscovery(data_file=data_file)
        self.matcher = Matcher()
        self.conversation = ConversationSimulator()

    async def scout_talent(self, job_description: str) -> TalentScoutResult:
        if not job_description or not job_description.strip():
            raise ValueError("job_description cannot be empty.")

        # ── Step 1: Parse JD ─────────────────────────────────────────────────
        logger.info("Step 1/5 — Parsing job description …")
        parsed_jd = await self.jd_parser.parse(job_description)
        logger.info(
            "Parsed JD: title=%s level=%s skills=%s",
            parsed_jd.job_title,
            parsed_jd.experience_level,
            parsed_jd.required_skills,
        )

        # ── Step 2: Discover candidates ──────────────────────────────────────
        logger.info("Step 2/5 — Discovering candidates …")
        candidates = await self.discovery.discover(parsed_jd)
        logger.info("Found %d candidates", len(candidates))

        # ── Step 3: Score all candidates ─────────────────────────────────────
        logger.info("Step 3/5 — Scoring %d candidates …", len(candidates))
        matched = await self.matcher.score_all(candidates, parsed_jd)

        # ── Step 4: Engage top 20 ────────────────────────────────────────────
        top_20 = matched[:20]
        logger.info("Step 4/5 — Simulating conversations with top %d candidates …", len(top_20))
        engaged = await self.conversation.engage_batch(top_20, parsed_jd)

        # ── Step 5: Combine scores and build ranked shortlist ────────────────
        logger.info("Step 5/5 — Building ranked shortlist …")
        shortlist = self._build_shortlist(engaged)

        return TalentScoutResult(
            job_title=parsed_jd.job_title,
            experience_level=parsed_jd.experience_level,
            location_type=parsed_jd.location_type,
            required_skills=parsed_jd.required_skills,
            total_candidates_reviewed=len(candidates),
            candidates_engaged=len(engaged),
            shortlist=shortlist[:10],
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def _build_shortlist(self, engaged: List[Dict[str, Any]]) -> List[RankedCandidate]:
        scored: List[Dict[str, Any]] = []
        for item in engaged:
            match_result = item["match_result"]
            engagement = item["engagement"]

            # Combined = 60% match + 40% interest
            combined = round(
                (match_result.match_score * 0.6) + (engagement.interest_score * 0.4), 2
            )
            scored.append(
                {
                    "match_result": match_result,
                    "engagement": engagement,
                    "combined": combined,
                }
            )

        scored.sort(key=lambda x: x["combined"], reverse=True)

        shortlist: List[RankedCandidate] = []
        for rank, item in enumerate(scored, start=1):
            mr = item["match_result"]
            eng = item["engagement"]
            shortlist.append(
                RankedCandidate(
                    rank=rank,
                    candidate_id=mr.candidate.id,
                    name=mr.candidate.name,
                    title=mr.candidate.title,
                    current_company=mr.candidate.current_company,
                    location=mr.candidate.location,
                    match_score=mr.match_score,
                    interest_score=eng.interest_score,
                    combined_score=item["combined"],
                    strengths=mr.strengths,
                    gaps=mr.gaps,
                    skills_reason=mr.breakdown.skills_reason,
                    experience_reason=mr.breakdown.experience_reason,
                    conversation_summary=eng.interest_breakdown,
                    conversation_turns=len(eng.conversation),
                )
            )

        return shortlist
