from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel

from agent.candidate_discovery import Candidate
from agent.jd_parser import ParsedJobDescription

logger = logging.getLogger(__name__)


class MatchBreakdown(BaseModel):
    skills_score: float
    skills_reason: str
    experience_score: float
    experience_reason: str
    location_score: float
    location_reason: str
    culture_score: float
    culture_reason: str


class MatchResult(BaseModel):
    candidate: Candidate
    match_score: float
    breakdown: MatchBreakdown
    strengths: List[str]
    gaps: List[str]


class Matcher:
    """Scores candidates against a parsed job description."""

    # Culture keywords that signal a good fit with startup/fast-paced JDs
    _CULTURE_SIGNALS = [
        "startup", "fast-paced", "scale", "distributed", "high-throughput",
        "ownership", "agile", "collaborative", "growth", "product",
    ]

    async def score_all(
        self, candidates: List[Candidate], jd: ParsedJobDescription
    ) -> List[MatchResult]:
        results: List[MatchResult] = []
        for candidate in candidates:
            result = self._score_candidate(candidate, jd)
            results.append(result)
        results.sort(key=lambda r: r.match_score, reverse=True)
        logger.info("Scored %d candidates. Top score: %.1f", len(results), results[0].match_score if results else 0)
        return results

    def _score_candidate(self, candidate: Candidate, jd: ParsedJobDescription) -> MatchResult:
        # ── Skills (40 pts) ─────────────────────────────────────────────────
        req = [s.lower() for s in jd.required_skills]
        nice = [s.lower() for s in jd.nice_to_have_skills]
        cand_skills = [s.lower() for s in candidate.skills]

        req_matched = [s for s in req if s in cand_skills]
        nice_matched = [s for s in nice if s in cand_skills]

        req_score = (len(req_matched) / max(len(req), 1)) * 30.0
        nice_score = (len(nice_matched) / max(len(nice), 1)) * 10.0 if nice else 5.0
        skills_score = round(req_score + nice_score, 2)

        skills_reason = (
            f"Matched {len(req_matched)}/{len(req)} required skills"
            + (f", {len(nice_matched)}/{len(nice)} nice-to-have" if nice else "")
        )

        # ── Experience (25 pts) ──────────────────────────────────────────────
        yrs = candidate.experience_years
        level = jd.experience_level.lower()

        if level == "senior":
            if yrs >= 5:
                exp_score, exp_reason = 25.0, f"{yrs} yrs meets senior requirement (5+)"
            elif yrs >= 3:
                exp_score, exp_reason = 15.0, f"{yrs} yrs is close to senior (5+ needed)"
            else:
                exp_score, exp_reason = 5.0, f"{yrs} yrs is below senior threshold"
        elif level == "mid":
            if 3 <= yrs <= 5:
                exp_score, exp_reason = 25.0, f"{yrs} yrs meets mid-level requirement"
            elif yrs > 5:
                exp_score, exp_reason = 18.0, f"{yrs} yrs is overqualified for mid role"
            else:
                exp_score, exp_reason = 8.0, f"{yrs} yrs is below mid threshold"
        elif level == "junior":
            if yrs <= 2:
                exp_score, exp_reason = 25.0, f"{yrs} yrs matches junior requirement"
            else:
                exp_score, exp_reason = 15.0, f"{yrs} yrs is overqualified for junior role"
        else:  # unknown
            exp_score, exp_reason = 15.0, "Experience level not specified in JD"

        # ── Location (15 pts) ────────────────────────────────────────────────
        jd_remote = "remote" in jd.location_type.lower()
        cand_remote = "remote" in candidate.location.lower()

        if jd_remote and cand_remote:
            location_score, location_reason = 15.0, "Both JD and candidate prefer remote"
        elif jd_remote and not cand_remote:
            location_score, location_reason = 8.0, "JD is remote, candidate is office-based"
        elif not jd_remote and cand_remote:
            location_score, location_reason = 10.0, "Candidate is remote, JD may allow it"
        else:
            location_score, location_reason = 12.0, "Both onsite — location compatible"

        # ── Culture fit (20 pts) ─────────────────────────────────────────────
        summary_lower = candidate.linkedin_summary.lower()
        jd_culture = [c.lower() for c in jd.company_culture]

        matched_culture = [kw for kw in self._CULTURE_SIGNALS if kw in summary_lower]
        jd_culture_matches = [c for c in jd_culture if c in summary_lower]

        if len(matched_culture) >= 3 or len(jd_culture_matches) >= 2:
            culture_score, culture_reason = 18.0, "Strong culture signal alignment"
        elif len(matched_culture) >= 1 or len(jd_culture_matches) >= 1:
            culture_score, culture_reason = 13.0, "Moderate culture alignment"
        else:
            culture_score, culture_reason = 8.0, "Limited culture signal in summary"

        # ── Final score ──────────────────────────────────────────────────────
        total = round(min(100.0, skills_score + exp_score + location_score + culture_score), 2)

        breakdown = MatchBreakdown(
            skills_score=skills_score,
            skills_reason=skills_reason,
            experience_score=exp_score,
            experience_reason=exp_reason,
            location_score=location_score,
            location_reason=location_reason,
            culture_score=culture_score,
            culture_reason=culture_reason,
        )

        strengths = [s for s in candidate.skills if s.lower() in req][:4]
        gaps = [s.title() for s in req if s not in cand_skills][:4]

        return MatchResult(
            candidate=candidate,
            match_score=total,
            breakdown=breakdown,
            strengths=strengths,
            gaps=gaps,
        )
