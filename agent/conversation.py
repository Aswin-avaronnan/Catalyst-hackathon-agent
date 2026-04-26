from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

from pydantic import BaseModel

from agent.candidate_discovery import Candidate
from agent.jd_parser import ParsedJobDescription

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    speaker: str   # "agent" | "candidate"
    message: str


class EngagementResult(BaseModel):
    candidate: Candidate
    conversation: List[ConversationTurn]
    interest_score: float   # 0–100
    interest_breakdown: str


# ── Personality response templates ───────────────────────────────────────────

_EAGER_REPLIES = [
    "Yes, definitely interested! Tell me more about the role and the team.",
    "Absolutely, this sounds exciting! What does the tech stack look like day-to-day?",
    "Great timing — I've been thinking about my next move. Tell me more!",
]

_CAUTIOUS_REPLIES = [
    "Maybe, depends on the details. What's the tech stack and team size?",
    "I'm somewhat open. Can you share more about the company stage and culture?",
    "I'd need more info before committing to a conversation. What's the opportunity?",
]

_PASSIVE_REPLIES = [
    "Not actively looking right now, but you can share the details.",
    "I'm pretty settled where I am, but feel free to send over the info.",
    "I'm not really in the market — what's the role exactly?",
]

_EAGER_FOLLOWUP = [
    "That sounds great! I'm free for a call this week. What's the next step?",
    "The stack aligns well with what I do. Happy to explore further — when can we connect?",
]

_CAUTIOUS_FOLLOWUP = [
    "Interesting. I'd want to know more about growth opportunities before I commit to a chat.",
    "That's not bad. I'll think about it — can you share a JD I can review?",
]

_PASSIVE_FOLLOWUP = [
    "I'll take a look. No promises though.",
    "Send it over. I'll get back to you if it makes sense.",
]


class ConversationSimulator:
    """Simulates a 3-turn recruiter ↔ candidate conversation."""

    async def engage_batch(
        self,
        match_results: List[Any],
        jd: ParsedJobDescription,
    ) -> List[Dict[str, Any]]:
        engaged = []
        for match_result in match_results:
            engagement = self._simulate_conversation(match_result, jd)
            engaged.append({"match_result": match_result, "engagement": engagement})
        logger.info("Simulated conversations with %d candidates", len(engaged))
        return engaged

    def _simulate_conversation(
        self, match_result: Any, jd: ParsedJobDescription
    ) -> EngagementResult:
        candidate: Candidate = match_result.candidate
        personality = candidate.personality
        first_name = candidate.name.split()[0]

        turns: List[ConversationTurn] = []

        # ── Turn 1: Agent outreach ────────────────────────────────────────────
        top_skills = ", ".join(candidate.skills[:3])
        outreach = (
            f"Hi {first_name}! I came across your profile and noticed your {top_skills} expertise. "
            f"We have an exciting {jd.job_title} role at a {', '.join(jd.company_culture[:1]) or 'fast-growing'} company. "
            "Would you be open to learning more?"
        )
        turns.append(ConversationTurn(speaker="agent", message=outreach))

        # ── Turn 2: Candidate response ────────────────────────────────────────
        if personality == "eager":
            reply = random.choice(_EAGER_REPLIES)
            interest_base = 78
        elif personality == "cautious":
            reply = random.choice(_CAUTIOUS_REPLIES)
            interest_base = 48
        else:  # passive
            reply = random.choice(_PASSIVE_REPLIES)
            interest_base = 18

        turns.append(ConversationTurn(speaker="candidate", message=reply))

        # ── Turn 3: Agent follow-up + candidate reaction (if engaged) ─────────
        if interest_base > 30:
            skills_preview = ", ".join(jd.required_skills[:3])
            salary = jd.salary_range or "competitive"
            loc = jd.location_type or "flexible"
            followup = (
                f"Great! The role is focused on {skills_preview}. "
                f"It is a {loc} position with {salary} compensation. "
                "What would be your ideal next step in terms of role or company stage?"
            )
            turns.append(ConversationTurn(speaker="agent", message=followup))

            if personality == "eager":
                cand_followup = random.choice(_EAGER_FOLLOWUP)
            else:
                cand_followup = random.choice(_CAUTIOUS_FOLLOWUP)
            turns.append(ConversationTurn(speaker="candidate", message=cand_followup))

        elif personality == "passive" and not candidate.open_to_opportunities:
            # Even passive candidates get a polite close
            turns.append(ConversationTurn(
                speaker="agent",
                message="Totally understand! I will send over the JD anyway — happy to reconnect whenever the timing is right.",
            ))
            turns.append(ConversationTurn(
                speaker="candidate",
                message=random.choice(_PASSIVE_FOLLOWUP),
            ))

        # ── Interest score ────────────────────────────────────────────────────
        interest_score = float(interest_base)

        if candidate.response_speed == "fast":
            interest_score += 12
        elif candidate.response_speed == "medium":
            interest_score += 6

        if candidate.open_to_opportunities:
            interest_score += 8

        interest_score = round(min(100.0, interest_score), 2)

        breakdown = (
            f"Personality: {personality} | "
            f"Response speed: {candidate.response_speed} | "
            f"Open to opps: {candidate.open_to_opportunities} | "
            f"Turns: {len(turns)}"
        )

        return EngagementResult(
            candidate=candidate,
            conversation=turns,
            interest_score=interest_score,
            interest_breakdown=breakdown,
        )
