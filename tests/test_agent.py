import pytest
import asyncio
from agent.jd_parser import ParsedJobDescription
from agent.candidate_discovery import Candidate, CandidateDiscovery
from agent.matcher import Matcher
from agent.conversation import ConversationSimulator

# Mock Data
MOCK_JD = ParsedJobDescription(
    job_title="Senior Python Engineer",
    required_skills=["Python", "Django", "PostgreSQL"],
    nice_to_have_skills=["Docker", "AWS"],
    experience_level="senior",
    experience_years_min=5,
    experience_years_max=None,
    location_type="remote",
    location=None,
    company_culture=["startup", "fast-paced"],
    salary_range="$140k-$180k",
    confidence_score=0.9,
    raw_jd="dummy text"
)

MOCK_CANDIDATE_PERFECT = Candidate(
    id="CAND999",
    name="Test Perfect",
    title="Senior Backend Engineer",
    current_company="TechInc",
    location="Remote",
    skills=["Python", "Django", "PostgreSQL", "Docker", "AWS", "Kubernetes"],
    experience_years=6,
    linkedin_summary="Startup focused engineer. Fast-paced builder.",
    github_repos=50,
    open_to_opportunities=True,
    response_speed="fast",
    personality="eager"
)

MOCK_CANDIDATE_POOR = Candidate(
    id="CAND998",
    name="Test Poor",
    title="Junior Java Developer",
    current_company="EnterpriseCorp",
    location="New York (Onsite)",
    skills=["Java", "Spring", "Oracle"],
    experience_years=1,
    linkedin_summary="Java developer.",
    github_repos=5,
    open_to_opportunities=False,
    response_speed="slow",
    personality="passive"
)

@pytest.mark.asyncio
async def test_matcher_perfect_candidate():
    matcher = Matcher()
    results = await matcher.score_all([MOCK_CANDIDATE_PERFECT, MOCK_CANDIDATE_POOR], MOCK_JD)
    
    assert len(results) == 2
    perfect_result = results[0] # Should be sorted highest first
    poor_result = results[1]
    
    assert perfect_result.candidate.id == "CAND999"
    assert perfect_result.match_score > 80.0
    
    assert poor_result.candidate.id == "CAND998"
    assert poor_result.match_score < 50.0

@pytest.mark.asyncio
async def test_conversation_simulator():
    simulator = ConversationSimulator()
    matcher = Matcher()
    # Need a MatchResult object to pass in
    match_results = await matcher.score_all([MOCK_CANDIDATE_PERFECT, MOCK_CANDIDATE_POOR], MOCK_JD)
    
    engaged = await simulator.engage_batch(match_results, MOCK_JD)
    
    assert len(engaged) == 2
    
    perfect_engagement = engaged[0]["engagement"]
    poor_engagement = engaged[1]["engagement"]
    
    assert perfect_engagement.interest_score > 80.0
    assert len(perfect_engagement.conversation) == 4 # Agent, Cand, Agent, Cand
    
    assert poor_engagement.interest_score < 40.0
    # Passive candidate not open to opps gets a polite close
    assert len(poor_engagement.conversation) == 4
