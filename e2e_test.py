"""End-to-end smoke test — run with:  python e2e_test.py"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.llm_manager import LLMManager
from agent.core import TalentScoutAgent

SAMPLE_JD = """
Senior Python Engineer

We are looking for a senior backend engineer with 5+ years Python experience.
Must have: Django or FastAPI, PostgreSQL, Redis, Docker, AWS
Nice to have: Kubernetes, React
Location: Remote (US timezone preferred)
Salary: $140k-$180k
We are a fast-paced startup building developer tools.
"""

async def main():
    print("\n" + "="*60)
    print("  AI TALENT SCOUT — END-TO-END SMOKE TEST")
    print("="*60)

    async with LLMManager() as llm:
        agent = TalentScoutAgent(llm)
        print("\n⏳ Running full pipeline (dry-run mode)...\n")
        result = await agent.scout_talent(SAMPLE_JD)

    print(f"✅ Job Title      : {result.job_title}")
    print(f"✅ Level          : {result.experience_level}")
    print(f"✅ Location       : {result.location_type}")
    print(f"✅ Required Skills: {result.required_skills}")
    print(f"✅ Reviewed       : {result.total_candidates_reviewed} candidates")
    print(f"✅ Engaged        : {result.candidates_engaged} candidates")
    print(f"✅ Shortlisted    : {len(result.shortlist)} candidates")

    print("\n── TOP 5 CANDIDATES ────────────────────────────────────")
    for c in result.shortlist[:5]:
        print(
            f"  #{c.rank} {c.name:<22} | Match:{c.match_score:5.1f} "
            f"| Interest:{c.interest_score:5.1f} | Combined:{c.combined_score:5.1f}"
        )
        print(f"      Strengths: {c.strengths}")
        print(f"      Gaps     : {c.gaps}")
        print(f"      Engage   : {c.conversation_summary}")
        print()

    print("="*60)
    print("✅ SMOKE TEST PASSED — Ready for demo!")
    print("="*60)
    print("\nTo start the full demo:")
    print("  bash run.sh")
    print("\nOr manually:")
    print("  Terminal 1: PYTHONPATH=. python -m uvicorn api.server:app --port 8000")
    print("  Terminal 2: PYTHONPATH=. streamlit run ui/app.py")

if __name__ == "__main__":
    asyncio.run(main())
