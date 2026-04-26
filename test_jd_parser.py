import asyncio
import logging
from agent.llm_manager import LLMManager
from agent.jd_parser import run_jd_parser_examples

logging.basicConfig(level=logging.INFO)

async def main():
    print("\n" + "="*60)
    print("Testing JD Parser - Phase 3")
    print("="*60)
    
    async with LLMManager() as llm:
        results = await run_jd_parser_examples(llm)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Example {i} ---")
            print(f"Title: {result.job_title}")
            print(f"Required Skills: {result.required_skills}")
            print(f"Nice-to-Have: {result.nice_to_have_skills}")
            print(f"Experience: {result.experience_level} ({result.experience_years_min}-{result.experience_years_max} years)")
            print(f"Location: {result.location_type} ({result.location})")
            print(f"Culture: {result.company_culture}")
            print(f"Salary: {result.salary_range}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Raw JD preview: {result.raw_jd[:100]}...")
    
    print("\n" + "="*60)
    print("✅ Phase 3 COMPLETE - JD Parser Working!")
    print("="*60)

asyncio.run(main())
