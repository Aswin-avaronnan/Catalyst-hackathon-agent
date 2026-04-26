import os
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
dry_run = os.getenv("LLM_DRY_RUN", "true")

print("\n" + "="*50)
print("Environment Variable Check")
print("="*50)

if gemini_key:
    masked = gemini_key[:10] + "..." + gemini_key[-4:]
    print(f"✅ GEMINI_API_KEY: {masked}")
else:
    print("❌ GEMINI_API_KEY: NOT FOUND")
    exit(1)

print(f"✅ LLM_DRY_RUN: {dry_run}")
print("="*50)

# Test Gemini API
print("\nTesting Gemini API connection...")

import asyncio
from agent.llm_manager import LLMManager

async def test():
    async with LLMManager() as llm:
        response = await llm.generate(
            prompt="Reply with exactly: 'API works!'",
            complexity="simple",
            max_tokens=20
        )
        print(f"✅ Response: {response.text}")
        print(f"✅ Provider: {response.provider}")
        print(f"✅ Model: {response.model}")
        print(f"✅ Tokens: {response.total_tokens}")
        print(f"✅ Cost: ${response.estimated_cost_usd}")
        print("\n✅ ALL TESTS PASSED - Ready for Phase 3!")

asyncio.run(test())
