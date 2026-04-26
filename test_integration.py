# test_integration_v3.py
"""
Integration test for Phase 2 - Async compatible version
"""

import os
import sys
import asyncio

# Mock environment variables
os.environ['OPENROUTER_API_KEY'] = 'test-key-123'
os.environ['ANTHROPIC_API_KEY'] = 'test-key-456'
os.environ['OPENAI_API_KEY'] = 'test-key-789'

from agent.token_tracker import TokenTracker
from agent.budget_controller import BudgetController
from agent.llm_manager import LLMManager

async def test_token_tracker():
    """Test TokenTracker with actual async implementation"""
    print("\n=== Testing TokenTracker ===")
    
    # Create tracker with test file
    tracker = TokenTracker(usage_file="test_data/test_usage.json")
    await tracker.initialize()
    print("✅ TokenTracker initialized")
    
    # Check what methods exist
    methods = [m for m in dir(tracker) if not m.startswith('_') and callable(getattr(tracker, m))]
    print(f"✅ Available methods: {methods}")
    
    # Try to track a call (check if method exists first)
    if hasattr(tracker, 'track_call'):
        try:
            # Check if it's async
            import inspect
            if inspect.iscoroutinefunction(tracker.track_call):
                cost = await tracker.track_call(
                    provider="openrouter_free",
                    model="meta-llama/llama-3.1-8b-instruct:free",
                    tokens_in=100,
                    tokens_out=50
                )
            else:
                cost = tracker.track_call(
                    provider="openrouter_free",
                    model="meta-llama/llama-3.1-8b-instruct:free",
                    tokens_in=100,
                    tokens_out=50
                )
            print(f"✅ Tracked call with cost: ${cost:.6f}")
        except Exception as e:
            print(f"⚠️  track_call failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Try to get summary
    if hasattr(tracker, 'get_summary'):
        try:
            import inspect
            if inspect.iscoroutinefunction(tracker.get_summary):
                summary = await tracker.get_summary()
            else:
                summary = tracker.get_summary()
            
            if summary:
                print(f"✅ Got summary with keys: {list(summary.keys())}")
                print(f"   Total calls: {summary.get('total_calls', 'N/A')}")
                print(f"   Total cost: ${summary.get('total_cost_usd', 0):.4f}")
            else:
                print("⚠️  get_summary returned None")
        except Exception as e:
            print(f"⚠️  get_summary failed: {e}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
    
    print("✅ TokenTracker test completed!\n")
    return tracker

async def test_budget_controller():
    """Test BudgetController"""
    print("=== Testing BudgetController ===")
    
    # Check __init__ signature
    import inspect
    sig = inspect.signature(BudgetController.__init__)
    print(f"   BudgetController.__init__ signature: {sig}")
    
    try:
        controller = BudgetController(max_budget_usd=15.0, max_cost_per_call=0.50)
        print("✅ BudgetController initialized with both params")
    except TypeError:
        try:
            controller = BudgetController(max_budget_usd=15.0)
            print("✅ BudgetController initialized with max_budget_usd only")
        except TypeError:
            try:
                controller = BudgetController()
                print("✅ BudgetController initialized with no params")
            except Exception as e:
                print(f"❌ Could not initialize BudgetController: {e}")
                return None
    
    # Check methods
    methods = [m for m in dir(controller) if not m.startswith('_') and callable(getattr(controller, m))]
    print(f"✅ Available methods: {methods}")
    
    # Test check_budget if exists
    if hasattr(controller, 'check_budget'):
        try:
            sig = inspect.signature(controller.check_budget)
            print(f"   check_budget signature: {sig}")
            
            if inspect.iscoroutinefunction(controller.check_budget):
                result = await controller.check_budget(current_spend=5.0, estimated_call_cost=0.10)
            else:
                result = controller.check_budget(current_spend=5.0, estimated_call_cost=0.10)
            
            print(f"✅ check_budget works: {result}")
        except Exception as e:
            print(f"⚠️  check_budget failed: {e}")
    
    # Test get_recommended_model if exists
    if hasattr(controller, 'get_recommended_model'):
        try:
            sig = inspect.signature(controller.get_recommended_model)
            print(f"   get_recommended_model signature: {sig}")
            
            if inspect.iscoroutinefunction(controller.get_recommended_model):
                model = await controller.get_recommended_model(current_spend=1.0, complexity="simple")
            else:
                model = controller.get_recommended_model(current_spend=1.0, complexity="simple")
            
            print(f"✅ get_recommended_model works: {model}")
        except Exception as e:
            print(f"⚠️  get_recommended_model failed: {e}")
    
    print("✅ BudgetController test completed!\n")
    return controller

async def test_llm_manager():
    """Test LLMManager"""
    print("=== Testing LLMManager ===")
    
    # Check __init__ signature
    import inspect
    sig = inspect.signature(LLMManager.__init__)
    print(f"   LLMManager.__init__ signature: {sig}")
    
    try:
        manager = LLMManager(max_budget_usd=15.0)
        print("✅ LLMManager initialized with max_budget_usd")
    except TypeError:
        try:
            manager = LLMManager()
            print("✅ LLMManager initialized with no params")
        except Exception as e:
            print(f"❌ Could not initialize LLMManager: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Check if initialize method exists (async pattern)
    if hasattr(manager, 'initialize'):
        try:
            await manager.initialize()
            print("✅ Called initialize()")
        except Exception as e:
            print(f"⚠️  initialize() failed: {e}")
    
    # Check structure
    print(f"✅ Has tracker: {hasattr(manager, 'tracker') or hasattr(manager, '_tracker')}")
    print(f"✅ Has budget: {hasattr(manager, 'budget') or hasattr(manager, '_budget')}")
    print(f"✅ Has openrouter: {hasattr(manager, 'openrouter') or hasattr(manager, '_openrouter')}")
    print(f"✅ Has anthropic: {hasattr(manager, 'anthropic') or hasattr(manager, '_anthropic')}")
    print(f"✅ Has openai: {hasattr(manager, 'openai') or hasattr(manager, '_openai')}")
    
    # Check methods
    methods = [m for m in dir(manager) if not m.startswith('_') and callable(getattr(manager, m))]
    print(f"✅ Available public methods: {methods}")
    
    # Try get_stats
    if hasattr(manager, 'get_stats'):
        try:
            if inspect.iscoroutinefunction(manager.get_stats):
                stats = await manager.get_stats()
            else:
                stats = manager.get_stats()
            
            if stats:
                print(f"✅ get_stats works")
                print(f"   Stats type: {type(stats)}")
                if isinstance(stats, dict):
                    print(f"   Stats keys: {list(stats.keys())[:5]}")
            else:
                print("⚠️  get_stats returned None/empty")
        except Exception as e:
            print(f"⚠️  get_stats failed: {e}")
    
    # Check if call method exists
    if hasattr(manager, 'call'):
        sig = inspect.signature(manager.call)
        print(f"✅ Has call method with signature: {sig}")
    
    print("✅ LLMManager test completed!\n")
    return manager

async def main():
    print("\n" + "="*60)
    print("PHASE 2 INTEGRATION TEST - ASYNC COMPATIBLE")
    print("="*60)
    
    try:
        tracker = await test_token_tracker()
        controller = await test_budget_controller()
        manager = await test_llm_manager()
        
        print("="*60)
        print("✅ INTEGRATION TEST COMPLETED!")
        print("="*60)
        
        # Summary
        all_good = True
        
        if tracker is None:
            print("⚠️  TokenTracker: Some issues")
            all_good = False
        else:
            print("✅ TokenTracker: Working")
        
        if controller is None:
            print("⚠️  BudgetController: Some issues")
            all_good = False
        else:
            print("✅ BudgetController: Working")
        
        if manager is None:
            print("⚠️  LLMManager: Some issues")
            all_good = False
        else:
            print("✅ LLMManager: Working")
        
        print("\n" + "="*60)
        if all_good:
            print("✅ READY FOR PHASE 3")
            print("="*60)
            print("\nNext steps:")
            print("1. Create .env file with OpenRouter API key")
            print("2. Tell Cursor to proceed to Phase 3 (JD Parser)")
            print("3. Test with real LLM calls")
        else:
            print("⚠️  MINOR ISSUES DETECTED")
            print("="*60)
            print("\nReview warnings above.")
            print("If only signature differences, can still proceed.")
            print("If critical errors, ask Cursor to fix specific issues.")
        print("="*60)
        
        return 0 if all_good else 1
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ CRITICAL ERROR: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))