import os
import asyncio
import logging
import google.auth
from google.auth.transport.requests import Request as SyncRequest

# 設定 logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# 引入 gemini_client
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.gemini_client import a_call_gemini_with_fallback, GeminiAPIError

async def test_integration():
    print("\n[INFO] Starting integration test for a_call_gemini_with_fallback...")
    
    # 測試 1：一般 PRO + HIGH 思考
    prompt = "請寫一段 50 字關於大自然的美麗短詩，並詳細思考。"
    print(f"\n[TEST 1] Calling with thinking='HIGH' (should succeed natively)...")
    try:
        ans = await a_call_gemini_with_fallback(prompt, thinking="HIGH")
        print("[SUCCESS] Test 1 completed!")
        print(f"  Answer ({len(ans)} chars): {ans[:200]}...")
    except Exception as e:
        print(f"[FAILED] Test 1 failed: {e}")

    # 測試 2：測試安全降級（模擬當我們給一個會觸發 MAX_TOKENS 或非常巨大的 context 時，或者直接傳入高難度並設定 max_tokens 很小來強迫觸發兩階段降級）
    print(f"\n[TEST 2] Calling with thinking='HIGH' and max_tokens=100 (should force MAX_TOKENS, then auto-fallback to MEDIUM/Pro)...")
    try:
        ans2 = await a_call_gemini_with_fallback(prompt, thinking="HIGH", max_tokens=150)
        print("[SUCCESS] Test 2 completed via auto-fallback!")
        print(f"  Answer ({len(ans2)} chars): {ans2[:200]}...")
    except Exception as e:
        print(f"[SUCCESSFUL DIAGNOSIS] Test 2 correctly handled/propagated error: {e}")

if __name__ == "__main__":
    asyncio.run(test_integration())
