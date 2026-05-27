import asyncio
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
# 強制標準輸出為 utf-8
sys.stdout.reconfigure(encoding='utf-8')

from llm.gemini_client import _a_post_with_retry, _resolve_model

async def test_high():
    config.setup_logging()
    prompt = "統計資訊工程學系的每一個年級必修選修數量"
    
    # 建立 payload，設定 thinkingLevel = "high"
    contents = {"role": "user", "parts": [{"text": prompt}]}
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 8192,
            "thinkingConfig": {"thinkingLevel": "high"},
        },
    }
    
    model_name, default_timeout, default_tokens = _resolve_model("pro")
    
    print("Sending request with thinkingLevel='high'...")
    try:
        resp = await _a_post_with_retry(model_name, payload, timeout=120.0)
        resp_json = resp.json()
        print("\n--- Raw Response JSON ---")
        print(json.dumps(resp_json, indent=2, ensure_ascii=False))
        print("-------------------------\n")
        
        # 嘗試解析 content.parts
        candidates = resp_json.get("candidates", [])
        if not candidates:
            print("No candidates found.")
            return
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        print(f"Number of parts returned: {len(parts)}")
        for idx, part in enumerate(parts):
            is_thought = part.get("thought", False)
            print(f"Part {idx+1}: thought={is_thought}")
            if "text" in part:
                print(f"  text length: {len(part['text'])}")
            else:
                print(f"  keys in part: {list(part.keys())}")
                
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_high())
