import os
import requests
import json
import google.auth
from google.auth.transport.requests import Request as SyncRequest

def test_thinking_levels():
    print("[INFO] Loading GCP ADC Credentials...")
    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(SyncRequest())
    token = credentials.token
    print(f"[INFO] Loaded. Project ID: {project_id}")
    
    # We test with the locations/global endpoint used in gemini_client.py
    url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/gemini-3.5-flash:generateContent"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    # Test levels
    levels_to_test = ["HIGH", "high", "MEDIUM", "medium"]
    
    for level in levels_to_test:
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": "請用繁體中文寫一首關於人工智慧的短詩，並詳細思考後回答。"}]
            }],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "thinkingConfig": {
                    "thinkingLevel": level
                }
            }
        }
        
        print(f"\n[TEST] Sending REST request to gemini-3.5-flash with thinkingLevel='{level}'...")
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            print("STATUS CODE:", resp.status_code)
            if resp.status_code == 200:
                print(f"[SUCCESS] level='{level}' processed successfully!")
                res_json = resp.json()
                
                # Check usageMetadata to see if thoughts exist
                usage = res_json.get("usageMetadata", {})
                thoughts_count = usage.get("thoughtsTokenCount", 0)
                print(f"  Thoughts tokens: {thoughts_count}")
                
                candidates = res_json.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    print(f"  Returned {len(parts)} parts in response content.")
                    for idx, part in enumerate(parts):
                        is_thought = part.get("thought", False)
                        text = part.get("text", "")
                        print(f"    Part {idx} (thought={is_thought}): {text[:100]}...")
                else:
                    print("  No candidates found in response.")
            else:
                print(f"[FAILED] level='{level}' returned status {resp.status_code}: {resp.text[:500]}")
        except Exception as e:
            print(f"[ERROR] request failed for level='{level}': {e}")

if __name__ == "__main__":
    test_thinking_levels()
