import os
import requests
import json
import google.auth
from google.auth.transport.requests import Request as SyncRequest

def print_thinking_response():
    print("[INFO] Loading GCP ADC Credentials...")
    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(SyncRequest())
    token = credentials.token
    print(f"[INFO] Loaded. Project ID: {project_id}")
    
    url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/gemini-3.5-flash:generateContent"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": "請用繁體中文寫一首關於人工智慧的短詩，並詳細思考後回答。"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 1024,
            "thinkingConfig": {
                "thinkingLevel": "HIGH"
            }
        }
    }
    
    print("\n[TEST] Sending request with level='HIGH'...")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        print("STATUS CODE:", resp.status_code)
        if resp.status_code == 200:
            res_json = resp.json()
            print("\n--- FULL RESPONSE JSON ---")
            print(json.dumps(res_json, indent=2, ensure_ascii=False))
            print("--------------------------")
        else:
            print(f"Failed with status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print_thinking_response()
