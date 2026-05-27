import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_dcard_queries():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] Missing OPENROUTER_API_KEY")
        return
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # We will test two queries: "英文教授" vs "應英系 英文老師"
    queries = ["英文教授", "應英系 英文老師"]
    
    for q in queries:
        prompt = f"""
請搜尋 site:dcard.tw/f/nqu
找出金門大學版中與「{q}」相關的文章與留言，特別是關於推薦或評價的部分。

整理：
1. 被討論/推薦的教授名字
2. 推薦/評價原因
3. 是否有負評或需注意的地方
4. 文章連結

來源文章連結請輸出為 Markdown 超連結格式。
"""
        data = {
            "model": "perplexity/sonar-pro",
            "messages": [
                {"role": "system", "content": "回答請使用繁體中文。你是一個專業的資料整理助手，專門整理 Dcard 上的課程與教授評價。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        print(f"\n[TEST] Sending Perplexity request for query='{q}'...")
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                res = resp.json()
                content = res["choices"][0]["message"]["content"]
                print(f"[SUCCESS] Results for '{q}':")
                print(content[:1000])
                print("-" * 50)
            else:
                print(f"[FAILED] status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")

if __name__ == "__main__":
    test_dcard_queries()
