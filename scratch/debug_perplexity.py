# -*- coding: utf-8 -*-
import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

async def test_perplexity(query: str, system_prompt: str, user_prompt: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Test-Dcard"
    }
    data = {
        "model": "perplexity/sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                res_json = response.json()
                content = res_json["choices"][0]["message"]["content"]
                citations = res_json.get("citations", [])
                print(f"Citations ({len(citations)}): {citations}")
                print(f"Content:\n{content}\n")
            else:
                print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

async def main():
    sys_prompt_complex = """你是一個專業的資料整理助手，專門整理 Dcard 上的課程與教授評價。
請根據搜尋到的 Dcard 金門大學板（site:dcard.tw/f/nqu）文章與留言，整理出相關的評價。

請整理並輸出以下格式的表格：
教授 | 評價/推薦原因 | 負評/注意事項 | 來源文章連結

來源文章連結請輸出為 Markdown 超連結格式，例如：
[文章標題](https://www.dcard.tw/f/nqu/p/123456)

重要規則：
1. 不要使用 [1][2] 引用編號。
2. 每個來源都要附上完整 URL 且為有效的超連結。
3. 只能整理與使用者查詢相關的資訊，過濾掉無建設性的留言（例如「推某某老師」、「+1」、「好過」）。
4. 如果真的找不到任何相關的教授評價或課程討論，請直接回答「目前在 Dcard 金門大學版上找不到與該關鍵字相關的詳細評價或推薦。」，絕對不要捏造或硬掰。
5. 回答請使用繁體中文。
"""

    print("=== TEST 4: Query '李錫捷' (User: query, System: instructions) ===")
    user_prompt_q1 = "site:dcard.tw/f/nqu 李錫捷"
    await test_perplexity("李錫捷", sys_prompt_complex, user_prompt_q1)

    print("=== TEST 5: Query '資工' (User: query, System: instructions) ===")
    user_prompt_q2 = "site:dcard.tw/f/nqu 資工系 資工老師 資訊工程學系 資工"
    await test_perplexity("資工", sys_prompt_complex, user_prompt_q2)

    print("=== TEST 6: Query '英文教授' (User: query, System: instructions) ===")
    user_prompt_q3 = "site:dcard.tw/f/nqu 應英系 英文老師 大一英文 英文課 應用英語學系"
    await test_perplexity("英文教授", sys_prompt_complex, user_prompt_q3)


if __name__ == "__main__":
    asyncio.run(main())
