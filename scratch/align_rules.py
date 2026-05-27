import os
import sys
import json
import time

# 將專案根目錄加入路徑以便引入 llm.gemini_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.gemini_client import call_gemini

RULES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rules")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules_aligned")
TEMPLATE_FILE = os.path.join(RULES_DIR, "資訊工程學系畢業門檻.json")

def read_template():
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        return f.read()

def process_file(filename, template_str):
    source_path = os.path.join(RULES_DIR, filename)
    target_path = os.path.join(OUTPUT_DIR, filename)
    
    # 已經轉過的跳過（支援斷點續傳）
    if os.path.exists(target_path):
        print(f"⏩ {filename} 已經存在，跳過。")
        return True

    print(f"🚀 正在處理 {filename} ...")
    with open(source_path, "r", encoding="utf-8") as f:
        raw_json_str = f.read()

    prompt = f"""你是一位極度嚴謹的 JSON 資料工程師。你的任務是將以下提供的「原始 JSON 資料」，無損地重構並對齊至我提供的「標準範本 JSON」的資料結構與欄位階層。

【標準範本 JSON (資訊工程學系)】：
```json
{template_str}
```

【極度重要規則，違反將導致系統崩潰】：
1. 頂層鍵 (Top-level Keys)：
   - 必須包含且嚴格對齊範本的鍵名：`_description`, `_source`, `_last_updated`, `department`, `applicable_from`, `total_required`, `categories`, `graduation_conditions`。
   - 若原始 JSON 包含額外的頂層鍵（如 `program`, `total_elective_credits` 等），請「保留」這些額外鍵以防資料遺失。
   - 若原始資料缺少某些必備頂層鍵，請盡可能從內文推導或填入預設空字串 `""`。

2. **多學年壓縮 (解決超大檔案問題)**：
   - 如果原始 JSON 包含了多個學年（例如 110, 111, 112 等多份規則），你 **只需要** 提取並重構「最新的一個學年（數字最大）」的規則。將所有舊學年的資料完全捨棄！這能確保輸出大小不會超過系統極限。

3. `categories` (課程分類) 重構指南：
   - 原有的群組名稱（如 `院必修`, `系必修`, `專業選修`）請原封不動保留。
   - 群組內部的陣列必須對齊範本，包含 `required_credits` 以及 `courses` 陣列。
   - 每個課程物件請對齊 `{{ "name": "...", "credits": X, "year": Y, "semester": Z, "note": "..." }}`。
   - 若遇到多層巢狀結構（例如群組內還有次群組），請將其攤平為一層的 `courses` 陣列，並將次群組的名稱寫進該課程的 `note` 欄位中。

3. `graduation_conditions` (畢業條件)：
   - 必須是一個單純的字串陣列 (Array of Strings)。如果原始資料是物件或其他格式，請將其條列化並轉換為字串陣列。

4. 🚫 嚴禁行為 (絕對禁止)：
   - 嚴禁偷懶省略資料！(在最新學年的範圍內) 不允許使用 `...` 或是 `等其他課程`。所有的課程、文字、條件都必須 100% 完整轉移。
   - 嚴禁刪減原始 JSON 中的任何學分規定或課程備註。
   - 嚴禁改變原始資料的數值（學分、學期數等）。

5. 輸出排版與格式 (極度重要)：
   - 必須是合法、乾淨的 JSON 格式字串。
   - 請直接輸出 JSON 大括號開頭，絕對不要加上 ````json` 外框。
   - **排版必須完美模仿範本**：特別是 `courses` 陣列中的每一個課程物件，請務必寫在 **同一行** 內（例如 `{{ "name": "...", "credits": X }}`），**絕對不要**把課程物件內部的鍵值對拆成多行。這攸關使用者的視覺對齊要求！

【原始 JSON 資料】：
```json
{raw_json_str}
```
"""
    
    consecutive_failures = 0
    while consecutive_failures < 3:
        try:
            # 呼叫 Gemini 3.1 Pro，並依據官方文件設定 thinking="low" (格式轉換不需要深層推論，可加速並降低幻覺)
            response_text = call_gemini(
                prompt=prompt,
                model="gemini-3.1-pro",
                thinking="low",
                timeout=180.0     # 轉換 100KB+ JSON 可能需要較長時間
            )
            
            # 清理可能的 markdown 殘留
            cleaned_json = response_text.strip()
            for tag in ["```json", "```"]:
                cleaned_json = cleaned_json.replace(tag, "")
            cleaned_json = cleaned_json.strip()
            
            # 驗證是否為合法 JSON (但我們只做驗證，保留 Gemini 生成的原始精美單行排版)
            json.loads(cleaned_json)
            
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(cleaned_json)
                
            print(f"✨ 成功：{filename}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Gemini 回傳了非法的 JSON 格式：{e}")
            consecutive_failures += 1
            time.sleep(2)
        except Exception as e:
            print(f"❌ API 呼叫失敗：{e}")
            consecutive_failures += 1
            time.sleep(2)
            
    print(f"🛑 {filename} 連續失敗超過 3 次，跳過。")
    return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    try:
        template_str = read_template()
    except Exception as e:
        print(f"讀取範本失敗：{e}")
        return

    all_files = [f for f in os.listdir(RULES_DIR) if f.endswith(".json")]
    target_files = [f for f in all_files if f != "資訊工程學系畢業門檻.json"]
    
    print(f"共找到 {len(target_files)} 個需要對齊的規則檔案。")
    
    for idx, f in enumerate(target_files):
        print(f"\n進度：[{idx+1}/{len(target_files)}]")
        process_file(f, template_str)
        time.sleep(1)  # 避免觸發 rate limit
        
    print("\n✅ 所有規則檔案對齊完成！請前往 scratch/rules_aligned 檢查。")

if __name__ == "__main__":
    main()
