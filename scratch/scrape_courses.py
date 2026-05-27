# -*- coding: utf-8 -*-
"""
scratch/scrape_courses.py
=========================
半自動校務系統課程爬蟲腳本。
使用 Playwright 啟動瀏覽器，由使用者手動完成登入、點擊選單並搜尋出列表。
腳本會讀取搜尋結果表格，遍歷每一列資料並點擊進入詳細內頁，
將「列表列資訊」與「詳細內頁 innerText」合併存入 JSON 暫存檔，實現重複排除。

執行前請先安裝：
    pip install playwright
    playwright install
"""

import os
import json
import time
from playwright.sync_api import sync_playwright

# ═══════════════════════════════════════════════════════════════
# ⚙️ 爬蟲設定參數（請根據學校網站結構微調）
# ═══════════════════════════════════════════════════════════════
TEMP_DIR = "scratch/temp_json"
SCHOOL_LOGIN_URL = "https://select.nqu.edu.tw/kmkuas/index_sky.html"  # 國立金門大學校務資訊系統網址

# 定位器設定 (Selectors)
TABLE_SELECTOR = "table#query_result_table"  # 搜尋結果表格的 CSS 定位器，請根據實際網頁修改
ROW_SELECTOR = "tr"  # 表格列定位器
CELL_SELECTOR = "td"  # 欄位定位器
COURSE_LINK_SELECTOR = "a"  # 開啟詳細資料的超連結定位器（通常在課程名稱欄位內）

# 列表欄位映射定義（必須與你的欄位順序一致）
FIELD_KEYS = [
    "部別", "科系", "班級", "選課代碼", "科目名稱", "科目英文名",
    "學分", "時數", "必/選", "授課教師", "教室", "修課人數",
    "上限人數", "上課時間", "限修備註", "備註"
]

def ensure_dirs():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"✅ 已建立暫存資料夾：{TEMP_DIR}")

def run_scraper():
    ensure_dirs()
    
    with sync_playwright() as p:
        print("🚀 正在啟動 Chrome 瀏覽器...")
        browser = p.chromium.launch(headless=False)  # 必須看得到瀏覽器以供手動登入
        context = browser.new_context()
        page = context.new_page()
        
        # 註冊全域對話框自動處理，避免 alert() 阻礙網頁讀取與關閉
        context.add_init_script("window.alert = () => {}; window.confirm = () => true;")
        context.on("page", lambda new_page: new_page.on("dialog", lambda dialog: dialog.accept()))
        page.on("dialog", lambda dialog: dialog.accept())
        
        print(f"🔗 正在前往網頁：{SCHOOL_LOGIN_URL}")
        page.goto(SCHOOL_LOGIN_URL)
        
        print("\n" + "=" * 60)
        print("💡 請在開啟的瀏覽器中完成登入。")
        print("=" * 60)
        input("👉 登入成功後，請在此處按 [Enter] 鍵繼續：")
        
        while True:
            print("\n" + "=" * 60)
            print("💡 請在瀏覽器中執行以下步驟：")
            print("  1. 點選左側選單的「課程資料查詢」。")
            print("  2. 選擇所需的「學年學期」（如：115-1、114-2 等）。")
            print("  3. 點擊「搜尋」按鈕，直到搜尋結果列表完整顯示在畫面上。")
            print("=" * 60)
            
            semester_input = input("👉 請輸入當前網頁搜尋的學期（例如 115-1，或輸入 exit 退出）：").strip()
            if semester_input.lower() == "exit":
                print("🚪 已退出爬取流程。")
                break
                
            if "-" not in semester_input:
                print("❌ 格式不正確！必須為「學年度-學期」（例如 115-1 或 114-2），請重新輸入。")
                continue
                
            # 尋找結果表格，自動支援 iframe/frame 巢狀結構，並以表頭內容進行「免設定」精準特徵定位
            table = None
            target_frame = page  # 預設為主要分頁
            
            # 定義特徵選擇器：尋找同時包含「部別」、「科系」、「科目名稱」之表格
            feature_selector = "table:has(td:has-text('部別')):has(td:has-text('科系')):has(td:has-text('科目名稱'))"
            
            try:
                # 1. 優先在主頁面嘗試特徵定位
                if page.locator(feature_selector).count() > 0:
                    table = page.locator(feature_selector).first.element_handle()
                else:
                    # 2. 使用備用選擇器
                    page.wait_for_selector(TABLE_SELECTOR, timeout=2000)
                    table = page.query_selector(TABLE_SELECTOR)
            except Exception:
                # 3. 遍歷所有 frame 嘗試定位
                for frame in page.frames:
                    try:
                        if frame.locator(feature_selector).count() > 0:
                            table = frame.locator(feature_selector).first.element_handle()
                            target_frame = frame
                            print(f"✨ 成功在 frame「{frame.name}」中自動定位目標表格。")
                            break
                        else:
                            table = frame.query_selector(TABLE_SELECTOR)
                            if table:
                                target_frame = frame
                                print(f"✨ 成功在 frame「{frame.name}」中透過備用定位器找到表格。")
                                break
                    except Exception:
                        continue
                        
            if not table:
                print(f"❌ 找不到課程結果表格，請確認網頁是否已完成查詢並顯示列表。")
                continue
                
            # 覆寫頁面中的 of_link 函數，強制將其改為在「新分頁」以 POST 方式開啟，防止主分頁被導航離開且避免 GET 請求導致資料空白
            try:
                target_frame.evaluate("""() => {
                    window.of_link = function(fncurl, year, sms, loginid, clssubdup) {
                        const form = document.createElement("form");
                        form.method = "POST";
                        form.action = fncurl;
                        form.target = "_blank";
                        
                        const createInput = (name, value) => {
                            const input = document.createElement("input");
                            input.type = "hidden";
                            input.name = name;
                            input.value = value;
                            form.appendChild(input);
                        };
                        
                        createInput("arg01", year);
                        createInput("arg02", sms);
                        createInput("arg03", loginid);
                        createInput("arg04", clssubdup);
                        
                        document.body.appendChild(form);
                        form.submit();
                        document.body.removeChild(form);
                    };
                }""")
                print("✨ 已成功將 of_link 覆寫為在新分頁以正確的 POST 欄位開啟。")
            except Exception as e:
                print(f"⚠️ 覆寫 of_link 失敗：{e}")
                
            rows = table.query_selector_all(ROW_SELECTOR)
            # 排除表頭列（通常第一或第二列是標題欄，請根據實際網頁調整起點，這裡假設從 index 1 開始為資料列）
            data_rows = rows[1:]
            total_rows = len(data_rows)
            print(f"📊 【學期 {semester_input}】偵測到 {total_rows} 筆課程資料列。")
            
            for index, row in enumerate(data_rows):
                cells = row.query_selector_all(CELL_SELECTOR)
                if not cells or len(cells) < len(FIELD_KEYS):
                    continue
                    
                # 1. 擷取列表列欄位資料
                row_data = {}
                for k_idx, key in enumerate(FIELD_KEYS):
                    try:
                        cell_text = cells[k_idx].inner_text().strip()
                    except Exception:
                        cell_text = ""
                    row_data[key] = cell_text
                    
                course_code = row_data.get("選課代碼", "").replace("\n", "").replace("\r", "").strip()
                course_name = row_data.get("科目名稱", "").strip()
                teacher = row_data.get("授課教師", "").strip()
                
                # 防呆：若選課代碼不是純數字，代表可能是表頭或裝飾行，直接跳過
                if not course_code or not course_code.isdigit():
                    continue
                    
                # 檔名加入學期前綴，避免跨學期重複選課代碼覆蓋
                temp_json_path = os.path.join(TEMP_DIR, f"raw_{semester_input}_{course_code}.json")
                if os.path.exists(temp_json_path):
                    # 讀取檢查是否為失敗的爬取（如果 detail_raw_text 為空或包含 "科目名稱 : ()" 等無效特徵，則重新爬取）
                    try:
                        with open(temp_json_path, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                        existing_detail = existing_data.get("detail_raw_text", "")
                        if "科目名稱 : ()" in existing_detail or existing_detail in ["詳細內頁抓取失敗", "無詳細內頁資訊", ""]:
                            print(f"🔄 [{semester_input}] 課程 [{course_code}] {course_name} 偵測到無效詳細資料，將重新爬取。")
                        else:
                            print(f"⏭️  [{semester_input}] 課程 [{course_code}] {course_name} 已存在且有效，跳過。")
                            continue
                    except Exception:
                        # 讀取失敗也重新爬取
                        pass
                    
                print(f"⏳ 正在爬取 [{index + 1}/{total_rows}]: {course_code} - {course_name} ({teacher})... ", end="", flush=True)
                start_time = time.time()
                
                # 2. 尋找並點擊詳細內頁連結
                link = cells[FIELD_KEYS.index("科目名稱")].query_selector(COURSE_LINK_SELECTOR)
                if not link:
                    # 嘗試在整列中尋找第一個超連結
                    link = row.query_selector(COURSE_LINK_SELECTOR)
                    
                if not link:
                    print(f"❌ 找不到點擊詳細頁面的超連結，跳過詳細資訊。")
                    detail_text = "無詳細內頁資訊"
                else:
                    detail_text = ""
                    try:
                        # 優先嘗試捕獲直接點擊（針對 of_link / window.open 等開啟新視窗）
                        with context.expect_page(timeout=3000) as new_page_info:
                            link.click()
                        detail_page = new_page_info.value
                        detail_page.wait_for_load_state(state="domcontentloaded")
                        detail_text = detail_page.evaluate("() => document.body.innerText").strip()
                        detail_page.close()
                    except Exception as e:
                        # 若未開啟新分頁，檢查是否可以直接獲取 href
                        try:
                            href = link.get_attribute("href")
                            if href and not href.strip().startswith("javascript:") and not href.strip().startswith("#"):
                                detail_url = target_frame.evaluate(f"href => new URL(href, document.baseURI).href", href)
                                detail_page = context.new_page()
                                detail_page.goto(detail_url)
                                detail_page.wait_for_load_state(state="domcontentloaded")
                                detail_text = detail_page.evaluate("() => document.body.innerText").strip()
                                detail_page.close()
                            else:
                                raise ValueError("No valid href and click did not open new window")
                        except Exception as ex_href:
                            print(f"❌ 詳細頁抓取失敗：{ex_href}")
                            detail_text = "詳細內頁抓取失敗"
                
                # 3. 合併資料並寫入暫存 JSON
                combined_data = {
                    "semester": semester_input,
                    "list_info": row_data,
                    "detail_raw_text": detail_text
                }
                
                with open(temp_json_path, "w", encoding="utf-8") as f:
                    json.dump(combined_data, f, ensure_ascii=False, indent=4)
                    
                elapsed = time.time() - start_time
                print(f"完成！[耗時 {elapsed:.2f} 秒]", flush=True)
                
                # 4. 強制分頁清理防護：關閉除了主頁面（page）之外的所有分頁，防止分頁殘留
                for open_page in context.pages:
                    if open_page != page:
                        try:
                            open_page.close()
                        except Exception:
                            pass
                
                # 設定溫和延遲 0.4 秒防止學校伺服器阻斷，平衡爬取效率
                time.sleep(0.4)
                
            print(f"🎉 學期 {semester_input} 爬取完成！")
            
        print("🎉 爬蟲程序結束！暫存檔案已儲存於 scratch/temp_json 中。")
        browser.close()

if __name__ == "__main__":
    run_scraper()
