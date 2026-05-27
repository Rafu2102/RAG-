# -*- coding: utf-8 -*-
"""
scratch/normalize_courses.py
============================
大語言模型批次課程正規化腳本。
讀取 scratch/temp_json 中的原始 JSON 檔案，利用專案現有的 llm.gemini_client 呼叫 Gemini API，
依據與資料庫格式 100% 相同之精準 Prompt 模板將資料格式化為標準的純文字課程檔案，
並將其分類存入新建的資料夾中（例如：scratch/courses_test/食品系114學年度第1學期課程資訊/）。
"""

import os
import sys
import json
import time
import re
from collections import defaultdict

# 確保 Windows 終端機下的輸出編碼支援 UTF-8 (以正常顯示 Emoji 與中文)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# 將專案根目錄加入 Python 搜尋路徑，以便匯入 llm.gemini_client 與 config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
try:
    # 根據使用者指示，不使用 call_gemini_with_fallback 降級機制，改為直接調用 call_gemini 核心函數
    from llm.gemini_client import call_gemini
except ImportError:
    print("❌ 無法匯入 llm.gemini_client，請確認是否在專案根目錄或專案虛擬環境中執行此腳本。")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# ⚙️ 設定參數
# ═══════════════════════════════════════════════════════════════
TEMP_DIR = "scratch/temp_json"
OUTPUT_ROOT = "scratch/courses_test"  # 測試用輸出目錄，放置於 scratch 下

def format_class_time(time_raw):
    """將簡寫的上課時間（如 (三)12-13、(五)2-4）轉換成中文完整描述格式。
    若已是中文完整描述或為空，則直接回傳原值。
    """
    if not time_raw:
        return ""
    time_str = time_raw.strip()
    
    # 若已經包含 "每周" 或 "星期"，代表已是完整描述版，不予處理
    if "每周" in time_str or "星期" in time_str:
        return time_str
        
    weekday_map = {
        "一": "每周星期一",
        "二": "每周星期二",
        "三": "每周星期三",
        "四": "每周星期四",
        "五": "每周星期五",
        "六": "每周星期六",
        "日": "每周星期日"
    }
    
    def get_section_desc(sec_str):
        sec_str = sec_str.strip()
        alpha_map = {"A": "10", "B": "11", "C": "12", "D": "13"}
        val_str = alpha_map.get(sec_str.upper(), sec_str)
        try:
            val = int(val_str)
            if 1 <= val <= 4:
                return f"上午第{val}節"
            elif 5 <= val <= 9:
                return f"下午第{val}節"
            elif 10 <= val <= 13:
                return f"晚上第{val}節"
            else:
                return f"第{val}節"
        except ValueError:
            return f"第{sec_str}節"

    # 1. 匹配多個不同的星期簡寫區段，如 (三)12-13 (五)2-4 或 (三)5, (五)2-4
    parts = re.split(r"[,，\s]+", time_str)
    formatted_parts = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 匹配 (星期)起-止 格式，如 (三)12-13 或 (三)12~13
        match_range = re.match(r"^\(([一二三四五六日])\)\s*(\d+|[A-D])\s*[-~]\s*(\d+|[A-D])$", part)
        if match_range:
            wday = weekday_map[match_range.group(1)]
            start_sec = get_section_desc(match_range.group(2))
            end_sec = get_section_desc(match_range.group(3))
            
            # 優化：若起止節次的前綴相同（如 "下午第5節" 與 "下午第7節"），則省略後者前綴，轉化為 "下午第5節~第7節"
            time_prefixes = ["上午", "下午", "晚上"]
            for prefix in time_prefixes:
                if start_sec.startswith(prefix) and end_sec.startswith(prefix):
                    end_sec = end_sec.replace(prefix, "")
                    break
                    
            formatted_parts.append(f"{wday}{start_sec}~{end_sec}")
            continue
            
        # 匹配單一節次，如 (三)5
        match_single = re.match(r"^\(([一二三四五六日])\)\s*(\d+|[A-D])$", part)
        if match_single:
            wday = weekday_map[match_single.group(1)]
            sec = get_section_desc(match_single.group(2))
            formatted_parts.append(f"{wday}{sec}")
            continue
            
        # 若匹配失敗，直接放入
        formatted_parts.append(part)
        
    return "、".join(formatted_parts)

def clean_department_name(dept_raw, class_raw=None):
    """將科系名稱清洗為標準簡稱，利用 config.py 中的 DEPT_REGISTRY。"""
    if not dept_raw:
        return "未知系"
    dept = dept_raw.replace("國立金門大學", "").strip()
    
    # 若為通識教育中心或體育室，直接以班級名稱（如「日大學通識」、「日大學國文」、「日大學體育」等）作為簡稱進行分流
    if "通識" in dept or "通識教育中心" in dept or "體育" in dept or "體育室" in dept:
        if class_raw:
            return class_raw.strip()
        return "日大學體育" if ("體育" in dept or "體育室" in dept) else "日大學通識"
        
    # 遍歷專案 config 中的 DEPT_REGISTRY 進行匹配
    if hasattr(config, "DEPT_REGISTRY"):
        for short_name, registry_info in config.DEPT_REGISTRY.items():
            full_name = registry_info.get("full_name", "")
            aliases = registry_info.get("aliases", [])
            keywords = registry_info.get("keywords", [])
            
            if dept == full_name or any(alias in dept for alias in aliases) or any(kw in dept.lower() for kw in keywords):
                return short_name
                
    # 備用模糊邏輯
    return dept.replace("學系", "系")

def scan_for_duplicates(json_files):
    """預先掃描所有課程，將相同科系且相同課程名稱的課程資料分組。"""
    course_groups = defaultdict(list)
    for file in json_files:
        path = os.path.join(TEMP_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            list_info = data.get("list_info", {})
            if "上課時間" in list_info:
                list_info["上課時間"] = format_class_time(list_info["上課時間"])
            if "授課教師" in list_info:
                list_info["授課教師"] = list_info["授課教師"].replace(",", "、").replace("，", "、").strip()
                
            dept_short = clean_department_name(list_info.get("科系"), list_info.get("班級"))
            course_name = list_info.get("科目名稱", "").strip()
            
            # 清理不合法字元，以便與 run_normalization 中的鍵值對齊
            invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
            for char in invalid_chars:
                course_name = course_name.replace(char, "_")
                
            if dept_short and course_name:
                course_groups[(dept_short, course_name)].append(list_info)
        except Exception:
            continue
    return course_groups

def extract_ge_theme(list_info):
    """從備註或限修備註中提取通識主題領域，例如「社會科學—社會類」"""
    remarks = [list_info.get("備註", ""), list_info.get("限修備註", "")]
    combined_remarks = " ".join(filter(None, remarks))
    
    # 1. 優先嘗試精準匹配常見的通識分類關鍵字
    ge_patterns = [
        r"社會科學[—-]社會類",
        r"人文藝術[—-]人文類",
        r"人文藝術[—-]藝術類",
        r"自然科學[—-]物質類",
        r"自然科學[—-]生命類",
        r"歷史與文化類",
        r"社會科學與管理類",
        r"思維與方法類",
        r"藝術與美感類",
        r"生命與健康類",
        r"生命與健康",
        r"物質科學",
        r"跨領域學科",
        r"跨領域",
        r"綜合領域"
    ]
    for pattern in ge_patterns:
        match = re.search(pattern, combined_remarks)
        if match:
            return match.group(0).strip()
            
    # 2. 備用匹配：使用較寬鬆的正則匹配
    match = re.search(r"通識(?:主題)?(?:領域)?(?:核心)?[：:\s]*([^\s，,;；、\)]+)", combined_remarks)
    if match:
        theme = match.group(1).strip()
        theme = re.sub(r"^[：:\s]+", "", theme)
        return theme
            
    return None

def build_prompt(list_info, detail_text, year, term):
    """建構傳送給 Gemini API 的 Prompt 模板，確保格式與資料庫檔案 100% 相同。"""
    try:
        y_val = int(year)
        t_val = int(term)
        west_year = y_val + 1911 if t_val == 1 else y_val + 1912
        season = "秋季" if t_val == 1 else "春季"
        semester_west = f"{west_year}年{season}學期"
    except Exception:
        semester_west = "未知學期"
        
    dept_short = clean_department_name(list_info.get("科系"), list_info.get("班級"))
    
    # 決定首行科系宣告名稱：通識教育中心與體育室用全稱，一般科系用簡稱
    raw_dept = list_info.get("科系", "").replace("國立金門大學", "").strip()
    if "通識" in raw_dept or "體育" in raw_dept:
        title_dept = raw_dept
    else:
        title_dept = dept_short
        
    course_name = list_info.get("科目名稱", "").strip()
    english_name = list_info.get("科目英文名", "").strip()
    english_name_display = f" ({english_name})" if english_name else ""
    
    # 取得清洗後的備註與通識主題
    ge_theme = extract_ge_theme(list_info)
    remark_raw = list_info.get("備註", "").strip()
    remark_clean = remark_raw
    if ge_theme:
        remark_clean = re.sub(rf"通識(?:主題)?(?:領域)?(?:核心)?[：:\s-]*{re.escape(ge_theme)}", "", remark_clean).strip()
        if remark_clean == ge_theme:
            remark_clean = ""
        remark_clean = remark_clean.strip("，。；;、 ")

    # 組合上課時間與備註/通識欄位
    time_str = list_info.get("上課時間", "").strip()
    ge_theme_line = f"\n通識主題領域：{ge_theme}" if ge_theme else ""
    remark_line = f"\n備註：{remark_clean}" if remark_clean else ""
    
    # 高精度上限人數格式微調，防禦性對齊專題研究範本
    limit_num = list_info.get("上限人數", "").strip()
    if limit_num.isdigit():
        limit_display = f"{limit_num}人"
    else:
        limit_display = limit_num
        
    if "依系上提供名單加選" in remark_raw and "依系上提供名單加選" not in limit_display:
        if limit_num == "0" or not limit_num:
            limit_display = "0人 (依系上提供名單加選)"
            
    # 格式化學分數與授課時數為統一保留一位小數格式，確保所有學期檔案高度規格一致化
    def format_to_float_str(val_raw):
        val_raw = val_raw.strip()
        try:
            val_float = float(val_raw)
            return f"{val_float:.1f}"
        except Exception:
            return val_raw
            
    credit_display = format_to_float_str(list_info.get("學分", ""))
    hours_display = format_to_float_str(list_info.get("時數", ""))
            
    # 建構基本資料區塊 (嚴格遵循無額外括號格式，開頭為「國立金門大學...」)
    basic_info_formatted = f"""國立金門大學{year}學年度第{term}學期{title_dept}課程
學年度：{year}學年度
學期：第{term}學期
課程名稱：{course_name}{english_name_display}
部別：{list_info.get("部別", "").strip()}
開課班級：{list_info.get("班級", "").strip()}
學分數：{credit_display}學分
授課時數：{hours_display}小時
授課教師：{list_info.get("授課教師", "").strip()}
必選修：{list_info.get("必/選", "").strip()}
教室：{list_info.get("教室", "").strip()}
修課上限人數：{limit_display}
上課時間：{time_str}{ge_theme_line}{remark_line}"""

    prompt = f"""請將以下的「詳細內頁原始文字」進行整合，並將其正規化，填入下方規定的模板中。
請直接輸出填充後的純文字，切勿在最外層加程式碼方塊（```）或任何 Markdown 符號（如 **、#、### 等）。

【必須遵循的輸出格式範本】：
課程教學目標：
1. <請從原始文字中整理出課程教學目標列在此處，請盡可能保留原始細節，不要過度簡略，資料越完整越好，若無則填寫「無詳細課程教學目標說明」>

課程教學綱要：
1. <請從原始文字中整理出課程教學綱要列在此處，請盡可能保留原始細節，不要過度簡略，資料越完整越好，若無則填寫「無詳細課程教學綱要說明」>

教科書資料：
1. 書名：<書名，若有多本，則改用 1-1.、1-2. 到 1-5. 格式，多本之間以空行分隔>
2. 出版日期：<出版日期，若無請填寫「未標註」>
3. 作者：<作者，若無請填寫「未標註」>
4. 出版社：<出版社，若無請填寫「未標註」>
5. 版本：<版本，若無請填寫「未標註」>
<說明：若只有 1 本教科書，請使用 1. 到 5. 順序編號，格式如上；若有 2 本以上，第一本使用 1-1. 到 1-5.，第二本使用 2-1. 到 2-5.。若無教科書或教科書書名為「無」，請直接填寫「無教科書」或「無」>

參考書資料：
1. 書名：<書名，格式規則與編號規則與教科書完全相同>
2. 出版日期：<出版日期，若無請填寫「未標註」>
3. 作者：<作者，若無請填寫「未標註」>
4. 出版社：<出版社，若無請填寫「未標註」>
5. 版本：<版本，若無請填寫「未標註」>
<說明：若無參考書或參考書書名為「無」，請直接填寫「無參考書」或「無」>

※請遵守智慧財產權觀念，依著作權法規定，教科書及教材不得非法影印與使用盜版軟體。

教學進度表（{semester_west}）：
第1週課程 (請填入原始進度表第1週日期區間，格式必須為 YYYY/MM/DD─YYYY/MM/DD)：<第一週課程內容，格式必須為：第N週課程 (YYYY/MM/DD─YYYY/MM/DD)：主題名稱>
第2週課程 (請填入原始進度表第2週日期區間，格式與上面相同)：<第二週課程內容，依此類推>
...
第18週課程 (請填入原始進度表第18週日期區間，格式與上面相同)：<第18週課程內容，中間不得有空行，若缺少週次需自行按順序補齊並填寫「彈性補充教學(自主學習、探究與實作)」>

成績評定方式：
1. <請從原始文字中整理出成績評定方式內容，請盡可能保留原始細節，不要過度簡略，資料越完整越好，若無則填寫「無詳細成績評定方式說明」>

課堂要求：
<請從原始文字中整理出課堂要求內容，請盡可能保留原始細節，不要過度簡略，資料越完整越好，若無則寫「無詳細課堂要求說明」>

---
【詳細內頁原始文字】：
{detail_text}
"""
    return prompt, basic_info_formatted

def run_normalization():
    import argparse
    parser = argparse.ArgumentParser(description="大語言模型批次課程正規化腳本。")
    parser.add_argument("--out", "-o", default="scratch/courses_test", help="輸出目錄路徑 (預設為 scratch/courses_test)")
    parser.add_argument("--delay", "-d", type=float, default=0.1, help="每次 API 呼叫的間隔延遲時間 (秒，預設為 0.1)")
    parser.add_argument("--limit", "-l", type=int, default=0, help="本次運行的最大 API 呼叫次數限制 (預設為 0，代表不限制)")
    
    args = parser.parse_args()
    output_root = args.out
    delay_sec = args.delay
    max_api_calls = args.limit if args.limit > 0 else 999999
    
    if not os.path.exists(TEMP_DIR):
        print(f"❌ 找不到暫存 JSON 目錄: {TEMP_DIR}")
        return
        
    json_files = [f for f in os.listdir(TEMP_DIR) if f.startswith("raw_") and f.endswith(".json")]
    if not json_files:
        print("ℹ️ 暫存目錄中無 raw_*.json 課程檔案。")
        return
        
    print(f"📊 找到 {len(json_files)} 個原始課程 JSON 檔案。")
    
    # 掃描課程以排除重名衝突
    course_groups = scan_for_duplicates(json_files)
    
    print("=" * 60)
    print("🤖 課程正規化批次處理腳本")
    print("=" * 60)
    print(f"分流輸出目錄設定為：{output_root}")
    print(f"每次請求延遲時間：{delay_sec} 秒")
    if args.limit > 0:
        print(f"本次最大呼叫上限：{max_api_calls} 次")
    else:
        print("本次最大呼叫上限：不限制")
    print("=" * 60)
        
    total = len(json_files)
    success_count = 0
    consecutive_failures = 0
    api_calls_made = 0
    
    for idx, file in enumerate(json_files):
        path = os.path.join(TEMP_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 讀取檔案 {file} 失敗: {e}")
            continue
            
        list_info = data.get("list_info", {})
        if "上課時間" in list_info:
            list_info["上課時間"] = format_class_time(list_info["上課時間"])
        if "授課教師" in list_info:
            list_info["授課教師"] = list_info["授課教師"].replace(",", "、").replace("，", "、").strip()
            
        detail_text = data.get("detail_raw_text", "")
        
        dept_raw = list_info.get("科系")
        dept_short = clean_department_name(dept_raw, list_info.get("班級"))
        
        # 清理科系名稱中的非法檔名路徑字元，防止 os.makedirs 錯誤
        invalid_path_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_path_chars:
            dept_short = dept_short.replace(char, "_")
            
        # 從 JSON 中讀取該課程所屬學期，若無則預設為 "114-2"
        semester = data.get("semester", "114-2")
        try:
            year, term = semester.split("-")
        except ValueError:
            year, term = "114", "2"
            
        # 決定科系資料夾名稱，格式與 data/courses 保持完全一致（無半形空格）
        # 例如：食品系114學年度第2學期課程資訊
        dept_folder = f"{dept_short}{year}學年度第{term}學期課程資訊"
        target_dir = os.path.join(output_root, dept_folder)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        # 計算不衝突的檔名
        course_name = list_info.get("科目名稱", "未知課程").strip()
        english_name = list_info.get("科目英文名", "").strip()
        
        # 清理檔名不合法字元
        invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            course_name = course_name.replace(char, "_")
            english_name = english_name.replace(char, "_")
            
        # 去括號與空格修剪
        english_name_display = f" ({english_name})" if english_name else ""
        
        # 取得同科系下同課程名稱的所有課程列表資訊
        group_items = course_groups.get((dept_short, course_name), [])
        occurrences = len(group_items)
        
        if occurrences > 1:
            # 判斷是否所有同名課程的「班級」都互不相同 (保留所有項目的班級長度進行比對)
            classes = [item.get("班級", "").strip() for item in group_items]
            unique_classes = set(classes)
            
            if len(unique_classes) == occurrences:
                # 班級皆不相同，僅加上班級：{課程名稱} ({英文}) ({班級}).txt
                class_name = list_info.get("班級", "未知班級").strip()
                filename = f"{course_name}{english_name_display} ({class_name}).txt"
            else:
                # 有相同班級，進一步判斷是否有「班級與教師」都完全相同的重複項目
                class_teachers = [(item.get("班級", "").strip(), item.get("授課教師", "").strip()) for item in group_items]
                curr_class = list_info.get("班級", "未知班級").strip()
                curr_teacher = list_info.get("授課教師", "未知教師").strip()
                
                # 計算目前這組 (班級, 授課教師) 在同名課程分組中出現的次數
                collision_count = class_teachers.count((curr_class, curr_teacher))
                
                if collision_count > 1:
                    # 班級與教師都相同，加上選課代碼以求絕對唯一：{課程名稱} ({英文}) ({班級}_{授課教師}_{選課代碼}).txt
                    course_code = list_info.get("選課代碼", "").strip()
                    if not course_code:
                        # 備用提取機制：自 raw_{semester}_{course_code}.json 檔名中解析出 4 位數代碼
                        match_code = re.search(r"raw_[^_]+_([^\s.]+)\.json", file)
                        course_code = match_code.group(1) if match_code else "未知代碼"
                    filename = f"{course_name}{english_name_display} ({curr_class}_{curr_teacher}_{course_code}).txt"
                else:
                    # 班級有重複，但此特定班級與教師組合是唯一的：{課程名稱} ({英文}) ({班級}_{授課教師}).txt
                    filename = f"{course_name}{english_name_display} ({curr_class}_{curr_teacher}).txt"
        else:
            # 唯一課程，檔名不加任何括號
            filename = f"{course_name}{english_name_display}.txt"
            
        target_file_path = os.path.join(target_dir, filename)
        
        # 斷點續傳：若檔案已存在，則直接跳過，省去 API Token
        if os.path.exists(target_file_path):
            print(f"⏭️  [{idx + 1}/{total}] {filename} 已存在於目標資料夾，跳過。")
            success_count += 1
            continue
            
        # 檢查是否即將達到自訂限制
        if api_calls_made >= max_api_calls:
            print(f"\n⚠️ 已達到本次運行的 API 限制 {max_api_calls} 次。為避免 API 超出配額，腳本自動暫停。")
            break
            
        print(f"🧠 [{idx + 1}/{total}] 正在呼叫 Gemini 正規化: {filename}...")
        prompt, basic_info_formatted = build_prompt(list_info, detail_text, year, term)
        
        api_calls_made += 1
        try:
            # 根據使用者指示，不使用 call_gemini_with_fallback 降級機制，改為直接調用 call_gemini 核心函數
            normalized_content = call_gemini(
                prompt,
                model="pro",
                thinking="low",
                timeout=60.0
            )
            
            # 最暴力的拔除法，絕對不傷內文，避免多重 Markdown 方塊導致資料被正則表達式腰斬
            cleaned_content = normalized_content
            for tag in ["```markdown", "```text", "```html", "```"]:
                cleaned_content = cleaned_content.replace(tag, "")
            cleaned_content = cleaned_content.strip()
                
            # 寫入目標檔案
            # 將絕對精確的 Python 拼接基礎資訊與 LLM 整理的下半部大綱合併
            final_content = f"{basic_info_formatted}\n\n{cleaned_content.strip()}"
            
            with open(target_file_path, "w", encoding="utf-8") as out_f:
                out_f.write(final_content)
                
            print(f"✨ 成功寫入檔案: {target_file_path}")
            success_count += 1
            consecutive_failures = 0  # 成功時重設計數器
            
        except Exception as e:
            print(f"❌ 呼叫 Gemini 失敗（課程代碼 {list_info.get('選課代碼')}）: {e}")
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("\n⚠️ 偵測到連續 5 次 API 呼叫失敗，判定為網路異常或 API 金鑰限制，終止批次程序。")
                sys.exit(1)
            
        # 配合 API 限制，每筆請求間隔延遲
        time.sleep(delay_sec)
        
    print(f"\n🎉 任務執行完畢！成功處理: {success_count}/{total} 門課程。")
    print(f"📁 檔案已整理至資料夾：{output_root}")

if __name__ == "__main__":
    run_normalization()
