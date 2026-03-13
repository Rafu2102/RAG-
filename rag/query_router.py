# -*- coding: utf-8 -*-
"""
query_router.py — Query Router 模組
=====================================
負責：
1. 判斷使用者問題類型（課程基本資訊 / 教學內容 / 教材 / 成績 / 一般）
2. 從問題中提取 metadata filter 條件（department, grade, teacher, course_type）
3. 為 Hybrid Retriever 提供路由結果與 metadata 過濾條件
"""

import logging
import json
import re
from dataclasses import dataclass, field

import requests

import config

logger = logging.getLogger(__name__)

# ── 動態名冊（啟動時從索引自動建構，取代硬編碼清單）──
_known_teachers: set[str] = set()
_known_courses: set[str] = set()


def init_known_registry(nodes: list) -> None:
    """
    從已索引的 Nodes metadata 中自動建構教師與課程名冊。
    應在 load_and_index() 完成後呼叫一次。
    """
    global _known_teachers, _known_courses
    teachers = set()
    courses = set()

    for node in nodes:
        meta = getattr(node, "metadata", {})

        # 教師：處理多人共授（如 "李錫捷、吳佳駿、柯志亨"）
        teacher_str = meta.get("teacher", "")
        if teacher_str and teacher_str != "未知":
            for t in re.split(r"[、,，/]", teacher_str):
                t = t.strip()
                if len(t) >= 2:
                    teachers.add(t)

        # 課程名稱：完整名 + 去括號的簡稱
        course_name = meta.get("course_name", "")
        if course_name and course_name != "未知":
            courses.add(course_name)
            # 提取去括號的簡稱（如 "微積分(一)" → "微積分"）
            short = re.sub(r"[（(][^）)]*[）)]", "", course_name).strip()
            if short and len(short) >= 2 and short != course_name:
                courses.add(short)

    _known_teachers = teachers
    _known_courses = courses
    logger.info(f"📋 動態名冊初始化完成 | 教師：{len(teachers)} 人 | 課程關鍵字：{len(courses)} 個")


# =============================================================================
# 📋 路由結果資料類別
# =============================================================================

@dataclass
class RouteResult:
    """Query Router 的輸出結果"""
    query_type: str              # 問題類型：course_info / syllabus / textbook / grading / schedule / general
    metadata_filters: dict = field(default_factory=dict)  # metadata 過濾條件
    confidence: float = 0.0      # 信心分數

    def __repr__(self):
        filters_str = ", ".join(f"{k}={v}" for k, v in self.metadata_filters.items())
        return f"RouteResult(type={self.query_type}, filters=[{filters_str}], conf={self.confidence:.2f})"


# =============================================================================
# 📝 Router Prompt
# =============================================================================

ROUTER_PROMPT = """你是一個校園課程助理的問題分類器。

你的任務是：
1. 判斷使用者問題的類型
2. 從問題中提取 metadata 過濾條件

## 問題類型
- course_info：詢問課程基本資訊（課程名稱、教師、學分、上課時間、教室、修課人數等）
- syllabus：詢問課程教學內容、教學目標、教學綱要
- textbook：詢問教科書、參考書、教材
- grading：詢問成績評定、考試方式、課堂要求
- schedule：詢問教學進度、每週課程安排
- calendar_action：把某堂課加到行事曆、記在日曆、設定提醒、刪除日曆行程、把事件從行事曆移除
- academic_calendar：詢問學校行事曆上的日期，如開學、期中考週、停修日期、放假等
- chitchat：日常打招呼、閒聊、感謝、無意義的問候（例如：「你好」、「你是誰」、「謝謝」、「早安」）
- general：其他一般性問題

## Metadata 條件
從問題中提取以下可能的過濾條件（若有提到）：
- dept_short：系所簡稱（如 "資工系", "電機系", "企管系", "觀光系", "護理系", "食品系" 等，碩士為 "資工碩" 等）
- grade：年級（如 "一", "二", "三", "四", "碩一", "碩二"）
- teacher：教師名稱（如 "馮玄明", "吳佳駿"）
- required_or_elective：必修 或 選修
- course_name_keyword：課程名稱關鍵字（如 "深度學習", "資料結構"）
- topic：主題關鍵字（如 "AI", "程式設計", "網路", 或是星期幾如 "星期一", "禮拜二"）

## 絕對禁止事項 (防幻覺)
1. 絕對不可自行猜測或編造答案！
2. 如果使用者的問題是在「詢問」某個資訊（例如：「誰教的？」、「是必修嗎？」），代表他不知道答案，此時對應的 filter（teacher, required_or_elective）必須留空！
3. 只能從問題中擷取「明確出現」的詞彙作為條件。

## 使用者問題
{question}

## 輸出格式（JSON）
{{
    "query_type": "類型",
    "metadata_filters": {{
        "欄位名": "值"
    }},
    "confidence": 0.0到1.0
}}

僅回傳 JSON，不要加任何其他文字。"""


# =============================================================================
# 🔀 合併式 Router+Rewrite Prompt（單次 LLM 呼叫）
# =============================================================================

COMBINED_ROUTER_REWRITE_PROMPT = """你是國立金門大學的校園課程助理問題分析器。
請同時完成以下兩個任務：

## 👤 使用者隱藏身分資料 (自動注入)
{user_profile_str}
當使用者在問題中使用代名詞（如：「我」、「我們系」、「我的」、「大幾」）或沒有明確指名科系/年級時，請**主動**將這些身分資訊填入 `metadata_filters` 的 `dept_short` 或 `grade` 中！
⚠️ 【絕對優先規則】：如果使用者在問題中**明確指定了其他的系所或年級**（例如：「企管系大一的課」、「財金系」），請**絕對優先**使用使用者句子中的條件，不可被隱藏身分覆蓋！

## 任務 1：問題分類與閒聊判定
判斷問題類型並提取 metadata 過濾條件。

### 問題類型
- course_info：課程基本資訊（教師、學分、上課時間、教室等）
- syllabus：教學內容、教學目標、教學綱要
- textbook：教科書、參考書、教材
- grading：成績評定、考試方式、課堂要求
- schedule：教學進度、每週課程安排
- calendar_action：任何涉及行事曆/日曆的操作！包含：加到行事曆、排進日曆、提醒我、刪除行事曆、列出行程、修改行程。只要提到「行事曆」「日曆」「提醒」「行程」就是 calendar_action！例：「3月15號是我生日加到行事曆」「刪除明天九點的」「我有什麼行程」
- academic_calendar：詢問學校行事曆上的日期（開學、期中考、停修申請、放假等）
- chitchat：純粹打招呼/閒聊
- general：其他問題

### 閒聊判定紅線
- "is_chitchat" 只有在使用者「純粹」打招呼、感謝，且【完全沒有】提到任何課程、老師、時間、規定等詢問時，才設為 true。
- ⚠️ 若使用者輸入包含「打招呼 + 課程問題」（例如：「你好，請問大二有什麼課」），is_chitchat 必須強制設為 false！
- ⚠️⚠️ 【絕對紅線】只要句子中出現「行事曆」「日曆」「加到」「提醒」「行程」「排進」任一關鍵字，is_chitchat 必須是 false，query_type 必須是 calendar_action！即使句子看起來像閒聊也一樣！

### Metadata 條件
從問題中提取：dept_short, grade, teacher, required_or_elective, course_name_keyword, topic
⚠️ 防幻覺：只能擷取問題中「明確出現」的詞彙。如果使用者在「詢問」某資訊（如「誰教的」），對應 filter 必須留空！

## 任務 2：搜尋查詢改寫
將使用者問題改寫為 {num_queries} 個不同角度的搜尋查詢，提高檢索覆蓋率。
⚠️ 最多只產生 {num_queries} 個查詢，不可超過！

## 對話歷史
{chat_history}

## 使用者問題
{question}

### 改寫規則
- 每個查詢都必須根據使用者的【實際問題】來改寫，不可照抄範例
- 從不同角度重新描述原始問題，例如換同義詞、換句型、加入相關關鍵字

### 範例（僅供理解改寫方式，不可照抄！）（僅供理解改寫方式，不可照抄！）
若原始問題是「圖書館今天幾點開」，正確的輸出應為：
{{
    "query_type": " general",
    "metadata_filters": {{}},
    "search_queries": ["圖書館開放時間", "今日圖書館營業時間", "圖書館幾點開門"],
    "is_chitchat": false
}}
（僅供理解改寫方式，不可照抄！）（僅供理解改寫方式，不可照抄！）

## 輸出格式（JSON）
{{
    "query_type": "問題類型",
    "metadata_filters": {{}},
    "search_queries": ["改寫查詢1", "改寫查詢2", "改寫查詢3"],
    "is_chitchat": false
}}

僅回傳 JSON。你的 search_queries 必須是根據使用者問題改寫的真實查詢。"""


# JSON Schema 強制鎖定（Ollama Structured Outputs）
COMBINED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string"},
        "metadata_filters": {"type": "object"},
        "search_queries": {
            "type": "array",
            "items": {"type": "string"}
        },
        "is_chitchat": {"type": "boolean"}
    },
    "required": ["query_type", "metadata_filters", "search_queries", "is_chitchat"]
}


# =============================================================================
# 🔀 Query Router 主函式
# =============================================================================

def route_query(question: str) -> RouteResult:
    """
    判斷使用者問題的類型並提取 metadata 過濾條件。

    結合 LLM 路由和規則路由兩種方法：
    - 先用規則快速匹配明確的 metadata
    - 再用 LLM 判斷問題類型和補充 metadata

    Args:
        question: 使用者問題

    Returns:
        RouteResult 包含問題類型和 metadata 過濾條件
    """
    # ── Step 1: 規則式 metadata 提取（快速且可靠） ──
    rule_filters = _extract_metadata_by_rules(question)
    rule_type = _detect_type_by_rules(question)

    # ── Step 2: LLM 路由（更準確的類型判斷） ──
    try:
        llm_result = _route_by_llm(question)

        # 合併：LLM 的提取（更聰明）優於簡單的規則，因此 LLM 蓋過 rule_filters
        merged_filters = {**rule_filters, **llm_result.metadata_filters}

        # 類型：如果規則有明確結果就用規則的，否則用 LLM 的
        final_type = rule_type if rule_type != "general" else llm_result.query_type

        result = RouteResult(
            query_type=final_type,
            metadata_filters=merged_filters,
            confidence=llm_result.confidence,
        )

    except Exception as e:
        logger.warning(f"LLM Router 失敗，使用規則路由：{e}")
        result = RouteResult(
            query_type=rule_type,
            metadata_filters=rule_filters,
            confidence=0.5,
        )

    logger.info(f"🔀 Query Router 結果：{result}")
    return result


def _extract_metadata_by_rules(question: str) -> dict:
    """
    使用規則從問題中提取 metadata 過濾條件。

    對每個 query 都檢查：department, grade, course_type, teacher, course_name, topic。
    """
    filters = {}

    import re
    
    # ── 學年度與學期檢查 (如 "114學年度第1學期", "114上", "114級上", "114-1") ──
    year_sem_match = re.search(r"(\d{3})\s*(?:年|學年度|級|級的)?\s*(?:第([一二12])學期|([上下])學期?|-([12])|的?([上下]))", question)
    if year_sem_match:
        filters["academic_year"] = year_sem_match.group(1)
        sem_str = year_sem_match.group(2) or year_sem_match.group(3) or year_sem_match.group(4) or year_sem_match.group(5)
        if sem_str:
            mapping = {"一": "1", "二": "2", "上": "1", "下": "2", "1": "1", "2": "2"}
            filters["semester"] = mapping.get(sem_str, sem_str)
    else:
        # 單獨檢查學期 (如 "大一上", "二下", "上學期", "第1學期")
        sem_match = re.search(r"(?:大[一二三四]|[一二三四]|碩[一二])([上下])|第([一二12])學期|([上下])學期", question)
        if sem_match:
            sem_str = sem_match.group(1) or sem_match.group(2) or sem_match.group(3)
            mapping = {"一": "1", "二": "2", "上": "1", "下": "2", "1": "1", "2": "2"}
            filters["semester"] = mapping.get(sem_str, sem_str)

    # ── 星期 / 上課時間檢查 (口語翻譯) ──
    day_patterns = {
        r"禮拜一|星期一|週一|周一": "星期一",
        r"禮拜二|星期二|週二|周二": "星期二",
        r"禮拜三|星期三|週三|周三": "星期三",
        r"禮拜四|星期四|週四|周四": "星期四",
        r"禮拜五|星期五|週五|周五": "星期五",
        r"禮拜六|星期六|週六|周六": "星期六",
        r"禮拜日|星期日|週日|周日|禮拜天|星期天": "星期日",
    }
    for pattern, d in day_patterns.items():
        if re.search(pattern, question):
            filters["day_of_week"] = d
            break

    # ── 系所檢查（支援全校 18 系所，簡稱 + 完整名稱 + 口語）──
    _DEPT_PATTERNS = {
        # 理工學院 (先長後短，避免「工」誤匹配)
        "資訊工程": "資工系", "資工": "資工系",
        "電機工程": "電機系", "電機": "電機系",
        "土木與工程管理": "土木系", "土木工程": "土木系", "土木": "土木系",
        "食品科學": "食品系", "食品": "食品系",
        # 管理學院
        "企業管理": "企管系", "企管": "企管系",
        "觀光管理": "觀光系", "觀光": "觀光系",
        "運動與休閒": "運休系", "運休": "運休系",
        "工業工程與管理": "工管系", "工業工程": "工管系", "工管": "工管系",
        # 人文社會學院
        "國際暨大陸事務": "國際系", "大陸事務": "國際系", "國際": "國際系",
        "建築": "建築系",
        "海洋與邊境管理": "海邊系", "海洋與邊境": "海邊系", "海邊": "海邊系", "邊境管理": "海邊系",
        "應用英語": "應英系", "應英": "應英系",
        "華語文": "華語系", "華語": "華語系",
        "都市計畫與景觀": "都景系", "都市計畫": "都景系", "都景": "都景系",
        # 健康護理學院
        "護理": "護理系",
        "長期照護": "長照系", "長照": "長照系",
        "社會工作": "社工系", "社工": "社工系",
        # 通識
        "通識": "通識中心",
    }
    # 先檢查是否包含碩士關鍵字
    is_master = any(kw in question for kw in ["碩", "研究所"])
    # 用最長匹配優先（避免「工管」被「工」先匹配到）
    sorted_dept_keys = sorted(_DEPT_PATTERNS.keys(), key=len, reverse=True)
    for kw in sorted_dept_keys:
        if kw in question:
            dept = _DEPT_PATTERNS[kw]
            if is_master:
                filters["dept_short"] = dept.replace("系", "碩") if "系" in dept else dept + "碩"
            else:
                filters["dept_short"] = dept
            break

    # ── 年級檢查 (支援五年制、口語) ──
    grade_patterns = {
        r"大一|一年級": "一",
        r"大二|二年級": "二",
        r"大三|三年級": "三",
        r"大四|四年級": "四",
        r"大五|五年級": "五",
        r"碩一|研一|碩班一年級": "碩一",
        r"碩二|研二|碩班二年級": "碩二",
    }
    for pattern, g in grade_patterns.items():
        if re.search(pattern, question):
            filters["grade"] = g
            break
    
    # ── 甲乙班檢查 (如「電機一甲」「一甲」「甲班」) ──
    group_match = re.search(r"[一二三四五]([甲乙丙丁])|([甲乙丙丁])班", question)
    if group_match:
        filters["class_group"] = group_match.group(1) or group_match.group(2)
    
    # ── 進修部檢查 ──
    if "進修" in question or "夜間" in question:
        filters["is_evening"] = True

    # ── 節次／時段提取（如「第二到四堂」→ time_period: "2-4"）──
    def _cn_to_num(s):
        mapping = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"十一":11}
        if s.isdigit(): return int(s)
        return mapping.get(s, 0)

    period_match = re.search(r"第([一二三四五六七八九十\d]+)[到至~\-]第?([一二三四五六七八九十\d]+)[堂節課]", question)
    if period_match:
        p_start = _cn_to_num(period_match.group(1))
        p_end = _cn_to_num(period_match.group(2))
        if p_start and p_end:
            filters["time_period"] = f"{p_start}-{p_end}"

    # ── 必修 / 選修檢查（防幻覺：若使用者在「詢問」而非「篩選」，不填入 filter）──
    # 例如：「資料結構是選修嗎？」→ 不填（使用者在問）
    #       「電機大一有什麼必修」→ 填入（使用者在篩選）
    _req_elec_word = None
    if "必修" in question:
        _req_elec_word = "必修"
    elif "選修" in question:
        _req_elec_word = "選修"
    
    if _req_elec_word:
        # 檢查是否為疑問句：後接 嗎/？/呢 或 前接 是/是否/算
        word_pos = question.index(_req_elec_word)
        after = question[word_pos + len(_req_elec_word):word_pos + len(_req_elec_word) + 3]
        before = question[max(0, word_pos - 3):word_pos]
        is_asking = bool(re.search(r"[嗎？呢]", after)) or "是不是" in question or "還是" in question
        is_asking = is_asking or bool(re.search(r"(是|是否|算)$", before))
        
        if not is_asking:
            filters["required_or_elective"] = _req_elec_word
    elif any(kw in question for kw in ["推薦", "建議", "推介", "適合"]):
        # 推薦情境：必修本來就要修，不需要推薦，自動篩選選修
        filters["required_or_elective"] = "選修"

    # ── 教師名稱提取（動態名冊，自動去除「老師」「教授」後綴）──
    clean_q = re.sub(r"(老師|教授)", "", question)
    for teacher in sorted(_known_teachers, key=len, reverse=True):
        if teacher in clean_q:
            filters["teacher"] = teacher
            break

    # ── 課程名稱關鍵字（動態清單，從索引自動建構）──
    for kw in sorted(_known_courses, key=len, reverse=True):
        if kw in question:
            filters["course_name_keyword"] = kw
            break

    return filters


def _detect_type_by_rules(question: str) -> str:
    """使用規則判斷問題類型"""
    
    # 行事曆相關 (動作類 Intent，優先級最高)
    calendar_keywords = [
        "行事曆", "日曆", "提醒", "加到", "排進", "排到",  # 新增
        "行程", "刪掉", "取消", "移除",                    # 刪除/列出
    ]
    if any(kw in question for kw in calendar_keywords):
        return "calendar_action"

    # 學校行事曆一般查詢 (如停修、開學、校慶、春假)
    academic_keywords = [
        "停修", "開學", "校慶", "春假", "寒假", "暑假", "放假", "補假", "連假", 
        "加退選", "選課", "退選", "選填", "畢業", "宿舍", "註冊", "繳費",
        "掃墓", "清明", "兒童節", "和平紀念", "端午", "中秋", "國慶", "元旦", "跨年"
    ]
    if any(kw in question for kw in academic_keywords):
        return "academic_calendar"
        
    # 教材相關
    if any(kw in question for kw in ["教科書", "課本", "參考書", "教材", "用書"]):
        return "textbook"

    # 成績與考試相關 (但若有時間詞，需視為行事曆查詢)
    if any(kw in question for kw in ["成績", "考試", "評分", "配分", "佔比", "期中考", "期末考", "小考"]):
        if any(wk in question for wk in ["什麼時候", "時間", "幾號", "何時", "哪天", "當週", "那週"]):
            return "academic_calendar"
        return "grading"

    # 進度相關
    if any(kw in question for kw in ["進度", "第幾週", "哪一週", "每週", "課程安排"]):
        return "schedule"

    # 教學目標 / 綱要
    if any(kw in question for kw in ["教學目標", "教學綱要", "學什麼", "教什麼", "課程內容", "大綱"]):
        return "syllabus"

    # 基本資訊（誰教、幾學分、什麼時候上課等）
    if any(kw in question for kw in ["誰教", "老師", "教授", "學分", "上課時間", "教室", "必修", "選修", "人數"]):
        return "course_info"

    return "general"


def _route_by_llm(question: str) -> RouteResult:
    """使用 LLM 進行問題路由"""
    prompt = ROUTER_PROMPT.format(question=question)

    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": getattr(config, "OLLAMA_FAST_MODEL", config.OLLAMA_MODEL),
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": 0,  # [VRAM 死亡交叉防護] 3B 用完立即卸載，釋放 2GB 空間
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 256,
            },
        },
        timeout=config.OLLAMA_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result_text = response.json()["response"].strip()
    logger.debug(f"LLM Router 原始回應：{result_text[:300]}")

    # 解析 JSON
    match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            raw_filters = data.get("metadata_filters", {})
            
            # 【防幻覺防線】確保 LLM 提取的 filter 真的有出現在問題中
            verified_filters = {}
            for k, v in raw_filters.items():
                if isinstance(v, str) and v and v in question:
                    verified_filters[k] = v
                elif isinstance(v, list):
                    valid_list = [item for item in v if item in question]
                    if valid_list:
                        verified_filters[k] = valid_list
                        
            return RouteResult(
                query_type=data.get("query_type", "general"),
                metadata_filters=verified_filters,
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"LLM Router JSON 解析失敗：{e}")

    return RouteResult(query_type="general", confidence=0.3)


# =============================================================================
# 🔗 合併式 Route + Rewrite（單次 LLM 呼叫）
# =============================================================================

def route_and_rewrite(
    question: str,
    chat_history: list[dict] = None,
    num_queries: int = None,
    user_profile: dict = None
) -> tuple:
    """
    合併 Query Router + Query Rewrite 為單次 LLM 呼叫。
    
    省去兩次獨立的 3B 模型載入/卸載開銷，節約 1~3 秒。
    使用 Ollama Structured Outputs (JSON Schema) 保證 100% JSON 格式正確。
    
    Args:
        question: 使用者問題
        chat_history: 對話歷史
        num_queries: 生成的查詢數量
    
    Returns:
        tuple: (RouteResult, search_queries: list[str])
    """
    if num_queries is None:
        num_queries = config.MULTI_QUERY_COUNT
    
    # ── Step 1: 規則式 metadata 快速提取 ──
    rule_filters = _extract_metadata_by_rules(question)
    rule_type = _detect_type_by_rules(question)
    
    # 格式化對話歷史
    history_str = "無"
    # 暫時停用對話歷史，避免 Router 受到舊問題干擾產生錯亂改寫
    # if chat_history and len(chat_history) > 0:
    #     history_lines = []
    #     for msg in chat_history[-config.MEMORY_WINDOW_SIZE * 2:]:
    #         role = "使用者" if msg["role"] == "user" else "助理"
    #         history_lines.append(f"{role}：{msg['content'][:200]}")
    #     history_str = "\n".join(history_lines)
    
    # 格式化使用者身分
    profile_str = "使用者尚未註冊身分。請依問題字面判斷。"
    if user_profile:
        dept = user_profile.get("department", "未知")
        grade = user_profile.get("grade", "未知")
        profile_str = f"該名使用者是【{dept}】的【{grade}年級】學生。"
    
    # ── Step 2: 合併式 LLM 呼叫 (Router + Rewrite 一次搞定) ──
    try:
        # 【企業級優化 4：課表語意/口語時間對齊】
        # 將學生的口語時間，隱式擴充為課表的標準節次，讓 BM25 和 LLM Rewrite 能精準命中
        time_mapper = {
            "早上": "第1節 第2節 第3節 第4節 上午",
            "上午": "第1節 第2節 第3節 第4節",
            "下午": "第5節 第6節 第7節 第8節 第9節",
            "晚上": "第10節 第11節 第12節 夜間",
        }
        expanded_question = question
        for spoken, formal in time_mapper.items():
            if spoken in question:
                expanded_question += f" ({formal})"
                break

        prompt = COMBINED_ROUTER_REWRITE_PROMPT.format(
            num_queries=num_queries,
            chat_history=history_str,
            question=expanded_question,
            user_profile_str=profile_str
        )
        
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": getattr(config, "OLLAMA_FAST_MODEL", config.OLLAMA_MODEL),
                "prompt": prompt,
                "format": COMBINED_OUTPUT_SCHEMA,  # JSON Schema 強制格式
                "stream": False,
                "keep_alive": 0,  # [VRAM 死亡交叉防護] 3B 用完立即卸載
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 2048,
                    "num_predict": 300,  # 限縮：防止 3B 模型生成過多 queries 導致 JSON 截斷
                },
            },
            timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result_text = response.json()["response"].strip()
        logger.info(f"合併 Router+Rewrite 原始回應：{result_text[:300]}")
        
        # 解析 JSON
        data = json.loads(result_text)
        
        # 提取路由結果
        llm_filters = data.get("metadata_filters", {})
        
        # 【防幻覺】驗證 LLM filters
        # 但若使用者有註冊身分，其 dept_short 和 grade 是由 Profile 注入的，
        # 不需要在問題原文出現也可以信任（因為 Prompt 明確指示了 LLM 自動補全）
        profile_trusted_keys = set()
        if user_profile:
            if user_profile.get("department"):
                profile_trusted_keys.add("dept_short")
            if user_profile.get("grade"):
                profile_trusted_keys.add("grade")
        
        verified_filters = {}
        for k, v in llm_filters.items():
            if k in profile_trusted_keys:
                # 由 Profile 注入的欄位，信任 LLM 的輸出
                if isinstance(v, str) and v:
                    verified_filters[k] = v
            elif isinstance(v, str) and v and v in question:
                verified_filters[k] = v
            elif isinstance(v, list):
                valid_list = [item for item in v if item in question]
                if valid_list:
                    verified_filters[k] = valid_list
        
        # 【正規化】LLM 可能回傳「三年級」但 metadata 只存「三」，統一格式
        if "grade" in verified_filters:
            g = verified_filters["grade"]
            # 去除多餘的「年級」「年」後綴
            g = g.replace("年級", "").replace("年", "").strip()
            # 也處理「大一」→「一」的口語
            g = g.replace("大", "")
            if g:
                verified_filters["grade"] = g
        
        # 合併 LLM + 規則 filters（規則較精準，優先覆蓋 LLM 的結果）
        merged_filters = {**verified_filters, **rule_filters}
        
        # 類型決定：若 LLM 判斷出 calendar 系列，則信任 LLM (比單純規則防呆聰明)
        llm_type = data.get("query_type", "general")
        if llm_type in ["academic_calendar", "calendar_action"]:
            final_type = llm_type
        else:
            final_type = rule_type if rule_type != "general" else llm_type
        
        # 閒聊判定 — 但若規則已判斷為行事曆操作，不允許 is_chitchat 覆蓋
        is_chitchat = data.get("is_chitchat", False)
        if is_chitchat and final_type not in ("calendar_action", "academic_calendar"):
            final_type = "chitchat"
        
        route_result = RouteResult(
            query_type=final_type,
            metadata_filters=merged_filters,
            confidence=0.8,
        )
        
        # 提取搜尋查詢
        search_queries = data.get("search_queries", [question])
        if not search_queries:
            search_queries = [question]
        # 確保原始問題在列表中
        if question not in search_queries:
            search_queries.insert(0, question)
        
        logger.info(f"🔀 合併 Router+Rewrite 結果：{route_result}")
        logger.info(f"   查詢：{search_queries}")
        return route_result, search_queries
        
    except Exception as e:
        logger.warning(f"合併 Router+Rewrite LLM 呼叫失敗，使用規則 fallback：{e}")
        
        route_result = RouteResult(
            query_type=rule_type,
            metadata_filters=rule_filters,
            confidence=0.5,
        )
        return route_result, [question]


# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    config.setup_logging()

    test_questions = [
        "深度學習是誰教的？",
        "資工系二年級有哪些必修課？",
        "馮玄明老師教哪些課？",
        "資料結構的教科書是什麼？",
        "機器學習的成績怎麼算？",
        "演算法這門課第 5 週教什麼？",
    ]

    for q in test_questions:
        print(f"\n❓ {q}")
        result = route_query(q)
        print(f"   {result}")
