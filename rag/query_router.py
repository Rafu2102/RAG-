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
    search_queries: list[str] = field(default_factory=list)  # Step-Back 擴充的搜尋問題
    is_chitchat: bool = False    # 是否為閒聊/純生活指引
    is_career_planning: bool = False # 是否為職涯大範圍探索問題

    def __repr__(self):
        filters_str = ", ".join(f"{k}={v}" for k, v in self.metadata_filters.items())
        return f"RouteResult(type={self.query_type}, filters=[{filters_str}], conf={self.confidence:.2f})"


# =============================================================================
# 🔀 合併式 Router+Rewrite Prompt（單次 LLM 呼叫）
# =============================================================================

COMBINED_ROUTER_REWRITE_PROMPT = """你是國立金門大學的校園資料庫檢索策略家。
你現在的引擎是極速且聰明的 Gemini 3.1 Flash Lite。請展現你的極致邏輯能力，分析問題並擬定最佳「檢索策略」。
你必須嚴格輸出符合 JSON 格式的內容，並在給出最終答案前，經過完整的自我詰問（Self-Correction CoT）。

## 👤 使用者身分 (系統自動注入，僅供參考)
{user_profile_str}

## 任務 1：意圖分類 (query_type)
判斷問題屬於哪種領域（單選，取最主要意圖）：
- course_info：【核心課程檢索】提供所有校內開設之課程細節。
  1. 「基礎資訊」：時間、地點、學分、必修選修、前置擋修科目、課程大綱。
  2. 「評價與建議」：給分甜度評估、作業與考試硬度比較、學長姐真實評價。
  3. 「空堂推薦與規劃」：針對特定閒置時段尋找適合的通識或選修（例如：「星期二下午推薦什麼課？」）。
  ⚠️ 排除機制一：若詢問國內外「研討會」、「高峰會」、「黑客松」，這並非實體學期課程，【必須交由】web_search 處理！
  ⚠️ 排除機制二：凡是詢問「推薦課程」、「哪堂課好過」，【絕對不可】歸類為 personal_schedule，必須在此進行全校課程搜索。
- professor_info：【教職員檔案檢索】涵蓋全校老師的學術與行政履歷。
  1. 「學術背景」：研究專長、最新發表論文、最高學歷。
  2. 「通訊與地標」：電子郵件(Email)、分機號碼、研究室確切實體位置。
  3. 「實驗室與指導」：研究生指導規範、專題生招募條件、實驗室風氣。
  ⚠️ 防呆提示：如果在詢問某門課的評分同時提到老師（如：「趙于翔的資料庫甜嗎？」），優先判定為 course_info，因為核心是問課程評價。
- policy_rules：【法規與教務制度】牽涉學校行政與學生權益的規章條文。
  1. 「畢業與學分」：資工系/全校畢業門檻、跨領域學程、雙主修條件。
  2. 「流程規範」：請假手續、加退選時程定義、停修規則、各類獎學金申請。
  ⚠️ 注意：若詢問「開學是哪一天」、「何時期中考」等具體【日期點】，請歸類至 academic_calendar。
  ⚠️ 注意：若詢問通用的「如何準備備審資料」、「如何寫履歷」等無關校內法規的廣泛技巧，必須放行至 web_search！
- calendar_action：【個人行事曆控制台】觸發 Google Calendar 實體時程操作。
  1. 「新增排程」：加入作業死線、建立開會提醒（例如：「把明天晚上的黑客松加進去」）。
  2. 「修改與刪除」：取消已存在的行程、變更時間地點。
  3. 「查詢大範圍行程」：搜索【不限於課表】的極度長期或過去的活動（例如：「未來三個月有什麼安排？」、「我去年五月去了哪裡？」）。
  ⚠️ 只要明確包含「行事曆」、「日曆」、「加行程」等命令，強制攔截至此意圖！
- academic_calendar：【校園通用節慶時刻表】查詢學校官方公布的行政時段。
  1. 「學期大節點」：開學日、註冊截止日、期中/期末考週、期末成績上傳日。
  2. 「國定校定假日」：春假、清明連假、校慶補假。
  ⚠️ 僅限「全校共同遵守」的時刻表，不包含學生自己的私人出遊。
- personal_schedule：【每週固定課表快查】限於學生本學期已被系統記錄的【規律性】每週課程。
  1. 「單日速查」：「我今天滿堂嗎？」、「禮拜三早上有什麼課？」
  2. 「下節定位」：「我等一下要去哪裡上課？」、「下一堂課的教室在哪？」
  ⚠️ 致命毒藥：本指令**絕對不支援**查詢「下個月」、「未來半年」等非規律性或大跨度行程，長期行程查詢必須導向 calendar_action！
  ⚠️ 【混合意圖核彈】：若使用者問「我星期X有空堂嗎，可以推薦課程嗎？」，這種「查課表 + 推薦」的複合問題，【必須歸類為 course_info】，因為推薦課程需要搜尋全校課程資料庫！個人課表會自動注入 LLM Prompt。
- personal_transcript：【成績與學分儀表板】限於學生已授權的私密教務成績檔。
  1. 「學分數結算」：「我目前修過多少學分？」、「我通識滿了嗎？」
  2. 「歷年成績」：「我大一微積分拿幾分？」、「我平均 GPA 多少？」
- chitchat：【情感樞紐與無效查詢】純粹打招呼（「早安」、「你好」）、無實質檢索標的之抱怨或誇獎（「這系統太棒了」、「我好累想休學」）。完全缺乏資訊價值時進入此冷卻區。
- web_search：【新增】明顯超出「金門大學」校園本地資訊的廣泛知識探索，或者使用者明確下達「聯網指令」時。這包含但不限於：
  1. 「強制聯網指令」：當問題中包含「網路查詢」、「上網搜尋」、「幫我 google」、「到網路查」等明確指令時，無論問題內容是否與校內有關，強制歸類為此項！
  2. 「全球與在地即時資訊」：天氣預報、新聞時事、科技產業動態（例如：「金門明天下雨嗎？」、「Nvidia 最新財報表現如何？」）。
  3. 「外部實體知識與名詞解釋」：專有名詞、歷史、科普（例如：「什麼是 Agentic RAG？」、「解釋量子力學」）。
  4. 「學術論文與專業期刊」：搜尋並講解國際期刊文獻、最新研究技術摘要與突破。
  5. 「大型研討會與學術會議」：若詢問國內外特定的學術研討會、競賽、非例行性大型講座（例如：「數位生活科技研討會介紹」、「全國大專IT競賽規則」），這些通常非本地常規修課課程，應透過網路爬取最新詳細。
  6. 「廣泛職涯與通識文件準備」：例如「如何準備備審資料」、「如何寫履歷」、「軟體工程師的面試準備」、「大學生怎麼找實習」。這類沒有針對特定一門課的純技巧/網路資源分享。
  7. 「生活娛樂與在地推薦」：美食餐廳推薦、旅遊景點規劃、影視文化（例如：「金城鎮有什麼必吃的牛肉麵？」、「最近有什麼好看的電影？」）。
  ⚠️ 除非使用者有下達如上述的「強制聯網指令」或明顯屬於「外部/大型非例行活動/廣泛通用的文件準備」，否則一般詢問「校內教授/一般學期課程/成績/個人行事曆」都不可落入此分類。

## 任務 2：退一步擴充查詢 (search_queries)
將問題轉化為 {num_queries} 個適合向量檢索的短句：
- 必須包含原始問題與「退一步 (Step-Back)」的泛化查詢。
- ⚠️ 絕對不要盲目加上「課程」、「時間」等字眼，除非使用者有問！這會污染搜尋結果！
- ⚠️ 核心指示：若當前問題含有代名詞(如他、這門課)，請參考上方『對話歷史』進行指代消解 (Coreference Resolution)，並將完整實體名稱用於擴充問題中。若問題是全新主題，請忽略對話歷史的干擾。
- ⚠️ 技術展開：遇到「前端/後端/AI/網管」等職涯規劃時，務必在擴充問題中將這些術語展開為具體的「技術堆疊」(例如：前端展開為 HTML/CSS/網頁設計，後端展開為 伺服器/資料庫)。
- ⚠️ 處理暱稱：如果使用者提到「cj」，請認知這代表「李錫捷」老師，並在擴充查詢中使用全名。

💡 退一步擴充範例：
- 「加入柯志亨實驗室條件」→ 擴充：「柯志亨 研究方向」、「柯志亨 實驗室」、「資工系 專題製作」
- 「畢業要修幾學分」→ 擴充：「畢業學分 門檻」、「資工系 修課規定」、「最低畢業學分」
- 「哪門課比較甜？」→ 擴充：「給分甜 課程」、「容易拿高分 推薦」、「哪門課比較好過」
- 「我想走前端後端要修什麼」→ 擴充：「網頁設計 HTML CSS JavaScript」、「後端伺服器 資料庫」、「軟體工程 專題」

## 任務 3：明確過濾條件與職涯判定 (Filters & Flags)
支援的鍵值：teacher, dept_short, grade, required_or_elective, course_name_keyword, semester, academic_year, ge_domain
⚠️⚠️ 預設封閉檢索圈與例外放行 ⚠️⚠️
1. 【強制預設】：除非有特例，你必須將使用者的科系、年級強制填入 `dept_short` 與 `grade`！並預設鎖定本學期 (`academic_year`: "{default_year}", `semester`: "{default_sem}")。
2. 【實體豁免】：若使用者明確指名「老師姓名」或「特定教室(如 E321)」，【必須清空】`dept_short` 與 `grade`，確保搜出全校範圍的老師與教室！
3. 【課名搜尋豁免年級】：若使用者提到「特定課程名稱」(如「微積分」、「資料結構」)，你【必須清空 `grade`】！因為使用者是問「這門課有沒有開」，而不是「我這個年級有什麼課」。指定課名時，保留 `dept_short` 與 `semester`，但【絕對不填 `grade`】！
4. 【跨域學習與職涯探索】：若使用者在問「我想學程式」、「未來想走前端」、「怎麼準備就業」，這屬於職涯與跨域探索。你【必須】將 `is_career_planning` 設為 true，並且【絕對清空】`academic_year`、`semester`。
   - ⚡ 跨域特判：如果使用者的科系（如企管系）明顯與想學的技能（如寫程式）無關，你【必須一併清空】`dept_short`，讓系統去全校（或資工系）撈課！
5. 【通識課程】：詢問通識時，`dept_short` 設為 `通識中心`，清空 `grade`。可設定 `ge_domain`（人文/社會/自然）。
6. 【防止幻覺】：未明確指定的欄位，請直接在 JSON 中設為 `null`（不要加雙引號！）或直接省略！嚴禁腦補發明內容。

## 對話歷史
{chat_history}

## 使用者問題
{question}

## 💡 【完美輸出範例】（雙重 Few-Shot 陷阱題）

👉 範例一：實體豁免陷阱
使用者問題：「幫我查那個教 Linux 的老師的 Gmail」
輸出：
{{
  "reasoning": {{
    "1_intent_analysis": "使用者想找老師聯絡方式，屬於 professor_info。",
    "2_condition_check": "只提到『Linux』，沒有明確老師姓名。沒有指定科系年級。",
    "3_cross_domain_and_time": "找特定專長老師，不需要限制學期與科系。",
    "4_hallucination_warning": "絕對不能把 'Linux' 填入 teacher 欄位，這會導致檢索失敗！",
    "5_step_back_logic": "我應該用 'Linux 老師'、'Linux 教授 聯絡方式' 作為搜尋字串。"
  }},
  "query_type": "professor_info",
  "metadata_filters": {{}},
  "search_queries": ["幫我查那個教 Linux 的老師的 Gmail", "Linux 教授 聯絡方式", "Linux 師資"],
  "is_chitchat": false,
  "is_career_planning": false
}}

👉 範例二：跨域學習與職涯探索陷阱
使用者身分：企管系 二年級
使用者問題：「我以後想當軟體工程師，有推薦修什麼課嗎？」
輸出：
{{
  "reasoning": {{
    "1_intent_analysis": "詢問修課建議與未來方向，屬於 course_info。",
    "2_condition_check": "無明確指定老師或特定課程名稱。",
    "3_cross_domain_and_time": "這是職涯探索！企管系想當軟體工程師屬於『跨域學習』。必須清空學期、科系、年級限制，去全校撈課。",
    "4_hallucination_warning": "過濾器必須保持乾淨，不可填入企管系，否則找不到工程師課程。",
    "5_step_back_logic": "展開『軟體工程師』技能樹，加入『程式設計』、『資料結構』等關鍵字。"
  }},
  "query_type": "course_info",
  "metadata_filters": {{}},
  "search_queries": ["我以後想當軟體工程師，有推薦修什麼課嗎？", "軟體工程師 推薦課程", "程式設計 基礎", "資料結構 演算法"],
  "is_chitchat": false,
  "is_career_planning": true
}}

請遵循上述邏輯，給我 JSON 輸出："""


# JSON Schema 強制鎖定（Gemini Structured Outputs）
COMBINED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "object",
            "properties": {
                "1_intent_analysis": {"type": "string"},
                "2_condition_check": {"type": "string"},
                "3_cross_domain_and_time": {"type": "string"},
                "4_hallucination_warning": {"type": "string"},
                "5_step_back_logic": {"type": "string"}
            },
            "required": ["1_intent_analysis", "2_condition_check", "3_cross_domain_and_time", "4_hallucination_warning", "5_step_back_logic"]
        },
        "query_type": {
            "type": "string",
            "enum": ["course_info", "professor_info", "policy_rules", "calendar_action", "academic_calendar", "personal_schedule", "personal_transcript", "chitchat", "web_search"]
        },
        "metadata_filters": {
            "type": "object",
            "properties": {
                "teacher": {"type": "string", "nullable": True},
                "dept_short": {"type": "string", "nullable": True},
                "grade": {"type": "string", "nullable": True},
                "required_or_elective": {"type": "string", "nullable": True},
                "course_name_keyword": {"type": "string", "nullable": True},
                "semester": {"type": "string", "nullable": True},
                "academic_year": {"type": "string", "nullable": True},
                "ge_domain": {"type": "string", "nullable": True}
            }
        },
        "search_queries": {
            "type": "array",
            "items": {"type": "string"}
        },
        "is_chitchat": {"type": "boolean"},
        "is_career_planning": {"type": "boolean"}
    },
    "required": ["reasoning", "query_type", "metadata_filters", "search_queries", "is_chitchat", "is_career_planning"]
}


def _extract_metadata_by_rules(question: str) -> dict:
    """
    使用規則從問題中提取 metadata 過濾條件。

    對每個 query 都檢查：department, grade, course_type, teacher, course_name, topic。
    """
    filters = {}
    
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
            
    # 如果上面沒配對到，檢查是否有「系所簡稱+年級」的說法（如：資工三、電機二）
    if "grade" not in filters and "dept_short" in filters:
        short_name = filters["dept_short"].replace("系", "").replace("碩", "")
        for g in ["一", "二", "三", "四", "五"]:
            if f"{short_name}{g}" in question:
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
        return int(mapping.get(s, 0))

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
        # 檢查是否為詢問「屬性」的疑問句（如「資料結構是必修嗎？」、「算選修嗎」）
        # 而不是篩選條件（如「大二有什麼必修課？」）
        # 避免被句末單純的問號「？」誤殺
        is_asking = bool(re.search(r"(是|是否|算|為).{0,3}?(必修|選修).{0,2}?[嗎呢]", question))
        is_asking = is_asking or bool(re.search(r"(必修|選修)嗎", question))
        
        if not is_asking:
            filters["required_or_elective"] = _req_elec_word
    elif any(kw in question for kw in ["推薦", "建議", "推介", "適合"]):
        # 推薦情境：必修本來就要修，不需要推薦，自動篩選選修
        filters["required_or_elective"] = "選修"

    # ── 通識課程偵測（自動設定 dept_short 和 ge_domain）──
    if "通識" in question:
        filters["dept_short"] = "通識中心"
        # 通識不分年級，移除 grade 限制
        filters.pop("grade", None)
        # 偵測通識領域
        if "人文" in question:
            filters["ge_domain"] = "人文"
        elif "社會" in question:
            filters["ge_domain"] = "社會"
        elif "自然" in question:
            filters["ge_domain"] = "自然"

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

    # ── 教室 / 地點提取（只匹配建築物代碼如 E321, I102, E405）──
    # 真實教室格式：E321電腦網路實驗室, E320多媒體實驗室, I102圖資電腦教室, E405教室(理工大樓)
    classroom_match = re.search(
        r"([A-Z]\d{3}(?:-\d)?)",   # 只匹配建築代碼：E321, I103-1, E405 等
        question
    )
    if classroom_match:
        filters["classroom"] = classroom_match.group(1)

    return filters


def _detect_type_by_rules(question: str) -> str:
    """使用規則判斷問題類型"""

    # ── 個人查詢偵測（最高優先：有主詞「我」就路由到個人資料）──
    # 防呆設計：避免「給我」的「我」誤判。需要有足夠的個人情境。
    has_personal_pronoun = False
    personal_pronouns = ["我自己的", "自己的", "我的", "幫我查", "我有", "我要", "我禮拜", "我星期", "我今天", "我明天", "我上學期", "我下學期"]
    has_personal_pronoun = any(p in question for p in personal_pronouns)
    if "我" in question and not any(kw in question for kw in ["給我", "為我", "幫我", "替我", "讓我"]): 
        has_personal_pronoun = True

    if has_personal_pronoun:
        # 個人成績/學分/畢業相關
        transcript_keywords = [
            "學分", "畢業", "還差", "修了", "修過", "沒修", "必修", "選修", "通識",
            "GPA", "平均", "成績", "不及格", "沒過", "停修", "擋修", "體育", "國文", "英文",
            "修課", "進度",
        ]
        if any(kw in question for kw in transcript_keywords):
            return "personal_transcript"

        # 個人課表相關（禮拜X有什麼課、空堂、今天的課、我的課表）
        schedule_keywords = [
            "課表", "有什麼課", "有課", "沒課", "空堂", "幾節課", "上什麼",
            "禮拜", "星期", "週", "周", "今天", "明天", "下午", "上午",
        ]
        if any(kw in question for kw in schedule_keywords):
            # 🆕 混合意圖安全閥：若同時包含推薦類關鍵字，強制走 course_info
            recommend_keywords = ["推薦", "建議", "有什麼可以上", "選什麼", "修什麼", "可以上什麼"]
            if any(kw in question for kw in recommend_keywords):
                return "course_info"
            return "personal_schedule"
    
    # ── 教授 / 實驗室 / 聯絡方式相關（優先級高於行事曆，避免「加進實驗室」被誤判）──
    professor_keywords = [
        "實驗室", "研究方向", "研究", "專長", "gmail", "email", "信箱", "聯絡方式",
        "指導", "收學生", "找老師",
    ]
    if any(kw in question.lower() for kw in professor_keywords):
        return "professor_info"

    # 行事曆相關 (動作類 Intent)
    calendar_keywords = [
        # 新增/加入
        "行事曆", "日曆", "提醒", "加到", "排進", "排到", "記到", "加進", "新增到",
        # 刪除
        "刪掉", "取消", "移除",
        # 查詢/列出
        "行程", "有什麼事", "有什麼安排",
        # 修改/更新（關鍵遺漏！）
        "改到", "改成", "更改", "改時間", "改日期", "搬到", "移到",
        "延後", "延期", "提前", "推遲", "調到", "調整時間",
        "換到", "換成", "變更", "挪到",
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

    # ── 規章 / 畢業門檻相關（無「我」主詞時才路由到 policy）──
    policy_keywords = [
        "畢業門檻", "畢業規定", "修課規定", "請假", "缺課", "系規", "規章",
    ]
    if any(kw in question for kw in policy_keywords):
        return "policy_rules"

    # ── 課程相關（合併原本的 textbook/grading/schedule/syllabus/course_info）──
    course_keywords = [
        "教科書", "課本", "參考書", "教材", "用書",
        "成績", "考試", "評分", "配分", "佔比", "期中考", "期末考", "小考",
        "進度", "第幾週", "哪一週", "每週", "課程安排",
        "教學目標", "教學綱要", "學什麼", "教什麼", "課程內容", "大綱",
        "誰教", "老師", "教授", "學分", "上課時間", "教室", "必修", "選修", "人數",
        "可以修", "能修", "能不能修", "要修", "想修", "衝堂", "擋修", "先修",
    ]
    if any(kw in question for kw in course_keywords):
        return "course_info"

    return "general"



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
    # 啟用對話歷史，協助 Router 進行指代消解
    if chat_history and len(chat_history) > 0:
        history_lines = []
        for msg in chat_history[-config.MEMORY_WINDOW_SIZE * 2:]:
            role = "使用者" if msg["role"] == "user" else "助理"
            history_lines.append(f"{role}：{msg['content'][:200]}")
        history_str = "\n".join(history_lines)
    
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
        expanded_question = str(question)
        for spoken, formal in time_mapper.items():
            if spoken in question:
                expanded_question = expanded_question + f" ({formal})" # type: ignore
                break

        prompt = COMBINED_ROUTER_REWRITE_PROMPT.format(
            num_queries=num_queries,
            chat_history=history_str,
            question=expanded_question,
            user_profile_str=profile_str,
            default_year=str(config.CURRENT_ACADEMIC_YEAR),
            default_sem=str(config.CURRENT_SEMESTER)
        )
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 1.0,
                "response_mime_type": "application/json",
                "response_schema": COMBINED_OUTPUT_SCHEMA,
                "maxOutputTokens": config.GEMINI_FLASH_MAX_TOKENS,
                "thinkingConfig": {
                    "thinkingLevel": "medium"
                }
            }
        }
        
        response = requests.post(
            config.GEMINI_FAST_API_URL,
            json=payload,
            timeout=config.GEMINI_FLASH_TIMEOUT,
        )
        response.raise_for_status()
        
        try:
            result_text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        except (KeyError, IndexError):
            result_text = "{}"
            logger.error(f"❌ Gemini JSON 解析失敗: {response.text}")
            
        logger.info(f"合併 Router+Rewrite 原始回應：{result_text[:4096]}")
        
        # ── 1. 強健 JSON 解析（防禦 LLM 偶發的 Markdown 包裝與多嘴）──
        try:
            # 尋找第一個 '{' 到最後一個 '}' 之間的所有內容
            match = re.search(r"\{[\s\S]*\}", result_text)
            if match:
                clean_json = match.group(0)
                data = json.loads(clean_json)
            else:
                raise ValueError("找不到 JSON 大括號")
        except Exception as e:
            logger.error(f"Router JSON 解析失敗：{e} | 原始：{result_text[:200]}")
            data = {}

        # ── 2. P1 意圖分類：LLM intent + 規則引擎合併決策 ──
        llm_type = data.get("query_type", "general")
        # 全面信賴 Gemini 3.1 判斷的意圖，只有在 LLM 判斷為 general 時才退回規則引擎
        if llm_type and llm_type != "general":
            final_type = llm_type
        else:
            final_type = rule_type if rule_type != "general" else "general"

        # 閒聊判定（行事曆不允許被覆蓋）
        if data.get("is_chitchat", False) and final_type not in ("calendar_action", "academic_calendar"):
            final_type = "chitchat"

        # ── 3. 明確過濾條件：只信任使用者「明確說出」的條件 ──
        llm_filters = data.get("metadata_filters", {})
        if not isinstance(llm_filters, dict):
            llm_filters = {}
        
        # 【關鍵修正】EXPLICIT_NULL 信號機制：
        # LLM 刻意回傳 null 的欄位代表「不要限制」，我們必須保留這個意圖，
        # 否則後面的 Python 保底邏輯會自作主張把預設值塞回去。
        _TEMPORAL_KEYS = {"semester", "academic_year", "grade"}
        _EXPLICIT_NULL = "__EXPLICIT_NULL__"
        for k in _TEMPORAL_KEYS:
            if k in llm_filters and llm_filters[k] is None:
                llm_filters[k] = _EXPLICIT_NULL
        
        # 清理空值（LLM 有時輸出 "grade": "" 等空字串），但保留 EXPLICIT_NULL
        llm_filters = {k: v for k, v in llm_filters.items() if v}

        # 正規化 grade 格式：「三年級」→「三」、「大三」→「三」、「3」→「三」
        _GRADE_TO_CN = {"1": "一", "2": "二", "3": "三", "4": "四", "5": "五"}
        if "grade" in llm_filters and llm_filters["grade"] != _EXPLICIT_NULL:
            g = llm_filters["grade"]
            if isinstance(g, str):
                g = g.replace("年級", "").replace("年", "").replace("大", "").strip()
                g = _GRADE_TO_CN.get(g, g)  # 阿拉伯數字 → 中文
                if g:
                    llm_filters["grade"] = g
                else:
                    del llm_filters["grade"]

        # 正規化 teacher：去除「老師」「教授」後綴，並過濾掉明顯的 LLM 幻覺（把教室當老師）
        if "teacher" in llm_filters:
            t = llm_filters["teacher"]
            if isinstance(t, str):
                t = re.sub(r"(老師|教授)$", "", t).strip()
                
                # 防呆：台灣老師名字不可能包含英文字母或數字。如果包含，通常是 LLM 把教室代碼（如 E320）誤認為老師了！
                if re.search(r'[A-Za-z0-9]', t):
                    logger.warning(f"  🛑 攔截到 LLM 幻覺：將包含英數字的字串 '{t}' 誤認為老師，已自動移除。")
                    t = ""
                    
                if t:
                    llm_filters["teacher"] = t
                else:
                    del llm_filters["teacher"]

        # 合併：規則與 LLM 互補
        merged_filters = llm_filters.copy()
        for k, v in rule_filters.items():
            # 針對容易受「空格/錯字」影響的動態實體（課程、教師），優先信任 LLM 判斷
            # 避免 "Linux 作業系統實務" 被笨笨的 substring 覆蓋成 "作業系統"
            if k in ("course_name_keyword", "teacher") and merged_filters.get(k):
                continue
            merged_filters[k] = v

        # 【關鍵修正】課名搜尋豁免年級：
        # 當使用者指定了特定課程名稱（如「微積分」），年級限制毫無意義。
        # 使用者問「資工有微積分嗎？」→ 不應該因為他是三年級就找不到一年級的微積分！
        if "course_name_keyword" in merged_filters and "grade" in merged_filters:
            removed_grade = merged_filters.pop("grade")
            logger.info(f"  🛡️ 課名搜尋安全閥：已自動移除年級限制 grade='{removed_grade}'（指定課程名稱時不應限制年級）")

        # 判斷是否為「職涯規劃 / 未來大範圍探索 / 跨時空學習路徑」問題
        is_career_planning = bool(data.get("is_career_planning", False))
        
        # (選擇性防禦機制) 避免極端情況下 LLM 忘記清空學期，如果是職涯探索，強制在 Python 端再清空一次
        if is_career_planning:
            merged_filters.pop("academic_year", None)
            merged_filters.pop("semester", None)
            logger.info("  🌟 觸發職涯/跨域探索模式，已自動卸除學期與年級限制")
        
        # (移除) 之前因為看到 _EXPLICIT_NULL 就強制切成職涯探索的邏輯，會造成普通跨學期問題被丟給 170+ chunk 運算，造成算力崩潰。
        # 現在 _EXPLICIT_NULL 僅單純用作防禦 Python fallback 覆寫。
        # Python 層級的強制保底：如果 LLM 和規則都沒拿到 semester/academic_year
        if final_type in ("course_info", "schedule") and not is_career_planning:
            if "academic_year" not in merged_filters:
                merged_filters["academic_year"] = str(config.CURRENT_ACADEMIC_YEAR)
            if "semester" not in merged_filters:
                merged_filters["semester"] = str(config.CURRENT_SEMESTER)
        
        # Python 層級保底：如果問課程，且沒有指定老師/教室/課程名/系所
        if final_type in ("course_info", "schedule"):
            if "teacher" not in merged_filters and "classroom" not in merged_filters and "course_name_keyword" not in merged_filters and "dept_short" not in merged_filters and user_profile:
                if "department" in user_profile:
                    merged_filters["dept_short"] = user_profile["department"]
                # 職涯規劃不該限制年級，以利跨年級搜尋
                if "grade" in user_profile and "grade" not in merged_filters and not is_career_planning:
                    profile_grade = str(user_profile["grade"])
                    merged_filters["grade"] = _GRADE_TO_CN.get(profile_grade, profile_grade)

        # ── 4. 淨化 EXPLICIT_NULL 並建構 RouteResult ──
        # 【關鍵清理】執行完所有的 Fallback 邏輯後，把防護罩 _EXPLICIT_NULL 卸除！
        # 為什麼要卸除？因為跑到 Retriever 時，如果 metadata_filters 裡面還活著 "__EXPLICIT_NULL__"，
        # 就會變成去找 `semester == "__EXPLICIT_NULL__"` 的課，導致 -100 死當！
        cleaned_filters = {}
        for k, v in merged_filters.items():
            if v == _EXPLICIT_NULL:
                continue
            if isinstance(v, list) and _EXPLICIT_NULL in v:
                v = [x for x in v if x != _EXPLICIT_NULL]
                if not v:
                    continue
            cleaned_filters[k] = v
            
        route_result = RouteResult(
            query_type=final_type,
            metadata_filters=cleaned_filters,
            confidence=0.9 if cleaned_filters else 0.7,
            search_queries=data.get("search_queries", []),
            is_chitchat=bool(data.get("is_chitchat", False)),
            is_career_planning=is_career_planning
        )
        
        # ── 5. P2 Step-Back 擴充查詢 ──
        search_queries = data.get("search_queries", [])
        if not isinstance(search_queries, list):
            search_queries = []
        
        # 永遠把原始問題放在第一位（保證 BM25 精準比對）
        if question not in search_queries:
            search_queries.insert(0, question)
        
        # VRAM 保護：限制查詢數量，避免過多 Embedding 請求塞爆 4060
        max_queries = getattr(config, "MULTI_QUERY_COUNT", 4) + 1
        search_queries = search_queries[:max_queries]
        
        logger.info(f"🔀 Agent 路由決策：Intent=[{final_type}], Filters={merged_filters}")
        logger.info(f"   🧠 Step-Back 擴充查詢 ({len(search_queries)} 個)：{search_queries}")
        return route_result, search_queries
        
    except Exception as e:
        logger.warning(f"Router LLM 呼叫失敗，降級為規則 fallback：{e}")
        
        # 終極防線：LLM 崩潰時只靠規則引擎 + 原始問題
        route_result = RouteResult(
            query_type=rule_type,
            metadata_filters=rule_filters,
            confidence=0.3,
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
        result, _ = route_and_rewrite(q)
        print(f"   {result}")
