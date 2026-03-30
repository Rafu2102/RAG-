import re
import json
import requests
import logging
from datetime import datetime, timedelta, timezone

import config
from tools.calendar_api import create_calendar_event, delete_calendar_events, list_calendar_events, update_calendar_event
from tools.search_event_tool import search_academic_events

logger = logging.getLogger(__name__)

# NQU 節次對應時間表 (依據金門大學一般作息)
NQU_PERIOD_START = {
    1: "08:10", 2: "09:10", 3: "10:10", 4: "11:10",
    5: "13:30", 6: "14:30", 7: "15:30", 8: "16:30",
    9: "17:30", 10: "18:30", 11: "19:30", 12: "20:30"
}
NQU_PERIOD_END = {
    1: "09:00", 2: "10:00", 3: "11:00", 4: "12:00",
    5: "14:20", 6: "15:20", 7: "16:20", 8: "17:20",
    9: "18:20", 10: "19:20", 11: "20:20", 12: "21:20"
}



def get_next_weekday(day_of_week: int) -> datetime:
    """取得下一個指定的星期幾的日期 (1=週一, 7=週日)"""
    # 台灣時區
    tz = timezone(timedelta(hours=8))
    today = datetime.now(tz)
    # Python weekday(): 0=Monday, 6=Sunday
    target_weekday = day_of_week - 1
    
    days_ahead = target_weekday - today.weekday()
    if days_ahead < 0: # 已經過了，往後找下週 (如果是今天則 direct start)
        days_ahead += 7
        
    return today + timedelta(days=days_ahead)

def extract_calendar_intent(query: str) -> dict:
    """僅負責透過 Gemini 判斷使用者的行事曆意圖 (不涉及 RAG 或建立 API)"""
    from llm.date_utils import normalize_chinese_datetime
    
    # 先正規化中文數字（三月16→3月16, 九點半→9點半）
    normalized_query = normalize_chinese_datetime(query)
    logger.info(f"📅 日期正規化：「{query}」→「{normalized_query}」")
    
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    
    # 建立精確的中文星期幾，讓 LLM 在推算相對時間(如下週二)時 100% 精準
    weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
    weekday_str = weekday_map[now.weekday()]
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S") + f" 星期{weekday_str}"

    intent_prompt = f"""你是一個精準的行事曆意圖分析助理。

【系統隱藏資訊】
現在的真實台灣時間是：{current_time_str}
請依據此基準時間，推算使用者口中的「明天」、「下週二」、「後天晚上」等相對時間，並轉換為絕對的 YYYY-MM-DDTHH:MM:SS 格式。

使用者要求對行事曆進行操作，請判斷使用者的意圖分類與動作類型，並嚴格依據 JSON Schema 回傳。

## 動作類型 (action_type)
- "add"：新增、加入、排進、設定提醒
- "remove"：刪除、移除、取消
- "list"：查看、列出、我的行事曆、有什麼行程
- "update"：修改、變更、更改時間

## 分類選項 (intent_type)
- "custom_event"：使用者明確提供新活動或事件（如「5月29是我生日」「明天三點開會」）。不管什麼事件，只要不是學校/課程就是 custom_event。
- "course_schedule_event"：提到課程名稱 + 考試活動（期中考/小考）。
- "academic_event"：學校行政事件（加退選/停修）。
- "weekly_course"：把某單一課程的「每週常態上課時間」加到行事曆。
- "import_schedule"：使用者明確指出要將「我的課表」、「本學期課表」、「所有課程」**整批**匯入或加入行事曆。

## ⚠️ 時間處理極重要規則
1. **下午必須加12**：「下午四點」= 16:00、「晚上七點」= 19:00。絕對不可寫成 04:00。
2. **end_dt 必須在 start_dt 之後**：如果使用者說「4點到 6 點」，start=16:00, end=18:00。
3. **target_date 的用途**：target_date 是用來定位「原始事件」的日期。如果使用者說「把數學課改到後天」，他不知道原本是哪天，target_date 必須為 null！「後天」是新時間，應填在 start_dt！
4. **不確定就留空**：如果使用者沒提到結束時間，end_dt 可以留 null。

## 💡 常見陷阱面試
- 「幫我把期中報告修改到後天下午 4:20 到 6:20」
  → action_type="update", event_name="期中報告", target_date=null, start_dt="後天T16:20:00", end_dt="後天T18:20:00"
- 「明天三點開會」
  → action_type="add", event_name="開會", start_dt="明天T15:00:00", end_dt=null

使用者輸入：{normalized_query}
"""

    gemini_schema = {
        "type": "OBJECT",
        "properties": {
            "action_type": {
                "type": "STRING",
                "enum": ["add", "remove", "list", "update"],
                "description": "判斷使用者是要新增、刪除、列出還是修改事件"
            },
            "intent_type": {
                "type": "STRING",
                "enum": ["custom_event", "course_schedule_event", "academic_event", "weekly_course", "import_schedule"]
            },
            "event_name": {
                "type": "STRING", 
                "description": "事件的乾淨名稱。必須移除所有時間、日期與請求詞。例如「幫我把明天的微積分加到行事曆」→「微積分」。「數學課改晚上」→「數學課」。若極簡無名稱才填 *"
            },
            "course_name": {"type": "STRING", "description": "課程名稱 (僅 course_schedule_event 時填寫)"},
            "schedule_keyword": {"type": "STRING", "description": "活動關鍵字如期中考/期末考 (僅 course_schedule_event 時)"},
            "target_date": {"type": "STRING", "description": "YYYY-MM-DD (要被操作的「原始」事件所在日期，用來幫助定位行程。如果使用者只說「把數學課改到後天」，代表不知道原本是哪天，此欄位必須保持為 null，絕對不可將新時間填這裡！)"},
            "start_dt": {"type": "STRING", "description": "YYYY-MM-DDTHH:MM:SS (新增或修改後的【新開始絕對時間】。下午必須加12！)"},
            "end_dt": {"type": "STRING", "description": "YYYY-MM-DDTHH:MM:SS (新增或修改後的【新結束絕對時間】。沒提到可留空。若使用者有提到結束時間或「到幾點」，務必填入，下午記得加12！)"},
            "is_time_missing": {
                "type": "BOOLEAN", 
                "description": "如果使用者要求新增事件(add)但完全沒提到在什麼時候（也非全天），請設為 true"
            }
        },
        "required": ["action_type", "intent_type", "event_name", "is_time_missing"]
    }

    try:
        logger.info("📅 正在呼叫 Gemini Flash-Lite 判斷加入行事曆的精確意圖...")
        payload = {
            "contents": [{"parts": [{"text": intent_prompt}]}],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": config.GEMINI_FLASH_MAX_TOKENS,
                "responseMimeType": "application/json",
                "responseSchema": gemini_schema,
                "thinkingConfig": {
                    "thinkingLevel": "medium"
                }
            }
        }
        
        response = requests.post(config.GEMINI_FAST_API_URL, json=payload, timeout=config.GEMINI_FLASH_TIMEOUT)
        response.raise_for_status()
        
        candidates = response.json().get("candidates", [])
        if not candidates:
            raise ValueError("Gemini 未回傳任何 candidate")
        
        text_content = candidates[0]["content"]["parts"][0]["text"].strip()
        res_data = json.loads(text_content)
        
        # 基本檢查與修正預設值
        action_type = res_data.get("action_type", "add")
        intent_type = res_data.get("intent_type", "weekly_course")
        e_name = res_data.get("event_name", "").strip()
        
        # 即使 Gemini 很準，若還是空字串，給原始 query 截斷
        if not e_name:
            e_name = query.split("，")[0].split(",")[0]
            res_data["event_name"] = e_name
            
        logger.info(f"📅 Gemini 意圖判定成功：{res_data}")

    except Exception as e:
        logger.warning(f"行事曆意圖分析失敗，退回預設參數：{e}")
        return {
            "action_type": "add",
            "intent_type": "custom_event",
            "event_name": query,
            "is_time_missing": False
        }

    return res_data


# === 教學進度表日期萃取 ===

def _extract_date_from_schedule(chunks: list, keyword: str) -> dict | None:
    # ... (existing code)
    import re
    
    # 先合併所有 chunk 的文字
    full_text = "\n".join([c.node.get_content() for c in chunks]) if chunks else ""
    
    if not full_text:
        return None
    
    # 正則匹配教學進度表中的日期行
    # 格式: 第X週課程 (YYYY/MM/DD─YYYY/MM/DD)：關鍵字
    pattern = rf'第\d+週課程\s*\((\d{{4}}/\d{{2}}/\d{{2}})─(\d{{4}}/\d{{2}}/\d{{2}})\)\s*[：:]\s*([^\n]*{re.escape(keyword)}[^\n]*)'
    
    matches = re.findall(pattern, full_text)
    
    if matches:
        start_date_str, end_date_str, activity = matches[0]
        # 轉為 ISO 格式
        start_iso = start_date_str.replace("/", "-")
        end_iso = end_date_str.replace("/", "-")
        logger.info(f"📅 教學進度表萃取成功 | 活動：{activity.strip()} | 日期：{start_iso} ~ {end_iso}")
        return {
            "start": f"{start_iso}T00:00:00",
            "end": f"{end_iso}T00:00:00",
            "activity": activity.strip(),
            "week_range": f"{start_date_str} ~ {end_date_str}"
        }
    
    return None

# === Calendar Action Handlers ===

def _handle_remove(discord_id: str, e_name: str, start_dt: str | None, target_date: str | None, intent_type: str = "") -> str:
    # 【強化】刪除課表意圖翻譯：如果使用者說「移除課表」，將關鍵字覆寫為「[課程]」以批次匹配所有匯入的課程行程
    is_batch_delete = False
    
    # 判斷是否要求「完全清空」
    clear_all_keywords = ["所有", "全部", "清空", "所有行程", "全部行程", "全部行事曆", "所有行事曆", "所有新增的行事曆", "全刪"]
    if e_name in clear_all_keywords or (e_name == "*" and intent_type in ["custom_event", "course_schedule_event"]):
        e_name = "__CLEAR_ALL__"
        is_batch_delete = True
        logger.info("🗑️ 偵測到『清空全部』意圖，將關鍵字設定為「__CLEAR_ALL__」清空所有系統行程")
        
    elif intent_type == "import_schedule" or e_name in ["課表", "所有課", "全部課", "課程表", "我的課表", "課程", "所有課程"]:
        e_name = "[課程]"
        is_batch_delete = True
        logger.info("🗑️ 偵測到刪除整份課表意圖，將關鍵字覆寫為「[課程]」進行批次刪除（含過去行程）")
    
    # 刪除時，盡量依賴 target_date 來做寬鬆搜尋，不再強制把 start_dt 當作 search_time
    search_time = start_dt if not target_date else None
    
    logger.info(f"📅 執行行事曆移除：名稱「{e_name}」，指定日期「{target_date}」，精確時間「{search_time}」")
    result = delete_calendar_events(
        discord_id=discord_id, keyword=e_name,
        target_date=target_date, target_time=search_time,
        include_past=True  # 移除指定事件時（無指定日期），一律往前追溯半年，以利刪除過去建立的錯誤日曆
    )
    return result["message"]

def _handle_list(discord_id: str, target_date: str | None) -> str:
    if target_date:
        logger.info(f"📅 執行行事曆列出：指定日期 {target_date}")
        result = list_calendar_events(discord_id=discord_id, target_date=target_date)
        date_desc = target_date
    else:
        days = 7
        logger.info(f"📅 執行行事曆列出：未來 {days} 天")
        result = list_calendar_events(discord_id=discord_id, days=days)
        date_desc = f"未來 {days} 天"
    
    if result["status"] == "success" and result["events"]:
        lines = [f"📅 **{date_desc}** 的行事曆事件：\n"]
        for ev in result["events"]:
            tag = "🤖" if ev["is_bot_created"] else "👤"
            start_str = ev["start"]
            if "T" in start_str:
                time_part = start_str.split("T")[1][:5]
                date_part = start_str.split("T")[0]
                display = f"{date_part} {time_part}"
            else:
                display = start_str
            lines.append(f"{tag} **{ev['title']}** ─ {display}")
        return "\n".join(lines)
    elif result["status"] == "success":
        return f"📅 在 {date_desc} 內沒有任何行事曆事件。"
    else:
        return result["message"]

def _handle_update(discord_id: str, intent_data: dict, e_name: str, start_dt: str, target_date: str) -> str:
    new_title = intent_data.get("event_name", "") # 如果使用者要改名
    new_start = start_dt # 新的開始時間
    new_end = intent_data.get("end_dt") # 新的結束時間
    
    logger.info(f"📅 執行行事曆修改：尋找「{e_name}」於日期「{target_date}」，新時間為 {new_start}")
    result = update_calendar_event(
        discord_id=discord_id, keyword=e_name,
        target_date=target_date,
        new_title=new_title if new_title and new_title != "*" else None,
        new_start=new_start if new_start else None,
        new_end=new_end if new_end else None
    )
    if result["status"] == "success" and "htmlLink" in result:
        return f"{result['message']}\n🔗 連結：{result['htmlLink']}"
    return result["message"]

def execute_calendar_action(query: str, intent_data: dict, retrieved_chunks: list | None = None, discord_id: str | None = None) -> str:
    """根據已經判斷出的意圖資料，實際執行 Google Calendar API 的增刪操作"""
    action_type = intent_data.get("action_type", "add")
    intent_type = intent_data.get("intent_type", "weekly_course")
    e_name = intent_data.get("event_name", query)
    target_date = intent_data.get("target_date")   # YYYY-MM-DD（list/remove/update 用）
    start_dt = intent_data.get("start_dt")          # YYYY-MM-DDTHH:MM:SS

    # === 【Fix 5】未註冊保護：提前檢測 ===
    if not discord_id:
        return "❌ 無法識別您的身分，請先使用 `/identity_login` 進行註冊與 Google 帳號綁定。"
    
    from tools.auth import get_user_token_path
    if not get_user_token_path(discord_id).exists():
        return "❌ 您尚未綁定 Google 帳號喔！請先使用 `/identity_login` 指令完成註冊與行事曆授權。"

    try:
        # === [流程 0a] 刪除行事曆事件 ===
        if action_type == "remove":
            return _handle_remove(discord_id, e_name, start_dt, target_date, intent_type=intent_type)
        
        # === [流程 0b] 列出行事曆事件 ===
        if action_type == "list":
            return _handle_list(discord_id, target_date)
        
        # === [流程 0c] 修改行事曆事件 ===
        if action_type == "update":
            return _handle_update(discord_id, intent_data, e_name, start_dt, target_date)
    
    except ValueError as e:
        # get_service() 會在 token 無效時 raise ValueError
        return f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"❌ 行事曆操作失敗 | discord_id={discord_id} | {e}")
        return f"❌ 行事曆操作時發生錯誤：{str(e)}"
    
    # === [流程 1] 課程教學進度表事件（如期中考、期末考）===
    if intent_type == "course_schedule_event":
        course_name = intent_data.get("course_name", e_name)
        schedule_keyword = intent_data.get("schedule_keyword", "期中考")
        
        logger.info(f"📅 課程進度表搜尋 | 課程：{course_name} | 搜尋：{schedule_keyword}")
        
        # 從 RAG 檢索結果中萃取日期
        date_info = _extract_date_from_schedule(retrieved_chunks, schedule_keyword)
        
        if date_info:
            event_title = f"[{schedule_keyword}] {course_name}"
            logger.info(f"📅 進度表日期找到 | {event_title} | {date_info['week_range']}")
            
            result = create_calendar_event(
                discord_id=discord_id,
                title=event_title,
                start=date_info["start"],
                end=date_info["end"]
            )
            
            if result["status"] == "success":
                return (
                    f"✅ 已從 **{course_name}** 的教學進度表中找到 **{schedule_keyword}** 的時間，並加入行事曆！\n\n"
                    f"📌 標題：{event_title}\n"
                    f"📆 日期：{date_info['week_range']}\n"
                    f"📝 進度表內容：{date_info['activity']}\n"
                    f"🔗 事件連結：{result['htmlLink']}"
                )
            elif result["status"] == "exists":
                return f"⚠️ 行事曆已有 **{event_title}**，已略過新增\n🔗 事件連結：{result['htmlLink']}"
            else:
                return f"❌ 建立失敗：{result.get('message', '未知錯誤')}"
        else:
            return (
                f"❌ 抱歉，我在 **{course_name}** 的教學進度表中找不到「{schedule_keyword}」的日期資訊。\n\n"
                f"💡 **建議**：您可以用自訂方式加入，例如：\n"
                f"「5月15日 {course_name}{schedule_keyword} 加到行事曆」"
            )

    # === 依據新增意圖分流處理 ===
    
    # [流程 A] 自訂時間事件
    if intent_type == "custom_event":
        e_start = intent_data.get("start_dt") or start_dt
        e_end = intent_data.get("end_dt")
        
        # 【新增】若 start_dt 為空但有 target_date，當作全天事件處理
        if not e_start and target_date:
            e_start = f"{target_date}T00:00:00"
            logger.info(f"📅 無具體時間但有 target_date={target_date}，視為全天事件")
        
        # 若有 start 但沒 end，自動 +2 小時（或全天事件）
        if e_start and not e_end:
            try:
                from tools.calendar_api import _parse_dt
                _s = _parse_dt(e_start)
                if _s.hour == 0 and _s.minute == 0:
                    e_end = e_start  # 全天事件，start == end
                else:
                    e_end = (_s + timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S')
            except Exception:
                e_end = e_start
        
        # === 【3. 缺漏資訊反問機制】 ===
        if intent_data.get("is_time_missing") and action_type == "add":
            return f"沒問題！想請問「{e_name}」是預計在哪一天、幾點開始呢？🏀"

        # 標題清理
        title = e_name if e_name not in ("*", "") else query

        
        if e_start and e_end:
            logger.info(f"📅 執行自訂時間事件建立：{title} {e_start} ~ {e_end}")
            result = create_calendar_event(discord_id=discord_id, title=f"[自訂] {title}", start=e_start, end=e_end)
            if result["status"] == "success":
                is_all_day = "T00:00:00" in e_start
                start_display = e_start.split('T')[0] if is_all_day else f"{e_start.split('T')[0]} {e_start.split('T')[1][:5]}"
                end_display = e_end.split('T')[0] if is_all_day else f"{e_end.split('T')[0]} {e_end.split('T')[1][:5]}"
                return (
                    f"✅ 已新增到 Google Calendar！\n\n"
                    f"> 📌 **{title}**\n"
                    f"> 🗓️ {start_display} ～ {end_display}\n"
                    f"> 🔗 {result['htmlLink']}"
                )
            elif result["status"] == "exists":
                return f"⚠️ 行事曆已有相同事件，已略過新增\n📌 標題：{title}\n🔗 事件連結：{result['htmlLink']}"
            else:
                return f"❌ 建立失敗：{result.get('message', '未知錯誤')}"
        else:
            # custom_event 但缺少時間 → 友善引導
            return (
                f"😅 我知道你想加「**{title}**」到行事曆，但我沒有偵測到具體的時間。\n\n"
                f"💡 **請提供具體時間**，例如：\n"
                f"「3月16日是我生日加到行事曆」\n"
                f"「明天下午三點開會幫我加行事曆」"
            )
                
    # [流程 B] 學校既定事件
    elif intent_type == "academic_event":
        academic_events = search_academic_events(query)
        if academic_events:
            event = academic_events[0]
            event_title = event.get("title", "")
            iso_start = event.get("start", "")
            iso_end = event.get("end", "")
            
            logger.info(f"📅 執行學校事件建立：{event_title} {iso_start} ~ {iso_end}")
            try:
                result = create_calendar_event(discord_id=discord_id, title=f"[金大] {event_title}", start=iso_start, end=iso_end)
                if result["status"] == "success":
                    display_date = iso_start.split("T")[0]
                    e_date_end = iso_end.split("T")[0]
                    return (
                        f"✅ 已新增到 Google Calendar\n"
                        f"📌 標題：{event_title}\n"
                        f"🕒 開始：{{'date': '{display_date}'}}\n"
                        f"🕒 結束：{{'date': '{e_date_end}'}}\n"
                        f"🔗 事件連結：{result['htmlLink']}"
                    )
                elif result["status"] == "exists":
                    return f"⚠️ 行事曆已有相同事件，已略過新增\n📌 標題：{event_title}\n🔗 事件連結：{result['htmlLink']}"
                else:
                    return f"❌ 建立失敗：{result.get('message', '未知錯誤')}"
            except Exception as e:
                logger.error(f"行事曆(學校事件)建立錯誤: {e}")
                return f"❌ 抱歉，在處理行事曆時發生錯誤：{e}。"
        else:
            # 找不到學校事件，直接回報找不到，不盲目 fallback 裝成每週課程！
            return f"❌ 抱歉，我在學校的行事曆上沒有找到名為「**{e_name}**」的事件，因此無法為您加入行事曆喔！"

    # [流程 C] 批次匯入個人課表
    elif intent_type == "import_schedule":
        from tools.schedule_manager import get_schedule, PERIOD_TIME_MAP
        import json
        
        schedule = get_schedule(discord_id)
        if not schedule or "courses" not in schedule:
            return "❌ 我找不到您的課表資料喔！請先使用 `/upload_schedule`（或附加快捷鍵）上傳課表截圖。"
            
        courses = schedule["courses"]
        if not courses:
            return "⚠️ 您的課表內目前沒有任何課程資料。"
            
        # 尋找 events.json 中的開學日，以做為 18 週循環的精準起點
        events_path = config.DATA_DIR + "/events.json"
        semester_start_date = None
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                events_data = json.load(f)
                for ev in events_data:
                    if "上課開始" in ev.get("title", ""):
                        # 將 UTC 或者 string 轉為本地 datetime
                        semester_start_date = datetime.fromisoformat(ev["start"]).replace(tzinfo=timezone(timedelta(hours=8)))
                        break
        except Exception as e:
            logger.warning(f"讀取 events.json 尋找開學日失敗: {e}")
            
        if not semester_start_date:
            logger.warning("⚠️ 在 events.json 找不到「上課開始」事件，退回使用下週推算")
        
        success_count = 0
        failure_count = 0
        last_success_link = ""
        last_error_msg = ""
        
        for course in courses:
            c_name = course.get("name", "未知課程")
            day_of_week = course.get("day")
            periods = course.get("periods", [])
            instructor = course.get("instructor", "")
            room = course.get("room", "")
            
            if not day_of_week or not periods:
                continue
                
            start_p = periods[0]
            end_p = periods[-1]
            
            # 推算該課程在第一週的正確日期
            if semester_start_date:
                target_weekday = day_of_week - 1
                days_ahead = target_weekday - semester_start_date.weekday()
                if days_ahead < 0:
                    days_ahead += 7
                target_date = semester_start_date + timedelta(days=days_ahead)
            else:
                target_date = get_next_weekday(day_of_week)
                
            date_str = target_date.strftime("%Y-%m-%d")
            start_time_str = PERIOD_TIME_MAP.get(start_p, ("08:10", "09:00"))[0]
            end_time_str = PERIOD_TIME_MAP.get(end_p, ("08:10", "09:00"))[1]
            
            iso_start = f"{date_str}T{start_time_str}:00"
            iso_end = f"{date_str}T{end_time_str}:00"
            
            event_title = f"[課程] {c_name} ({instructor})" if instructor else f"[課程] {c_name}"
            if room:
                event_title += f" @ {room}"
                
            logger.info(f"📅 準備批次建立課程：{event_title} {iso_start}")
            
            # 【強化】生成 EXDATE 以跳過國定假日
            exdates = []
            from datetime import date as _date
            for week_i in range(18):
                event_date = (target_date + timedelta(weeks=week_i)).date() if isinstance(target_date, datetime) else target_date + timedelta(weeks=week_i)
                date_str_check = event_date.isoformat() if isinstance(event_date, _date) else str(event_date)[:10]
                if date_str_check in config.HOLIDAYS:
                    exdates.append(f"{date_str_check}T{start_time_str}:00")
                    logger.info(f"🎌 跳過國定假日：{date_str_check} ({c_name})")
            
            recurrence_rules = ["RRULE:FREQ=WEEKLY;COUNT=18"]
            if exdates:
                for exd in exdates:
                    recurrence_rules.append(f"EXDATE;TZID=Asia/Taipei:{exd.replace('-', '').replace(':', '')}")
            
            result = create_calendar_event(
                discord_id=discord_id,
                title=event_title,
                start=iso_start,
                end=iso_end,
                recurrence=recurrence_rules
            )
            
            if result["status"] in ("success", "exists"):
                success_count += 1
                last_success_link = result.get('htmlLink', last_success_link)
            else:
                failure_count += 1
                last_error_msg = result.get('message', '未知錯誤')
                
        if success_count > 0:
            return (
                f"✅ **已成功為您匯入整份課表至 Google Calendar！**\n"
                f"一共排入了 **{success_count}** 個課程時段（每週自動重複 18 週）。\n"
                f"🔗 前往查看行事曆：{last_success_link}"
                + (f"\n❌ (部分課程匯入失敗: {last_error_msg})" if failure_count > 0 else "")
            )
        else:
            return f"❌ 課表匯入全數失敗：{last_error_msg}"


    # === 2. 退回原有的「每週課程」時間擷取邏輯 ===
    if not retrieved_chunks:
        return "🤔 我不太確定你想加哪一堂課或活動，可以請您提供更多課程名稱或老師細節嗎？"
        
    # 取最相關的那門課
    top_chunk = retrieved_chunks[0]
    meta = top_chunk.node.metadata
    
    course_name = meta.get("course_name", "未知課程")
    schedule_str = meta.get("schedule", "")
    teacher = meta.get("teacher", "")
    
    if not schedule_str or schedule_str == "?":
        return f"😅 抱歉，我找到了【{course_name}】，但它的資訊裡沒有寫明上課時間，我無法幫你加到行事曆喔。"
        
    # 使用 3B/8B LLM 搭配 Structured Output 安全提取節次
    prompt = f"""你是一個精準的時間解析器。
請分析這堂課的上課時間字串，找出所有的上課時段，並純粹回傳 JSON。
若該課程一週有多次上課時間（例如：星期一與星期三），請將所有時段各自拆解為物件，放入 schedules 陣列中。
如果沒有寫上課時間，請回傳空的 schedules 陣列。

上課時間字串：{schedule_str}

## 節次對照表（金門大學）
第1節 08:10-09:00、第2節 09:10-10:00、第3節 10:10-11:00、第4節 11:10-12:00
第5節 13:30-14:20、第6節 14:30-15:20、第7節 15:30-16:20、第8節 16:30-17:20
第N節(晚) 18:30-19:20、第10節 19:25-20:15、第11節 20:20-21:10、第12節 21:15-22:05

請擷取出每一個時段的：
1. day_of_week: 星期幾 (1=星期一, 2=星期二... 7=星期日)
2. start_period: 開始節次 (如第5節則為 5，第N節則為 9)
3. end_period: 結束節次 (如有波浪號如 5~7 則 end 為 7，否則同 start)

ℹ️ 注意：如果看到「第N節」，請將 N 視為第 9 節（晚間第一節）。
"""

    logger.info(f"📅 呼叫 Gemini 解析行事曆時間: {schedule_str}")
    
    schedule_schema = {
        "type": "OBJECT",
        "properties": {
            "schedules": {
                "type": "ARRAY",
                "description": "這堂課所有的上課時段列表 (若跨多天則會有多筆)",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "day_of_week": {
                            "type": "INTEGER",
                            "description": "星期幾 (1=星期一, 2=星期二... 7=星期日)"
                        },
                        "start_period": {
                            "type": "INTEGER",
                            "description": "開始節次 (例如 1~12)"
                        },
                        "end_period": {
                            "type": "INTEGER",
                            "description": "結束節次 (例如 1~12)"
                        }
                    },
                    "required": ["day_of_week", "start_period", "end_period"]
                }
            }
        },
        "required": ["schedules"]
    }

    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": config.GEMINI_SHORT_MAX_TOKENS,
                "responseMimeType": "application/json",
                "responseSchema": schedule_schema,
                "thinkingConfig": {
                    "thinkingLevel": "low"
                }
            }
        }
        response = requests.post(config.GEMINI_FAST_API_URL, json=payload, timeout=config.GEMINI_FLASH_TIMEOUT)
        response.raise_for_status()
        
        text_content = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        data = json.loads(text_content)
        
        schedules = data.get("schedules", [])
        if not schedules:
             return f"😅 解析失敗：【{course_name}】雖然有上課時間字串 ({schedule_str})，但我無法辨識其內容，無法為您加到行事曆。"
             
        event_title = f"[課程] {course_name} ({teacher})"
        
        success_count = 0
        failure_count = 0
        last_success_link = ""
        last_error_msg = ""
        skipped_holidays = 0
        
        for sched in schedules:
            day_of_week = sched.get("day_of_week")
            start_p = sched.get("start_period")
            end_p = sched.get("end_period")
            
            if day_of_week is None or start_p is None or end_p is None:
                continue
            
            # 組裝 IsoFormat 日期
            target_date = get_next_weekday(day_of_week)
            date_str = target_date.strftime("%Y-%m-%d")
            
            start_time_str = NQU_PERIOD_START.get(start_p, "08:10")
            end_time_str = NQU_PERIOD_END.get(end_p, "09:00")
            
            iso_start = f"{date_str}T{start_time_str}:00"
            iso_end = f"{date_str}T{end_time_str}:00"
            
            # 【強化】生成 EXDATE 以跳過國定假日
            exdates = []
            from datetime import date as _date
            for week_i in range(18):
                event_date = target_date + timedelta(weeks=week_i)
                date_str_check = event_date.strftime("%Y-%m-%d")
                if date_str_check in config.HOLIDAYS:
                    exdates.append(f"{date_str_check}T{start_time_str}:00")
                    skipped_holidays += 1
                    logger.info(f"🎌 跳過國定假日：{date_str_check} ({course_name})")
            
            recurrence_rules = ["RRULE:FREQ=WEEKLY;COUNT=18"]
            if exdates:
                for exd in exdates:
                    recurrence_rules.append(f"EXDATE;TZID=Asia/Taipei:{exd.replace('-', '').replace(':', '')}")
            
            logger.info(f"📅 準備呼叫 Google API：{event_title} {iso_start} ~ {iso_end} (週期：18週)")
            
            # 呼叫 Google API
            result = create_calendar_event(
                discord_id=discord_id,
                title=event_title,
                start=iso_start,
                end=iso_end,
                recurrence=recurrence_rules
            )
            
            if result["status"] == "success":
                success_count = success_count + 1
                last_success_link = result['htmlLink']
            elif result["status"] == "exists":
                success_count = success_count + 1
                last_success_link = result['htmlLink']
            else:
                failure_count = failure_count + 1
                last_error_msg = result.get('message', '未知錯誤')
        
        if success_count > 0:
            holiday_msg = f"\n🎌 已自動跳過 {skipped_holidays} 個國定假日的上課時段" if skipped_holidays > 0 else ""
            return (
                f"✅ 已成功將課程加入 Google Calendar，為您設定為**每週重複 (共18週)**！\n"
                f"📌 課程：{course_name}\n"
                f"🕒 共新增了 {success_count} 個上課時段\n"
                f"🔗 前往查看行事曆：{last_success_link}"
                + holiday_msg
                + (f"\n❌ (部分時段失敗: {last_error_msg})" if failure_count > 0 else "")
            )
        else:
            return f"❌ 建立失敗：{last_error_msg}"
            
    except Exception as e:
        logger.error(f"行事曆建立發生錯誤: {e}")
        return f"❌ 抱歉，在處理行事曆時發生錯誤：{e}。這可能是因為我還沒獲得授權，或是沒有這台主機的 tokens 捷徑喔！"


def generate_academic_event_answer(query: str, events: list) -> str:
    """單純查詢學校行事曆事件時，由 LLM 包裝友善的回應 (非寫入行事曆)"""
    
    events_str = "\n".join([f"- {e.get('title', '未知事件')}: {e.get('start', '').split('T')[0]} 到 {e.get('end', '').split('T')[0]}" for e in events])
    
    prompt = f"""你是一個熱心助人的金門大學校園助理。
使用者問了一個關於學校行事曆的問題。我已經從資料庫為你找到了相關的學校行程。
請根據找到的資料，直接簡潔地回答使用者的問題。

【學校行程資料】
{events_str}

【使用者的問題】
{query}

【回答要求】
- 語氣友善、簡潔，像學長姐在 Discord 聊天室裡回答。
- 只需回答與使用者問題最相關的事件時間，不要囉嗦。
- 不用提及「根據資料」、「我找到」等 AI 機器人用語。
- 如果行程橫跨多天（如連假），請明確寫出「從 X月X日 到 X月X日」。
- 如果找到多個相關行程，請用 Emoji 清單列出，不要只回答一個。
- 【禁止捏造】：只能用上方提供的行事曆資料回答，絕對不可編造日期。"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": config.GEMINI_SHORT_MAX_TOKENS,
            "thinkingConfig": {
                "thinkingLevel": "low"
            }
        }
    }

    try:
        response = requests.post(
            config.GEMINI_FAST_API_URL,
            json=payload,
            timeout=config.GEMINI_FLASH_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
    except Exception as e:
        logger.error(f"Academic Event LLM 發生錯誤: {e}")
        # Fallback 回覆
        return f"為您找到以下學校行事曆資料：\n{events_str}"

