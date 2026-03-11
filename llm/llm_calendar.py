import re
import json
import requests
import logging
from datetime import datetime, timedelta, timezone

import config
from tools.calendar_tool import create_calendar_event, delete_calendar_events
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

CALENDAR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "day_of_week": {
            "type": "integer",
            "description": "星期幾 (1=星期一, 2=星期二... 7=星期日)"
        },
        "start_period": {
            "type": "integer",
            "description": "開始節次 (例如 1~12)"
        },
        "end_period": {
            "type": "integer",
            "description": "結束節次 (例如 1~12)"
        }
    },
    "required": ["day_of_week", "start_period", "end_period"]
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
    """僅負責透過 LLM 判斷使用者的行事曆意圖 (不涉及 RAG 或建立 API)"""
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    
    # 建立精確的中文星期幾，讓 LLM 在推算相對時間(如下週二)時 100% 精準
    weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
    weekday_str = weekday_map[now.weekday()]
    today_str = f"{now.strftime('%Y-%m-%d')} (星期{weekday_str})"

    # === 1. 使用 LLM 判斷加入行事曆的意圖 ===
    intent_prompt = f"""你是一個精準的行事曆意圖分析助理。今天是 {today_str}。
使用者要求對行事曆進行操作（新增或刪除），請判斷使用者的意圖分類與動作類型，並純粹回傳 JSON。

分類選項 (intent_type)：
- "custom_event"：使用者「已經明確給出具體日期或時間」(例如：「5月29是我生日」、「明天下午開會」、「5月13日期中考」)。不管事件是什麼，只要有給時間就是 custom_event！
- "academic_event"：詢問或要求把學校的「節日、行事曆事件」操作日曆，且「完全沒有給具體時間」 (例如：「什麼時候加退選加到行事曆」、「停修申請幫我加到日曆」、「宿舍開館加到行事曆」)
- "weekly_course"：要求把某個「課程」的每週上課時間操作行事曆 (例如：「把線性代數加到日曆」、「幫我刪除程式設計」、「移除微積分」)

動作類型 (action_type)：
- "add"：新增、加入、排進、設定提醒
- "remove"：刪除、移除、取消

【特別注意】
1. 若判斷為 "custom_event" 且 action_type 為 "add"，必須同時擷取 start_dt 與 end_dt (格式 YYYY-MM-DDTHH:MM:SS)。
2. 若使用者句子裡有「什麼時候」、「何時」，代表他不知道時間，這時絕對不能選 "custom_event"，必須選 "academic_event"。
3. 反之，只要句子裡出現「X月X日」、「明天」、「下週三」，就代表他知道時間，請務必選擇 "custom_event"！
4. 如果是要「刪除」事件，日期不重要，只要擷取 event_name 即可。

使用者輸入：{query}

請純粹回傳 JSON。"""

    INTENT_SCHEMA = {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["add", "remove"],
                "description": "判斷使用者是要新增還是刪除事件"
            },
            "intent_type": {
                "type": "string",
                "enum": ["custom_event", "academic_event", "weekly_course"]
            },
            "event_name": {"type": "string", "description": "事件或課程名稱"},
            "start_dt": {"type": "string", "description": "YYYY-MM-DDTHH:MM:SS (僅 custom_event 新增時需要)"},
            "end_dt": {"type": "string", "description": "YYYY-MM-DDTHH:MM:SS (僅 custom_event 新增時需要)"}
        },
        "required": ["action_type", "intent_type", "event_name", "start_dt", "end_dt"]
    }

    try:
        logger.info("📅 正在呼叫 LLM 判斷加入行事曆的精確意圖...")
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": getattr(config, "OLLAMA_FAST_MODEL", config.OLLAMA_MODEL),
                "prompt": intent_prompt,
                "format": INTENT_SCHEMA,
                "stream": False,
                "keep_alive": 0,
                "options": {"temperature": 0.0, "num_ctx": 1024, "num_predict": 150},
            },
            timeout=15,
        )
        response.raise_for_status()
        res_data = json.loads(response.json()["response"].strip())
        
        action_type = res_data.get("action_type", "add")
        intent_type = res_data.get("intent_type", "weekly_course")
        e_name = res_data.get("event_name", query)
        
        logger.info(f"📅 意圖判定：{action_type} - {intent_type} (目標：{e_name})")

    except Exception as e:
        logger.warning(f"行事曆意圖分析失敗，退回預設每週課程流程：{e}")
        return {
            "action_type": "add",
            "intent_type": "weekly_course",
            "event_name": query
        }

    return res_data

def execute_calendar_action(query: str, intent_data: dict, retrieved_chunks: list = None) -> str:
    """根據已經判斷出的意圖資料，實際執行 Google Calendar API 的增刪操作"""
    action_type = intent_data.get("action_type", "add")
    intent_type = intent_data.get("intent_type", "weekly_course")
    e_name = intent_data.get("event_name", query)

    # === [流程 0] 刪除行事曆事件 ===
    if action_type == "remove":
        logger.info(f"📅 執行行事曆移除：尋找包含「{e_name}」的事件")
        result = delete_calendar_events(keyword=e_name)
        return result["message"]

    # === 依據新增意圖分流處理 ===
    
    # [流程 A] 自訂時間事件
    if intent_type == "custom_event":
        e_start = intent_data.get("start_dt")
        e_end = intent_data.get("end_dt")
        
        if e_start and e_end:
            logger.info(f"📅 執行自訂時間事件建立：{e_name} {e_start} ~ {e_end}")
            result = create_calendar_event(title=f"[自訂] {e_name}", start=e_start, end=e_end)
            if result["status"] == "success":
                is_all_day = "T00:00:00" in e_start
                start_display = f"{{'date': '{e_start.split('T')[0]}'}}" if is_all_day else f"{{'dateTime': '{e_start}'}}"
                end_display = f"{{'date': '{e_end.split('T')[0]}'}}" if is_all_day else f"{{'dateTime': '{e_end}'}}"
                return (
                    f"✅ 已新增到 Google Calendar\n"
                    f"📌 標題：{e_name}\n"
                    f"🕒 開始：{start_display}\n"
                    f"🕒 結束：{end_display}\n"
                    f"🔗 事件連結：{result['htmlLink']}"
                )
            elif result["status"] == "exists":
                return f"⚠️ 行事曆已有相同事件，已略過新增\n📌 標題：{e_name}\n🔗 事件連結：{result['htmlLink']}"
            else:
                return f"❌ 建立失敗：{result.get('message', '未知錯誤')}"
                
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
                result = create_calendar_event(title=f"[金大] {event_title}", start=iso_start, end=iso_end)
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
請分析這堂課的上課時間字串，並純粹回傳 JSON。
上課時間字串：{schedule_str}

請擷取出：
1. day_of_week: 星期幾 (1-7)
2. start_period: 開始節次 (如第5節則為 5)
3. end_period: 結束節次 (如有波浪號如 5~7 則 end 為 7，否則同 start)
"""

    logger.info(f"📅 呼叫 LLM 解析行事曆時間: {schedule_str}")
    
    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": getattr(config, "OLLAMA_FAST_MODEL", config.OLLAMA_MODEL),
                "prompt": prompt,
                "format": CALENDAR_JSON_SCHEMA,
                "stream": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 1024,
                    "num_predict": 100,
                },
            },
            timeout=15,
        )
        response.raise_for_status()
        result_text = response.json()["response"].strip()
        data = json.loads(result_text)
        
        day_of_week = data["day_of_week"]
        start_p = data["start_period"]
        end_p = data["end_period"]
        
        # 組裝 IsoFormat 日期
        target_date = get_next_weekday(day_of_week)
        date_str = target_date.strftime("%Y-%m-%d")
        
        start_time_str = NQU_PERIOD_START.get(start_p, "08:10")
        end_time_str = NQU_PERIOD_END.get(end_p, "09:00")
        
        iso_start = f"{date_str}T{start_time_str}:00"
        iso_end = f"{date_str}T{end_time_str}:00"
        
        event_title = f"[課程] {course_name} ({teacher})"
        
        # 加上每週重複規則，連上 18 週 (一個正常學期的長度)
        recurrence_rule = ["RRULE:FREQ=WEEKLY;COUNT=18"]
        
        logger.info(f"📅 準備呼叫 Google API：{event_title} {iso_start} ~ {iso_end} (週期：18週)")
        
        # 呼叫 Google API
        result = create_calendar_event(
            title=event_title,
            start=iso_start,
            end=iso_end,
            recurrence=recurrence_rule
        )
        
        if result["status"] == "success":
            return (
                f"✅ 已成功將課程加入 Google Calendar，為您設定為**每週重複 (共18週)**！\n"
                f"📌 課程：{course_name}\n"
                f"🕒 首堂開始：{{'dateTime': '{iso_start}'}}\n"
                f"🕒 首堂結束：{{'dateTime': '{iso_end}'}}\n"
                f"🔗 事件連結：{result['htmlLink']}"
            )
        elif result["status"] == "exists":
            return (
                f"⚠️ 行事曆已有相同事件，已略過新增\n"
                f"📌 標題：{course_name}\n"
                f"🔗 事件連結：{result['htmlLink']}"
            )
        else:
            return f"❌ 建立失敗：{result.get('message', '未知錯誤')}"
            
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
- 語氣友善、簡潔。
- 只需回答與使用者問題最相關的事件時間，不要囉嗦。
- 不用提及「根據資料」、「我找到」等 AI 機器人用語。"""

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": getattr(config, "OLLAMA_FAST_MODEL", config.OLLAMA_MODEL),
                "prompt": prompt,
                "stream": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 1024,
                    "num_predict": 150,
                },
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        logger.error(f"Academic Event LLM 發生錯誤: {e}")
        # Fallback 回覆
        return f"為您找到以下學校行事曆資料：\n{events_str}"

