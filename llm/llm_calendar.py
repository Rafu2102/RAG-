import re
import json
import requests
import logging
from datetime import datetime, timedelta, timezone

import config
from tools.calendar_tool import create_calendar_event, delete_calendar_events, list_calendar_events, update_calendar_event
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
        "schedules": {
            "type": "array",
            "description": "這堂課所有的上課時段列表 (若跨多天則會有多筆)",
            "items": {
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
        }
    },
    "required": ["schedules"]
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
    from llm.date_utils import normalize_chinese_datetime
    
    # 先正規化中文數字（三月16→3月16, 九點半→9點半）
    normalized_query = normalize_chinese_datetime(query)
    logger.info(f"📅 日期正規化：「{query}」→「{normalized_query}」")
    
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    
    # 建立精確的中文星期幾，讓 LLM 在推算相對時間(如下週二)時 100% 精準
    weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
    weekday_str = weekday_map[now.weekday()]
    today_str = f"{now.strftime('%Y-%m-%d')} (星期{weekday_str})"
    tomorrow_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')

    # === 1. 使用 LLM 判斷加入行事曆的意圖 ===
    intent_prompt = f"""你是一個精準的行事曆意圖分析助理。今天是 {today_str}。
使用者要求對行事曆進行操作，請判斷使用者的意圖分類與動作類型，並純粹回傳 JSON。

## 動作類型 (action_type)
- "add"：新增、加入、排進、設定提醒
- "remove"：刪除、移除、取消
- "list"：查看、列出、我的行事曆、有什麼行程
- "update"：修改、變更、更改時間

## 分類選項 (intent_type)
- "custom_event"：使用者已給出具體日期或時間（如「5月29是我生日」「明天下午三點開會」「下週三第5節小考」）。不管什麼事件，只要有給時間 = custom_event。注意「第X節」也算有給時間！
- "course_schedule_event"：提到課程名稱 + 考試活動（期中考/期末考/小考/報告），但完全沒給日期。例：「微積分期中考加到行事曆」
- "academic_event"：學校行政事件（加退選/停修/開學/宿舍），且沒給日期。例：「停修日加到行事曆」
- "weekly_course"：把某課程的「每週上課時間」加到行事曆。例：「把線性代數加到日曆」

## 欄位填寫規則
1. **event_name**（必填）：提取事件核心名稱，去掉時間和「加到行事曆」等詞。
   範例：「明天九點十分要上通識課加到行事曆」→ "通識課"
   範例：「刪除三月十四號九點的」→ "*"（刪除/查詢時若無明確名稱，填 "*"）
2. **start_dt / end_dt**（add 必填，remove/update 盡量填）：
   - 必須是完整 YYYY-MM-DDTHH:MM:SS！
   - 「明天上午十點」→ "{tomorrow_str}T10:00:00"
   - 「第5節到第7節」→ 用金大節次表：第1節=08:10, 第2節=09:10, 第3節=10:10, 第4節=11:10, 第5節=13:30, 第6節=14:30, 第7節=15:30, 第8節=16:30
   - 刪除時若有時間（如「刪除明天九點的」），start_dt 也要填！
3. **target_date**（list/remove/update 時填寫）：
   - 格式 YYYY-MM-DD，用於定位特定日期
   - 「我下週五有什麼行程」→ 計算出下週五日期
   - 「刪除明天的行事曆」→ "{tomorrow_str}"
4. **course_name / schedule_keyword**：僅 course_schedule_event 時填寫

使用者輸入：{normalized_query}

請純粹回傳 JSON。"""

    INTENT_SCHEMA = {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["add", "remove", "list", "update"],
                "description": "判斷使用者是要新增、刪除、列出還是修改事件"
            },
            "intent_type": {
                "type": "string",
                "enum": ["custom_event", "course_schedule_event", "academic_event", "weekly_course"]
            },
            "event_name": {"type": "string", "description": "事件核心名稱，刪除/查詢時若無明確名稱填 *"},
            "course_name": {"type": "string", "description": "課程名稱 (僅 course_schedule_event 時填寫)"},
            "schedule_keyword": {"type": "string", "description": "搜尋活動關鍵字如期中考/期末考 (僅 course_schedule_event 時填寫)"},
            "start_dt": {"type": "string", "description": "YYYY-MM-DDTHH:MM:SS（add 必填，remove/update 若有時間線索也填）"},
            "end_dt": {"type": "string", "description": "YYYY-MM-DDTHH:MM:SS（add 必填）"},
            "target_date": {"type": "string", "description": "YYYY-MM-DD 用於 list/remove/update 定位特定日期"}
        },
        "required": ["action_type", "intent_type", "event_name"]
    }

    try:
        logger.info("📅 正在呼叫 LLM 判斷加入行事曆的精確意圖（8B 模型）...")
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,  # 使用 8B 主模型，提高意圖判斷準確度
                "prompt": intent_prompt,
                "format": INTENT_SCHEMA,
                "stream": False,
                "keep_alive": 0,  # 用完立即卸載，為後續騰出 VRAM
                "options": {"temperature": 0.0, "num_ctx": 1024, "num_predict": 250},
            },
            timeout=25,  # 8B 較慢，給多一點時間
        )
        response.raise_for_status()
        res_data = json.loads(response.json()["response"].strip())
        
        action_type = res_data.get("action_type", "add")
        intent_type = res_data.get("intent_type", "weekly_course")
        e_name = res_data.get("event_name", "").strip()
        
        # 防呆：若 LLM 回傳空的 event_name，從原始 query 提取
        if not e_name:
            import re as _re
            # 先移除時間表達式（九點十分、十一點半、下午三點等）
            e_name = _re.sub(r'[一二三四五六七八九十\d]+點[十二三四五六七八九\d]*分?半?', '', normalized_query)
            # 移除日期/時間/行事曆關鍵字（含斜線日期如 3/16）
            e_name = _re.sub(r'(明天|今天|後天|大後天|下週.?|這週.?|\d+[/月]\d+[日號]?|上午|下午|晚上|早上)', '', e_name)
            # 增強版：過濾各種口語冗言贅字，避免「明天是阿澄澄痢疾日請當我加到行事曆」抓到整句話
            e_name = _re.sub(r'(第[一二三四五六七八九十\d]+節|到|~|請|當我|幫我|加到|新增|行事曆|日曆|提醒|裡面|裡|我的|要上|要去|把|的|刪除|刪掉|移除|取消|修改|變更|查看|列出|可以|嗎|是|可不可以)', '', e_name)
            # 清理殘留的單字「我」和多餘空白
            e_name = _re.sub(r'^[我你他她們\s]+|[我你他她們\s]+$', '', e_name).strip()
            if not e_name:
                if action_type == "add":
                    e_name = query  # add 操作若被洗空，才退回原始 query
                else:
                    e_name = "*"  # 刪除/查詢時無法提取 → 通配
            res_data["event_name"] = e_name
            logger.warning(f"⚠️ LLM 未回傳 event_name，自動提取：{e_name}")
        
        # 【Fix 7 最後防線】remove/update 時若 event_name 仍是垃圾（含動作詞），設為 *
        if action_type in ("remove", "update", "list"):
            _action_words = ["刪除", "刪掉", "移除", "取消", "修改", "變更", "查看", "列出", "行事曆", "日曆"]
            if any(w in e_name for w in _action_words):
                logger.warning(f"⚠️ event_name「{e_name}」含動作詞，重設為通配 *")
                e_name = "*"
                res_data["event_name"] = "*"
        
        logger.info(f"📅 意圖判定：{action_type} - {intent_type} (目標：{e_name})")

    except Exception as e:
        logger.warning(f"行事曆意圖分析失敗，退回預設每週課程流程：{e}")
        return {
            "action_type": "add",
            "intent_type": "weekly_course",
            "event_name": query
        }

    return res_data

# === 教學進度表日期萃取 ===

def _extract_date_from_schedule(chunks: list, keyword: str) -> dict | None:
    """
    從 RAG 檢索到的教學進度表中，用正則萃取指定活動（如期中考）的日期。
    教學進度表格式: 第9週課程 (2025/11/09─2025/11/15)：期中考
    """
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


def execute_calendar_action(query: str, intent_data: dict, retrieved_chunks: list = None, discord_id: str = None) -> str:
    """根據已經判斷出的意圖資料，實際執行 Google Calendar API 的增刪操作"""
    action_type = intent_data.get("action_type", "add")
    intent_type = intent_data.get("intent_type", "weekly_course")
    e_name = intent_data.get("event_name", query)
    target_date = intent_data.get("target_date")   # YYYY-MM-DD（list/remove/update 用）
    start_dt = intent_data.get("start_dt")          # YYYY-MM-DDTHH:MM:SS

    # === 【Fix 5】未註冊保護：提前檢測 ===
    if not discord_id:
        return "❌ 無法識別您的身分，請先使用 `/identity_login` 進行註冊與 Google 帳號綁定。"
    
    from tools.calendar_tool import get_user_token_path
    if not get_user_token_path(discord_id).exists():
        return "❌ 您尚未綁定 Google 帳號喔！請先使用 `/identity_login` 指令完成註冊與行事曆授權。"

    try:
        # === [流程 0a] 刪除行事曆事件 ===
        if action_type == "remove":
            # 優先用 start_dt，其次用 target_date 轉換，最後用名稱
            search_time = start_dt
            if not search_time and target_date:
                search_time = f"{target_date}T00:00:00"  # 用全天範圍搜尋
            logger.info(f"📅 執行行事曆移除：名稱「{e_name}」，時間「{search_time}」，日期「{target_date}」")
            result = delete_calendar_events(discord_id=discord_id, keyword=e_name, target_time=search_time)
            return result["message"]
        
        # === [流程 0b] 列出行事曆事件 ===
        if action_type == "list":
            # 支持指定日期查詢（如「我下週五有什麼行程」）
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
                    # 顯示時間而不只是日期
                    start_str = ev["start"]
                    if "T" in start_str:
                        time_part = start_str.split("T")[1][:5]  # HH:MM
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
        
        # === [流程 0c] 修改行事曆事件 ===
        if action_type == "update":
            new_title = intent_data.get("event_name", "")
            new_start = start_dt
            new_end = intent_data.get("end_dt")
            # 定位用的時間：用 target_date 或 start_dt
            search_time = start_dt if start_dt else (f"{target_date}T00:00:00" if target_date else None)
            logger.info(f"📅 執行行事曆修改：關鍵字「{e_name}」，定位時間「{search_time}」")
            result = update_calendar_event(
                discord_id=discord_id, keyword=e_name,
                new_title=new_title if new_title and new_title != "*" else None,
                new_start=new_start if new_start else None,
                new_end=new_end if new_end else None,
                target_time=search_time
            )
            return result["message"]
    
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
        
        # 若有 start 但沒 end，自動 +2 小時（或全天事件）
        if e_start and not e_end:
            try:
                from tools.calendar_tool import _parse_dt
                _s = _parse_dt(e_start)
                if _s.hour == 0 and _s.minute == 0:
                    e_end = e_start  # 全天事件，start == end
                else:
                    e_end = (_s + timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S')
            except Exception:
                e_end = e_start
        
        # 標題清理：避免用 * 或原始 query 做標題
        title = e_name if e_name not in ("*", "") else query
        
        if e_start and e_end:
            logger.info(f"📅 執行自訂時間事件建立：{title} {e_start} ~ {e_end}")
            result = create_calendar_event(discord_id=discord_id, title=f"[自訂] {title}", start=e_start, end=e_end)
            if result["status"] == "success":
                is_all_day = "T00:00:00" in e_start
                start_display = f"{{'date': '{e_start.split('T')[0]}'}}" if is_all_day else f"{{'dateTime': '{e_start}'}}"
                end_display = f"{{'date': '{e_end.split('T')[0]}'}}" if is_all_day else f"{{'dateTime': '{e_end}'}}"
                return (
                    f"✅ 已新增到 Google Calendar\n"
                    f"📌 標題：{title}\n"
                    f"🕒 開始：{start_display}\n"
                    f"🕒 結束：{end_display}\n"
                    f"🔗 事件連結：{result['htmlLink']}"
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

請擷取出每一個時段的：
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
        
        schedules = data.get("schedules", [])
        if not schedules:
             return f"😅 解析失敗：【{course_name}】雖然有上課時間字串 ({schedule_str})，但我無法辨識其內容，無法為您加到行事曆。"
             
        event_title = f"[課程] {course_name} ({teacher})"
        recurrence_rule = ["RRULE:FREQ=WEEKLY;COUNT=18"]
        
        success_count = 0
        failure_count = 0
        last_success_link = ""
        last_error_msg = ""
        
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
            
            logger.info(f"📅 準備呼叫 Google API：{event_title} {iso_start} ~ {iso_end} (週期：18週)")
            
            # 呼叫 Google API
            result = create_calendar_event(
                discord_id=discord_id,
                title=event_title,
                start=iso_start,
                end=iso_end,
                recurrence=recurrence_rule
            )
            
            if result["status"] == "success":
                success_count += 1
                last_success_link = result['htmlLink']
            elif result["status"] == "exists":
                success_count += 1
                last_success_link = result['htmlLink']
            else:
                failure_count += 1
                last_error_msg = result.get('message', '未知錯誤')
        
        if success_count > 0:
            return (
                f"✅ 已成功將課程加入 Google Calendar，為您設定為**每週重複 (共18週)**！\n"
                f"📌 課程：{course_name}\n"
                f"🕒 共新增了 {success_count} 個上課時段\n"
                f"🔗 前往查看行事曆：{last_success_link}"
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

