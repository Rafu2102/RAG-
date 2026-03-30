# tools/calendar_api.py — Google Calendar CRUD 模組
# ================================================
# 負責 Google Calendar 事件的建立、刪除、列出、修改
# ================================================
from __future__ import annotations

import re
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# 使用 config 裡面的 TIMEZONE，如果沒有就預設 Asia/Taipei
try:
    from config import TIMEZONE
except ImportError:
    TIMEZONE = "Asia/Taipei"

from tools.auth import get_service

BOT_SOURCE_TAG = "NQU_agent"
TAIPEI_TZ = timezone(timedelta(hours=8), name="Asia/Taipei")

# 事件標題前綴（建立時會加上）
_TITLE_PREFIXES = ["[自訂]", "[課程]", "[學校]"]

def _title_matches_keyword(ev_title: str, keyword: str) -> bool:
    """
    智慧標題匹配：將事件標題去掉 [自訂]/[課程] 等前綴後，
    比對是否包含 keyword（不區分大小寫）。
    例：ev_title="[自訂] 數學課"  keyword="數學課" → True
    """
    if keyword == "*":
        return True
    title_clean = ev_title
    for prefix in _TITLE_PREFIXES:
        title_clean = title_clean.replace(prefix, "").strip()
    kw_lower = keyword.lower()
    
    if kw_lower in title_clean.lower() or kw_lower in ev_title.lower():
        return True
        
    # 如果關鍵字包含「與、跟、和、、」等連接詞，拆解並允許部分匹配
    import re
    parts = re.split(r'跟|與|和|及|、|,|\s+', kw_lower)
    parts = [p.strip() for p in parts if len(p.strip()) >= 2] # 關鍵字至少需 2 字防誤刪
    if parts:
        for p in parts:
            if p in title_clean.lower() or p in ev_title.lower():
                return True
                
    return False


# === 日期時間工具 ===

def _parse_dt(s: str) -> datetime:
    """解析 ISO datetime，支援時間-only 字串的防呆處理"""
    s = s.strip()
    
    # 防呆：若 LLM 只回傳時間如 "10:00" 或 "10:00:00"，自動補上明天日期
    if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s):
        tomorrow = datetime.now(TAIPEI_TZ).date() + timedelta(days=1)
        s = f"{tomorrow.isoformat()}T{s}"
        if s.count(":") == 1:
            s += ":00"
        logger.warning(f"⚠️ _parse_dt 收到時間-only 字串，自動補上明天日期: {s}")
    
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TAIPEI_TZ)
    return dt

def _to_utc_rfc3339(dt: datetime) -> str:
    """轉為 UTC + RFC3339 格式供 Google API 原生使用"""
    dt_utc = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt_utc.isoformat().replace("+00:00", "Z")

def _get_event_source_tag(ev: dict) -> str | None:
    return (
        ev.get("extendedProperties", {})
        .get("private", {})
        .get("source")
    )

def find_duplicate_event(service, calendar_id: str, title: str, time_min: str, time_max: str) -> dict | None:
    """防呆：在指定的區間尋找是否已經有來源為 NQU_agent 的同樣標題行程"""
    resp = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
        maxResults=100,
    ).execute()

    for ev in resp.get("items", []):
        if ev.get("summary") != title:
            continue
        if _get_event_source_tag(ev) == BOT_SOURCE_TAG:
            return ev

    return None


# === 行事曆 CRUD：建立 (Create) ===

def create_calendar_event(discord_id: str, title: str, start: str, end: str, calendar_id: str = "primary", recurrence: list[str] = None):
    """
    建立 Google 行事曆事件
    若參數為單純日期 (YYYY-MM-DD)，會建立全天事件。
    """
    service = get_service(discord_id)

    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)

    is_all_day_like = (
        start_dt.hour == 0 and start_dt.minute == 0 and start_dt.second == 0
        and end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0
    )

    # 處理全天事件 (只有日期)
    if is_all_day_like:
        start_date = start_dt.date()
        end_date = end_dt.date()

        if end_date <= start_date:
            end_date = start_date + timedelta(days=1)
        else:
            end_date = end_date + timedelta(days=1)

        time_min = _to_utc_rfc3339(datetime.combine(start_date, datetime.min.time(), tzinfo=TAIPEI_TZ))
        time_max = _to_utc_rfc3339(datetime.combine(end_date, datetime.min.time(), tzinfo=TAIPEI_TZ))

        dup = find_duplicate_event(service, calendar_id, title, time_min, time_max)
        if dup:
            return {
                "status": "exists",
                "message": f"【{title}】在此時段已存在於行事曆 (防重複建立)",
                "eventId": dup.get("id"),
                "htmlLink": dup.get("htmlLink")
            }

        body = {
            "summary": title,
            "start": {"date": start_date.isoformat()},
            "end": {"date": end_date.isoformat()},
            "extendedProperties": {"private": {"source": BOT_SOURCE_TAG}},
        }
    
    # 處理特定時間事件
    else:
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(hours=1)

        time_min = _to_utc_rfc3339(start_dt)
        time_max = _to_utc_rfc3339(end_dt)

        dup = find_duplicate_event(service, calendar_id, title, time_min, time_max)
        if dup:
            return {
                "status": "exists",
                "message": f"【{title}】在此時段已存在於行事曆 (防重複建立)",
                "eventId": dup.get("id"),
                "htmlLink": dup.get("htmlLink")
            }

        body = {
            "summary": title,
            "start": {"dateTime": start_dt.replace(microsecond=0).isoformat(), "timeZone": TIMEZONE},
            "end": {"dateTime": end_dt.replace(microsecond=0).isoformat(), "timeZone": TIMEZONE},
            "extendedProperties": {"private": {"source": BOT_SOURCE_TAG}},
        }

    if recurrence:
        body["recurrence"] = recurrence

    created = service.events().insert(calendarId=calendar_id, body=body).execute()
    logger.info(f"✅ 行事曆建立成功 | 標題：{title} | ID: {discord_id} | eventId: {created.get('id')}")

    return {
        "status": "success",
        "message": f"成功將【{title}】加入行事曆！",
        "eventId": created.get("id"),
        "htmlLink": created.get("htmlLink")
    }


# === 行事曆 CRUD：刪除 (Delete) ===

def delete_calendar_events(discord_id: str, keyword: str, calendar_id: str = "primary", target_time: str = None, target_date: str = None, include_past: bool = False) -> dict:
    """
    找出標題含有 keyword 且是防呆來源(NQU_agent)的未來事件並刪除。
    """
    service = get_service(discord_id)
    now_dt = datetime.now(TAIPEI_TZ)
    start_of_today_str = _to_utc_rfc3339(datetime.combine(now_dt.date(), datetime.min.time(), tzinfo=TAIPEI_TZ))
    
    try:
        def _do_search(t_min, t_max):
            resp = service.events().list(
                calendarId=calendar_id,
                privateExtendedProperty=f"source={BOT_SOURCE_TAG}",
                timeMin=t_min,
                timeMax=t_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=250,
            ).execute()
            
            d_count = 0
            s_count = 0
            d_titles = set()
            is_wild = (keyword == "*")
            is_clear = (keyword == "__CLEAR_ALL__")
            
            for ev in resp.get("items", []):
                if _get_event_source_tag(ev) != BOT_SOURCE_TAG:
                    s_count += 1
                    continue
                
                ev_title = ev.get("summary", "")
                if not is_wild and not is_clear and not _title_matches_keyword(ev_title, keyword):
                    continue
                
                service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
                d_titles.add(ev_title or "未知事件")
                d_count += 1
                
                if is_wild:
                    logger.warning(f"⚠️ keyword 為通配符 *，僅刪除第一筆匹配事件以防止誤刪")
                    break
            
            return d_count, s_count, d_titles


        # Initial exact search
        time_min_str = start_of_today_str
        time_max_str = None
        
        if target_date and str(target_date).lower() not in ("null", "none"):
            d = datetime.fromisoformat(target_date[:10]).date()
            time_min_str = _to_utc_rfc3339(datetime.combine(d, datetime.min.time(), tzinfo=TAIPEI_TZ))
            time_max_str = _to_utc_rfc3339(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=TAIPEI_TZ))
        elif target_time:
            target_dt = _parse_dt(target_time)
            time_min_str = _to_utc_rfc3339(target_dt - timedelta(hours=1))
            time_max_str = _to_utc_rfc3339(target_dt + timedelta(hours=1))
        elif include_past:
            time_min_str = _to_utc_rfc3339(datetime.combine(now_dt.date() - timedelta(days=180), datetime.min.time(), tzinfo=TAIPEI_TZ))
            
        deleted_count, skipped_count, deleted_titles = _do_search(time_min_str, time_max_str)
        
        # 智慧回退 (Smart Fallback)：如果限定時間找不到，且使用者給了具體的標題（非 wildcard），則放寬上下半年搜尋
        if deleted_count == 0 and (target_date or target_time) and keyword not in ("*", "__CLEAR_ALL__"):
            logger.info(f"⚠️ 在 {target_date or target_time} 找不到「{keyword}」，啟動智慧回退，搜尋全時段...")
            t_min = _to_utc_rfc3339(datetime.combine(now_dt.date() - timedelta(days=180), datetime.min.time(), tzinfo=TAIPEI_TZ))
            t_max = _to_utc_rfc3339(datetime.combine(now_dt.date() + timedelta(days=180), datetime.min.time(), tzinfo=TAIPEI_TZ))
            deleted_count, skipped_count, deleted_titles = _do_search(t_min, t_max)
                
        if deleted_count > 0:
            titles_str = "、".join(deleted_titles)
            logger.info(f"✅ 行事曆刪除成功 | ID: {discord_id} | 共 {deleted_count} 筆 | {titles_str}")
            return {
                "status": "success",
                "message": f"✅ 已成功為您從行事曆中清理了 {deleted_count} 筆相關行程：{titles_str}"
            }
        else:
            search_desc = f"時間 {target_time}" if target_time else f"名稱「{keyword}」"
            if skipped_count > 0:
                logger.info(f"⚠️ 行事曆刪除跳過 | ID: {discord_id} | {search_desc} | 找到 {skipped_count} 筆但非機器人建立")
                return {
                    "status": "not_found",
                    "message": f"⚠️ 我有在行事曆找到符合 {search_desc} 的行程，但它們似乎是您私下建立的，為了安全起見，小助手不會越權刪除喔！"
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"❌ 抱歉，在您未來的行事曆中，找不到符合 {search_desc} 且是由我建立的行程！"
                }
                
    except Exception as e:
        logger.error(f"❌ 行事曆刪除錯誤 | ID: {discord_id} | {e}")
        return {
            "status": "error",
            "message": f"❌ 刪除行事曆時發生錯誤：{str(e)}"
        }


# === 行事曆 CRUD：讀取 (List) ===

def list_calendar_events(discord_id: str, days: int = 7, calendar_id: str = "primary", target_date: str = None) -> dict:
    """列出使用者的行事曆事件。"""
    try:
        service = get_service(discord_id)
        now_dt = datetime.now(TAIPEI_TZ)
        
        if target_date and str(target_date).lower() not in ("null", "none"):
            from datetime import date as _date
            d = _date.fromisoformat(target_date[:10])
            time_min = _to_utc_rfc3339(datetime.combine(d, datetime.min.time(), tzinfo=TAIPEI_TZ))
            time_max = _to_utc_rfc3339(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=TAIPEI_TZ))
        else:
            time_min = _to_utc_rfc3339(now_dt)
            time_max = _to_utc_rfc3339(now_dt + timedelta(days=days))
        
        resp = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=30,
        ).execute()
        
        events = []
        for ev in resp.get("items", []):
            start = ev.get("start", {})
            end = ev.get("end", {})
            is_bot = _get_event_source_tag(ev) == BOT_SOURCE_TAG
            events.append({
                "id": ev.get("id"),
                "title": ev.get("summary", "無標題"),
                "start": start.get("dateTime", start.get("date", "")),
                "end": end.get("dateTime", end.get("date", "")),
                "is_bot_created": is_bot,
                "htmlLink": ev.get("htmlLink", "")
            })
        
        logger.info(f"📅 行事曆讀取 | ID: {discord_id} | 未來 {days} 天 | 共 {len(events)} 筆事件")
        return {
            "status": "success",
            "events": events,
            "message": f"找到 {len(events)} 筆未來 {days} 天內的行事曆事件"
        }
    except Exception as e:
        logger.error(f"❌ 行事曆讀取錯誤 | ID: {discord_id} | {e}")
        return {
            "status": "error",
            "events": [],
            "message": f"❌ 讀取行事曆失敗：{str(e)}"
        }


# === 行事曆 CRUD：修改 (Update) ===

def update_calendar_event(discord_id: str, keyword: str, new_title: str = None, new_start: str = None, new_end: str = None, calendar_id: str = "primary", target_time: str = None, target_date: str = None) -> dict:
    """修改行事曆中由機器人建立的事件。"""
    try:
        service = get_service(discord_id)
        now_dt = datetime.now(TAIPEI_TZ)
        
        def _do_update_search(t_min, t_max):
            resp = service.events().list(
                calendarId=calendar_id,
                privateExtendedProperty=f"source={BOT_SOURCE_TAG}",
                timeMin=t_min,
                timeMax=t_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=50,
            ).execute()
            
            u_count = 0
            u_link = ""
            is_wild = (keyword == "*")
            
            for ev in resp.get("items", []):
                if _get_event_source_tag(ev) != BOT_SOURCE_TAG:
                    continue
                
                ev_title = ev.get("summary", "")
                if not is_wild and not _title_matches_keyword(ev_title, keyword):
                    continue
                
                if new_title:
                    ev["summary"] = new_title
                if new_start:
                    start_dt = _parse_dt(new_start)
                    old_start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
                    old_end = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date")
                    
                    old_duration = timedelta(hours=2)
                    if old_start and old_end:
                        try:
                            os_dt = datetime.fromisoformat(str(old_start).replace('Z', '+00:00'))
                            oe_dt = datetime.fromisoformat(str(old_end).replace('Z', '+00:00'))
                            old_duration = oe_dt - os_dt
                            if old_duration < timedelta(minutes=1):
                                old_duration = timedelta(hours=2)
                        except Exception:
                            pass
                            
                    ev.pop("start", None)
                    ev["start"] = {"dateTime": start_dt.replace(microsecond=0).isoformat(), "timeZone": TIMEZONE}
                    
                    if not new_end:
                        end_dt = start_dt + old_duration
                        ev.pop("end", None)
                        ev["end"] = {"dateTime": end_dt.replace(microsecond=0).isoformat(), "timeZone": TIMEZONE}

                if new_end:
                    end_dt = _parse_dt(new_end)
                    ev.pop("end", None)
                    ev["end"] = {"dateTime": end_dt.replace(microsecond=0).isoformat(), "timeZone": TIMEZONE}
                
                try:
                    updated_ev = service.events().update(calendarId=calendar_id, eventId=ev["id"], body=ev).execute()
                    u_count += 1
                    u_link = updated_ev.get("htmlLink")
                    logger.info(f"✅ 行事曆修改成功 | ID: {discord_id} | {ev.get('summary')} | eventId: {ev['id']}")
                    break
                except Exception as update_err:
                    logger.error(f"⚠️ 更新單一事件失敗 {ev['id']}: {update_err}")
            
            return u_count, u_link

        # Initial exact search
        time_min_str = _to_utc_rfc3339(datetime.combine(now_dt.date(), datetime.min.time(), tzinfo=TAIPEI_TZ))
        time_max_str = None
        
        if target_date and str(target_date).lower() not in ("null", "none"):
            from datetime import date as _date
            d = _date.fromisoformat(target_date[:10])
            time_min_str = _to_utc_rfc3339(datetime.combine(d, datetime.min.time(), tzinfo=TAIPEI_TZ))
            time_max_str = _to_utc_rfc3339(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=TAIPEI_TZ))
        elif target_time:
            target_dt = _parse_dt(target_time)
            time_min_str = _to_utc_rfc3339(target_dt - timedelta(hours=1))
            time_max_str = _to_utc_rfc3339(target_dt + timedelta(hours=1))
            
        updated_count, updated_link = _do_update_search(time_min_str, time_max_str)
        
        # 智慧回退 (Smart Fallback)：如果限定時間找不到，啟動半年大範圍搜尋
        if updated_count == 0 and (target_date or target_time) and keyword != "*":
            logger.info(f"⚠️ 在 {target_date or target_time} 找不到「{keyword}」以進行修改，啟動智慧回退，搜尋全時段...")
            t_min = _to_utc_rfc3339(datetime.combine(now_dt.date() - timedelta(days=180), datetime.min.time(), tzinfo=TAIPEI_TZ))
            t_max = _to_utc_rfc3339(datetime.combine(now_dt.date() + timedelta(days=180), datetime.min.time(), tzinfo=TAIPEI_TZ))
            updated_count, updated_link = _do_update_search(t_min, t_max)
            
        search_desc = f"時間 {target_time}" if target_time else f"名稱「{keyword}」"
        if updated_count > 0:
            return {
                "status": "success",
                "message": f"✅ 已成功為您修改了行事曆中符合 {search_desc} 的事件",
                "htmlLink": updated_link
            }
        else:
            return {
                "status": "not_found",
                "message": f"❌ 找不到由機器人建立且符合 {search_desc} 的事件"
            }
    except Exception as e:
        logger.error(f"❌ 行事曆修改錯誤 | ID: {discord_id} | {e}")
        return {
            "status": "error",
            "message": f"❌ 修改行事曆時發生錯誤：{str(e)}"
        }
