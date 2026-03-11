# tools/calendar_tool.py
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# 使用 config 裡面的 TIMEZONE，如果沒有就預設 Asia/Taipei
try:
    from config import TIMEZONE
except ImportError:
    TIMEZONE = "Asia/Taipei"

SCOPES = ["https://www.googleapis.com/auth/calendar"]
BOT_SOURCE_TAG = "NQU_agent"

# 指向 tools 目錄下的憑證檔
BASE_DIR = Path(__file__).resolve().parent
TOKEN_PATH = BASE_DIR / "token.json"
CREDS_PATH = BASE_DIR / "credentials.json"

TAIPEI_TZ = timezone(timedelta(hours=8), name="Asia/Taipei")

def get_service():
    """取得 Google Calendar API Service，如果沒有 token 會自動開網頁要求授權"""
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_PATH.exists():
                raise FileNotFoundError(f"Google 驗證檔 {CREDS_PATH} 不存在。請先將你的 credentials.json 放入 tools 資料夾。")
            
            # port=0 會自動找一個空閒的 port，並打開瀏覽器進行 Oauth 登入
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
            
        # 寫入 token 供未來使用
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        
    return build("calendar", "v3", credentials=creds)

def _parse_dt(s: str) -> datetime:
    """解析 ISO datetime，沒有提供時區的話固定補上台北時區"""
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

def create_calendar_event(title: str, start: str, end: str, calendar_id: str = "primary", recurrence: list[str] = None):
    """
    建立 Google 行事曆事件
    若參數為單純日期 (YYYY-MM-DD)，會建立全天事件。
    """
    service = get_service()

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

        # Google API 裡的 end_date 必須是 Exclusive (不包含結尾當天)，所以如果相同要 +1 天
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

    # 如果有週期性規則，將其加入 payload
    if recurrence:
        body["recurrence"] = recurrence

    created = service.events().insert(calendarId=calendar_id, body=body).execute()

    return {
        "status": "success",
        "message": f"成功將【{title}】加入行事曆！",
        "eventId": created.get("id"),
        "htmlLink": created.get("htmlLink")
    }

def delete_calendar_events(keyword: str, calendar_id: str = "primary") -> dict:
    """
    找出標題含有 keyword 且是防呆來源(NQU_agent)的未來事件並刪除。
    用來處理使用者要求「刪除」或「移除」指定課程或事件的情況。
    """
    service = get_service()
    # 為了安全與實用性，預設只搜尋/刪除從今天起未來的事件
    now_dt = datetime.now(TAIPEI_TZ)
    # 取今天的 00:00:00
    start_of_today_str = _to_utc_rfc3339(datetime.combine(now_dt.date(), datetime.min.time(), tzinfo=TAIPEI_TZ))
    
    try:
        resp = service.events().list(
            calendarId=calendar_id,
            q=keyword,
            timeMin=start_of_today_str,
            singleEvents=True,
            orderBy="startTime",
            maxResults=50,
        ).execute()

        deleted_count = 0
        skipped_count = 0
        deleted_titles = set()
        
        for ev in resp.get("items", []):
            # 為了防止誤刪使用者自己的私人事件，嚴格檢查 Tag
            if _get_event_source_tag(ev) == BOT_SOURCE_TAG:
                service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
                deleted_titles.add(ev.get("summary", "未知事件"))
                deleted_count += 1
            else:
                skipped_count += 1
                
        if deleted_count > 0:
            titles_str = "、".join(deleted_titles)
            return {
                "status": "success",
                "message": f"✅ 已成功為您從行事曆中清理了 {deleted_count} 筆相關行程：{titles_str}"
            }
        else:
            if skipped_count > 0:
                return {
                    "status": "not_found",
                    "message": f"⚠️ 我有在行事曆找到包含「{keyword}」的行程，但它們似乎是您私下建立的，為了安全起見，小助手不會越權刪除喔！"
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"❌ 抱歉，在您未來的行事曆中，找不到名稱包含「{keyword}」且是由我建立的行程！"
                }
                
    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ 刪除行事曆時發生錯誤：{str(e)}"
        }
