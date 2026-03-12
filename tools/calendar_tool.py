# tools/calendar_tool.py
from __future__ import annotations

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
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
CREDS_PATH = BASE_DIR / "credentials.json"
TOKENS_DIR = BASE_DIR / "tokens"
TOKENS_DIR.mkdir(exist_ok=True)

TAIPEI_TZ = timezone(timedelta(hours=8), name="Asia/Taipei")

# === 授權與身分模組 ===

# 【PKCE 關鍵】快取 Flow 物件，保存 code_verifier
# get_auth_url() 產生的 Flow 內含 code_verifier (PKCE)，
# 如果 verify_and_save_token() 建立「全新」的 Flow，就會遺失 code_verifier，
# 導致 Google 回報 "Missing code verifier" (invalid_grant)。
_pending_flows: dict[str, "Flow"] = {}

def get_user_token_path(discord_id: str) -> Path:
    """取得特定使用者的 Token 檔案路徑"""
    return TOKENS_DIR / f"{discord_id}_token.json"

def get_auth_url(discord_id: str) -> str:
    """產生給使用者點擊的 Google 授權網址，並快取 Flow 物件以保留 PKCE code_verifier"""
    import os
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
    
    flow = Flow.from_client_secrets_file(
        str(CREDS_PATH),
        scopes=SCOPES,
        redirect_uri="http://localhost"
    )
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
    
    # 將 Flow 存入快取，等 verify_and_save_token 時取回
    _pending_flows[discord_id] = flow
    return auth_url

def verify_and_save_token(discord_id: str, auth_response_url: str, department: str, grade: str, class_group: str = ""):
    """【升級版】接收網址、科系、年級、班級分組，並合併儲存為使用者的數位身分"""
    import os
    from urllib.parse import urlparse, parse_qs
    
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
    
    # 【PKCE 核心】取回與 auth_url 相同的 Flow 物件（內含 code_verifier）
    flow = _pending_flows.pop(discord_id, None)
    if flow is None:
        # 如果快取已過期或不存在，備用建立新 Flow（不含 PKCE，可能失敗）
        flow = Flow.from_client_secrets_file(
            str(CREDS_PATH),
            scopes=SCOPES,
            redirect_uri="http://localhost"
        )
    
    # 從 URL 解析 code，繞過 OAuth2Session 的 MismatchingStateError 檢查
    parsed = urlparse(auth_response_url)
    qs = parse_qs(parsed.query)
    code = qs.get("code", [None])[0]
    
    if code:
        flow.fetch_token(code=code)
    else:
        flow.fetch_token(authorization_response=auth_response_url)
        
    creds = flow.credentials
    creds_dict = json.loads(creds.to_json())
    
    # 【保留 nickname】若使用者重新註冊，保留舊的 nickname
    existing_nickname = ""
    token_path = get_user_token_path(discord_id)
    if token_path.exists():
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                existing_nickname = old_data.get("profile", {}).get("nickname", "")
        except Exception:
            pass
    
    # 科系簡稱 → 完整名稱映射
    _DEPT_MAP = {
        "資工系": "資訊工程學系", "電機系": "電機工程學系",
        "土木系": "土木與工程管理學系", "食品系": "食品科學系",
        "企管系": "企業管理學系", "觀光系": "觀光管理學系",
        "運休系": "運動與休閒學系", "工管系": "工業工程與管理學系",
        "國際系": "國際暨大陸事務學系", "建築系": "建築學系",
        "海邊系": "海洋與邊境管理學系", "應英系": "應用英語學系",
        "華語系": "華語文學系", "都景系": "都市計畫與景觀學系",
        "護理系": "護理學系", "長照系": "長期照護學系",
        "社工系": "社會工作學系", "通識中心": "通識教育中心",
    }
    dept_full = _DEPT_MAP.get(department, department)
    
    profile = {
        "department": department,
        "department_full": dept_full,
        "grade": grade,
        "class_group": class_group,
    }
    if existing_nickname:
        profile["nickname"] = existing_nickname
    
    user_data = {
        "profile": profile,
        "credentials": creds_dict
    }
    
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=4)
    
    group_label = f" {class_group}班" if class_group else ""
    logger.info(f"✅ Token 儲存成功 | ID: {discord_id} | {department} {grade}年級{group_label}")
    return True

def get_user_profile(discord_id: str) -> dict | None:
    """【新增】讀取使用者的基本身分資料"""
    token_path = get_user_token_path(discord_id)
    if not token_path.exists():
        return None
    
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("profile", None)
    except Exception:
        return None

def delete_user_token(discord_id: str) -> bool:
    """刪除使用者的註冊資料與 Token 檔案"""
    token_path = get_user_token_path(discord_id)
    if token_path.exists():
        token_path.unlink()
        return True
    return False

def get_service(discord_id: str):
    """【升級版】根據 discord_id 讀取合併後的 JSON，並還原 Google Service"""
    token_path = get_user_token_path(discord_id)
    if not token_path.exists():
        raise ValueError("尚未註冊！請先使用 `/calendar_login` 指令綁定身分與 Google 帳號。")

    with open(token_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)
        
    creds_dict = user_data.get("credentials")
    if not creds_dict:
        raise ValueError("憑證格式錯誤，請重新授權。")

    creds = Credentials.from_authorized_user_info(creds_dict, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                user_data["credentials"] = json.loads(creds.to_json())
                with open(token_path, "w", encoding="utf-8") as f:
                    json.dump(user_data, f, ensure_ascii=False, indent=4)
                logger.info(f"🔄 Token 已自動刷新 | ID: {discord_id}")
            except Exception as e:
                logger.error(f"❌ Token 刷新失敗 | ID: {discord_id} | {e}")
                token_path.unlink(missing_ok=True)
                raise ValueError("授權已過期，請重新使用 `/identity_login` 進行綁定。")
        else:
            raise ValueError("憑證無效，請重新授權。")

    return build('calendar', 'v3', credentials=creds)

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
    logger.info(f"✅ 行事曆建立成功 | 標題：{title} | ID: {discord_id} | eventId: {created.get('id')}")

    return {
        "status": "success",
        "message": f"成功將【{title}】加入行事曆！",
        "eventId": created.get("id"),
        "htmlLink": created.get("htmlLink")
    }

def delete_calendar_events(discord_id: str, keyword: str, calendar_id: str = "primary", target_time: str = None) -> dict:
    """
    找出標題含有 keyword 且是防呆來源(NQU_agent)的未來事件並刪除。
    若 target_time 有值（ISO 格式），優先用時間窗口（±1小時）搜尋，
    不依賴 keyword 名稱匹配。
    """
    service = get_service(discord_id)
    now_dt = datetime.now(TAIPEI_TZ)
    start_of_today_str = _to_utc_rfc3339(datetime.combine(now_dt.date(), datetime.min.time(), tzinfo=TAIPEI_TZ))
    
    try:
        # 若有指定時間，用時間窗口搜尋（±1 小時）
        if target_time:
            target_dt = _parse_dt(target_time)
            time_min_str = _to_utc_rfc3339(target_dt - timedelta(hours=1))
            time_max_str = _to_utc_rfc3339(target_dt + timedelta(hours=1))
            resp = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min_str,
                timeMax=time_max_str,
                singleEvents=True,
                orderBy="startTime",
                maxResults=20,
            ).execute()
        else:
            # 用關鍵字名稱搜尋（若 keyword 為 * 通配符則不限名稱）
            q_keyword = keyword if keyword != "*" else None
            resp = service.events().list(
                calendarId=calendar_id,
                q=q_keyword,
                timeMin=start_of_today_str,
                singleEvents=True,
                orderBy="startTime",
                maxResults=50,
            ).execute()

        deleted_count = 0
        skipped_count = 0
        deleted_titles = set()
        
        for ev in resp.get("items", []):
            if _get_event_source_tag(ev) == BOT_SOURCE_TAG:
                service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
                deleted_titles.add(ev.get("summary", "未知事件"))
                deleted_count += 1
            else:
                skipped_count += 1
                
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
    """
    列出使用者的行事曆事件。
    若 target_date 有值（YYYY-MM-DD），列出該日所有事件；否則列出未來 N 天。
    """
    try:
        service = get_service(discord_id)
        now_dt = datetime.now(TAIPEI_TZ)
        
        if target_date:
            # 指定日期：列出該天 00:00 ~ 23:59
            from datetime import date as _date
            d = _date.fromisoformat(target_date)
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

def update_calendar_event(discord_id: str, keyword: str, new_title: str = None, new_start: str = None, new_end: str = None, calendar_id: str = "primary", target_time: str = None) -> dict:
    """
    修改行事曆中由機器人建立的事件。
    若 target_time 有值，用時間窗口（±1小時）定位事件；否則用關鍵字搜尋。
    """
    try:
        service = get_service(discord_id)
        now_dt = datetime.now(TAIPEI_TZ)
        
        if target_time:
            # 時間定位：±1 小時窗口
            target_dt = _parse_dt(target_time)
            time_min = _to_utc_rfc3339(target_dt - timedelta(hours=1))
            time_max = _to_utc_rfc3339(target_dt + timedelta(hours=1))
            resp = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=10,
            ).execute()
        else:
            start_of_today = _to_utc_rfc3339(datetime.combine(now_dt.date(), datetime.min.time(), tzinfo=TAIPEI_TZ))
            q_keyword = keyword if keyword != "*" else None
            resp = service.events().list(
                calendarId=calendar_id,
                q=q_keyword,
                timeMin=start_of_today,
                singleEvents=True,
                orderBy="startTime",
                maxResults=10,
            ).execute()
        
        updated_count = 0
        for ev in resp.get("items", []):
            if _get_event_source_tag(ev) != BOT_SOURCE_TAG:
                continue
            
            # 更新欄位
            if new_title:
                ev["summary"] = new_title
            if new_start:
                start_dt = _parse_dt(new_start)
                ev["start"] = {"dateTime": start_dt.replace(microsecond=0).isoformat(), "timeZone": TIMEZONE}
            if new_end:
                end_dt = _parse_dt(new_end)
                ev["end"] = {"dateTime": end_dt.replace(microsecond=0).isoformat(), "timeZone": TIMEZONE}
            
            service.events().update(calendarId=calendar_id, eventId=ev["id"], body=ev).execute()
            updated_count += 1
            logger.info(f"✅ 行事曆修改成功 | ID: {discord_id} | {ev.get('summary')} | eventId: {ev['id']}")
            break  # 只修改第一筆匹配的
        
        search_desc = f"時間 {target_time}" if target_time else f"名稱「{keyword}」"
        if updated_count > 0:
            return {
                "status": "success",
                "message": f"✅ 已成功修改行事曆中符合 {search_desc} 的事件"
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
