# tools/auth.py — 授權與身分管理模組
# ================================================
# 負責 Google OAuth 授權、Token 管理、使用者身分資料、廣播名單篩選
# ================================================
from __future__ import annotations

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from googleapiclient.discovery import build  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore
from google_auth_oauthlib.flow import Flow  # type: ignore
from google.auth.transport.requests import Request  # type: ignore

SCOPES = ["https://www.googleapis.com/auth/calendar"]

# 指向 tools/data 目錄下的憑證檔與 Token
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CREDS_PATH = DATA_DIR / "credentials.json"

# ── 雙軌 Token 路由 ──
# Discord ID: 純數字 (18碼) → discord_tokens/
# Telegram ID: "tg_" 前綴    → telegram_tokens/
DISCORD_TOKENS_DIR = DATA_DIR / "discord_tokens"
DISCORD_TOKENS_DIR.mkdir(exist_ok=True)
TELEGRAM_TOKENS_DIR = DATA_DIR / "telegram_tokens"
TELEGRAM_TOKENS_DIR.mkdir(exist_ok=True)

# 向後相容：舊程式碼若直接引用 TOKENS_DIR，指向 discord 區
TOKENS_DIR = DISCORD_TOKENS_DIR


# === OAuth 授權 ===

# 【PKCE 關鍵】快取 Flow 物件，保存 code_verifier
_pending_flows: dict[str, "Flow"] = {}

def get_user_token_path(user_id: str) -> Path:
    """取得特定使用者的 Token 檔案路徑（自動依 tg_ 前綴路由至 Telegram 區）"""
    if user_id.startswith("tg_"):
        real_id = user_id[3:]
        return TELEGRAM_TOKENS_DIR / f"{real_id}_token.json"
    return DISCORD_TOKENS_DIR / f"{user_id}_token.json"

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
    
    # 【保留 nickname 與 groups】若使用者重新註冊，保留舊的 nickname 與群組標籤
    existing_nickname = ""
    existing_groups = []
    token_path = get_user_token_path(discord_id)
    if token_path.exists():
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                existing_nickname = old_data.get("profile", {}).get("nickname", "")
                existing_groups = old_data.get("profile", {}).get("groups", [])
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
        "groups": existing_groups,
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


# === 使用者身分讀取/刪除 ===

def get_user_profile(discord_id: str) -> dict | None:
    """讀取使用者的基本身分資料"""
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
    """根據 discord_id 讀取合併後的 JSON，並還原 Google Calendar Service"""
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


# === 管理員廣播：使用者名單篩選 ===

def get_targeted_users(target_dept: str = None, target_grade: str = None, target_group: str = None) -> list[dict]:
    """
    根據科系、年級、群組標籤，篩選出符合條件的已註冊 Discord User ID 名單。
    若參數為 None，則不限制該條件 (全體廣播)。
    
    Returns:
        list[dict]: [{"discord_id": "123", "department": "資工系", "grade": "三", "groups": [...]}, ...]
    """
    targets = []
    for token_file in DISCORD_TOKENS_DIR.glob("*_token.json"):
        discord_id = token_file.stem.replace("_token", "")
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                profile = data.get("profile", {})
                
                # 科系過濾 (支援模糊比對，如 "資工" 可匹配 "資工系")
                user_dept = profile.get("department", "")
                if target_dept and target_dept not in user_dept:
                    continue
                
                # 年級過濾 (精確比對)
                user_grade = profile.get("grade", "")
                if target_grade and target_grade != user_grade:
                    continue
                
                # 群組標籤過濾
                user_groups = profile.get("groups", [])
                if target_group and target_group not in user_groups:
                    continue
                
                targets.append({
                    "discord_id": discord_id,
                    "department": user_dept,
                    "grade": user_grade,
                    "groups": user_groups,
                })
        except Exception as e:
            logger.error(f"讀取使用者 {discord_id} 資料失敗: {e}")
            continue
    
    group_tag = f" 群組={target_group}" if target_group else ""
    logger.info(f"📢 廣播名單篩選 | 科系={target_dept or '全校'} 年級={target_grade or '全系'}{group_tag} | 符合 {len(targets)} 人")
    return targets
