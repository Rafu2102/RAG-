# tools/group_manager.py — 群組標籤與邀請碼管理模組
# ================================================
# 負責群組標籤操作、groups.json 資料庫管理、邀請碼生成
# ================================================
from __future__ import annotations

import json
import random
import string
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

from tools.auth import get_user_token_path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GROUPS_PATH = DATA_DIR / "groups.json"
TAIPEI_TZ = timezone(timedelta(hours=8), name="Asia/Taipei")


# === 群組標籤操作 (寫入使用者 Token 檔案) ===

def add_user_group(discord_id: str, group_name: str) -> bool:
    """將學生加入特定的群組標籤。回傳是否成功。"""
    token_path = get_user_token_path(discord_id)
    if not token_path.exists():
        return False
    
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        
        profile = user_data.get("profile", {})
        groups = profile.get("groups", [])
        
        already_in = group_name in groups
        if not already_in:
            groups.append(group_name)
        
        profile["groups"] = groups
        user_data["profile"] = profile
        
        with open(token_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=4)
        
        # 同步更新 groups.json 的 member_count
        if not already_in:
            _update_member_count(group_name, delta=1)
        
        logger.info(f"🏷️ 群組標籤新增 | 使用者={discord_id} | 加入「{group_name}」| 目前群組={groups}")
        return True
    except Exception as e:
        logger.error(f"為 {discord_id} 添加群組標籤失敗: {e}")
        return False


def remove_user_group(discord_id: str, group_name: str) -> bool:
    """將學生從特定群組標籤移除。回傳是否成功。"""
    token_path = get_user_token_path(discord_id)
    if not token_path.exists():
        return False
    
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        
        profile = user_data.get("profile", {})
        groups = profile.get("groups", [])
        
        was_in = group_name in groups
        if was_in:
            groups.remove(group_name)
        
        profile["groups"] = groups
        user_data["profile"] = profile
        
        with open(token_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=4)
        
        # 同步更新 groups.json 的 member_count
        if was_in:
            _update_member_count(group_name, delta=-1)
        
        logger.info(f"🏷️ 群組標籤移除 | 使用者={discord_id} | 離開「{group_name}」| 目前群組={groups}")
        return True
    except Exception as e:
        logger.error(f"為 {discord_id} 移除群組標籤失敗: {e}")
        return False


def _update_member_count(group_name: str, delta: int):
    """更新 groups.json 中指定群組的 member_count"""
    try:
        groups = _load_groups()
        if group_name in groups:
            current = groups[group_name].get("member_count", 0)
            groups[group_name]["member_count"] = max(0, current + delta)
            _save_groups(groups)
    except Exception as e:
        logger.error(f"更新群組人數失敗 ({group_name}): {e}")


# === 群組資料庫 (groups.json) ===

def _load_groups() -> dict:
    """載入群組資料庫"""
    if not GROUPS_PATH.exists():
        return {}
    try:
        with open(GROUPS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_groups(data: dict):
    """儲存群組資料庫"""
    with open(GROUPS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def create_group(group_name: str, creator_id: str, creator_name: str) -> str:
    """
    建立一個新群組，自動產生 6 碼邀請碼。
    若群組已存在則回傳現有邀請碼。
    
    Returns:
        邀請碼 (str)
    """
    groups = _load_groups()
    
    # 若群組已存在，直接回傳邀請碼
    if group_name in groups:
        return groups[group_name]["invite_code"]
    
    # 產生不重複的 6 碼邀請碼
    existing_codes = {g["invite_code"] for g in groups.values()}
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if code not in existing_codes:
            break
    
    groups[group_name] = {
        "invite_code": code,
        "creator_id": creator_id,
        "creator_name": creator_name,
        "created_at": datetime.now(TAIPEI_TZ).isoformat(),
        "member_count": 0,
    }
    _save_groups(groups)
    logger.info(f"🏷️ 群組建立 | 名稱={group_name} 邀請碼={code} 建立者={creator_name}")
    return code


def get_group_by_code(invite_code: str) -> str | None:
    """根據邀請碼查找群組名稱。找不到回傳 None。"""
    groups = _load_groups()
    for name, data in groups.items():
        if data.get("invite_code", "").upper() == invite_code.upper():
            return name
    return None


def list_all_groups() -> list[str]:
    """列出所有群組名稱"""
    groups = _load_groups()
    return list(groups.keys())


def get_group_info(group_name: str) -> dict | None:
    """取得群組完整資訊"""
    groups = _load_groups()
    return groups.get(group_name)


def delete_group(group_name: str) -> bool:
    """刪除群組"""
    groups = _load_groups()
    if group_name not in groups:
        return False
    del groups[group_name]
    _save_groups(groups)
    logger.info(f"🏷️ 群組刪除 | 名稱={group_name}")
    return True
