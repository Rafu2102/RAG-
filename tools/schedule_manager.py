# -*- coding: utf-8 -*-
"""
tools/schedule_manager.py — 課表資料管理
==========================================
管理學生個人課表的 JSON 儲存、讀取與查詢。

存放位置：tools/data/tokens/{discord_id}.json 的 "schedule" 欄位

OCR 方案：
    - 模型：Gemini 多模態 API（中文 OCR 表格辨識）
    - 方式：透過 Gemini Vision API 將課表截圖轉為 JSON
    - 目前先支援手動 JSON 匯入
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("discord_bot")

# 時段對照表（NQU 標準 9 節課）
PERIOD_TIME_MAP = {
    1: ("08:10", "09:00"),
    2: ("09:10", "10:00"),
    3: ("10:10", "11:00"),
    4: ("11:10", "12:00"),
    5: ("13:30", "14:20"),
    6: ("14:30", "15:20"),
    7: ("15:30", "16:20"),
    8: ("16:30", "17:20"),
    9: ("17:30", "18:20"),
}

DAY_NAMES = {1: "星期一", 2: "星期二", 3: "星期三", 4: "星期四", 5: "星期五", 6: "星期六", 7: "星期日"}

TOKEN_DIR = Path(__file__).parent / "data" / "discord_tokens"

# ── 雙軌路由：tg_ 前綴 → telegram_tokens/ ──
_TG_TOKEN_DIR = Path(__file__).parent / "data" / "telegram_tokens"


def _get_user_token_path(user_id: str) -> Path:
    if user_id.startswith("tg_"):
        return _TG_TOKEN_DIR / f"{user_id[3:]}_token.json"
    return TOKEN_DIR / f"{user_id}_token.json"


def _load_user_data(discord_id: str) -> dict | None:
    """讀取使用者 JSON 資料"""
    path = _get_user_token_path(discord_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_user_data(discord_id: str, data: dict):
    """儲存使用者 JSON 資料"""
    path = _get_user_token_path(discord_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# =========================================================================
# 📅 課表存取
# =========================================================================

def save_schedule(discord_id: str, schedule_data: dict) -> bool:
    """
    儲存課表資料到使用者 token 的 schedule 欄位。
    
    schedule_data 應包含：
    {
        "academic_year": "114",
        "semester": "2",
        "courses": [
            {
                "name": "Linux系統自動化運維",
                "name_en": "...",
                "instructor": "柯志亨",
                "room": "E321電腦網路實驗室",
                "day": 2,
                "periods": [2, 3, 4],
                "credits": 3,
                "type": "選修"
            }
        ]
    }
    """
    user_data = _load_user_data(discord_id)
    if user_data is None:
        logger.error(f"找不到使用者 {discord_id} 的 token 檔案")
        return False

    # 自動建立 timetable 與 free_periods
    timetable = {}
    for course in schedule_data.get("courses", []):
        day_str = str(course["day"])
        if day_str not in timetable:
            timetable[day_str] = {}
        for period in course["periods"]:
            timetable[day_str][str(period)] = course["name"]

    free_periods = {}
    for day in range(1, 6):  # 只算星期一到五
        day_str = str(day)
        day_schedule = timetable.get(day_str, {})
        free = [p for p in range(1, 10) if str(p) not in day_schedule]
        if free:
            free_periods[day_str] = free

    total_credits = sum(c.get("credits", 0) for c in schedule_data.get("courses", []))

    tz = timezone(timedelta(hours=8))
    schedule = {
        "academic_year": schedule_data.get("academic_year", ""),
        "semester": schedule_data.get("semester", ""),
        "updated_at": datetime.now(tz).isoformat(),
        "courses": schedule_data["courses"],
        "timetable": timetable,
        "free_periods": free_periods,
        "total_credits": total_credits,
    }

    user_data["schedule"] = schedule
    _save_user_data(discord_id, user_data)
    logger.info(f"📅 課表儲存 | 使用者={discord_id} | {len(schedule_data['courses'])} 門課 {total_credits} 學分")
    return True


def get_schedule(discord_id: str) -> dict | None:
    """取得使用者的課表資料"""
    user_data = _load_user_data(discord_id)
    if user_data is None:
        return None
    return user_data.get("schedule")


# =========================================================================
# 🔍 課表查詢工具
# =========================================================================

def query_day_schedule(discord_id: str, day: int) -> str:
    """查詢某天的課表，返回人類可讀的文字"""
    schedule = get_schedule(discord_id)
    if not schedule:
        return "❌ 您還沒有匯入課表資料。請先使用 `/upload_schedule` 上傳您的課表。"

    day_str = str(day)
    day_name = DAY_NAMES.get(day, f"第{day}天")
    timetable = schedule.get("timetable", {})
    day_courses = timetable.get(day_str, {})

    if not day_courses:
        return f"📅 **{day_name}** — 沒有課！整天都是空堂 🎉"

    lines = [f"📅 **{day_name}的課表**："]
    # 合併連續課時
    courses_by_name = {}
    for period_str, name in sorted(day_courses.items(), key=lambda x: int(x[0])):
        period = int(period_str)
        if name not in courses_by_name:
            courses_by_name[name] = []
        courses_by_name[name].append(period)

    for name, periods in courses_by_name.items():
        # 找到這門課的詳細資訊
        course_detail = None
        for c in schedule.get("courses", []):
            if c["name"] == name and c["day"] == day:
                course_detail = c
                break
        
        period_range = f"第{periods[0]}-{periods[-1]}節"
        time_start = PERIOD_TIME_MAP.get(periods[0], ("?", "?"))[0]
        time_end = PERIOD_TIME_MAP.get(periods[-1], ("?", "?"))[1]
        
        info = f"  📖 **{name}** | {period_range} ({time_start}~{time_end})"
        if course_detail:
            info += f" | 👩‍🏫 {course_detail.get('instructor', '?')} | 📍 {course_detail.get('room', '?')}"
        lines.append(info)

    # 空堂
    free = schedule.get("free_periods", {}).get(day_str, [])
    if free:
        free_str = ", ".join([f"第{p}節" for p in free])
        lines.append(f"  🕐 **空堂**：{free_str}")

    return "\n".join(lines)


def query_free_periods(discord_id: str) -> str:
    """查詢所有空堂"""
    schedule = get_schedule(discord_id)
    if not schedule:
        return "❌ 您還沒有匯入課表資料。"

    free_periods = schedule.get("free_periods", {})
    lines = ["📅 **您的空堂總覽**："]
    for day in range(1, 6):
        day_str = str(day)
        day_name = DAY_NAMES[day]
        free = free_periods.get(day_str, [])
        if free:
            free_items = []
            for p in free:
                time_range = PERIOD_TIME_MAP.get(p, ("?", "?"))
                free_items.append(f"第{p}節({time_range[0]})")
            lines.append(f"  {day_name}：{', '.join(free_items)}")
        else:
            lines.append(f"  {day_name}：沒有空堂 😢")
    return "\n".join(lines)


def query_credit_summary(discord_id: str) -> str:
    """查詢本學期學分摘要"""
    schedule = get_schedule(discord_id)
    if not schedule:
        return "❌ 您還沒有匯入課表資料。"

    courses = schedule.get("courses", [])
    total = schedule.get("total_credits", 0)

    # 按類型分類
    by_type = {}
    for c in courses:
        t = c.get("type", "其他")
        if t not in by_type:
            by_type[t] = {"count": 0, "credits": 0}
        by_type[t]["count"] += 1
        by_type[t]["credits"] += c.get("credits", 0)

    lines = [f"📊 **本學期學分摘要**（{schedule.get('academic_year', '?')}學年度 第{schedule.get('semester', '?')}學期）"]
    lines.append(f"  📚 **總修學分**：{total}")
    lines.append(f"  📖 **課程數**：{len(courses)}")
    for t, info in by_type.items():
        lines.append(f"  • {t}：{info['count']} 門 ({info['credits']} 學分)")

    return "\n".join(lines)


def get_schedule_context_for_llm(discord_id: str, query: str = "") -> str:
    """
    為 LLM 生成精簡的課表 context（避免 token 浪費）。
    根據查詢內容只提供相關部分。
    """
    schedule = get_schedule(discord_id)
    if not schedule:
        return ""

    query_lower = query.lower()

    # 判斷查詢是關於哪天
    day_keywords = {
        "星期一": 1, "週一": 1, "禮拜一": 1, "monday": 1,
        "星期二": 2, "週二": 2, "禮拜二": 2, "tuesday": 2,
        "星期三": 3, "週三": 3, "禮拜三": 3, "wednesday": 3,
        "星期四": 4, "週四": 4, "禮拜四": 4, "thursday": 4,
        "星期五": 5, "週五": 5, "禮拜五": 5, "friday": 5,
    }

    # 空堂查詢 → 只傳空堂資料
    if any(k in query for k in ["空堂", "有空", "沒課", "沒有課", "下課"]):
        return f"【學生的空堂】\n{query_free_periods(discord_id)}"

    # 特定日期查詢 → 只傳該天
    for keyword, day in day_keywords.items():
        if keyword in query:
            return f"【學生的課表】\n{query_day_schedule(discord_id, day)}"

    # 學分查詢 → 只傳學分摘要
    if any(k in query for k in ["學分", "幾學分", "修了多少"]):
        return f"【學生的學分】\n{query_credit_summary(discord_id)}"

    # 查特定課程/教授 → 只傳該課程資訊
    courses = schedule.get("courses", [])
    for course in courses:
        if course["name"] in query or course.get("instructor", "") in query:
            day_name = DAY_NAMES.get(course["day"], "?")
            periods = course["periods"]
            time_start = PERIOD_TIME_MAP.get(periods[0], ("?", "?"))[0]
            time_end = PERIOD_TIME_MAP.get(periods[-1], ("?", "?"))[1]
            return (
                f"【學生的課程資訊】\n"
                f"課程：{course['name']} ({course.get('name_en', '')})\n"
                f"教授：{course.get('instructor', '?')}\n"
                f"時間：{day_name} 第{periods[0]}-{periods[-1]}節 ({time_start}~{time_end})\n"
                f"教室：{course.get('room', '?')}\n"
                f"學分：{course.get('credits', '?')} ({course.get('type', '?')})"
            )

    # 一般課表查詢 → 傳完整詳細課表
    lines = [f"【學生本學期詳細課表】"]
    for day in range(1, 6):
        lines.append(query_day_schedule(discord_id, day))
    return "\n\n".join(lines)
