# -*- coding: utf-8 -*-
"""
bot/telegram/tg_ui_menus.py — Telegram 奢華互動介面
====================================================
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, ForceReply
from telegram.ext import ContextTypes

from bot.telegram import get_tg_user_id
from tools.auth import get_user_profile, get_user_token_path, delete_user_token

logger = logging.getLogger("telegram_bot")

# ── 科系對照表 ──
DEPT_MAPPING = {
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

COLLEGE_DEPTS = {
    "⚡ 理工學院": ["資工系", "電機系", "土木系", "食品系"],
    "💼 管理學院": ["企管系", "觀光系", "運休系", "工管系"],
    "📚 人文社會": ["國際系", "應英系", "華語系", "社工系"],
    "🏗️ 其他學院": ["建築系", "海邊系", "都景系", "護理系", "長照系", "通識中心"],
}

GRADE_LABELS = {
    "一": "大一", "二": "大二", "三": "大三", "四": "大四",
    "五": "大五(建築)", "碩": "碩士班", "進修": "進修部",
}


# =========================================================
# 輔助函式
# =========================================================
def _get_user_full_status(tg_id: str) -> dict:
    """讀取使用者完整狀態（profile + schedule + transcript）"""
    path = get_user_token_path(tg_id)
    if not path.exists():
        return {"registered": False}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    profile = data.get("profile", {})
    sch = data.get("schedule", {})
    tr = data.get("transcript", {})
    return {
        "registered": True,
        "profile": profile,
        "has_schedule": "schedule" in data,
        "has_transcript": "transcript" in data,
        "has_credentials": "credentials" in data,
        "schedule_courses": len(sch.get("courses", [])),
        "schedule_credits": sch.get("total_credits", 0),
        "transcript_semesters": len(tr.get("semesters", [])),
    }


def _clear_all_user_states(context: ContextTypes.DEFAULT_TYPE):
    """清除所有輸入狀態，避免回到主頁後仍攔截對話"""
    keys = [
        "awaiting_upload", "awaiting_calendar_url", "awaiting_profile_info",
        "awaiting_search", "temp_dept", "temp_grade", "temp_class_group",
        "pending_schedule", "pending_transcript"
    ]
    for k in keys:
        context.user_data.pop(k, None)


def _build_dashboard_text(user_name: str, status: dict) -> str:
    """建構主控台狀態卡片（手機友好寬度）"""
    if not status["registered"]:
        return (
            f"🤖 NQU 校園智慧助理\n"
            f"━━━━━━━━━━━━━━\n\n"
            f"歡迎來到國立金門大學\n"
            f"智慧校園 AI 助理！\n\n"
            f"嗨 {user_name}！我可以幫你：\n\n"
            f"📚 查詢課程資訊與選課建議\n"
            f"📅 管理課表與行事曆\n"
            f"📊 追蹤學分與畢業進度\n"
            f"🔍 搜尋 Dcard 評價與公告\n\n"
            f"━━━━━━━━━━━━━━\n"
            f"⚠️ 尚未設定身分\n"
            f"━━━━━━━━━━━━━━\n\n"
            f"💬 直接打字即可提問\n"
            f"🎓 點下方設定身分，AI 更懂你 ⬇️"
        )
    p = status["profile"]
    name = p.get("student_name", "")
    dept = p.get("department", "未知")
    grade = p.get("grade", "?")
    sid = p.get("student_id", "")
    cls = p.get("class_group", "")

    name_disp = name if name else user_name
    grade_disp = GRADE_LABELS.get(grade, grade + '年級')
    cls_disp = f" {cls}班" if cls else ""

    cal = "✅ 已連線" if status["has_credentials"] else "❌ 未綁定"
    sch = f"✅ {status['schedule_courses']}門 {status['schedule_credits']}學分" if status["has_schedule"] else "❌ 未上傳"
    tr = f"✅ {status['transcript_semesters']}學期" if status["has_transcript"] else "❌ 未上傳"

    lines = [
        f"🤖 NQU 校園智慧助理",
        f"━━━━━━━━━━━━━━",
        f"👤 {name_disp}",
        f"🏫 {dept} · {grade_disp}{cls_disp}",
    ]
    if sid:
        lines.append(f"🆔 {sid}")
    lines += [
        f"",
        f"📅 行事曆 {cal}",
        f"📚 課表　 {sch}",
        f"📊 成績單 {tr}",
        f"━━━━━━━━━━━━━━",
        f"",
        f"💬 直接打字提問，或選功能 ⬇️",
    ]
    return "\n".join(lines)


def _build_dashboard_keyboard() -> InlineKeyboardMarkup:
    """建構主控台按鈕面板（手機友好排版）"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎓 設定身分", callback_data="menu_profile"),
         InlineKeyboardButton("📋 我的課表", callback_data="menu_schedule_query")],
        [InlineKeyboardButton("📅 上傳課表", callback_data="menu_upload_schedule"),
         InlineKeyboardButton("📊 上傳成績單", callback_data="menu_upload_transcript")],
        [InlineKeyboardButton("📈 學分與成績", callback_data="menu_grades"),
         InlineKeyboardButton("🔍 搜尋功能", callback_data="menu_search")],
        [InlineKeyboardButton("📆 綁定行事曆", callback_data="menu_calendar")],
        [InlineKeyboardButton("❓ 使用教學", callback_data="menu_help"),
         InlineKeyboardButton("🗑️ 刪除帳號", callback_data="menu_delete")],
    ])


# =========================================================
# 1. Bot Commands 註冊
# =========================================================
async def setup_bot_commands(application):
    """啟動時主動向 Telegram 伺服器註冊左下角 Menu 指令表"""
    commands = [
        BotCommand("start", "🏠 開啟主控台"),
        BotCommand("profile", "🎓 設定科系與年級"),
        BotCommand("my_schedule", "📋 查詢今日課表"),
        BotCommand("my_free", "🕐 查詢空堂"),
        BotCommand("my_credits", "📊 本學期學分"),
        BotCommand("my_gpa", "📈 歷年 GPA"),
        BotCommand("calendar", "📆 綁定 Google 行事曆"),
        BotCommand("dcard_search", "🔍 搜尋 Dcard 評價"),
        BotCommand("nqu_news", "🏛️ 搜尋金大公告"),
        BotCommand("help", "❓ 使用教學"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("✅ 已向 Telegram 伺服器註冊 10 項 Bot Commands 選單")


# =========================================================
# 2. /start 主控台
# =========================================================
async def tg_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /start — 彈出主控台"""
    tg_id = get_tg_user_id(update.effective_chat.id)
    user_name = update.effective_user.first_name if update.effective_user else "同學"
    status = _get_user_full_status(tg_id)
    text = _build_dashboard_text(user_name, status)
    kb = _build_dashboard_keyboard()

    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb)


# =========================================================
# 3. Callback 路由器
# =========================================================
async def tg_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理所有 InlineKeyboard 按鈕點擊"""
    query = update.callback_query
    await query.answer()
    data = query.data
    tg_id = get_tg_user_id(update.effective_chat.id)

    # ── 返回主控台 ──
    if data == "menu_main":
        _clear_all_user_states(context)
        await tg_start_command(update, context)

    # ── 科系設定：選擇學院 ──
    elif data == "menu_profile":
        text = "🎓 請選擇您的所屬學院：\n\n💡 設定後 AI 會根據您的身分給予個人化推薦"
        keyboard = []
        for college_name in COLLEGE_DEPTS:
            keyboard.append([InlineKeyboardButton(college_name, callback_data=f"college_{college_name}")])
        keyboard.append([InlineKeyboardButton("⬅️ 返回主頁", callback_data="menu_main")])
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("college_"):
        college = data[8:]
        depts = COLLEGE_DEPTS.get(college, [])
        text = f"🏫 {college}\n\n請選擇您的科系："
        keyboard = [[InlineKeyboardButton(f"{d} ({DEPT_MAPPING.get(d, '')})", callback_data=f"dept_{d}")] for d in depts]
        keyboard.append([InlineKeyboardButton("⬅️ 返回選擇學院", callback_data="menu_profile")])
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("dept_"):
        dept = data[5:]
        context.user_data["temp_dept"] = dept
        text = f"🎓 您選擇了 {dept}\n\n請選擇您的就讀年級："
        keyboard = [
            [InlineKeyboardButton("一年級", callback_data="grade_一"),
             InlineKeyboardButton("二年級", callback_data="grade_二")],
            [InlineKeyboardButton("三年級", callback_data="grade_三"),
             InlineKeyboardButton("四年級", callback_data="grade_四")],
            [InlineKeyboardButton("五年級(建築)", callback_data="grade_五"),
             InlineKeyboardButton("碩士班", callback_data="grade_碩")],
            [InlineKeyboardButton("進修部", callback_data="grade_進修")],
            [InlineKeyboardButton("⬅️ 返回選擇科系", callback_data="menu_profile")],
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("grade_"):
        grade = data[6:]
        context.user_data["temp_grade"] = grade
        # 甲乙班選擇
        text = f"📝 甲乙班分組\n\n部分科系（如電機、企管）有分甲乙班，請選擇："
        keyboard = [
            [InlineKeyboardButton("🅰️ 甲班", callback_data="class_甲"),
             InlineKeyboardButton("🅱️ 乙班", callback_data="class_乙")],
            [InlineKeyboardButton("➖ 不適用 / 跳過", callback_data="class_none")],
            [InlineKeyboardButton("⬅️ 返回選擇年級", callback_data=f"dept_{context.user_data.get('temp_dept', '')}")],
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("class_"):
        class_group = "" if data == "class_none" else data[6:]
        context.user_data["temp_class_group"] = class_group
        # 詢問姓名與學號
        await query.edit_message_text(
            "📝 最後一步！請輸入您的姓名與學號\n\n"
            "━━━━━━━━━━━━━━\n"
            "📌 請直接回覆下方訊息\n"
            "📌 格式：姓名 學號\n"
            "📌 範例：王冠程 111210527\n"
            "📌 如不想填寫，輸入「跳過」\n"
            "━━━━━━━━━━━━━━"
        )
        # 發送 ForceReply 訊息
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="請輸入：姓名 學號（或輸入「跳過」）",
            reply_markup=ForceReply(selective=True),
        )
        context.user_data["awaiting_profile_info"] = True

    # ── 課表查詢子菜單 ──
    elif data == "menu_schedule_query":
        text = "📋 課表查詢\n━━━━━━━━━━━━━━\n請選擇查詢項目："
        keyboard = [
            [InlineKeyboardButton("📅 今日課表", callback_data="qry_schedule_today"),
             InlineKeyboardButton("🕐 空堂查詢", callback_data="qry_free")],
            [InlineKeyboardButton("📊 本學期學分", callback_data="qry_credits")],
            [InlineKeyboardButton("⬅️ 返回主頁", callback_data="menu_main")],
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    # ── 學分與成績子菜單 ──
    elif data == "menu_grades":
        text = "📈 學分與成績查詢\n━━━━━━━━━━━━━━\n請選擇查詢項目："
        keyboard = [
            [InlineKeyboardButton("🎓 畢業學分進度", callback_data="qry_credits_total")],
            [InlineKeyboardButton("📈 歷年 GPA", callback_data="qry_gpa"),
             InlineKeyboardButton("❌ 不及格課程", callback_data="qry_failed")],
            [InlineKeyboardButton("⬅️ 返回主頁", callback_data="menu_main")],
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    # ── 搜尋功能子菜單 ──
    elif data == "menu_search":
        text = "🔍 搜尋功能\n━━━━━━━━━━━━━━\n請選擇搜尋來源："
        keyboard = [
            [InlineKeyboardButton("🔍 Dcard 教授評價", callback_data="search_dcard")],
            [InlineKeyboardButton("🏛️ 金大官網公告", callback_data="search_nqu")],
            [InlineKeyboardButton("⬅️ 返回主頁", callback_data="menu_main")],
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "search_dcard":
        keyboard = [[InlineKeyboardButton("⬅️ 取消 / 返回主頁", callback_data="menu_main")]]
        await query.edit_message_text("🔍 請輸入想搜尋的教授或關鍵字：", reply_markup=InlineKeyboardMarkup(keyboard))
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="請輸入搜尋關鍵字（例如：英文教授、王大明）",
            reply_markup=ForceReply(selective=True),
        )
        context.user_data["awaiting_search"] = "dcard"

    elif data == "search_nqu":
        keyboard = [[InlineKeyboardButton("⬅️ 取消 / 返回主頁", callback_data="menu_main")]]
        await query.edit_message_text("🏛️ 請輸入想搜尋的公告關鍵字：", reply_markup=InlineKeyboardMarkup(keyboard))
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="請輸入搜尋關鍵字（例如：獎學金、招生、住宿）",
            reply_markup=ForceReply(selective=True),
        )
        context.user_data["awaiting_search"] = "nqu"

    # ── 上傳課表 ──
    elif data == "menu_upload_schedule":
        status = _get_user_full_status(tg_id)
        if status.get("has_schedule"):
            text = (
                "⚠️ 您已經上傳過課表了\n"
                "━━━━━━━━━━━━━━\n\n"
                "如果您上傳新的課表，原本的課程紀錄將會被覆蓋。\n"
                "請問要繼續重新上傳嗎？"
            )
            keyboard = [
                [InlineKeyboardButton("✅ 確認重新上傳", callback_data="force_upload_schedule")],
                [InlineKeyboardButton("⬅️ 取消 / 返回主頁", callback_data="menu_main")],
            ]
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await _prompt_upload_schedule(query, context)

    elif data == "force_upload_schedule":
        await _prompt_upload_schedule(query, context)

    # ── 上傳成績單 ──
    elif data == "menu_upload_transcript":
        status = _get_user_full_status(tg_id)
        if status.get("has_transcript"):
            text = (
                "⚠️ 您已經上傳過成績單了\n"
                "━━━━━━━━━━━━━━\n\n"
                "如果您上傳新的成績單，原本的學分紀錄將會被更新或覆蓋。\n"
                "請問要繼續重新上傳嗎？"
            )
            keyboard = [
                [InlineKeyboardButton("✅ 確認重新上傳", callback_data="force_upload_transcript")],
                [InlineKeyboardButton("⬅️ 取消 / 返回主頁", callback_data="menu_main")],
            ]
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await _prompt_upload_transcript(query, context)

    elif data == "force_upload_transcript":
        await _prompt_upload_transcript(query, context)

    # ── 查詢指令路由 ──
    elif data.startswith("qry_"):
        await _handle_query_callback(update, context, data)

    # ── 帳號刪除 ──
    elif data == "menu_delete":
        text = (
            "⚠️ 確認刪除帳號\n"
            "━━━━━━━━━━━━━━\n\n"
            "此操作將會：\n"
            "❌ 移除您的科系、年級身分資料\n"
            "❌ 清除已上傳的課表與成績單\n"
            "❌ 撤銷 Google 行事曆授權\n\n"
            "⚠️ 此操作無法復原！"
        )
        keyboard = [
            [InlineKeyboardButton("⚠️ 確認刪除", callback_data="confirm_delete"),
             InlineKeyboardButton("✖ 取消", callback_data="menu_main")],
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "confirm_delete":
        deleted = delete_user_token(tg_id)
        if deleted:
            logger.info(f"🗑️ [Telegram] 帳號已刪除 | {tg_id}")
            text = "✅ 帳號已成功刪除\n\n您的所有資料已全部移除。\n💡 如需重新註冊，請使用 /start"
        else:
            text = "❓ 找不到您的註冊資料，可能已經被刪除了。"
        keyboard = [[InlineKeyboardButton("🏠 返回主頁", callback_data="menu_main")]]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    # ── 確認/取消上傳 ──
    elif data == "confirm_schedule":
        from bot.telegram.tg_cmd_schedule import save_pending_schedule
        await save_pending_schedule(update, context)

    elif data == "confirm_transcript":
        from bot.telegram.tg_cmd_transcript import save_pending_transcript
        await save_pending_transcript(update, context)

    elif data in ("cancel_schedule", "cancel_transcript"):
        context.user_data.pop("pending_schedule", None)
        context.user_data.pop("pending_transcript", None)
        await query.edit_message_text("❌ 已取消上傳。")

    # ── Google Calendar 綁定 ──
    elif data == "menu_calendar":
        status = _get_user_full_status(tg_id)
        if status.get("has_credentials"):
            text = (
                "⚠️ 您已經綁定過行事曆了\n"
                "━━━━━━━━━━━━━━\n\n"
                "目前 AI 已具備讀寫您 Google 行事曆的權限。\n"
                "重新綁定將會重新取得授權，請問要繼續嗎？"
            )
            keyboard = [
                [InlineKeyboardButton("✅ 重新綁定", callback_data="force_bind_calendar")],
                [InlineKeyboardButton("⬅️ 取消 / 返回主頁", callback_data="menu_main")],
            ]
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await _prompt_bind_calendar(update, context, query, tg_id)

    elif data == "force_bind_calendar":
        await _prompt_bind_calendar(update, context, query, tg_id)

    # ── 使用教學（升級版） ──
    elif data in ("menu_help", "help_1", "help_2", "help_3"):
        page = {"menu_help": 1, "help_1": 1, "help_2": 2, "help_3": 3}.get(data, 1)
        await _show_help_page(query, page)


# =========================================================
# 查詢指令處理
# =========================================================
async def _handle_query_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str):
    """處理 qry_* 查詢按鈕"""
    query = update.callback_query
    tg_id = get_tg_user_id(update.effective_chat.id)
    # 查詢結果底部：返回上一層 + 主頁
    parent = "menu_schedule_query" if data.startswith("qry_schedule") or data == "qry_free" or data == "qry_credits" else "menu_grades"
    back_kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ 返回上一層", callback_data=parent),
         InlineKeyboardButton("🏠 回主頁", callback_data="menu_main")],
    ])

    if data == "qry_schedule_today":
        from tools.schedule_manager import query_day_schedule
        tz = timezone(timedelta(hours=8))
        day = datetime.now(tz).isoweekday()
        if day > 5:
            await query.edit_message_text("🎉 今天是假日，沒有課！", reply_markup=back_kb)
            return
        result = query_day_schedule(tg_id, day)
        await query.edit_message_text(result, reply_markup=back_kb)

    elif data == "qry_free":
        from tools.schedule_manager import query_free_periods
        result = query_free_periods(tg_id)
        await query.edit_message_text(result, reply_markup=back_kb)

    elif data == "qry_credits":
        from tools.schedule_manager import query_credit_summary
        result = query_credit_summary(tg_id)
        await query.edit_message_text(result, reply_markup=back_kb)

    elif data == "qry_credits_total":
        from tools.transcript_manager import query_credit_progress
        result = query_credit_progress(tg_id)
        # Telegram 訊息上限 4096 字元
        if len(result) > 4000:
            result = result[:3990] + "\n..."
        await query.edit_message_text(result, reply_markup=back_kb)

    elif data == "qry_gpa":
        from tools.transcript_manager import query_gpa
        result = query_gpa(tg_id)
        await query.edit_message_text(result, reply_markup=back_kb)

    elif data == "qry_failed":
        from tools.transcript_manager import query_failed_courses
        result = query_failed_courses(tg_id)
        await query.edit_message_text(result, reply_markup=back_kb)


# =========================================================
# ForceReply 處理：姓名/學號
# =========================================================
async def process_profile_name_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 ForceReply 回覆：解析姓名與學號並完成註冊"""
    context.user_data["awaiting_profile_info"] = False
    tg_id = get_tg_user_id(update.effective_chat.id)
    text = update.message.text.strip()

    dept = context.user_data.pop("temp_dept", "未知")
    grade = context.user_data.pop("temp_grade", "?")
    class_group = context.user_data.pop("temp_class_group", "")

    student_name = ""
    student_id = ""

    if text != "跳過":
        parts = text.split()
        if len(parts) >= 2:
            student_name = parts[0]
            student_id = parts[1]
        elif len(parts) == 1:
            student_name = parts[0]

    # 寫入 JSON
    token_path = get_user_token_path(tg_id)
    existing = {}
    if token_path.exists():
        with open(token_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    dept_full = DEPT_MAPPING.get(dept, dept)
    tz = timezone(timedelta(hours=8))

    profile = {
        "department": dept,
        "department_full": dept_full,
        "grade": grade,
        "class_group": class_group,
        "registered_at": datetime.now(tz).isoformat(),
    }
    if student_name:
        profile["student_name"] = student_name
    if student_id:
        profile["student_id"] = student_id

    existing["profile"] = profile
    token_path.parent.mkdir(parents=True, exist_ok=True)
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)

    name_display = f" {student_name}" if student_name else ""
    sid_display = f" (學號：{student_id})" if student_id else ""
    group_display = f" {class_group}班" if class_group else ""

    logger.info(f"✅ [Telegram] 身分設定完成 | {tg_id} | {dept} {grade}年級{group_display}{name_display}{sid_display}")

    from bot.telegram.tg_audit import tg_send_audit_event
    await tg_send_audit_event(
        title="身分註冊完成",
        description=f"✅ {name_display.strip()} ({sid_display.replace(' (學號：', '').replace(')', '')})\n🏫 {dept} {GRADE_LABELS.get(grade, grade + '年級')}{group_display}",
        color_val=3447003  # blue
    )

    text = (
        f"✅ 身分設定完成！\n"
        f"━━━━━━━━━━━━━━\n"
        f"👤{name_display}{sid_display}\n"
        f"🏫 {dept} ({dept_full})\n"
        f"📚 {GRADE_LABELS.get(grade, grade + '年級')}{group_display}\n"
        f"━━━━━━━━━━━━━━\n\n"
        f"💡 AI 已記住您的身分！\n"
        f"使用 /start 返回主控台"
    )
    await update.message.reply_text(text)


# =========================================================
# 多頁使用教學
# =========================================================
async def _show_help_page(query, page: int):
    """分頁式使用教學"""
    pages = {
        1: (
            "❓ 使用教學  1️⃣ / 3\n"
            "━━━━━━━━━━━━━━\n\n"
            "💬 基本對話\n\n"
            "直接輸入問題即可對話：\n\n"
            '• 「資工系有什麼涼課？」\n'
            '• 「柯志亨的實驗室在哪？」\n'
            '• 「什麼時候期末考？」\n'
            '• 「我跟機器學習衝堂嗎？」\n\n'
            "💡 AI 會自動搜尋校園\n"
            "　 知識庫為您解答"
        ),
        2: (
            "❓ 使用教學  2️⃣ / 3\n"
            "━━━━━━━━━━━━━━\n\n"
            "🎓 身分設定 & 課表\n\n"
            "➀ 設定身分\n"
            "　 AI 會根據您的系級\n"
            "　 提供專屬建議\n\n"
            "➁ 上傳課表\n"
            "　 截圖或 JSON 皆可\n"
            "　 AI 自動辨識課程\n\n"
            "➂ 上傳成績單\n"
            "　 PDF 或截圖皆可\n"
            "　 追蹤畢業進度"
        ),
        3: (
            "❓ 使用教學  3️⃣ / 3\n"
            "━━━━━━━━━━━━━━\n\n"
            "🔍 搜尋 & 行事曆\n\n"
            "🔍 Dcard 搜尋\n"
            "　 查看學長姐對教授\n"
            "　 和課程的評價\n\n"
            "🏩 金大官網\n"
            "　 搜尋最新公告\n"
            "　 招生、獎學金等\n\n"
            "📆 Google 行事曆\n"
            "　 綁定後可讓 AI\n"
            "　 自動加入事件"
        ),
    }
    text = pages.get(page, pages[1])
    buttons = []
    if page > 1:
        buttons.append(InlineKeyboardButton(f"⬅️ 上一頁", callback_data=f"help_{page-1}"))
    if page < 3:
        buttons.append(InlineKeyboardButton(f"下一頁 ➡️", callback_data=f"help_{page+1}"))
    keyboard = []
    if buttons:
        keyboard.append(buttons)
    keyboard.append([InlineKeyboardButton("🏠 回主頁", callback_data="menu_main")])
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


# =========================================================
# ForceReply 處理：Google Calendar URL
# =========================================================
async def process_calendar_url_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 ForceReply 回覆：解析 OAuth 網址並存入 Token"""
    context.user_data["awaiting_calendar_url"] = False
    tg_id = get_tg_user_id(update.effective_chat.id)
    url_text = update.message.text.strip()

    if not url_text.startswith("http"):
        await update.message.reply_text("❌ 格式不正確！請貼上完整的 http://localhost/?code=... 網址")
        return

    try:
        from tools.auth import verify_and_save_token, get_user_token_path
        # 讀取現有 profile
        token_path = get_user_token_path(tg_id)
        dept, grade, cls = "", "", ""
        if token_path.exists():
            with open(token_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            p = existing.get("profile", {})
            dept = p.get("department", "")
            grade = p.get("grade", "")
            cls = p.get("class_group", "")

        verify_and_save_token(tg_id, url_text, dept, grade, cls)
        logger.info(f"✅ [Telegram] Google Calendar 綁定成功 | {tg_id}")
        from bot.telegram.tg_audit import tg_send_audit_event
        await tg_send_audit_event(
            title="Google 行事曆綁定成功",
            description=f"✅ 使用者 {tg_id} 已成功綁定 Google Calendar",
            color_val=5763719  # green
        )
        await update.message.reply_text(
            "✅ Google 行事曆綁定成功！\n\n"
            "現在 AI 可以幫您將課程\n"
            "和學校事件加入行事曆\n\n"
            "💡 試試問：「幫我把線性代\n"
            "　 數加到行事曆」\n\n"
            "使用 /start 返回主控台"
        )
    except Exception as e:
        logger.exception(f"❌ [Telegram] Calendar 綁定失敗")
        await update.message.reply_text(
            f"❌ 綁定失敗：{e}\n\n"
            "💡 請確認您貼的是完整的\n"
            "http://localhost/?code=... 網址\n\n"
            "使用 /start 重新嘗試"
        )


async def _prompt_upload_schedule(query, context: ContextTypes.DEFAULT_TYPE):
    await query.edit_message_text(
        "📅 上傳課表\n"
        "━━━━━━━━━━━━━━\n\n"
        "請傳送以下任一格式：\n"
        "📷 課表截圖（PNG / JPG）→ AI 自動辨識\n"
        "📄 課表 JSON 檔案 → 直接匯入\n\n"
        "💡 截圖請從選課系統擷取完整的課表表格",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ 取消 / 返回主頁", callback_data="menu_main")]])
    )
    context.user_data["awaiting_upload"] = "schedule"


async def _prompt_upload_transcript(query, context: ContextTypes.DEFAULT_TYPE):
    await query.edit_message_text(
        "📊 上傳成績單\n"
        "━━━━━━━━━━━━━━\n\n"
        "請傳送以下任一格式：\n"
        "📄 成績單 PDF → AI 自動辨識\n"
        "📷 成績單截圖（PNG / JPG）\n"
        "📄 成績單 JSON 檔案 → 直接匯入\n\n"
        "💡 請上傳「歷年成績表」PDF",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ 取消 / 返回主頁", callback_data="menu_main")]])
    )
    context.user_data["awaiting_upload"] = "transcript"


async def _prompt_bind_calendar(update: Update, context: ContextTypes.DEFAULT_TYPE, query, tg_id: str):
    from tools.auth import get_auth_url
    try:
        auth_url = get_auth_url(tg_id)
        text = (
            "📆 綁定 Google 行事曆\n"
            "━━━━━━━━━━━━━━\n\n"
            "➀ 點下方按鈕登入 Google\n"
            "➁ 完成授權後複製網址\n"
            "➂ 貼回對話即完成！\n\n"
            "💡 網址開頭像這樣：\n"
            "http://localhost/?code=..."
        )
        keyboard = [
            [InlineKeyboardButton("🔗 點我登入 Google", url=auth_url)],
            [InlineKeyboardButton("⬅️ 返回主頁", callback_data="menu_main")],
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        # 發送 ForceReply 讓使用者貼回網址
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="請貼上授權後的網址（http://localhost/?code=...)",
            reply_markup=ForceReply(selective=True),
        )
        context.user_data["awaiting_calendar_url"] = True
    except Exception as e:
        logger.exception("Calendar OAuth URL 產生失敗")
        await query.edit_message_text(f"❌ 產生授權連結失敗：{e}")
