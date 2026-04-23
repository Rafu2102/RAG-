# -*- coding: utf-8 -*-
"""
bot/telegram/tg_cmd_schedule.py — 課表上傳與查詢
=================================================
"""

import json
import asyncio
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.telegram import get_tg_user_id
from tools.schedule_manager import (
    save_schedule, query_day_schedule, query_free_periods, query_credit_summary,
    DAY_NAMES,
)

logger = logging.getLogger("telegram_bot")


async def process_schedule_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理課表上傳（圖片 OCR 或 JSON 文件）"""
    tg_id = get_tg_user_id(update.effective_chat.id)
    context.user_data.pop("awaiting_upload", None)

    # ── 圖片 OCR ──
    if update.message.photo:
        progress = await update.message.reply_text("⏳ Gemini 3.1 Pro 正在辨識課表截圖...\n請稍候約 10-20 秒 🔍")
        try:
            photo = update.message.photo[-1]
            file = await photo.get_file()
            image_bytes = await file.download_as_bytearray()

            from tools.ocr_engine import ocr_schedule, enrich_schedule_with_course_db
            schedule_data = await ocr_schedule(
                image_bytes_list=[bytes(image_bytes)],
                mime_types=["image/jpeg"],
            )

            if not schedule_data.get("courses"):
                await progress.edit_text("⚠️ 辨識完成但沒有找到任何課程！\n請確認截圖包含完整的課表表格。")
                return

            # 用 RAG 資料庫增強
            global_nodes = context.bot_data.get("global_nodes")
            if global_nodes:
                schedule_data = enrich_schedule_with_course_db(schedule_data, global_nodes)

            # 暫存 + 預覽 + 確認
            context.user_data["pending_schedule"] = schedule_data
            courses = schedule_data.get("courses", [])
            total_credits = sum(c.get("credits", 0) for c in courses)
            preview = _build_schedule_preview(schedule_data, courses, total_credits)
            keyboard = [
                [InlineKeyboardButton("✅ 確認儲存", callback_data="confirm_schedule"),
                 InlineKeyboardButton("❌ 取消", callback_data="cancel_schedule")],
            ]
            await progress.edit_text(preview, reply_markup=InlineKeyboardMarkup(keyboard))

        except Exception as e:
            logger.exception(f"❌ [Telegram] 課表 OCR 錯誤")
            await progress.edit_text(f"❌ 辨識失敗：{e}\n💡 請改用 JSON 檔案匯入。")
        return

    # ── JSON 文件 ──
    if update.message.document:
        doc = update.message.document
        filename = (doc.file_name or "").lower()

        if not filename.endswith(".json"):
            await update.message.reply_text("❌ 請上傳 .json 格式的課表檔案。")
            return

        try:
            file = await doc.get_file()
            content = await file.download_as_bytearray()
            data = json.loads(content.decode("utf-8"))

            if "schedule" in data:
                data = data["schedule"]
            if "courses" not in data:
                await update.message.reply_text("❌ JSON 格式錯誤：找不到 courses 欄位。")
                return

            success = save_schedule(tg_id, data)
            if success:
                courses = data.get("courses", [])
                total = sum(c.get("credits", 0) for c in courses)
                await update.message.reply_text(
                    f"✅ 課表匯入成功！\n"
                    f"📚 {len(courses)} 門課 · {total} 學分\n\n"
                    f"💡 使用 /start 返回主控台查詢課表"
                )
            else:
                await update.message.reply_text("❌ 儲存失敗！請先使用 /start → 設定個人身分。")
        except json.JSONDecodeError as e:
            await update.message.reply_text(f"❌ JSON 解析失敗：{e}")
        except Exception as e:
            logger.exception("❌ [Telegram] 課表 JSON 匯入錯誤")
            await update.message.reply_text(f"❌ 匯入失敗：{e}")
        return

    await update.message.reply_text("📷 請傳送課表截圖或 .json 檔案。")


async def save_pending_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """確認按鈕回調：儲存暫存的 OCR 課表"""
    query = update.callback_query
    tg_id = get_tg_user_id(update.effective_chat.id)
    schedule_data = context.user_data.pop("pending_schedule", None)

    if not schedule_data:
        await query.edit_message_text("⚠️ 找不到暫存資料，請重新上傳。")
        return

    success = save_schedule(tg_id, schedule_data)
    if success:
        courses = schedule_data.get("courses", [])
        total = sum(c.get("credits", 0) for c in courses)
        await query.edit_message_text(
            f"✅ 課表儲存成功！\n"
            f"📚 {len(courses)} 門課 · {total} 學分\n\n"
            f"💡 使用 /start 返回主控台查詢"
        )
    else:
        await query.edit_message_text("❌ 儲存失敗！請先使用 /start → 設定個人身分。")


def _build_schedule_preview(schedule_data: dict, courses: list, total_credits: int) -> str:
    """建構課表預覽文字"""
    lines = [
        f"🔍 課表辨識結果預覽",
        f"━━━━━━━━━━━━━━",
        f"📚 {len(courses)} 門課 · {total_credits} 學分",
        f"🗓️ {schedule_data.get('academic_year', '?')}學年度 第{schedule_data.get('semester', '?')}學期",
        "",
    ]

    for day in range(1, 6):
        day_courses = [c for c in courses if c.get("day") == day]
        if day_courses:
            lines.append(f"📅 {DAY_NAMES.get(day, '')}：")
            for c in day_courses:
                periods = c.get("periods", [])
                p_str = f"第{periods[0]}-{periods[-1]}節" if periods else "?"
                lines.append(f"  • {c['name']} ({p_str})")
            lines.append("")

    lines.append("⚠️ 請確認是否正確：")
    return "\n".join(lines)


# ── 直接指令處理器 ──
async def cmd_my_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /my_schedule 指令"""
    from datetime import datetime, timezone, timedelta
    tg_id = get_tg_user_id(update.effective_chat.id)
    tz = timezone(timedelta(hours=8))
    day = datetime.now(tz).isoweekday()
    if day > 5:
        await update.message.reply_text("🎉 今天是假日，沒有課！\n💡 使用 /start → 我的課表 查詢其他天")
        return
    result = query_day_schedule(tg_id, day)
    await update.message.reply_text(result)


async def cmd_my_free(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /my_free 指令"""
    tg_id = get_tg_user_id(update.effective_chat.id)
    result = query_free_periods(tg_id)
    await update.message.reply_text(result)


async def cmd_my_credits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /my_credits 指令"""
    tg_id = get_tg_user_id(update.effective_chat.id)
    result = query_credit_summary(tg_id)
    await update.message.reply_text(result)
