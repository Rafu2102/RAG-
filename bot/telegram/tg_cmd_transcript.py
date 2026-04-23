# -*- coding: utf-8 -*-
"""
bot/telegram/tg_cmd_transcript.py — 成績單上傳與查詢
=====================================================
"""

import json
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.telegram import get_tg_user_id
from tools.transcript_manager import (
    save_transcript, query_credit_progress, query_failed_courses, query_gpa,
)

logger = logging.getLogger("telegram_bot")


async def process_transcript_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理成績單上傳（PDF/圖片 OCR 或 JSON 文件）"""
    tg_id = get_tg_user_id(update.effective_chat.id)
    context.user_data.pop("awaiting_upload", None)

    # ── 圖片 OCR ──
    if update.message.photo:
        progress = await update.message.reply_text("⏳ Gemini 3.1 Pro 正在辨識成績單...\n請稍候約 15-30 秒 🔍")
        try:
            photo = update.message.photo[-1]
            file = await photo.get_file()
            image_bytes = await file.download_as_bytearray()

            from tools.auth import get_user_profile
            user_profile = get_user_profile(tg_id)

            from tools.ocr_engine import ocr_transcript
            transcript_data = await ocr_transcript(
                file_bytes_list=[bytes(image_bytes)],
                mime_types=["image/jpeg"],
                user_profile=user_profile,
            )

            if not transcript_data.get("semesters"):
                await progress.edit_text("⚠️ 辨識完成但沒有找到任何學期資料！\n請確認上傳的是「歷年成績表」。")
                return

            context.user_data["pending_transcript"] = transcript_data
            preview = _build_transcript_preview(transcript_data)
            keyboard = [
                [InlineKeyboardButton("✅ 確認儲存", callback_data="confirm_transcript"),
                 InlineKeyboardButton("❌ 取消", callback_data="cancel_transcript")],
            ]
            await progress.edit_text(preview, reply_markup=InlineKeyboardMarkup(keyboard))

        except Exception as e:
            logger.exception("❌ [Telegram] 成績單 OCR 錯誤")
            await progress.edit_text(f"❌ 辨識失敗：{e}\n💡 請改用 JSON 檔案匯入。")
        return

    # ── 文件（PDF 或 JSON）──
    if update.message.document:
        doc = update.message.document
        filename = (doc.file_name or "").lower()
        content_type = doc.mime_type or ""

        is_json = filename.endswith(".json") or "json" in content_type
        is_pdf = filename.endswith(".pdf") or "pdf" in content_type

        if is_json:
            try:
                file = await doc.get_file()
                content = await file.download_as_bytearray()
                data = json.loads(content.decode("utf-8"))

                if "transcript" in data:
                    transcript_data = data["transcript"]
                elif "semesters" in data:
                    transcript_data = data
                else:
                    await update.message.reply_text("❌ JSON 格式錯誤：找不到 semesters 欄位。")
                    return

                success = save_transcript(tg_id, {"transcript": transcript_data})
                if success:
                    semesters = transcript_data.get("semesters", [])
                    total_courses = sum(len(s.get("courses", [])) for s in semesters)
                    await update.message.reply_text(
                        f"✅ 成績單匯入成功！\n"
                        f"📚 {len(semesters)} 學期 · {total_courses} 門課\n\n"
                        f"💡 使用 /start 返回主控台查詢成績"
                    )
                else:
                    await update.message.reply_text("❌ 儲存失敗！請先使用 /start → 設定個人身分。")
            except json.JSONDecodeError as e:
                await update.message.reply_text(f"❌ JSON 解析失敗：{e}")
            except Exception as e:
                logger.exception("❌ [Telegram] 成績單 JSON 匯入錯誤")
                await update.message.reply_text(f"❌ 匯入失敗：{e}")
            return

        if is_pdf:
            progress = await update.message.reply_text("⏳ Gemini 3.1 Pro 正在辨識成績單 PDF...\n📄 PDF 會先轉為高畫質圖片再辨識，約需 15-30 秒 🔍")
            try:
                import tempfile
                from pathlib import Path as _Path

                file = await doc.get_file()
                # 使用暫存檔下載，避免 bytearray 截斷導致 PyMuPDF 解析失敗
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = tmp.name
                await file.download_to_drive(tmp_path)
                file_bytes = _Path(tmp_path).read_bytes()

                # 清理暫存檔
                try:
                    _Path(tmp_path).unlink()
                except Exception:
                    pass

                from tools.auth import get_user_profile
                user_profile = get_user_profile(tg_id)

                from tools.ocr_engine import ocr_transcript
                transcript_data = await ocr_transcript(
                    file_bytes_list=[file_bytes],
                    mime_types=["application/pdf"],
                    user_profile=user_profile,
                )

                if not transcript_data.get("semesters"):
                    await progress.edit_text("⚠️ 辨識完成但沒有找到任何學期資料！\n請確認上傳的是「歷年成績表」PDF。")
                    return

                context.user_data["pending_transcript"] = transcript_data
                preview = _build_transcript_preview(transcript_data)
                keyboard = [
                    [InlineKeyboardButton("✅ 確認儲存", callback_data="confirm_transcript"),
                     InlineKeyboardButton("❌ 取消", callback_data="cancel_transcript")],
                ]
                await progress.edit_text(preview, reply_markup=InlineKeyboardMarkup(keyboard))

            except Exception as e:
                logger.exception("❌ [Telegram] 成績單 PDF OCR 錯誤")
                await progress.edit_text(f"❌ 辨識失敗：{e}\n💡 請改用 JSON 檔案匯入。")
            return

        await update.message.reply_text("❌ 不支援的檔案格式。\n請上傳 PDF、PNG/JPG 或 JSON。")
        return

    await update.message.reply_text("📄 請傳送成績單 PDF、截圖或 .json 檔案。")


async def save_pending_transcript(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """確認按鈕回調：儲存暫存的 OCR 成績單"""
    query = update.callback_query
    tg_id = get_tg_user_id(update.effective_chat.id)
    transcript_data = context.user_data.pop("pending_transcript", None)

    if not transcript_data:
        await query.edit_message_text("⚠️ 找不到暫存資料，請重新上傳。")
        return

    success = save_transcript(tg_id, {"transcript": transcript_data})
    if success:
        semesters = transcript_data.get("semesters", [])
        total_courses = sum(len(s.get("courses", [])) for s in semesters)
        await query.edit_message_text(
            f"✅ 成績單儲存成功！\n"
            f"📚 {len(semesters)} 學期 · {total_courses} 門課\n\n"
            f"💡 使用 /start 返回主控台查詢"
        )
    else:
        await query.edit_message_text("❌ 儲存失敗！請先使用 /start → 設定個人身分。")


def _build_transcript_preview(transcript_data: dict) -> str:
    """建構成績單預覽文字"""
    semesters = transcript_data.get("semesters", [])
    total_courses = sum(len(s.get("courses", [])) for s in semesters)
    overall_gpa = transcript_data.get("overall_gpa", "?")
    student_id = transcript_data.get("student_id", "")

    lines = [
        "🔍 成績單辨識結果預覽",
        "━━━━━━━━━━━━━━",
    ]
    if student_id:
        lines.append(f"🆔 學號：{student_id}")
    lines.extend([
        f"📚 {len(semesters)} 學期 · {total_courses} 門課",
        f"📊 學業總平均：{overall_gpa}",
        "",
    ])

    for sem in semesters[:6]:
        year = sem.get("year", "?")
        semester = sem.get("semester", "?")
        courses = sem.get("courses", [])
        gpa = sem.get("gpa", "?")
        lines.append(f"📅 {year}-{semester} | {len(courses)}門 | GPA {gpa}")

    if len(semesters) > 6:
        lines.append(f"  ...等共 {len(semesters)} 學期")

    lines.extend(["", "⚠️ 請確認是否正確："])
    return "\n".join(lines)


# ── 直接指令處理器 ──
async def cmd_my_credits_total(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_id = get_tg_user_id(update.effective_chat.id)
    result = query_credit_progress(tg_id)
    if len(result) > 4000:
        result = result[:3990] + "\n..."
    await update.message.reply_text(result)


async def cmd_my_gpa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_id = get_tg_user_id(update.effective_chat.id)
    result = query_gpa(tg_id)
    await update.message.reply_text(result)


async def cmd_my_failed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_id = get_tg_user_id(update.effective_chat.id)
    result = query_failed_courses(tg_id)
    await update.message.reply_text(result)
