# -*- coding: utf-8 -*-
"""
bot/telegram/tg_events.py — Telegram 訊息處理（文字 + 檔案）
============================================================
"""

import re
import html
import asyncio
import logging
import time

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.telegram import get_user_memory, get_tg_user_id, gpu_semaphore
from main import rag_pipeline
from tools.auth import get_user_profile
from bot.telegram.tg_audit import tg_send_audit_dm

logger = logging.getLogger("telegram_bot")


# =========================================================
# Markdown → Telegram HTML
# =========================================================
def discord_md_to_tg_html(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r'```(?:\w+\n)?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'^#+\s+(.*)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    return text


def smart_tg_split(text: str, max_len: int = 4000) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        cut = text.rfind('\n', 0, max_len)
        if cut <= 0:
            cut = max_len
        parts.append(text[:cut])
        text = text[cut:].lstrip('\n')
    return parts


# =========================================================
# 檔案上傳路由器
# =========================================================
async def handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """路由圖片/文件上傳至課表或成績單處理器"""
    upload_type = context.user_data.get("awaiting_upload")

    if upload_type == "schedule":
        from bot.telegram.tg_cmd_schedule import process_schedule_upload
        await process_schedule_upload(update, context)
    elif upload_type == "transcript":
        from bot.telegram.tg_cmd_transcript import process_transcript_upload
        await process_transcript_upload(update, context)
    else:
        await update.message.reply_text(
            "📌 請先透過 /start 主控台選擇上傳類型\n"
            "（📅 上傳課表 或 📊 上傳成績單）"
        )


# =========================================================
# 搜尋指令處理
# =========================================================
async def cmd_dcard_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /dcard_search 指令"""
    args = update.message.text.replace("/dcard_search", "").strip()
    if not args:
        from telegram import ForceReply
        await update.message.reply_text(
            "請輸入搜尋關鍵字（例如：英文教授、王大明）",
            reply_markup=ForceReply(selective=True),
        )
        context.user_data["awaiting_search"] = "dcard"
        return
    await _do_dcard_search(update, args)


async def cmd_nqu_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /nqu_news 指令"""
    args = update.message.text.replace("/nqu_news", "").strip()
    if not args:
        from telegram import ForceReply
        await update.message.reply_text(
            "請輸入搜尋關鍵字（例如：獎學金、招生、住宿）",
            reply_markup=ForceReply(selective=True),
        )
        context.user_data["awaiting_search"] = "nqu"
        return
    await _do_nqu_search(update, args)


async def _do_dcard_search(update: Update, query: str):
    progress = await update.message.reply_text(f"🔍 正在搜尋 Dcard：{query}...")
    try:
        from tools.dcard_search_tool import search_dcard_professor
        result = await asyncio.to_thread(search_dcard_professor, query)
        parts = smart_tg_split(str(result))
        await progress.edit_text(parts[0])
        for part in parts[1:]:
            await update.message.reply_text(part)
    except Exception as e:
        logger.exception(f"❌ [Telegram] Dcard 搜尋錯誤")
        await progress.edit_text(f"❌ 搜尋失敗：{e}")


async def _do_nqu_search(update: Update, query: str):
    progress = await update.message.reply_text(f"🏛️ 正在搜尋金大官網：{query}...")
    try:
        from tools.dcard_search_tool import search_nqu_news
        result = await asyncio.to_thread(search_nqu_news, query)
        parts = smart_tg_split(str(result))
        await progress.edit_text(parts[0])
        for part in parts[1:]:
            await update.message.reply_text(part)
    except Exception as e:
        logger.exception(f"❌ [Telegram] 金大官網搜尋錯誤")
        await progress.edit_text(f"❌ 搜尋失敗：{e}")


# =========================================================
# 文字訊息主處理器
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理一般文字訊息"""
    if not update.message or not update.message.text:
        return

    # ── 優先檢查 ForceReply 狀態 ──
    if context.user_data.get("awaiting_profile_info"):
        from bot.telegram.tg_ui_menus import process_profile_name_input
        await process_profile_name_input(update, context)
        return

    if context.user_data.get("awaiting_calendar_url"):
        from bot.telegram.tg_ui_menus import process_calendar_url_input
        await process_calendar_url_input(update, context)
        return

    search_type = context.user_data.get("awaiting_search")
    if search_type:
        context.user_data.pop("awaiting_search")
        if search_type == "dcard":
            await _do_dcard_search(update, update.message.text.strip())
        elif search_type == "nqu":
            await _do_nqu_search(update, update.message.text.strip())
        return

    # ── RAG 問答 Pipeline ──
    global_nodes = context.bot_data.get("global_nodes")
    global_faiss = context.bot_data.get("global_faiss")
    global_bm25 = context.bot_data.get("global_bm25")

    question = update.message.text
    start_time = time.time()
    tg_id = get_tg_user_id(update.effective_chat.id)
    user_name = update.effective_user.first_name if update.effective_user else "同學"

    logger.info(f"🔍 [Telegram] 處理問題：{question[:60]} | 使用者：{user_name} (ID: {tg_id})")

    user_profile = get_user_profile(tg_id)
    profile_tag = ""
    if user_profile:
        profile_tag = f" | 身分：{user_profile.get('department', '?')} {user_profile.get('grade', '?')}年級"
        logger.info(f"  👤 [Telegram] 身分已識別{profile_tag}")

    memory = get_user_memory(update.effective_chat.id)
    progress_msg = await update.message.reply_text("⏳ AI 正在思考中，請稍候...")

    try:
        import bot.telegram as _tg_bot
        _tg_bot.active_gpu_requests += 1

        async with gpu_semaphore:
            try:
                await progress_msg.edit_text("🔍 正在搜尋校園知識庫...")
            except Exception:
                pass

            answer = await asyncio.to_thread(
                rag_pipeline, question,
                global_nodes, global_faiss, global_bm25,
                memory, False, user_profile, tg_id
            )

        _tg_bot.active_gpu_requests -= 1

        # 加上暱稱前綴
        nick = user_profile.get("student_name", "") if user_profile else ""
        if nick:
            answer = f"{nick}，{answer}"

        tg_html = discord_md_to_tg_html(answer)
        parts = smart_tg_split(tg_html)

        logger.info(f"✅ [Telegram] 回答完成 | {user_name}{profile_tag} | {len(answer)} 字 | {len(parts)} 段")

        try:
            await progress_msg.edit_text(parts[0], parse_mode=ParseMode.HTML)
        except Exception:
            try:
                await progress_msg.edit_text(parts[0])
            except Exception:
                await progress_msg.edit_text(answer[:4000])

        for part in parts[1:]:
            try:
                await update.message.reply_text(part, parse_mode=ParseMode.HTML)
            except Exception:
                await update.message.reply_text(part)

        # 傳送審計記錄
        await tg_send_audit_dm(
            user_id=f"tg_{update.effective_user.id}",
            user_name=update.effective_user.full_name,
            question=question,
            answer=answer,
            channel_name="Telegram私訊",
            user_profile=user_profile,
            elapsed=(time.time() - start_time)
        )

    except Exception as e:
        import bot.telegram as _tg_bot
        _tg_bot.active_gpu_requests -= 1
        logger.exception(f"❌ [Telegram] 錯誤 | 使用者：{user_name}")
        try:
            await progress_msg.edit_text("❌ 糟糕！查詢時發生錯誤，請稍後再試。")
        except Exception:
            pass
