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

from bot.telegram import get_user_memory, get_tg_user_id
from main import rag_pipeline
from tools.dcard_search_tool import search_dcard_professor, search_nqu_news
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
    
    # 處理 Markdown 連結，過濾 URL 中的 &lt;、&gt;、<、> 符號
    def replace_link(match):
        title = match.group(1)
        url = match.group(2)
        url = url.replace("&lt;", "").replace("&gt;", "").replace("<", "").replace(">", "").strip()
        return f'<a href="{url}">{title}</a>'
        
    text = re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, text)
    
    # 處理被 &lt; 和 &gt; 包裹的裸露 URL
    text = re.sub(
        r'(?<!href=")&lt;(https?://[^\s&<>]+)&gt;',
        r'<a href="\1">\1</a>',
        text
    )
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


async def safe_edit_tg_message(progress_msg, text_html: str, text_raw: str):
    """安全編輯 Telegram 訊息，支援 HTML 與純文字雙重保底，並具備 Rate Limit 防禦"""
    try:
        await progress_msg.edit_text(text_html, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.warning(f"⚠️ [Telegram] HTML 編輯失敗，退化至純文字發送: {e}")
        try:
            await progress_msg.edit_text(text_raw)
        except Exception as e2:
            logger.error(f"❌ [Telegram] 純文字編輯也失敗: {e2}")
            try:
                await progress_msg.edit_text(text_raw[:3500])
            except Exception as e3:
                logger.error(f"🚨 [Telegram] 編輯極致保底也失敗: {e3}")


async def safe_reply_tg_message(update: Update, text_html: str, text_raw: str):
    """安全發送 Telegram 新訊息，支援 HTML 與純文字雙重保底，帶有頻率限制重試"""
    for attempt in range(3):
        try:
            await update.message.reply_text(text_html, parse_mode=ParseMode.HTML)
            return
        except Exception as e:
            err_msg = str(e).lower()
            if "retry after" in err_msg or "too many requests" in err_msg:
                wait_secs = 2.0
                match = re.search(r"retry after (\d+)", err_msg)
                if match:
                    wait_secs = float(match.group(1)) + 0.5
                logger.warning(f"⚠️ [Telegram] 觸發頻率限制，等待 {wait_secs} 秒後重試...")
                await asyncio.sleep(wait_secs)
                continue
            
            logger.warning(f"⚠️ [Telegram] HTML 發送失敗，嘗試退化至純文字: {e}")
            try:
                await update.message.reply_text(text_raw)
                return
            except Exception as e2:
                err_msg2 = str(e2).lower()
                if "retry after" in err_msg2 or "too many requests" in err_msg2:
                    wait_secs = 2.0
                    match = re.search(r"retry after (\d+)", err_msg2)
                    if match:
                        wait_secs = float(match.group(1)) + 0.5
                    logger.warning(f"⚠️ [Telegram] 純文字發送觸發頻率限制，等待 {wait_secs} 秒後重試...")
                    await asyncio.sleep(wait_secs)
                    continue
                
                logger.error(f"❌ [Telegram] 純文字發送也失敗: {e2}")
                break
    
    logger.error("🚨 [Telegram] 該段訊息經過 3 次重試與雙重保底後依然發送失敗，略過此段以保護後續發送！")


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
        result = await search_dcard_professor(query)
        md_parts = smart_tg_split(str(result), max_len=3500)
        parts_html = [discord_md_to_tg_html(p) for p in md_parts]
        
        await safe_edit_tg_message(progress, parts_html[0], md_parts[0])
        for idx, (p_html, p_raw) in enumerate(zip(parts_html[1:], md_parts[1:]), start=2):
            await asyncio.sleep(0.5)
            await safe_reply_tg_message(update, p_html, p_raw)
    except Exception as e:
        logger.exception(f"❌ [Telegram] Dcard 搜尋錯誤")
        try:
            await progress.edit_text(f"❌ 搜尋失敗：{e}")
        except Exception:
            pass


async def _do_nqu_search(update: Update, query: str):
    progress = await update.message.reply_text(f"🏛️ 正在搜尋金大官網：{query}...")
    try:
        result = await search_nqu_news(query)
        md_parts = smart_tg_split(str(result), max_len=3500)
        parts_html = [discord_md_to_tg_html(p) for p in md_parts]
        
        await safe_edit_tg_message(progress, parts_html[0], md_parts[0])
        for idx, (p_html, p_raw) in enumerate(zip(parts_html[1:], md_parts[1:]), start=2):
            await asyncio.sleep(0.5)
            await safe_reply_tg_message(update, p_html, p_raw)
    except Exception as e:
        logger.exception(f"❌ [Telegram] 金大官網搜尋錯誤")
        try:
            await progress.edit_text(f"❌ 搜尋失敗：{e}")
        except Exception:
            pass


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

    # 🆕 雙進程無重啟熱重載監控 (Hot Reload Alignment)
    from rag.index_manager import check_and_reload_index_if_needed
    global_nodes, global_faiss, global_bm25, has_changed = check_and_reload_index_if_needed(
        global_nodes, global_faiss, global_bm25
    )
    if has_changed:
        context.bot_data["global_nodes"] = global_nodes
        context.bot_data["global_faiss"] = global_faiss
        context.bot_data["global_bm25"] = global_bm25
        logger.info("⚡ [Telegram] 偵測到索引已在磁碟更新，Telegram 內存已同步完成熱更新！")

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
        try:
            await progress_msg.edit_text("🔍 正在搜尋校園知識庫...")
        except Exception:
            pass

        answer = await rag_pipeline(
            question,
            global_nodes, global_faiss, global_bm25,
            memory, False, user_profile, tg_id
        )

        # 加上暱稱前綴
        nick = user_profile.get("student_name", "") if user_profile else ""
        if nick:
            answer = f"{nick}，{answer}"

        # 🆕 完美 HTML 安全發送機制：先對原始 Markdown 切分，再單獨轉為 HTML！
        md_parts = smart_tg_split(answer, max_len=3500)  # 使用較保守的 3500 字元限制
        parts_html = [discord_md_to_tg_html(p) for p in md_parts]

        logger.info(f"✅ [Telegram] 回答完成 | {user_name}{profile_tag} | {len(answer)} 字 | {len(md_parts)} 段")

        # 編輯第一段
        await safe_edit_tg_message(progress_msg, parts_html[0], md_parts[0])

        # 依序發送後續段落
        for idx, (p_html, p_raw) in enumerate(zip(parts_html[1:], md_parts[1:]), start=2):
            # 主動加入微小延遲防止 Rate Limit
            await asyncio.sleep(0.5)
            logger.info(f"  📥 [Telegram] 正在發送第 {idx}/{len(md_parts)} 段...")
            await safe_reply_tg_message(update, p_html, p_raw)

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
        logger.exception(f"❌ [Telegram] 錯誤 | 使用者：{user_name}")
        try:
            await progress_msg.edit_text("❌ 糟糕！查詢時發生錯誤，請稍後再試。")
        except Exception:
            pass
