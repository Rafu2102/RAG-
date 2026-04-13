# -*- coding: utf-8 -*-
"""
telegram_bot.py — Telegram Bot 啟動入口
=========================================
"""

import logging
import asyncio
import io
import time
import re
import html
from telegram import Update
from telegram.constants import ParseMode, ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

import config
from main import load_and_index, rag_pipeline
from rag.query_router import init_known_registry
from llm.llm_answer import ConversationMemory

user_memories: dict[int, ConversationMemory] = {}

def get_user_memory(user_id: int) -> ConversationMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationMemory()
    return user_memories[user_id]

# ── Logging 設定 ──
logger = logging.getLogger("telegram_bot")
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 全域資源
global_nodes = None
global_faiss = None
global_bm25 = None
gpu_semaphore = asyncio.Semaphore(1)


def discord_md_to_tg_html(text: str) -> str:
    """將 Discord 格式的 Markdown 轉換為 Telegram 支援的 HTML"""
    # 1. 逃脫 HTML 特殊字元 (<, >, &)
    text = html.escape(text)
    
    # 2. Code blocks ```...```
    text = re.sub(r'```(?:\w+\n)?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    
    # 3. Inline code `...`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # 4. Bold **...**
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # 5. Italic *...* 
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    
    # 6. Headers # Header
    text = re.sub(r'^#+\s+(.*)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    
    # 7. Blockquotes > quote
    text = re.sub(r'^>\s+(.*)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    
    # 8. Links [text](url) Note: url got html escaped, so we safely construct HTML tag
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    
    return text


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /start 指令"""
    user_name = update.effective_user.first_name if update.effective_user else "未知"
    logger.info(f"🚀 [Telegram] 收到 /start 指令來自使用者：{user_name}")
    try:
        welcome_text = (
            "哈囉！我是 **NQU AI 校園智慧助理** 👋\n\n"
            "你可以直接問我關於金門大學的：\n"
            "📖 課程資訊 (例如：有哪些好玩的通識課？)\n"
            "👨‍🏫 教授資訊 (例如：柯志亨的實驗室在哪？)\n"
            "📅 校園行事曆 (例如：什麼時候放春假？)\n\n"
            "目前我還在測試階段，請隨便問我問題吧！"
        )
        html_text = discord_md_to_tg_html(welcome_text)
        await update.message.reply_text(html_text, parse_mode=ParseMode.HTML)
        logger.info(f"✅ [Telegram] 成功回覆 /start 指令給 {user_name}")
    except Exception as e:
        logger.exception(f"❌ [Telegram] 處理 /start 時發生嚴重錯誤：{e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理一般文字訊息"""
    global global_nodes, global_faiss, global_bm25
    
    if not update.message or not update.message.text:
        return

    question = update.message.text
    chat_id = str(update.message.chat_id)
    user_name = update.effective_user.first_name

    logger.info(f"🔍 [Telegram] 處理問題：{question[:60]} | 使用者：{user_name} (ID: {chat_id})")

    # 顯示「輸入中...」的打字狀態
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    # 取得這個使用者的記憶庫
    # 注意：這裡使用 chat_id 的 hash 或者強轉 int 來確保相容
    uid_hash = int(abs(hash(chat_id)) % (10 ** 12))  # 防止 TG ID 太短或包含負號
    memory = get_user_memory(uid_hash)
    
    try:
        async with gpu_semaphore:
            logger.info(f"🚀 [Telegram] 取得執行權 | {user_name} 開始處理問題")
            
            # 使用 RAG Pipeline
            answer = await asyncio.to_thread(
                rag_pipeline, question,
                global_nodes, global_faiss, global_bm25,
                memory, False, None, chat_id
            )
            
        tg_html_answer = discord_md_to_tg_html(answer)
        logger.info(f"✅ [Telegram] 回答完成 | 使用者：{user_name} | {len(answer)} 字")
        
        # 由於 Telegram 每則訊息有 4096 字元限制，這裡需做分片（如果過長）
        if len(tg_html_answer) <= 4000:
            await update.message.reply_text(tg_html_answer, parse_mode=ParseMode.HTML)
        else:
            # 簡易的字串分段 (不中斷 HTML 標籤有難度，簡單切分)
            for i in range(0, len(tg_html_answer), 4000):
                await update.message.reply_text(tg_html_answer[i:i+4000], parse_mode=ParseMode.HTML)
                
    except Exception as e:
        logger.exception(f"❌ [Telegram] 錯誤 | 使用者：{user_name}")
        await update.message.reply_text("❌ 糟糕！在查詢時發生錯誤，請稍後再試。")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """捕捉 Telegram 所有未預期的錯誤"""
    logger.error(f"❌ [Telegram] 發生未預期錯誤:", exc_info=context.error)

def main():
    global global_nodes, global_faiss, global_bm25
    
    logger.info("啟動 Telegram Bot...")
    
    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("❌ 找不到 TELEGRAM_BOT_TOKEN，請確認 .env 設定！")
        return

    logger.info("📂 載入資料與索引...")
    try:
        global_nodes, global_faiss, global_bm25 = load_and_index()
        init_known_registry(global_nodes)
        logger.info(f"✅ 索引載入完成！共 {len(global_nodes)} 個文件區段")
    except Exception as e:
        logger.error(f"❌ 索引載入失敗：{e}")
        return

    # 建立 Application
    app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()

    # 註冊處理器
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    # 開始輪詢
    logger.info("⚡ Telegram Bot 開始監聽...")
    app.run_polling()


if __name__ == "__main__":
    main()
