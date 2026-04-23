# -*- coding: utf-8 -*-
"""
telegram_bot.py — Telegram Bot 啟動入口
=========================================
"""

import logging
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters,
)

import config
from main import load_and_index
from rag.query_router import init_known_registry

# ── 匯入 Telegram 子模組 ──
from bot.telegram.tg_ui_menus import setup_bot_commands, tg_start_command, tg_button_callback
from bot.telegram.tg_events import handle_message, handle_file_upload, cmd_dcard_search, cmd_nqu_news
from bot.telegram.tg_cmd_schedule import cmd_my_schedule, cmd_my_free, cmd_my_credits
from bot.telegram.tg_cmd_transcript import cmd_my_credits_total, cmd_my_gpa, cmd_my_failed

# ── Logging ──
logger = logging.getLogger("telegram_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [TG] %(name)-13s | %(levelname)-7s | %(message)s",
    datefmt=config.LOG_DATE_FORMAT,
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def error_handler(update, context):
    """捕捉 Telegram 所有未預期的錯誤"""
    logger.error("❌ [Telegram] 發生未預期錯誤:", exc_info=context.error)


def main():
    logger.info("啟動 Telegram Bot...")

    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("❌ 找不到 TELEGRAM_BOT_TOKEN，請確認 .env 設定！")
        return

    # ── 載入 AI 核心 ──
    logger.info("📂 載入資料與索引...")
    try:
        global_nodes, global_faiss, global_bm25 = load_and_index()
        init_known_registry(global_nodes)
        logger.info(f"✅ 索引載入完成！共 {len(global_nodes)} 個文件區段")

        # 預先載入 Reranker 模型（非致命）
        try:
            from rag.reranker import get_reranker
            get_reranker()
            logger.info("✅ Reranker 模型預載完成")
        except Exception as e:
            logger.warning(f"⚠️ Reranker 預載失敗（首次查詢時會自動重試）：{e}")
        
        # 預先載入 CKIP 斷詞模型（消除首次查詢的冷啟動延遲）
        try:
            from nlp_utils import get_ws_model
            get_ws_model()
            logger.info("✅ CKIP 斷詞模型預載完成")
        except Exception as e:
            logger.warning(f"⚠️ CKIP 預載失敗（首次查詢時會自動重試）：{e}")

    except Exception as e:
        logger.error(f"❌ 啟動載入失敗：{e}")
        return

    # ── 建立 Application ──
    app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.bot_data["global_nodes"] = global_nodes
    app.bot_data["global_faiss"] = global_faiss
    app.bot_data["global_bm25"] = global_bm25

    # ── 註冊處理器（順序重要！）──
    # 1. 指令
    app.add_handler(CommandHandler("start", tg_start_command))
    app.add_handler(CommandHandler("help", tg_start_command))
    app.add_handler(CommandHandler("profile", tg_start_command))
    app.add_handler(CommandHandler("my_schedule", cmd_my_schedule))
    app.add_handler(CommandHandler("my_free", cmd_my_free))
    app.add_handler(CommandHandler("my_credits", cmd_my_credits))
    app.add_handler(CommandHandler("my_gpa", cmd_my_gpa))
    app.add_handler(CommandHandler("my_credits_total", cmd_my_credits_total))
    app.add_handler(CommandHandler("my_failed", cmd_my_failed))
    app.add_handler(CommandHandler("dcard_search", cmd_dcard_search))
    app.add_handler(CommandHandler("nqu_news", cmd_nqu_news))
    app.add_handler(CommandHandler("calendar", tg_start_command))

    # 2. InlineKeyboard 回調
    app.add_handler(CallbackQueryHandler(tg_button_callback))

    # 3. 檔案上傳（圖片 + 文件，排在文字之前）
    app.add_handler(MessageHandler(filters.PHOTO, handle_file_upload))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file_upload))

    # 4. 一般文字（RAG + ForceReply 路由）
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    # ── 啟動時註冊左下角藍色 Menu 按鈕 ──
    app.post_init = setup_bot_commands

    # ── 開始輪詢 ──
    logger.info("⚡ Telegram Bot 開始監聽...")
    app.run_polling()


if __name__ == "__main__":
    main()
