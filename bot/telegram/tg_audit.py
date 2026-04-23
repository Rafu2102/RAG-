# -*- coding: utf-8 -*-
"""
bot/telegram/tg_audit.py — Telegram 審計通訊模組
==================================================
透過本地 HTTP POST 請求，將 Telegram 發生的事件
發送到 Discord 機器人的 IPC Server，再由 Discord 轉發到監控頻道。
這保證了兩邊完全非同步、無阻塞隔離。
"""

import os
import logging
import json
import asyncio
import aiohttp

logger = logging.getLogger("telegram_bot")

async def tg_send_audit_dm(
    user_id: str,
    user_name: str,
    question: str,
    answer: str,
    channel_name: str = "私訊",
    user_profile: dict = None,
    pipeline_log: str = None,
    elapsed: float = 0.0,
):
    """將使用者的問答紀錄轉發到 Discord 監控系統"""
    if os.environ.get("ENABLE_BOT_IPC") != "1":
        return

    port = int(os.environ.get("IPC_PORT", 50505))
    url = f"http://127.0.0.1:{port}/audit"

    payload = {
        "type": "ask",
        "user_id": user_id,
        "user_name": user_name,
        "source": "[Telegram]",
        "channel_name": channel_name,
        "question": question,
        "answer": answer,
        "user_profile": user_profile,
        "pipeline_log": pipeline_log,
        "elapsed": elapsed
    }

    try:
        # 使用 asyncio.create_task 放後台，避免卡住回應使用者
        asyncio.create_task(_send_ipc_post(url, payload))
    except Exception as e:
        logger.error(f"❌ 傳送 IPC 審計紀錄失敗: {e}")


async def tg_send_audit_event(
    title: str,
    description: str,
    fields: list[tuple[str, str, bool]] = None,
    color_val: int = 3447003, # 預設 blurple
):
    """將系統事件轉發到 Discord 監控系統"""
    if os.environ.get("ENABLE_BOT_IPC") != "1":
        return

    port = int(os.environ.get("IPC_PORT", 50505))
    url = f"http://127.0.0.1:{port}/audit"

    payload = {
        "type": "event",
        "title": f"[Telegram] {title}",
        "description": description,
        "fields": fields or [],
        "color": color_val
    }

    try:
        asyncio.create_task(_send_ipc_post(url, payload))
    except Exception as e:
        logger.error(f"❌ 傳送 IPC 審計事件失敗: {e}")


async def _send_ipc_post(url: str, payload: dict):
    """實際執行非同步 POST 請求"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=5.0) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.warning(f"⚠️ IPC 伺服器回傳錯誤: {text}")
    except Exception as e:
        logger.warning(f"⚠️ 無法連接到 Discord IPC 伺服器，略過審計發送 ({e})")
