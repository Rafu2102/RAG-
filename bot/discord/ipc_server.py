# -*- coding: utf-8 -*-
"""
bot/discord/ipc_server.py — Discord 本地 IPC 伺服器
===================================================
負責接收來自 Telegram 機器人的本地 HTTP 請求，
並透過 Discord Bot 的連線發送到監控頻道。
"""

import os
import logging
from aiohttp import web
import discord

from bot.discord.audit import send_audit_dm, send_audit_log, send_debug_log

logger = logging.getLogger("discord_bot")

class MockAvatar:
    @property
    def url(self):
        return "https://telegram.org/img/t_logo.png"

class MockUser:
    """偽造一個 Discord User 物件，欺騙 audit_dm 函數"""
    def __init__(self, user_id: str, display_name: str):
        self.id = user_id
        self.display_name = display_name
        self.display_avatar = MockAvatar()


async def handle_audit(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        req_type = data.get("type")
        
        if req_type == "ask":
            # 這是 RAG 問答紀錄
            student = MockUser(data.get("user_id", "Unknown"), data.get("user_name", "Unknown"))
            
            # 發送給 send_audit_dm
            await send_audit_dm(
                student=student,
                question=data.get("question", ""),
                answer=data.get("answer", ""),
                source=data.get("source", "[Telegram]"),
                channel_name=data.get("channel_name", "私訊"),
                user_profile=data.get("user_profile")
            )
            
            # 如果有 debug_log，一併發送
            pipeline_log = data.get("pipeline_log")
            if pipeline_log:
                await send_debug_log(
                    student=student,
                    question=data.get("question", ""),
                    answer=data.get("answer", ""),
                    pipeline_log=pipeline_log,
                    source=data.get("source", "[Telegram]"),
                    elapsed=data.get("elapsed", 0.0)
                )
                
        elif req_type == "event":
            # 一般系統事件 (如綁定、註冊等)
            title = data.get("title", "事件")
            description = data.get("description", "")
            
            # 重組 fields
            raw_fields = data.get("fields", [])
            fields = []
            for f in raw_fields:
                if len(f) == 3:
                    fields.append((f[0], f[1], f[2]))
            
            # 重組顏色
            color_val = data.get("color")
            color = discord.Color(color_val) if color_val is not None else None
            
            await send_audit_log(
                title=title,
                description=description,
                color=color,
                fields=fields
            )
        
        return web.json_response({"status": "ok"})
        
    except Exception as e:
        logger.error(f"❌ IPC 伺服器處理錯誤: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


async def start_ipc_server():
    """啟動本地 IPC 伺服器"""
    if os.environ.get("ENABLE_BOT_IPC") != "1":
        return

    port = int(os.environ.get("IPC_PORT", 50505))
    app = web.Application()
    app.router.add_post('/audit', handle_audit)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', port)
    
    try:
        await site.start()
        logger.info(f"🔌 Discord IPC 伺服器已啟動於 127.0.0.1:{port}，等待 Telegram 訊號...")
    except Exception as e:
        logger.error(f"❌ 啟動 IPC 伺服器失敗: {e}")
