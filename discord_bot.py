# -*- coding: utf-8 -*-
"""
discord_bot.py — Discord Bot 啟動入口
=======================================
此檔案已模組化拆分到 bot/ 目錄下：
  bot/__init__.py    → 共用 client、tree、全域狀態
  bot/audit.py       → 監控審計 Log
  bot/cmd_identity.py → OAuth 身分註冊 (/identity_login)
  bot/cmd_admin.py   → 管理員指令 (/rebuild, /admin_broadcast, ...)
  bot/cmd_groups.py  → 群組邀請 (/join_group, GroupInviteView)
  bot/cmd_ask.py     → 問答指令 (/ask, /add_calendar, /dcard_search)
  bot/events.py      → Discord Events (on_ready, on_message)

本檔案僅作為啟動入口，匯入所有子模組後執行 client.run()。
"""

import logging

# ── 匯入 bot 核心 (建立 client、tree、全域狀態) ──
from bot import client, DISCORD_BOT_TOKEN

# ── 匯入所有子模組，讓 @tree.command 和 @client.event 被註冊 ──
import bot.cmd_identity   # noqa: F401 — /identity_login
import bot.cmd_admin      # noqa: F401 — /rebuild, /clear, /admin_*
import bot.cmd_groups     # noqa: F401 — /join_group, GroupInviteView
import bot.cmd_ask        # noqa: F401 — /ask, /add_calendar, /dcard_search
import bot.cmd_schedule   # noqa: F401 — /upload_schedule, /my_schedule, /my_free, /my_credits
import bot.cmd_transcript # noqa: F401 — /upload_transcript, /my_credits_total, /my_gpa, /my_failed
import bot.events         # noqa: F401 — on_ready, on_message

logger = logging.getLogger("discord_bot")

if __name__ == "__main__":
    logger.info("啟動 Discord Bot...")
    client.run(DISCORD_BOT_TOKEN)
