# -*- coding: utf-8 -*-
"""
bot/audit.py — 監控審計 Log 系統
=================================
將所有系統事件（問答、廣播、管理操作）發送到伺服器的監控頻道。

頻道偵測優先順序：
1. .env 的 AUDIT_CHANNEL_ID（最可靠）
2. 自動搜尋頻道名稱包含 "bot" 或 "監控" 或 "log" 的文字頻道
"""

import logging
import discord  # type: ignore

from bot import client, AUDIT_CHANNEL_ID

logger = logging.getLogger("discord_bot")

# 快取頻道物件，避免每次都搜尋
_audit_channel_cache: discord.TextChannel | None = None

# 可匹配的頻道名稱關鍵字（只要包含任一個就算匹配）
_CHANNEL_NAME_KEYWORDS = [
    "bot_modify", "bot-modify", "bot_監控", "bot-監控",
    "bot_log", "bot-log", "監控大廳", "audit", "bot_audit",
]


async def _get_audit_channel() -> discord.TextChannel | None:
    """
    找到伺服器中的監控頻道（快取）。
    優先用 AUDIT_CHANNEL_ID（精確），否則搜尋名稱匹配。
    """
    global _audit_channel_cache  # type: ignore
    if _audit_channel_cache:
        return _audit_channel_cache
    
    # 方法 1：用 .env 中的頻道 ID（最精確）
    if AUDIT_CHANNEL_ID:
        try:
            ch = client.get_channel(int(AUDIT_CHANNEL_ID))
            if ch and isinstance(ch, discord.TextChannel):
                _audit_channel_cache = ch
                logger.info(f"📋 審計頻道已綁定 (ID 模式): #{ch.name} (ID: {ch.id})")
                return ch
        except (ValueError, TypeError):
            pass
    
    # 方法 2：搜尋頻道名稱
    for guild in client.guilds:
        for channel in guild.text_channels:
            ch_name = channel.name.lower()
            for keyword in _CHANNEL_NAME_KEYWORDS:
                if keyword.lower() in ch_name:
                    _audit_channel_cache = channel
                    logger.info(f"📋 審計頻道已綁定 (名稱匹配): #{channel.name} (ID: {channel.id})")
                    logger.info(f"   💡 建議在 .env 加上 AUDIT_CHANNEL_ID={channel.id} 以確保穩定偵測")
                    return channel
    
    # 方法 3：找不到就列出所有可用頻道供管理員參考
    all_channels = []
    for guild in client.guilds:
        all_channels.extend([f"#{c.name} (ID: {c.id})" for c in guild.text_channels[:20]])
    
    logger.warning(
        f"📋 找不到監控頻道！請在 .env 加入 AUDIT_CHANNEL_ID=<頻道ID>\n"
        f"   可用的文字頻道：{', '.join(all_channels[:10])}"
    )
    return None


def reset_audit_cache():
    """重設快取（用於 on_ready 時重新偵測）"""
    global _audit_channel_cache  # type: ignore
    _audit_channel_cache = None


# =========================================================================
# 🕵️ 問答審計 — 學生問答紀錄
# =========================================================================

async def send_audit_dm(
    student: discord.User | discord.Member,
    question: str,
    answer: str,
    source: str = "@tag",
    channel_name: str = "私訊",
    user_profile: dict = None,
):
    """將學生的問答紀錄發送到監控頻道。"""
    audit_channel = await _get_audit_channel()
    if not audit_channel:
        return

    profile_info = "未註冊"
    if user_profile:
        dept = user_profile.get("department", "?")
        grade = user_profile.get("grade", "?")
        profile_info = f"{dept} {grade}年級"

    answer_preview = answer[:800] + "..." if len(answer) > 800 else answer

    audit_embed = discord.Embed(
        title="🕵️ 系統監控紀錄",
        color=discord.Color.dark_embed(),
        timestamp=discord.utils.utcnow()
    )
    audit_embed.add_field(name="👤 提問者", value=f"{student.display_name} (`{student.id}`)", inline=True)
    audit_embed.add_field(name="🏫 身分", value=profile_info, inline=True)
    audit_embed.add_field(name="📡 來源", value=f"`{source}` · {channel_name}", inline=True)
    audit_embed.add_field(name="❓ 問題內容", value=f"```\n{question[:500]}\n```", inline=False)
    audit_embed.add_field(name="🤖 AI 回應", value=f"```\n{answer_preview[:1000]}\n```", inline=False)
    audit_embed.set_thumbnail(url=student.display_avatar.url)
    audit_embed.set_footer(text="NQU 校園智慧助理 · 監控系統")

    try:
        await audit_channel.send(embed=audit_embed)
        logger.info(f"📋 審計 Log 已發送到 #{audit_channel.name} | 學生={student.display_name} 來源={source}")
    except Exception as e:
        logger.error(f"📋 審計 Log 發送失敗: {e}")


# =========================================================================
# 📢 通用審計 — 管理員操作、廣播、系統事件
# =========================================================================

async def send_audit_log(
    title: str,
    description: str,
    color: discord.Color = None,
    fields: list[tuple[str, str, bool]] = None,
    admin_user: discord.User | discord.Member = None,
):
    """
    發送一般審計紀錄到監控頻道。
    
    用途：廣播完成、管理員操作、群組變更、系統事件等。
    
    Args:
        title: 審計 Embed 標題
        description: 說明文字
        color: Embed 顏色（預設為 blurple）
        fields: [(name, value, inline), ...] 額外欄位
        admin_user: 操作的管理員（可選，用於顯示頭貼）
    """
    audit_channel = await _get_audit_channel()
    if not audit_channel:
        return

    embed = discord.Embed(
        title=title,
        description=description,
        color=color or discord.Color.blurple(),
        timestamp=discord.utils.utcnow()
    )
    if fields:
        for name, value, inline in fields:
            embed.add_field(name=name, value=value, inline=inline)
    if admin_user:
        embed.set_thumbnail(url=admin_user.display_avatar.url)
    embed.set_footer(text="NQU 校園智慧助理 · 監控系統")

    try:
        await audit_channel.send(embed=embed)
    except Exception as e:
        logger.error(f"📋 審計 Log 發送失敗: {e}")
