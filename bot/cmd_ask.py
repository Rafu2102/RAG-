# -*- coding: utf-8 -*-
"""
bot/cmd_ask.py — 問答相關斜線指令
====================================
包含：
- /ask (課程問答)
- /add_calendar (加入行事曆)
- /dcard_search (搜尋 Dcard 教授評價)
- /nqu_news (搜尋金大官網公告與新聞)
"""

import asyncio
import logging
import discord  # type: ignore
from discord import app_commands  # type: ignore

import bot as _bot
from bot import tree, logger, bot_ready, gpu_semaphore, get_channel_memory
from main import rag_pipeline
from tools.auth import get_user_profile
from tools.dcard_search_tool import search_dcard_professor, search_nqu_news, search_campus_info
from utils import smart_split_message


# =========================================================================
# 🤖 /ask — 課程問答
# =========================================================================

@tree.command(name="ask", description="🤖 呼叫智慧校園助理為您解答課程問題")
@app_commands.describe(question="請輸入你想詢問的課程問題 (例如：星期二有什麼課？)")
@app_commands.checks.cooldown(1, 10, key=lambda i: (i.user.id))
async def slash_ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    
    if not bot_ready.is_set():
        await interaction.followup.send("⏳ 機器人正在啟動中，請稍後再試！")
        return
    
    channel_id = interaction.channel_id
    memory = get_channel_memory(channel_id)

    try:
        discord_id = str(interaction.user.id)
        user_profile = get_user_profile(discord_id)

        try:
            _bot.active_gpu_requests += 1
            if _bot.active_gpu_requests > 1:
                await interaction.followup.send(f"💬 前方還有 {_bot.active_gpu_requests - 1} 位同學在排隊，請稍等喔！")
            
            async with gpu_semaphore:
                logger.info(f"🚀 GPU 取得 | {interaction.user.display_name} 開始處理問題：{str(question)[:40]}")
                
                answer = await asyncio.to_thread(
                    rag_pipeline, question,
                    _bot.global_nodes, _bot.global_faiss, _bot.global_bm25,
                    memory, False, user_profile, discord_id
                )
        finally:
            _bot.active_gpu_requests -= 1
        
        nick = user_profile.get("nickname", "") if user_profile else ""
        name_tag = f"{nick} " if nick else ""
        logger.info(f"✅ /ask 回答完成 | 使用者：{interaction.user.display_name} | 問題：{str(question)[:50]} | 回答長度：{len(answer)} 字")
        final_reply = f"**👤 {name_tag}你問：** {question}\n\n**🤖 助理回答：**\n{answer}"
        parts = smart_split_message(final_reply)
        for part in parts:
            await interaction.followup.send(part)
        
    except Exception as e:
        logger.exception(f"❌ /ask 錯誤 | 使用者：{interaction.user.display_name} | 問題：{str(question)[:50]}")
        await interaction.followup.send("❌ 抱歉，系統現在有點忙碌，請稍後再試！(內部錯誤)")

@slash_ask.error
async def slash_ask_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(f"⏳ 技能冷卻中！請等待 {error.retry_after:.1f} 秒後再試。", ephemeral=True)
    else:
        logger.error(f"❌ /ask 未知錯誤: {error}")


# =========================================================================
# 📅 /add_calendar — 加入行事曆
# =========================================================================

@tree.command(name="add_calendar", description="📅 專屬指令：將課程或學校事件快速加入 Google 行事曆")
@app_commands.describe(event="請輸入想加入的課程或事件 (例如：什麼時候停修申請 / 程式設計 / 演算法)")
@app_commands.checks.cooldown(1, 10, key=lambda i: (i.user.id))
async def slash_add_calendar(interaction: discord.Interaction, event: str):
    await interaction.response.defer()
    
    if not bot_ready.is_set():
        await interaction.followup.send("⏳ 機器人正在啟動中，請稍後再試！")
        return
    
    channel_id = interaction.channel_id
    memory = get_channel_memory(channel_id)

    try:
        discord_id = str(interaction.user.id)
        user_profile = get_user_profile(discord_id)
        augmented_query = f"{event} 幫我加到行事曆"
        
        try:
            _bot.active_gpu_requests += 1
            if _bot.active_gpu_requests > 1:
                await interaction.followup.send(f"💬 前方還有 {_bot.active_gpu_requests - 1} 位同學在排隊，請稍等喔！")
            
            async with gpu_semaphore:
                logger.info(f"🚀 GPU 取得 | {interaction.user.display_name} 開始處理行事曆：{str(event)[:40]}")
                
                answer = await asyncio.to_thread(
                    rag_pipeline, augmented_query,
                    _bot.global_nodes, _bot.global_faiss, _bot.global_bm25,
                    memory, False, user_profile, discord_id
                )
        finally:
            _bot.active_gpu_requests -= 1
        
        nick = user_profile.get("nickname", "") if user_profile else ""
        name_tag = f"{nick} " if nick else ""
        logger.info(f"✅ /add_calendar 回答完成 | 使用者：{interaction.user.display_name} | 事件：{str(event)[:50]} | 回答長度：{len(answer)} 字")
        final_reply = f"**👤 {name_tag}你要求加入行事曆：** {event}\n\n**🤖 助理回報：**\n{answer}"
        parts = smart_split_message(final_reply)
        for part in parts:
            await interaction.followup.send(part)
        
    except Exception as e:
        logger.exception(f"❌ /add_calendar 錯誤 | 使用者：{interaction.user.display_name} | 事件：{str(event)[:50]}")
        await interaction.followup.send("❌ 抱歉，行事曆新增發生錯誤，請稍後再試！(內部錯誤)")

@slash_add_calendar.error
async def slash_add_calendar_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(f"⏳ 技能冷卻中！請等待 {error.retry_after:.1f} 秒後再試。", ephemeral=True)
    else:
        logger.error(f"❌ /add_calendar 未知錯誤: {error}")


# =========================================================================
# 🔍 /dcard_search — Dcard 教授搜尋
# =========================================================================

@tree.command(name="dcard_search", description="🔍 搜尋 Dcard 金門大學版的教授評價與推薦")
@app_commands.describe(query="請輸入想搜尋的關鍵字 (例如：英文教授、王大明)")
@app_commands.checks.cooldown(1, 15, key=lambda i: (i.user.id))
async def slash_dcard_search(interaction: discord.Interaction, query: str):
    await interaction.response.defer()
    
    try:
        logger.info(f"🔍 /dcard_search | 使用者：{interaction.user.display_name} | 關鍵字：{query}")
        result = await asyncio.to_thread(search_dcard_professor, query)
        
        parts = smart_split_message(str(result))
        for part in parts:
            await interaction.followup.send(part)
        logger.info(f"✅ /dcard_search 完成 | 關鍵字：{query} | 回答長度：{len(result)} 字")
    except Exception as e:
        logger.exception(f"❌ /dcard_search 錯誤 | 關鍵字：{query}")
        await interaction.followup.send("❌ 搜尋 Dcard 時發生錯誤，請稍後再試！(內部錯誤)")

@slash_dcard_search.error
async def slash_dcard_search_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(f"⏳ 技能冷卻中！請等待 {error.retry_after:.1f} 秒後再試。", ephemeral=True)
    else:
        logger.error(f"❌ /dcard_search 未知錯誤: {error}")


# =========================================================================
# 🏛️ /nqu_news — 金大官網公告搜尋
# =========================================================================

@tree.command(name="nqu_news", description="🏛️ 搜尋金門大學官網最新公告、招生資訊、學術新聞")
@app_commands.describe(query="請輸入想搜尋的關鍵字 (例如：招生、獎學金、轉系、住宿)")
@app_commands.checks.cooldown(1, 10, key=lambda i: (i.user.id))
async def slash_nqu_news(interaction: discord.Interaction, query: str):
    await interaction.response.defer()
    
    try:
        logger.info(f"🏛️ /nqu_news | 使用者：{interaction.user.display_name} | 關鍵字：{query}")
        result = await asyncio.to_thread(search_nqu_news, query)
        
        parts = smart_split_message(str(result))
        for part in parts:
            await interaction.followup.send(part)
        logger.info(f"✅ /nqu_news 完成 | 關鍵字：{query} | 回答長度：{len(result)} 字")
    except Exception as e:
        logger.exception(f"❌ /nqu_news 錯誤 | 關鍵字：{query}")
        await interaction.followup.send("❌ 搜尋金大官網時發生錯誤，請稍後再試！")

@slash_nqu_news.error
async def slash_nqu_news_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(f"⏳ 技能冷卻中！請等待 {error.retry_after:.1f} 秒後再試。", ephemeral=True)
    else:
        logger.error(f"❌ /nqu_news 未知錯誤: {error}")


# =========================================================================
# 🔗 /search — 統一校園資訊搜尋（NQU官網 + Dcard + Gemini 總結）
# =========================================================================

@tree.command(name="search", description="🔍 統一搜尋金大官網公告 + Dcard 學生討論，由 AI 智慧總結")
@app_commands.describe(query="請輸入想搜尋的關鍵字 (例如：英文教授、招生、獎學金、轉系)")
@app_commands.checks.cooldown(1, 20, key=lambda i: (i.user.id))
async def slash_search(interaction: discord.Interaction, query: str):
    await interaction.response.defer()
    
    try:
        logger.info(f"🔗 /search | 使用者：{interaction.user.display_name} | 關鍵字：{query}")
        await interaction.followup.send(f"🔍 正在同時搜尋**金大官網** + **Dcard 金門大學版**，並由 AI 總結中... 請稍候 ☕")
        result = await asyncio.to_thread(search_campus_info, query)
        
        parts = smart_split_message(str(result))
        for part in parts:
            await interaction.followup.send(part)
        logger.info(f"✅ /search 完成 | 關鍵字：{query} | 回答長度：{len(result)} 字")
    except Exception as e:
        logger.exception(f"❌ /search 錯誤 | 關鍵字：{query}")
        await interaction.followup.send("❌ 搜尋時發生錯誤，請稍後再試！")

@slash_search.error
async def slash_search_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(f"⏳ 搜尋冷卻中！請等待 {error.retry_after:.1f} 秒後再試。", ephemeral=True)
    else:
        logger.error(f"❌ /search 未知錯誤: {error}")
