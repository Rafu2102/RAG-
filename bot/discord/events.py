# -*- coding: utf-8 -*-
"""
bot/events.py — Discord Events (on_ready, on_message)
======================================================
包含：
- on_ready: 啟動載入索引、同步指令, 持久化按鈕
- on_message: @tag / !ask / DM 問答 + 審計 Log
"""

import re
import asyncio
import logging
import time
import io
import discord  # type: ignore

import bot as _bot
from bot import (
    client, tree, logger,
    ADMIN_DISCORD_IDS,
    bot_ready, get_user_memory,
    user_memories,
)
from rag.index_manager import load_and_index, check_data_changes
from rag.query_router import init_known_registry
from main import rag_pipeline
from tools.auth import get_user_profile
from utils import smart_split_message
from bot.discord.ui_utils import safe_send_parts_message
from bot.discord.audit import send_audit_dm, send_debug_log
from bot.discord.cmd_groups import GroupInviteView, handle_dynamic_group_invite
from bot.discord.ipc_server import start_ipc_server


# =========================================================================
# 🟢 on_ready — 啟動事件
# =========================================================================

@client.event
async def on_ready():
    logger.info(f"✅ 機器人 {client.user} 已成功登入 Discord！")
    
    # 持久化按鈕：不再需要 add_view placeholder，改由 on_interaction 事件統一調度
    
    # 啟動跨平台 IPC 通訊伺服器
    client.loop.create_task(start_ipc_server())
    
    # 同步斜線指令
    try:
        synced = await tree.sync()
        logger.info(f"✅ 已成功同步 {len(synced)} 個斜線指令！")
    except Exception as e:
        logger.error(f"❌ 斜線指令同步失敗：{e}")
    
    # 載入索引
    logger.info("📂 正在檢查課程資料與載入索引...")
    try:
        data_status = check_data_changes()
        force_rebuild = data_status["has_changes"]
        
        if force_rebuild:
            change_summary = []
            if data_status["new_files"]:
                change_summary.append(f"新增 {len(data_status['new_files'])} 個檔案")
            if data_status["modified_files"]:
                change_summary.append(f"修改 {len(data_status['modified_files'])} 個檔案")
            if data_status["deleted_files"]:
                change_summary.append(f"刪除 {len(data_status['deleted_files'])} 個檔案")
            change_desc = "、".join(change_summary)
            logger.info(f"🔄 偵測到課程資料變更（{change_desc}），自動重建索引...")
            await client.change_presence(activity=discord.Game(name=f"🔄 重建索引中（{change_desc}）"))
        else:
            logger.info("✅ 課程資料無變更，載入既有索引")
        
        nodes, faiss_idx, bm25_idx = await asyncio.to_thread(load_and_index, force_rebuild=force_rebuild)
        _bot.global_nodes = nodes
        _bot.global_faiss = faiss_idx
        _bot.global_bm25 = bm25_idx
        
        init_known_registry(nodes)
        
        logger.info(f"✅ 索引載入完成！共 {len(nodes)} 個文件區段")
        
        # 預先載入 Reranker 模型（非致命：失敗時會在首次查詢時 lazy-load）
        try:
            from rag.reranker import get_reranker
            await asyncio.to_thread(get_reranker)
            logger.info("✅ Reranker 模型預載完成")
        except Exception as e:
            logger.warning(f"⚠️ Reranker 預載失敗（首次查詢時會自動重試）：{e}")
        
        # 預先載入 CKIP 斷詞模型（消除首次查詢的 TensorFlow + CKIP 冷啟動延遲 ~9 秒）
        try:
            from nlp_utils import get_ws_model
            await asyncio.to_thread(get_ws_model)
            logger.info("✅ CKIP 斷詞模型預載完成")
        except Exception as e:
            logger.warning(f"⚠️ CKIP 預載失敗（首次查詢時會自動重試）：{e}")
        
        await client.change_presence(activity=discord.Game(name="✅ 已就緒 · 輸入 / 查詢課程"))
        
        bot_ready.set()
        logger.info("🟢 伺服器就緒完畢，開始接受指令！")
    except Exception as e:
        logger.error(f"❌ 索引載入失敗：{e}")
        await client.close()


# =========================================================================
# 🔘 on_interaction — 無狀態按鈕調度器
# =========================================================================

@client.event
async def on_interaction(interaction: discord.Interaction):
    """Discord Client 的全域 Interaction 攔截器，將動態按鈕事件分派到對應的 handler"""
    # 先讓 CommandTree 處理斜線指令和 autocomplete
    if interaction.type in (
        discord.InteractionType.application_command,
        discord.InteractionType.autocomplete,
    ):
        return  # 交給 tree 處理，不在這裡攝截

    # 群組邀請按鈕（無狀態動態 custom_id）
    await handle_dynamic_group_invite(interaction)


# =========================================================================
# 💬 on_message — @tag / !ask / DM 問答
# =========================================================================

@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    # 處理開發者 !sync 指令
    if "!sync" in message.content.lower():
        if ADMIN_DISCORD_IDS and str(message.author.id) not in ADMIN_DISCORD_IDS:
            await message.reply("🚫 抱歉，只有管理員才能執行同步指令喔！")
            return
        if message.guild:
            try:
                tree.clear_commands(guild=message.guild)
                await tree.sync(guild=message.guild)
                synced = await tree.sync()
                await message.reply(f"✅ 開發者模式：已為機器人強制執行全域同步 ({len(synced)} 個指令)！\n\n💡 **重要提示**：Discord 全域指令同步可能需要 1~2 分鐘到 1 小時不等才能完全生效。")
                logger.info(f"✅ 手動 Global Sync 完成： {len(synced)} 個指令")
            except Exception as e:
                await message.reply(f"❌ 同步失敗：{e}")
        else:
            await message.reply("⚠️ 這個指令只能在伺服器頻道中使用喔！")
        return

    # 檢查是否是被 tag、以前綴開頭、或是私訊 (DM)
    is_dm = message.guild is None
    is_mentioned = client.user in message.mentions
    is_prefixed = message.content.startswith("!ask")
    
    if not (is_mentioned or is_prefixed or is_dm):
        return

    # 決定來源標籤
    if is_dm:
        audit_source = "DM"
        audit_channel = "私訊"
    elif is_mentioned:
        audit_source = "@tag"
        audit_channel = f"#{message.channel.name}" if hasattr(message.channel, 'name') else "未知頻道"
    else:
        audit_source = "!ask"
        audit_channel = f"#{message.channel.name}" if hasattr(message.channel, 'name') else "未知頻道"

    # 清除 tag 殘留字串，取得真正的問題
    content_clean = re.sub(r'<@&?\d+>', '', message.content)
    question = content_clean.strip()
    
    if question.startswith("!ask"):
        question = question.removeprefix("!ask").strip()
    
    if not question:
        await message.reply("請輸入你想要查詢的課程問題喔！")
        return

    discord_id = str(message.author.id)
    user_profile = get_user_profile(discord_id)
    memory = get_user_memory(int(discord_id))  # 🆕 每位學生獨立記憶，不再按頻道共用
    
    profile_tag = ""
    if user_profile:
        profile_tag = f" | 身分：{user_profile.get('department', '?')} {user_profile.get('grade', '?')}年級"
    logger.info(f"🔍 處理問題：{question[:60]} | 使用者：{message.author.display_name} (ID: {discord_id}){profile_tag} | 來源：{audit_source}")

    # 🆕 雙進程無重啟熱重載監控 (Hot Reload Alignment)
    from rag.index_manager import check_and_reload_index_if_needed
    nodes, faiss_idx, bm25_idx, has_changed = check_and_reload_index_if_needed(
        _bot.global_nodes, _bot.global_faiss, _bot.global_bm25
    )
    if has_changed:
        _bot.global_nodes = nodes
        _bot.global_faiss = faiss_idx
        _bot.global_bm25 = bm25_idx
        logger.info("⚡ [Discord] 偵測到索引已在磁碟更新，Discord 內存已同步完成熱更新！")

    async with message.channel.typing():
        try:
            logger.info(f"🚀 開始處理問題：{str(question)[:40]} | 使用者：{message.author.display_name}")
            
            # 【Debug Log 捕獲】在 pipeline 執行期間捕獲所有 log
            log_capture = io.StringIO()
            log_handler = logging.StreamHandler(log_capture)
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(logging.Formatter('%(name)s | %(message)s'))
            
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)
            
            pipeline_start = time.time()
            try:
                answer = await rag_pipeline(
                    question,
                    _bot.global_nodes, _bot.global_faiss, _bot.global_bm25,
                    memory, False, user_profile, discord_id
                )
            finally:
                root_logger.removeHandler(log_handler)
            pipeline_elapsed = time.time() - pipeline_start
            captured_log = log_capture.getvalue()
            
            # 智慧分段回傳
            nick = user_profile.get("nickname", "") if user_profile else ""
            if nick:
                answer = f"**{nick}**，{answer}"
            logger.info(f"✅ {audit_source} 回答完成 | 使用者：{message.author.display_name} | 問題：{str(question)[:50]} | 回答長度：{len(answer)} 字")
            parts = smart_split_message(answer)
            await safe_send_parts_message(message, parts)
            
            # 【審計推播】（Bot 擁有者的問答會被 send_audit_dm 內部跳過）
            client.loop.create_task(
                send_audit_dm(
                    student=message.author,
                    question=question,
                    answer=answer,
                    source=audit_source,
                    channel_name=audit_channel,
                    user_profile=user_profile,
                )
            )
            
            # 【Debug Log 推播】完整 pipeline log 發到 debug 頻道
            client.loop.create_task(
                send_debug_log(
                    student=message.author,
                    question=question,
                    answer=answer,
                    pipeline_log=captured_log,
                    source=audit_source,
                    elapsed=pipeline_elapsed,
                )
            )
            
        except Exception as e:
            logger.exception(f"❌ {audit_source} 錯誤 | 使用者：{message.author.display_name} | 問題：{str(question)[:50]}")
            error_msg = str(e)
            if "Connection refused" in error_msg or "Failed to establish a new connection" in error_msg:
                await message.reply("❌ 糟糕！我的 AI 核心 (Gemini API) 似乎連線失敗，請管理員檢查一下伺服器喔！")
            else:
                await message.reply("❌ 查詢時發生內部錯誤，請稍後再試！(內部錯誤)")

