# -*- coding: utf-8 -*-
"""
discord_bot.py — Discord Bot 串接介面
======================================
負責監聽 Discord 訊息並將其傳入 RAG Pipeline。
包含獨立的對話記憶管理，確保不同頻道/使用者的對話不互相干擾。
"""

import os
import sys
import logging
import asyncio
from typing import Dict
import discord
from discord import app_commands
from dotenv import load_dotenv

import config
from rag.data_loader import load_and_index
from llm.llm_answer import ConversationMemory
from main import rag_pipeline

# 載入環境變數
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

if not DISCORD_BOT_TOKEN:
    print("❌ 錯誤：找不到 DISCORD_BOT_TOKEN。請確保目錄下有 .env 檔案並設定正確的 Token。")
    sys.exit(1)

# 初始化 logging
config.setup_logging()
logger = logging.getLogger("discord_bot")

# 準備 Discord Client
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

# ── 全域狀態 ──
# 儲存不同頻道的對話記憶，Key 是 Channel ID
channel_memories: Dict[int, ConversationMemory] = {}
# 儲存全域向量索引
global_nodes = None
global_faiss = None
global_bm25 = None

def get_channel_memory(channel_id: int) -> ConversationMemory:
    """取得特定頻道的對話記憶，若無則建立"""
    if channel_id not in channel_memories:
        channel_memories[channel_id] = ConversationMemory()
    return channel_memories[channel_id]


# =========================================================================
# 🌟 註冊原生斜線指令 (Slash Commands)
# =========================================================================

@tree.command(name="rebuild", description="🔄 重建並熱更新課程索引庫 (專題小組專用)")
async def slash_rebuild(interaction: discord.Interaction):
    # 小組開發模式：移除權限限制，所有人皆可觸發

    # 【防當機關鍵】呼叫 defer() 會先在聊天室顯示「機器人正在思考中...」
    # 這能幫我們向 Discord 爭取突破 3 秒限制的無限運算時間！
    await interaction.response.defer()
    
    try:
        # 在背景執行緒跑重建，避免卡死 Discord 機器人
        nodes, faiss_idx, bm25_idx = await asyncio.to_thread(load_and_index, True)
        global global_nodes, global_faiss, global_bm25
        global_nodes = nodes
        global_faiss = faiss_idx
        global_bm25 = bm25_idx
        
        # 處理完成後，使用 followup.send 傳送最終結果
        await interaction.followup.send(f"✅ 索引重建與載入無縫完成！最新資料已上線（共 {len(nodes)} 個區段）。")
    except Exception as e:
        await interaction.followup.send(f"❌ 索引重建失敗：{e}")


@tree.command(name="clear", description="🗑️ 清除當前頻道的對話記憶")
async def slash_clear(interaction: discord.Interaction):
    if interaction.channel_id in channel_memories:
        channel_memories[interaction.channel_id].clear()
    await interaction.response.send_message("✅ 已經為您清除這個頻道的對話記憶囉！")


# =========================================================================
# 🌟 原生問答斜線指令 (附帶精美文字提示框)
# =========================================================================

@tree.command(name="ask", description="🤖 呼叫智慧校園助理為您解答課程問題")
@app_commands.describe(question="請輸入你想詢問的課程問題 (例如：星期二有什麼課？)")
async def slash_ask(interaction: discord.Interaction, question: str):
    
    # 1. 告訴 Discord 我們正在思考，爭取處理時間
    await interaction.response.defer()
    
    channel_id = interaction.channel_id
    memory = get_channel_memory(channel_id)

    try:
        # 2. 呼叫我們強大的 RAG Pipeline
        answer = await asyncio.to_thread(
            rag_pipeline,
            question,
            global_nodes,
            global_faiss,
            global_bm25,
            memory,
            False
        )
        
        # 確保回傳長度不會超過 Discord 的 2000 字限制
        if len(answer) > 1950:
            answer = answer[:1950] + "\n... (訊息過長已截斷)"
            
        # 3. 回傳答案 (這裡可以加上使用者的問題，讓排版更好看)
        final_reply = f"**👤 你問：** {question}\n\n**🤖 助理回答：**\n{answer}"
        await interaction.followup.send(final_reply)
        
    except Exception as e:
        logger.exception("Discord Slash Ask Error")
        await interaction.followup.send(f"❌ 抱歉，系統發生錯誤：{e}")


@tree.command(name="add_calendar", description="📅 專屬指令：將課程或學校事件快速加入 Google 行事曆")
@app_commands.describe(event="請輸入想加入的課程或事件 (例如：什麼時候停修申請 / 程式設計 / 演算法)")
async def slash_add_calendar(interaction: discord.Interaction, event: str):
    
    # 1. 告訴 Discord 我們正在思考
    await interaction.response.defer()
    
    channel_id = interaction.channel_id
    memory = get_channel_memory(channel_id)

    try:
        # 強制附加強烈意圖字詞，確保 Query Router 100% 走 calendar_action 路由
        augmented_query = f"{event} 幫我加到行事曆"
        
        answer = await asyncio.to_thread(
            rag_pipeline,
            augmented_query,
            global_nodes,
            global_faiss,
            global_bm25,
            memory,
            False
        )
        
        # 確保回傳長度合法
        if len(answer) > 1950:
            answer = answer[:1950] + "\n... (訊息過長已截斷)"
            
        final_reply = f"**👤 你要求加入行事曆：** {event}\n\n**🤖 助理回報：**\n{answer}"
        await interaction.followup.send(final_reply)
        
    except Exception as e:
        logger.exception("Discord Slash Add Calendar Error")
        await interaction.followup.send(f"❌ 抱歉，行事曆新增發生錯誤：{e}")


# =========================================================================
# 🤖 Discord Events
# =========================================================================

@client.event
async def on_ready():
    global global_nodes, global_faiss, global_bm25
    logger.info(f"✅ 機器人 {client.user} 已成功登入 Discord！")
    
    # 【修正】直接呼叫 sync() 即可，它會自動把我們寫好的指令覆蓋到伺服器上
    try:
        synced = await tree.sync()
        logger.info(f"✅ 已成功同步 {len(synced)} 個斜線指令！")
    except Exception as e:
        logger.error(f"❌ 斜線指令同步失敗：{e}")
    
    # 啟動時自動載入向量資料庫與 Reranker 模型
    logger.info("📂 正在載入課程資料與 Reranker 索引...")
    try:
        nodes, faiss_idx, bm25_idx = load_and_index()
        global_nodes = nodes
        global_faiss = faiss_idx
        global_bm25 = bm25_idx
        
        # 預先載入 Reranker 模型，避免第一次查詢時卡頓超過 15 秒
        from rag.reranker import get_reranker
        get_reranker()
        
        logger.info(f"✅ 索引與模型載入完成！共 {len(nodes)} 個文件區段")
        await client.change_presence(activity=discord.Game(name="輸入 /rebuild 重建 · @機器人 查詢"))
    except Exception as e:
        logger.error(f"❌ 索引載入失敗：{e}")
        await client.close()

@client.event
async def on_message(message: discord.Message):
    # 忽略機器人自己發送的訊息
    if message.author == client.user:
        return

    # 檢查是否是被 tag，或是以特定前綴開頭 (例如 !ask)
    is_mentioned = client.user in message.mentions
    is_prefixed = message.content.startswith("!ask")  # 把空格拿掉，增加容錯
    
    if not (is_mentioned or is_prefixed):
        return

    # 清除 tag 殘留字串或 !ask 前綴，取得真正的問題
    question = message.content.replace(f"<@{client.user.id}>", "").replace("!ask", "").strip()
    
    if not question:
        await message.reply("請輸入你想要查詢的課程問題喔！")
        return

    # 系統指令全部交給 @tree.command 斜線指令處理，on_message 只負責課程問答

    channel_id = message.channel.id
    memory = get_channel_memory(channel_id)

    # 顯示正在輸入的狀態
    async with message.channel.typing():
        try:
            # 將耗時的同步 rag_pipeline 放到背景執行緒 (Background Thread) 中執行
            # 這樣才不會卡死 Discord 的 Event Loop 導致 Heartbeat Timeout 斷線
            answer = await asyncio.to_thread(
                rag_pipeline,
                question,
                global_nodes,
                global_faiss,
                global_bm25,
                memory,
                False
            )
            
            # 確保回傳長度不會超過 Discord 的 2000 字限制
            if len(answer) > 1950:
                answer = answer[:1950] + "\n... (訊息過長已截斷)"
                
            await message.reply(answer)
            
        except Exception as e:
            logger.exception("Discord Pipeline Error")
            error_msg = str(e)
            if "Connection refused" in error_msg or "Failed to establish a new connection" in error_msg:
                await message.reply("❌ 糟糕！我的 AI 核心 (Ollama) 似乎沒有啟動，請管理員檢查一下伺服器喔！")
            else:
                await message.reply(f"❌ 查詢時發生內部錯誤，請稍後再試：{e}")

if __name__ == "__main__":
    logger.info("啟動 Discord Bot...")
    client.run(DISCORD_BOT_TOKEN)
