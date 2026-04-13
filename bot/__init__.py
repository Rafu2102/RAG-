# -*- coding: utf-8 -*-
"""
bot/__init__.py — Bot Package 初始化
====================================
匯出共用的 client、tree、全域狀態，
讓子模組都能 `from bot import client, tree, ...`
"""

import os
import sys
import logging
import asyncio
from typing import Dict

import discord  # type: ignore
from discord import app_commands  # type: ignore
from dotenv import load_dotenv  # type: ignore

import config
from llm.llm_answer import ConversationMemory

# ── 載入環境變數 ──
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ADMIN_DISCORD_IDS = set(filter(None, os.getenv("ADMIN_DISCORD_IDS", "").split(",")))
AUDIT_CHANNEL_ID = os.getenv("AUDIT_CHANNEL_ID")  # 監控頻道 ID（可選，填了就直接用，不用搜尋名稱）

if not DISCORD_BOT_TOKEN:
    print("❌ 錯誤：找不到 DISCORD_BOT_TOKEN。請確保目錄下有 .env 檔案並設定正確的 Token。")
    sys.exit(1)

# ── 初始化 logging ──
config.setup_logging()
logger = logging.getLogger("discord_bot")

# ── 準備 Discord Client ──
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

# ── 科系簡稱 ↔ 完整名稱對照表 ──
DEPT_MAPPING = {
    "資工系": "資訊工程學系", "電機系": "電機工程學系",
    "土木系": "土木與工程管理學系", "食品系": "食品科學系",
    "企管系": "企業管理學系", "觀光系": "觀光管理學系",
    "運休系": "運動與休閒學系", "工管系": "工業工程與管理學系",
    "國際系": "國際暨大陸事務學系", "建築系": "建築學系",
    "海邊系": "海洋與邊境管理學系", "應英系": "應用英語學系",
    "華語系": "華語文學系", "都景系": "都市計畫與景觀學系",
    "護理系": "護理學系", "長照系": "長期照護學系",
    "社工系": "社會工作學系", "通識中心": "通識教育中心",
}
DEPT_REVERSE = {v: k for k, v in DEPT_MAPPING.items()}

import time
from collections import OrderedDict

# ── 安全的對話記憶體 (TTL / LRU 快取) ──
class TTLMemoryCache:
    """含有時間存活(TTL)與最近最少使用(LRU)淘汰機制的快取，防止 RAM 被撐爆"""
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 1800):
        self.cache: OrderedDict[int, tuple[ConversationMemory, float]] = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds

    def get(self, key: int) -> ConversationMemory:
        now = time.time()
        self._cleanup(now)
        
        if key in self.cache:
            item, timestamp = self.cache.pop(key)
            if now - timestamp < self.ttl:
                self.cache[key] = (item, now)  # 更新存活時間，移到最新
                return item
        
        # 鍵值不存在或已過期，建立新的並存入
        new_memory = ConversationMemory()
        self.cache[key] = (new_memory, now)
        
        # 超出最大容量，移除最舊的一筆 (FIFO)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
            
        return new_memory

    def _cleanup(self, now: float):
        """清除已經過期的記憶"""
        keys_to_delete = []
        for k, (item, timestamp) in self.cache.items():
            if now - timestamp >= self.ttl:
                keys_to_delete.append(k)
            else:
                break  # 因為是按照存取時間排序，後面的一定比較新
        for k in keys_to_delete:
            del self.cache[k]

user_memories = TTLMemoryCache(maxsize=500, ttl_seconds=3600)  # 保留 1 小時，最多 500 位使用者（每人獨立記憶）
global_nodes = None
global_faiss = None
global_bm25 = None

# 【伺服器就緒門欄】確保機器人完全載入完畢後才接受指令
bot_ready = asyncio.Event()

# 【GPU 佇列排程】限制同時只有 1 個 RAG Pipeline 接觸 GPU
gpu_semaphore = asyncio.Semaphore(1)
active_gpu_requests = 0


def get_user_memory(user_id: int) -> ConversationMemory:
    """取得特定使用者的對話記憶，若無則建立。每位學生有獨立的記憶空間，不會交叉污染。"""
    return user_memories.get(user_id)
