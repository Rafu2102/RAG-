# -*- coding: utf-8 -*-
"""
bot/telegram/__init__.py — Telegram Bot 共用狀態
=================================================
Telegram 專用的全域狀態管理，與 Discord 的 bot/__init__.py 完全隔離。
"""

import time
import asyncio
import logging
from collections import OrderedDict
from llm.llm_answer import ConversationMemory

logger = logging.getLogger("telegram_bot")

# ── 安全的對話記憶體 (TTL / LRU 快取) ──
class TTLMemoryCache:
    """含有時間存活(TTL)與最近最少使用(LRU)淘汰機制的快取"""
    def __init__(self, maxsize: int = 500, ttl_seconds: int = 3600):
        self.cache: OrderedDict[int, tuple[ConversationMemory, float]] = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds

    def get(self, key: int) -> ConversationMemory:
        now = time.time()
        self._cleanup(now)
        if key in self.cache:
            item, timestamp = self.cache.pop(key)
            if now - timestamp < self.ttl:
                self.cache[key] = (item, now)
                return item
        new_memory = ConversationMemory()
        self.cache[key] = (new_memory, now)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return new_memory

    def _cleanup(self, now: float):
        keys_to_delete = [k for k, (_, ts) in self.cache.items() if now - ts >= self.ttl]
        for k in keys_to_delete:
            del self.cache[k]

user_memories = TTLMemoryCache(maxsize=500, ttl_seconds=3600)
gpu_semaphore = asyncio.Semaphore(1)
active_gpu_requests = 0

def get_user_memory(user_id: int) -> ConversationMemory:
    """取得特定 Telegram 使用者的對話記憶"""
    return user_memories.get(user_id)

def get_tg_user_id(chat_id: int) -> str:
    """取得帶 tg_ 前綴的使用者 ID（用於 Token 路由）"""
    return f"tg_{chat_id}"
