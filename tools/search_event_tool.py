# -*- coding: utf-8 -*-
"""
search_event_tool.py — 學校行事曆事件搜尋工具
================================================
使用 CKIP Tagger 萃取關鍵字來搜尋 events.json 中的學校行事曆事件。
"""

import json
import logging
from pathlib import Path
from typing import Optional

from nlp_utils import tokenize_texts_ckip

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
EVENTS_PATH = BASE_DIR / "data" / "events.json"

# 【Bug 3 + Opt 5 修復】Lazy Loading — 避免模組載入時 events.json 不存在直接 crash
_events_cache: Optional[list] = None


def _get_events() -> list:
    """延遲載入 events.json，第一次呼叫才讀取"""
    global _events_cache
    if _events_cache is None:
        if EVENTS_PATH.exists():
            try:
                with EVENTS_PATH.open(encoding="utf-8") as f:
                    _events_cache = json.load(f)
                logger.info(f"📅 events.json 載入完成：{len(_events_cache)} 個事件")
            except Exception as e:
                logger.error(f"❌ 讀取 events.json 失敗：{e}")
                _events_cache = []
        else:
            logger.warning(f"⚠️ events.json 不存在：{EVENTS_PATH}，學校行事曆功能將不可用")
            _events_cache = []
    return _events_cache


def search_academic_events(query: str) -> list[dict]:
    """使用 jieba 萃取關鍵字來搜尋學校行事曆事件"""
    
    events = _get_events()
    if not events:
        return []
    
    # 【同義詞對照表】解決學生俗稱與官方行事曆名稱不匹配的問題
    SYNONYM_MAP = {
        "退選": "停修",
        "加退選": "選課 停修",
        "加選": "選課",
        "選課": "選課",
        "宿舍": "宿舍",
        "住宿": "宿舍",
        "開學": "上課開始",
        "註冊": "繳費",
        "學費": "繳費",
        "期中": "期中考試",
        "期末": "期末考試"
    }
    
    # 進行同義詞替換擴充
    expanded_query = query
    for slang, official in SYNONYM_MAP.items():
        if slang in query:
            expanded_query += f" {official}"
            
    # 【改用 CKIP 斷詞】（因為是單一句子，包裝成 list 傳入）
    ckip_results = tokenize_texts_ckip([expanded_query])
    keywords = ckip_results[0] if ckip_results else []
    
    stopwords = {"什麼時候", "幫我", "加到", "行事曆", "的", "請問", "日期", "時間", "是", "何時", "查詢", "有", "嗎", "我想"}
    valid_keywords = [kw for kw in keywords if kw not in stopwords and len(kw) >= 2]
    
    if not valid_keywords:
        # 如果無法切出有效關鍵字，嘗試直接 substring 搜尋
        results = [e for e in events if query in e.get("title", "")]
        return results[:3]

    scored_events = []
    for e in events:
        title = e.get("title", "")
        # 計算匹配分數：只要有效關鍵字出現在官方行事曆標題中即加分
        score = sum(1 for kw in valid_keywords if kw in title)
        if score > 0:
            scored_events.append((score, e))
            
    scored_events.sort(key=lambda x: x[0], reverse=True)
    return [e[1] for e in scored_events[:3]]