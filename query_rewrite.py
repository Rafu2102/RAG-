# -*- coding: utf-8 -*-
"""
query_rewrite.py — Query Rewrite + Multi-query RAG 模組
=========================================================
負責：
1. 改寫使用者問題，提升檢索準確度
2. Multi-query：從一個問題生成多個搜尋查詢
3. 整合對話歷史進行上下文感知的 query 改寫
"""

import logging
import json
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)

# =============================================================================
# 📝 Prompt Templates
# =============================================================================

QUERY_REWRITE_PROMPT = """你是一個專業的校園課程助理問題改寫器。

## 對話歷史
{chat_history}

## 使用者目前的問題
{question}

## 嚴格改寫規則
1. 你的唯一目標是針對「使用者目前的問題」生成 {num_queries} 個搜尋查詢。
2. 【注意力切換】：如果使用者的問題是一個全新的需求（例如：請幫我推薦課程、我有空堂），絕對不可以把對話歷史中的「老師名稱」或「課程名稱」混入新的查詢中！

## 輸出格式 (必須是 JSON)
請務必嚴格依照以下 JSON 格式輸出，先判斷是否為延續話題，再給出查詢：
{{
  "is_follow_up": false,
  "reasoning": "判斷這個問題是否與歷史紀錄的特定課程/老師有關的理由",
  "search_queries": [
    "查詢1",
    "查詢2"
  ]
}}
"""


# =============================================================================
# 🔄 Query Rewrite 主函式
# =============================================================================

def rewrite_query(
    question: str,
    chat_history: Optional[list[dict]] = None,
    num_queries: int = None,
) -> list[str]:
    """
    將使用者問題改寫為多個搜尋查詢（Multi-query RAG）。

    使用 Ollama Llama 3.1 進行改寫，結合對話歷史進行上下文感知。

    Args:
        question: 使用者原始問題
        chat_history: 對話歷史 [{"role": "user/assistant", "content": "..."}]
        num_queries: 生成的查詢數量（預設使用 config 設定）

    Returns:
        改寫後的搜尋查詢列表（包含原始問題）
    """
    if num_queries is None:
        num_queries = config.MULTI_QUERY_COUNT

    # 格式化對話歷史
    history_str = "無"
    if chat_history and len(chat_history) > 0:
        history_lines = []
        for msg in chat_history[-config.MEMORY_WINDOW_SIZE * 2:]:  # 最近 N 輪
            role = "使用者" if msg["role"] == "user" else "助理"
            history_lines.append(f"{role}：{msg['content'][:200]}")
        history_str = "\n".join(history_lines)

    # 組合 prompt
    prompt = QUERY_REWRITE_PROMPT.format(
        num_queries=num_queries,
        chat_history=history_str,
        question=question,
    )

    try:
        # 呼叫 Ollama API 生成改寫查詢
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": getattr(config, "OLLAMA_FAST_MODEL", config.OLLAMA_MODEL),
                "prompt": prompt,
                "format": "json",  # 強制 LLM 僅輸出 JSON 格式
                "stream": False,
                "keep_alive": "0s",  # [防爆] 瞬間卸載，不佔用 VRAM
                "options": {
                    "temperature": 0.3,  # 稍高溫度增加多樣性
                    "num_ctx": 2048,     # Rewrite 不需要太大的 Context，2k 夠了
                    "num_predict": 256,
                },
            },
            timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        result_text = response.json()["response"].strip()
        logger.info(f"Query Rewrite 原始回應：{result_text[:200]}")

        # 解析 JSON 回應
        queries = _parse_queries(result_text)

        if queries and len(queries) > 0:
            # 確保原始問題也在列表中（去重）
            if question not in queries:
                queries.insert(0, question)
            logger.info(f"✅ 生成 {len(queries)} 個搜尋查詢")
            for i, q in enumerate(queries):
                logger.info(f"   查詢 {i+1}：{q}")
            return queries

    except Exception as e:
        logger.warning(f"Query Rewrite 失敗，將使用原始問題：{e}")

    # 若改寫失敗，返回原始問題
    return [question]


def _parse_queries(text: str) -> list[str]:
    """
    解析 LLM 回傳的 JSON 格式查詢列表。
    處理各種格式：
    - ["查詢1", "查詢2"]
    - [{"query": "查詢1"}, {"query": "查詢2"}]
    - 逐行文字 fallback
    """
    # 嘗試解析新版 Chain-of-Thought JSON
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            queries = data.get("search_queries", [])
            if isinstance(queries, list) and len(queries) > 0:
                result = []
                for q in queries:
                    if isinstance(q, str) and q.strip():
                        result.append(q.strip())
                return result
        except json.JSONDecodeError:
            pass

    # 嘗試直接 JSON Array 解析 (Fallback)
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            queries = json.loads(text[start:end])
            if isinstance(queries, list):
                result = []
                for q in queries:
                    # 處理 dict 格式：{"query": "..."} 或 {"搜尋查詢": "..."}
                    if isinstance(q, dict):
                        # 取第一個 value
                        val = next(iter(q.values()), "")
                        if isinstance(val, str) and val.strip():
                            result.append(val.strip())
                    elif isinstance(q, str) and q.strip():
                        result.append(q.strip())
                return result
    except json.JSONDecodeError:
        pass

    # 嘗試逐行解析（fallback）
    lines = text.strip().split("\n")
    queries = []
    for line in lines:
        line = line.strip().strip('"').strip("'").strip(",").strip()
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if line and len(line) > 3 and not line.startswith("[") and not line.startswith("]"):
            queries.append(line)

    return queries


# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    config.setup_logging()

    # 測試 1：簡單問題
    print("=== 測試 1：簡單問題 ===")
    queries = rewrite_query("深度學習是誰教的？")
    for q in queries:
        print(f"  → {q}")

    # 測試 2：帶對話歷史
    print("\n=== 測試 2：帶對話歷史 ===")
    history = [
        {"role": "user", "content": "資工系有哪些課？"},
        {"role": "assistant", "content": "資工系有很多課程..."},
    ]
    queries = rewrite_query("那必修的有哪些？", chat_history=history)
    for q in queries:
        print(f"  → {q}")
