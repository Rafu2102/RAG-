# -*- coding: utf-8 -*-
"""
mcp_server.py — MCP 伺服器入口
===============================
將 NQU 校園助理的核心 RAG 與 LLM 查詢能力曝露為 MCP 工具。
此伺服器啟動後，支援 stdio 通訊，可掛載於 Cursor, Claude Desktop 等 MCP Client。
"""

import sys
import logging
import asyncio

from mcp.server.fastmcp import FastMCP

import config
from rag.index_manager import load_and_index
from rag.query_router import init_known_registry

# 設置基礎 Logging
config.setup_logging()
logger = logging.getLogger("mcp_server")

# 建立 FastAPI MCP app
mcp = FastMCP("NQU Campus Assistant")

# 全域單例變數（因為 FastMCP 是異步或單執行緒模式運行，這樣快取比較安全）
class ServerState:
    nodes = None
    faiss_idx = None
    bm25_idx = None
    is_ready = False

# ---------------------------------------------------------------------------
# 🛠️ 啟動與初始化邏輯
# ---------------------------------------------------------------------------
def _ensure_loaded():
    """確保 RAG 索引與模型已載入，供 Tools 同步/非同步調用"""
    if not ServerState.is_ready:
        logger.info("📂 正在啟動 MCP Server 並預先載入 RAG Index...")
        try:
            nodes, faiss_idx, bm25_idx = load_and_index()
            init_known_registry(nodes)
            ServerState.nodes = nodes
            ServerState.faiss_idx = faiss_idx
            ServerState.bm25_idx = bm25_idx
            ServerState.is_ready = True
            logger.info("✅ MCP Server 準備就緒！")
        except Exception as e:
            logger.error(f"❌ 索引載入失敗: {e}")
            raise RuntimeError(f"伺服器初始化失敗: {e}")

# ---------------------------------------------------------------------------
# 🛠️ MCP 工具定義
# ---------------------------------------------------------------------------

@mcp.tool()
def query_campus_info(question: str, department: str = "", grade: str = "") -> str:
    """
    透過 NQU 校園助理的 RAG 系統查詢總合校園資訊。
    支援查詢：課程大綱、成績評定方式、上課時間、系所規定、教授資訊、行事曆等。

    Args:
        question (str): 學生的自然語言提問（例如："資工系三年級禮拜二有什麼課？", "介紹柯志亨教授"）
        department (str, optional): 學生所屬科系（如 "資工系"）。若提供，將縮小課程查詢範圍。
        grade (str, optional): 學生年級（如 "三"）。若提供，將精準匹配年級。
    """
    _ensure_loaded()
    
    # 動態引入 main.py 的 RAG pipeline
    from main import rag_pipeline
    from llm.llm_answer import ConversationMemory
    
    # 建立一個拋棄式的短暫記憶體（MCP 是 Stateless 的，此處僅供本次對話格式使用）
    temp_memory = ConversationMemory(window_size=0)
    
    user_profile = None
    if department or grade:
        user_profile = {
            "department": department or "未知",
            "grade": grade or "未知",
            "nickname": "MCP Client User"
        }
        
    try:
        # 呼叫底層代理執行 (Query Router -> Hybrid Search -> Reranker -> Gemini LLM Response)
        answer = rag_pipeline(
            question=question,
            nodes=ServerState.nodes,
            faiss_index=ServerState.faiss_idx,
            bm25_index=ServerState.bm25_idx,
            memory=temp_memory,
            debug=False,
            user_profile=user_profile,
            discord_id="mcp_user"
        )
        return answer
    except Exception as e:
        logger.exception("MCP 工具執行錯誤")
        return f"❌ 系統發生內部錯誤：{e}"


@mcp.tool()
def read_personal_schedule(discord_id: str) -> str:
    """
    讀取使用者事先上傳的個人課表 JSON 檔案，以字串形式返回供 AI 解析。
    
    Args:
        discord_id (str): 使用者的唯一識別碼
    """
    from tools.schedule_manager import get_schedule_context_for_llm
    
    context, error = get_schedule_context_for_llm(discord_id)
    if error:
        return f"無法取得課表：{error}"
    return context


@mcp.tool()
def read_personal_transcript(discord_id: str) -> str:
    """
    讀取使用者事先上傳的歷史成績單 JSON 檔案，以字串形式返回供 AI 解析 GPA 或學分。
    
    Args:
        discord_id (str): 使用者的唯一識別碼
    """
    from tools.transcript_manager import get_transcript_context_for_llm
    
    context, error = get_transcript_context_for_llm(discord_id)
    if error:
        return f"無法取得成績單：{error}"
    return context


if __name__ == "__main__":
    # 使用 stdio 作為傳輸協議，此為 Cursor / Claude Desktop 最標準的對接方式
    logger.info("啟動 NQU Campus FastMCP Server (stdio)...")
    _ensure_loaded()
    mcp.run(transport="stdio")
