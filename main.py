# -*- coding: utf-8 -*-
"""
main.py — 主 Pipeline（CLI 互動介面）
========================================
校園課程助理機器人的主要入口。

Pipeline 流程：
    1. 載入 / 建立索引
    2. 使用者輸入問題
    3. Query Rewrite（Multi-query RAG）
    4. Query Router（判斷問題類型 + 提取 metadata filter）
    5. Hybrid Retriever（Vector + BM25 + Metadata → α/β/γ 融合）
    6. Reranker（Top-30 → Top-5）
    7. LLM Answer（Gemini 3.1 Pro + Source Grounding）
    8. 更新對話記憶
    9. 回到步驟 2

指令：
    /quit   - 退出程式
    /rebuild - 重建索引
    /clear  - 清除對話歷史
    /debug  - 切換 debug 模式（顯示詳細檢索資訊）
"""

import sys
import re
import time
import logging
import gc

import torch
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

import config
from rag.index_manager import load_and_index
from rag.query_router import route_and_rewrite
from rag.retriever import hybrid_retrieve, intent_inject_chunks
from rag.reranker import rerank
from llm.llm_answer import generate_answer, format_sources, ConversationMemory
from llm.chitchat import generate_chitchat_answer
from llm.coreference import resolve_coreference

logger = logging.getLogger(__name__)
console = Console()


# ═══════════════════════════════════════════════════════════════
# 🔧 Pipeline 短路 helpers (B1/B2 DRY 修復)
# ═══════════════════════════════════════════════════════════════

def _shortcircuit_chitchat(question: str, memory, user_profile: dict | None = None) -> str:
    """閒聊短路：跳過 RAG，直接用 Flash Lite 回應並更新記憶。"""
    answer_obj = generate_chitchat_answer(question, memory, user_profile)
    memory.add_user_message(question)
    memory.add_assistant_message(answer_obj.answer)
    return answer_obj.answer


def _shortcircuit_message(question: str, memory, message: str) -> str:
    """固定訊息短路：回傳固定文字並更新記憶。"""
    memory.add_user_message(question)
    memory.add_assistant_message(message)
    return message


# =============================================================================
# 🎨 UI 相關
# =============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║            🎓 NQU 校園課程助理機器人 v2.0                   ║
║            國立金門大學 · 資訊工程學系                       ║
║                                                              ║
║  📚 Hybrid RAG Pipeline                                     ║
║  🤖 Gemini 3.1 Pro + Gemini Embedding 2 + bge-reranker      ║
║  🔤 Embedding: 3072d · task_type 非對稱檢索 · Cloud API     ║
║                                                              ║
║  指令：/quit 退出 | /rebuild 重建索引 | /clear 清除對話      ║
║        /debug 切換詳細模式                                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """顯示程式橫幅"""
    console.print(BANNER, style="bold cyan")


def print_step(step_num: int, description: str, elapsed: float | None = None):
    """顯示 Pipeline 步驟"""
    time_str = f" ({elapsed:.2f}s)" if elapsed is not None else ""
    console.print(f"  [dim]Step {step_num}[/dim] {description}{time_str}")


# =============================================================================
# 🚀 RAG Pipeline
# =============================================================================

def rag_pipeline(
    question: str,
    nodes: list,
    faiss_index,
    bm25_index,
    memory: ConversationMemory,
    debug: bool = False,
    user_profile: dict | None = None,
    discord_id: str | None = None,
) -> str:
    """
    執行完整的 RAG Pipeline。

    Pipeline:
        User Question
        → Step 1: Query Rewrite (Multi-query)
        → Step 2: Query Router (type + metadata filter)
        → Step 3: Hybrid Retriever (Vector + BM25 + Metadata → α·V + β·B + γ·M)
        → Step 4: Reranker (Top-30 → Top-5, cross-encoder)
        → Step 5: LLM Answer (Gemini 3.1 Pro + source grounding)
        → Return answer with sources

    Args:
        question: 使用者問題
        nodes: 全部 Nodes
        faiss_index: FAISS 索引
        bm25_index: BM25 索引
        memory: 對話記憶
        debug: 是否顯示詳細資訊

    Returns:
        格式化的回答字串
    """
    total_start = time.time()
    console.print(f"\n[bold yellow]🔍 處理問題：[/bold yellow]{question}\n")

    # ═════════════════════════════════════
    # Step 0: 規則式閒聊快速攔截（在 LLM Router 之前，零延遲）
    # ═════════════════════════════════════
    stripped = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27BF\u2700-\u27BF\uFE00-\uFE0F\u200d❤♥☺✨⭐🌟💕😀-🙏]+', '', question, flags=re.UNICODE).strip()
    common_greetings = {"你好", "哈囉", "嗨", "hi", "hello", "早安", "午安", "晚安", "謝謝", "感謝", "掰掰", "bye", "嗯", "ok", "好"}
    is_trivial = (
        len(stripped) == 0 or                       # 純 emoji
        len(stripped) <= 2 or                        # 太短沒意義（如「是」「好」）
        stripped.lower() in common_greetings         # 常見打招呼
    )
    if is_trivial:
        logger.info("  👋 規則式閒聊攔截：跳過 Router+Retriever，直接回應")
        return _shortcircuit_chitchat(question, memory, user_profile)

    # ═════════════════════════════════════
    # Step 0.5: 指代消解 (Coreference Resolution)
    # ═════════════════════════════════════
    # 當有對話歷史時，先用 Flash Lite 極速消解代名詞
    # 例：「那老師是誰？」→「微積分的老師是誰？」
    chat_hist = memory.get_history()
    if chat_hist:
        question = resolve_coreference(question, chat_hist)

    # ═════════════════════════════════════
    # Step 1: 合併式 Router + Rewrite（單次 LLM 呼叫）
    # ═════════════════════════════════════
    step_start = time.time()
    route_result, queries = route_and_rewrite(
        question=question,
        chat_history=memory.get_history(),
        user_profile=user_profile,
    )

    # 【學期預設】若使用者未指定學期且並非職涯大範圍探索，自動注入當前學期 filter
    if not getattr(route_result, "is_career_planning", False):
        if "semester" not in route_result.metadata_filters:
            route_result.metadata_filters["semester"] = str(config.CURRENT_SEMESTER)
        if "academic_year" not in route_result.metadata_filters:
            route_result.metadata_filters["academic_year"] = str(config.CURRENT_ACADEMIC_YEAR)

    print_step(1, f"Router+Rewrite → type={route_result.query_type}, filters={route_result.metadata_filters}, {len(queries)} queries", time.time() - step_start)

    # ═════════════════════════════════════
    # Step 2: Agentic 意圖攔截與短路機制
    # ═════════════════════════════════════
    # 若被短路，則不進入 RAG Retriever 階段，直接處理並返回
    
    # [攔截 1] 日常閒聊 (Chitchat)
    if route_result.query_type == "chitchat":
        logger.info("  [dim]👋 偵測到日常閒聊，跳過檢索直接回應[/dim]")
        return _shortcircuit_chitchat(question, memory, user_profile)

    # [攔截 1.2] 聯網搜尋 (Web Search)
    if route_result.query_type == "web_search":
        logger.info("  [dim]🌐 偵測到校外/即時大眾問題，啟動 Google Search 聯網搜尋（跳過本地 RAG 資料庫）[/dim]")
        from llm.llm_answer import generate_web_search_answer
        
        step_start_web = time.time()
        answer_obj = generate_web_search_answer(question, memory)
        
        memory.add_user_message(question)
        memory.add_assistant_message(answer_obj.answer)
        
        print_step(2, "Google Search 聯網解答", time.time() - step_start_web)
        total_time = time.time() - total_start
        console.print(f"\n  [dim]⏱️ 總耗時：{total_time:.2f}s[/dim]\n")
        
        return answer_obj.answer

    # ══ 🛡️ 混合意圖安全閥：偵測推薦/搜尋關鍵字，避免個人資料攔截吃掉推薦需求 ══
    _RECOMMEND_KEYWORDS = [
        "推薦", "建議", "有什麼可以上", "選什麼", "修什麼", "可以上什麼",
        "適合", "好過", "甜", "涼", "容易", "簡單",
        "哪門課", "哪堂課", "什麼課比較",
    ]
    def _has_recommend_intent(q: str) -> bool:
        return any(kw in q for kw in _RECOMMEND_KEYWORDS)

    # [攔截 1.5] 個人課表查詢 (Personal Schedule)
    if route_result.query_type == "personal_schedule" and discord_id:
        if _has_recommend_intent(question):
            logger.info("  📅➡️🔍 個人課表查詢中偵測到推薦意圖，改走 RAG Pipeline（課表將自動注入 LLM Prompt）")
            route_result.query_type = "course_info"
            # 不 return，讓流程繼續往下走到 Retriever + generate_answer（已內建課表注入）
        elif route_result.metadata_filters.get("course_name_keyword"):
            # 🆕 衝堂檢查：需要同時取得個人課表 + 目標課程的 RAG 資料
            target_course = route_result.metadata_filters["course_name_keyword"]
            logger.info(f"  📅🔍 偵測到衝堂/課程資格查詢，同時查詢個人課表 + RAG 課程資料：{target_course}")
            from tools.schedule_manager import get_schedule_context_for_llm
            personal_ctx = get_schedule_context_for_llm(discord_id, question)
            if personal_ctx:
                # 改走 course_info RAG，讓 generate_answer 內建的課表注入機制一併處理
                route_result.query_type = "course_info"
                logger.info("  📅➡️🔍 改走 course_info RAG Pipeline（個人課表 + 課程 RAG 雙資料源）")
                # 不 return，繼續走到 Retriever + generate_answer
            else:
                logger.info("  ⚠️ 使用者尚未匯入課表，fallback 提示")
                fallback = "❌ 您還沒有匯入個人課表資料喔！請先使用 `/upload_schedule` 上傳您的課表 JSON。"
                memory.add_user_message(question)
                memory.add_assistant_message(fallback)
                return fallback
        else:
            logger.info("  📅 偵測到個人課表查詢，跳過 RAG 直接查詢個人資料")
            from tools.schedule_manager import get_schedule_context_for_llm
            personal_ctx = get_schedule_context_for_llm(discord_id, question)
            if personal_ctx:
                logger.info("  ✨ 課表資料取得成功，交由 LLM 進行友善包裝")
                from llm.llm_answer import generate_personal_info_answer
                answer = generate_personal_info_answer(question, personal_ctx, "課表")
                memory.add_user_message(question)
                memory.add_assistant_message(answer)
                total_time = time.time() - total_start
                print_step(2, f"個人課表直接回答 (跳過 RAG)", total_time)
                return answer
            else:
                logger.info("  ⚠️ 使用者尚未匯入課表，fallback 提示")
                fallback = "❌ 您還沒有匯入個人課表資料喔！請先使用 `/upload_schedule` 上傳您的課表 JSON。"
                memory.add_user_message(question)
                memory.add_assistant_message(fallback)
                return fallback

    # [攔截 1.6] 個人成績/學分查詢 (Personal Transcript)
    if route_result.query_type == "personal_transcript" and discord_id:
        if _has_recommend_intent(question):
            logger.info("  📊➡️🔍 成績查詢中偵測到推薦意圖（如：被當了推薦補什麼），改走 RAG Pipeline")
            route_result.query_type = "course_info"
            # 不 return，讓流程繼續往下走
        else:
            logger.info("  📊 偵測到個人成績/學分查詢，跳過 RAG 直接查詢成績單")
            from tools.transcript_manager import get_transcript_context_for_llm
            personal_ctx = get_transcript_context_for_llm(discord_id, question)
            if personal_ctx:
                logger.info("  ✨ 成績資料取得成功，交由 LLM 進行友善包裝")
                from llm.llm_answer import generate_personal_info_answer
                answer = generate_personal_info_answer(question, personal_ctx, "成績單")
                memory.add_user_message(question)
                memory.add_assistant_message(answer)
                total_time = time.time() - total_start
                print_step(2, f"個人成績單直接回答 (跳過 RAG)", total_time)
                return answer
            else:
                logger.info("  ⚠️ 使用者尚未匯入成績單，fallback 提示")
                fallback = "❌ 您還沒有匯入歷年成績單資料喔！請先使用 `/upload_transcript` 上傳您的成績單 JSON。"
                memory.add_user_message(question)
                memory.add_assistant_message(fallback)
                return fallback

    # [攔截 2] 學校行事曆 (Academic Calendar)
    if route_result.query_type == "academic_calendar":
        # 如果同時帶有課程名稱，代表問的是「某門課的第幾週日期」而非全校行事曆
        # 此時應走 RAG 搜尋 schedule_table，而非翻 events.json
        has_course = bool(route_result.metadata_filters.get("course_name_keyword"))
        if has_course:
            logger.info("  📅→📚 行事曆查詢帶有課程名稱，改走 RAG 搜尋課程進度表")
            route_result.query_type = "course_info"
        else:
            logger.info("  [dim]📅 偵測到學校行事曆查詢，攔截 RAG 檢索直接查詢 events.json[/dim]")
            from tools.search_event_tool import search_academic_events
            academic_events = search_academic_events(question)
            
            if academic_events:
                from llm.llm_calendar import generate_academic_event_answer
                answer_str = generate_academic_event_answer(question, academic_events)
                memory.add_user_message(question)
                memory.add_assistant_message(answer_str)
                return answer_str
            else:
                fallback_msg = "🤔 抱歉，我在學校的行事曆上沒有找到與您問題相關的特定行程或節日喔！"
                logger.info("  [dim]📅 找不到對應學校事件，直接觸發 Fallback 回應[/dim]")
                return _shortcircuit_message(question, memory, fallback_msg)

    # [攔截 3] 缺乏課程特徵的通用查詢 (General fallback)
    core_filters = {k: v for k, v in route_result.metadata_filters.items() if k not in ["semester", "academic_year"]}
    if route_result.query_type == "general" and not core_filters:
        logger.info("  [dim]⚠️ 偵測到無具體課程過濾條件的一般提問，啟動直接對答防護（跳過 RAG 檢索）[/dim]")
        return _shortcircuit_chitchat(question, memory, user_profile)

    # [攔截 4] 行事曆操作預先判斷與短路 (Calendar Actions)
    calendar_intent_data = None
    if route_result.query_type == "calendar_action":
        from llm.llm_calendar import extract_calendar_intent, execute_calendar_action
        logger.info("  [dim]📅 偵測到行事曆操作，預先解析意圖...[/dim]")
        calendar_intent_data = extract_calendar_intent(question)
        
        # 自訂時間、節慶、增刪改查皆無需搜尋課堂資料庫
        if calendar_intent_data.get("action_type") in ["remove", "list", "update"] or calendar_intent_data.get("intent_type") in ["custom_event", "academic_event"]:
            logger.info(f"  [dim]⚡ 行事曆意圖為 {calendar_intent_data.get('intent_type')} ({calendar_intent_data.get('action_type')})，直接攔截跳過 RAG 檢索！[/dim]")
            answer_str = execute_calendar_action(question, calendar_intent_data, discord_id=discord_id)
            memory.add_user_message(question)
            memory.add_assistant_message(answer_str)
            return answer_str
        elif calendar_intent_data.get("intent_type") == "course_schedule_event":
            logger.info("  [dim]📅 意圖為 course_schedule_event，需進入 RAG 搜尋教學進度表[/dim]")
        else:
            logger.info("  [dim]📅 意圖為 weekly_course，需要進入 RAG 檢索課表[/dim]")



    # 【優化】精確查詢跳過 Multi-query 擴充
    if "course_name_keyword" in route_result.metadata_filters and route_result.query_type in ["course_info", "grading", "schedule", "textbook", "syllabus"]:
        queries = [question]  # 精確查詢，捨棄 LLM 擴充的 queries
        logger.info("  📍 精確查詢，跳過 Multi-query 擴充以降低空泛雜訊")
    else:
        # 【關鍵】直接使用 Step 1 route_and_rewrite() 已算好的 queries，不再呼叫 API！
        if question not in queries:
            queries.insert(0, question)
        logger.info(f"  📍 採用 LLM 擴充查詢：{len(queries)} 個")

    if debug:
        for i, q in enumerate(queries):
            console.print(f"      [dim]Q{i+1}: {q}[/dim]")

    # ═════════════════════════════════════
    # Step 3: Hybrid Retriever
    # ═════════════════════════════════════
    step_start = time.time()
    retrieved_chunks = hybrid_retrieve(
        queries=queries,
        route_result=route_result,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        nodes=nodes,
        top_k=config.RETRIEVER_TOP_K,
    )
    print_step(3, f"Hybrid Retriever → {len(retrieved_chunks)} chunks "
               f"(α={config.HYBRID_ALPHA}, β={config.HYBRID_BETA}, γ={config.HYBRID_GAMMA})",
               time.time() - step_start)

    # ═════════════════════════════════════
    # Step 3.5: Intent-Driven Injection（意圖驅動注入）
    # ═════════════════════════════════════
    injected = intent_inject_chunks(
        intent=route_result.query_type,
        filters=route_result.metadata_filters,
        nodes=nodes,
    )
    if injected:
        existing_map = {c.node.node_id: c for c in retrieved_chunks}
        new_chunks = []
        for inc in injected:
            if inc.node.node_id in existing_map:
                # 【修正：強制給予保送加分】如果該 chunk 已經被搜尋到了，必須覆寫它可憐的原始分數，賦予它 10.0 滿級特權！
                existing_map[inc.node.node_id].metadata_score = max(existing_map[inc.node.node_id].metadata_score, inc.metadata_score)
                existing_map[inc.node.node_id].source = inc.source
            else:
                new_chunks.append(inc)
                
        retrieved_chunks.extend(new_chunks)
        logger.info(f"  🎯 Step 3.5 Intent Injection：成功保送 {len(injected)} 個神聖節點 (新增 {len(new_chunks)}/提拔 {len(injected) - len(new_chunks)})")

    if debug and retrieved_chunks:
        table = Table(title="Retriever Results (Top-10)", box=box.SIMPLE)
        table.add_column("#", style="dim")
        table.add_column("Course")
        table.add_column("Section")
        table.add_column("V-Score", justify="right")
        table.add_column("B-Score", justify="right")
        table.add_column("M-Score", justify="right")
        table.add_column("Final", justify="right", style="bold")
        for i, c in enumerate(retrieved_chunks[:10]):
            table.add_row(
                str(i+1),
                c.node.metadata.get("course_name", c.node.metadata.get("professor_name", "?"))[:15],
                c.node.metadata.get("section", c.node.metadata.get("info_type", "?"))[:15],
                f"{c.vector_score:.3f}",
                f"{c.bm25_score:.3f}",
                f"{c.metadata_score:.3f}",
                f"{c.final_score:.3f}",
            )
        console.print(table)

    # ═════════════════════════════════════
    # Step 4: Reranker (精細重排序與斬草除根)
    # ═════════════════════════════════════
    step_start = time.time()
    # 處理特例：若是找教授，因為會強制注入所有教授履歷(佔用名額)，所以要把 Top-N 加大，確保相關課程不會被擠掉
    actual_top_n = 25 if route_result.query_type == "professor_info" else config.RERANKER_TOP_N
    
    reranked_chunks = rerank(
        query=question,
        chunks=retrieved_chunks,
        top_n=actual_top_n,
        route_result=route_result,
    )
    print_step(4, f"Reranker → Top-{len(reranked_chunks)} "
               f"(from {len(retrieved_chunks)} candidates)", time.time() - step_start)

    # 【新增：防爆短路機制】如果過濾後沒有任何合格資料，直接中斷 Pipeline！
    if not reranked_chunks:
        fallback_msg = "📭 抱歉！根據您的條件（例如特定星期或老師），目前的資料庫中沒有找到符合的課程喔。您要不要試著放寬條件再問一次呢？"
        logger.info("  [dim]📍 零檢索結果，直接觸發 Fallback 攔截[/dim]")
        memory.add_user_message(question)
        memory.add_assistant_message(fallback_msg)
        return fallback_msg

    if debug and reranked_chunks:
        for i, c in enumerate(reranked_chunks):
            course = c.node.metadata.get("course_name", "?")
            section = c.node.metadata.get("section", "?")
            console.print(f"      [dim]#{i+1} [{course}][{section}] score={c.final_score:.4f}[/dim]")

    # ═════════════════════════════════════
    # Step 5: LLM Answer Generation / Action Execution
    # ═════════════════════════════════════
    step_start = time.time()
    
    if route_result.query_type == "calendar_action" and calendar_intent_data:
        logger.info("  [dim]📅 觸發行事曆建立流程 (weekly_course RAG 提供支援)[/dim]")
        from llm.llm_calendar import execute_calendar_action
        final_answer = execute_calendar_action(question, calendar_intent_data, reranked_chunks, discord_id=discord_id)
        print_step(5, f"Calendar Action → 完成", time.time() - step_start)
    else:
        answer_result = generate_answer(
            query=question,
            chunks=reranked_chunks,
            memory=memory,
            route_result=route_result,
            all_nodes=nodes,
            user_profile=user_profile,
            discord_id=discord_id,
        )
        final_answer = answer_result.answer
        print_step(5, f"LLM Answer → {len(final_answer)} 字", time.time() - step_start)

    total_time = time.time() - total_start
    console.print(f"\n  [dim]⏱️ 總耗時：{total_time:.2f}s[/dim]\n")

    # ══ 更新對話記憶 ══
    memory.add_user_message(question)
    memory.add_assistant_message(final_answer)

    # 【VRAM 防爆】每次 pipeline 結束後清理 GPU 殘留記憶體
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_answer


# =============================================================================
# 🎮 CLI 主迴圈
# =============================================================================

def main():
    """
    主程式入口。啟動 CLI 互動介面。
    """
    # 初始化 logging
    config.setup_logging()

    # 顯示橫幅
    print_banner()

    # ══ 載入索引 ══
    console.print("[bold green]📂 載入資料與索引...[/bold green]\n")
    try:
        nodes, faiss_index, bm25_index = load_and_index()
        from rag.query_router import init_known_registry
        init_known_registry(nodes)
        # 預載 CKIP 斷詞模型（消除首次查詢的 TensorFlow 冷啟動 ~9 秒）
        from nlp_utils import get_ws_model
        get_ws_model()
        console.print(f"[green]✅ 索引載入完成！共 {len(nodes)} 個文件區段[/green]\n")
    except Exception as e:
        console.print(f"[red]❌ 索引載入失敗：{e}[/red]")
        console.print("[yellow]請確認：[/yellow]")
        console.print("  1. 網路連線是否正常（Gemini Cloud API 需要網路）")
        console.print("  2. GEMINI_API_KEY 是否有效且有 Embedding 權限")
        console.print(f"  3. 資料目錄是否存在：{config.DATA_DIR}")
        sys.exit(1)

    # ══ 初始化對話記憶 ══
    memory = ConversationMemory()
    debug_mode = False

    # ══ 互動主迴圈 ══
    console.print("[bold]💬 請輸入你的問題（輸入 /quit 退出）：[/bold]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]🧑‍🎓 你：[/bold cyan]").strip()

            if not user_input:
                continue

            # ══ 指令處理 ══
            if user_input.lower() == "/quit":
                console.print("\n[yellow]👋 感謝使用，再見！[/yellow]\n")
                break

            elif user_input.lower() == "/rebuild":
                console.print("\n[yellow]🔨 重建索引中...[/yellow]\n")
                nodes, faiss_index, bm25_index = load_and_index(force_rebuild=True)
                console.print(f"[green]✅ 索引重建完成！共 {len(nodes)} 個文件區段[/green]\n")
                continue

            elif user_input.lower() == "/clear":
                memory.clear()
                console.print("\n[green]🗑️ 對話歷史已清除[/green]\n")
                continue

            elif user_input.lower() == "/debug":
                debug_mode = not debug_mode
                status = "開啟" if debug_mode else "關閉"
                console.print(f"\n[yellow]🔧 Debug 模式已{status}[/yellow]\n")
                continue

            # ══ 執行 RAG Pipeline ══
            answer = rag_pipeline(
                question=user_input,
                nodes=nodes,
                faiss_index=faiss_index,
                bm25_index=bm25_index,
                memory=memory,
                debug=debug_mode,
            )

            # 顯示回答
            console.print(Panel(
                Markdown(answer),
                title="🤖 助理回答",
                border_style="green",
                padding=(1, 2),
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 感謝使用，再見！[/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]❌ 發生錯誤：{e}[/red]")
            logger.exception("Pipeline error")
            console.print("[dim]請重新輸入問題，或輸入 /quit 退出[/dim]\n")


if __name__ == "__main__":
    main()
