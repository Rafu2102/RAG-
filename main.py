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
    7. LLM Answer（Llama 3.1 + Source Grounding）
    8. 更新對話記憶
    9. 回到步驟 2

指令：
    /quit   - 退出程式
    /rebuild - 重建索引
    /clear  - 清除對話歷史
    /debug  - 切換 debug 模式（顯示詳細檢索資訊）
"""

import sys
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
from data_loader import load_and_index
from query_router import route_and_rewrite, route_query
from retriever import hybrid_retrieve
from reranker import rerank
from llm_answer import generate_answer, format_sources, ConversationMemory

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# 🎨 UI 相關
# =============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║            🎓 NQU 校園課程助理機器人 v1.0                   ║
║            國立金門大學 · 資訊工程學系                       ║
║                                                              ║
║  📚 Hybrid RAG Pipeline                                     ║
║  🤖 Llama 3.1 8B + multilingual-e5-large + bge-reranker     ║
║                                                              ║
║  指令：/quit 退出 | /rebuild 重建索引 | /clear 清除對話      ║
║        /debug 切換詳細模式                                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """顯示程式橫幅"""
    console.print(BANNER, style="bold cyan")


def print_step(step_num: int, description: str, elapsed: float = None):
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
) -> str:
    """
    執行完整的 RAG Pipeline。

    Pipeline:
        User Question
        → Step 1: Query Rewrite (Multi-query)
        → Step 2: Query Router (type + metadata filter)
        → Step 3: Hybrid Retriever (Vector + BM25 + Metadata → α·V + β·B + γ·M)
        → Step 4: Reranker (Top-30 → Top-5, cross-encoder)
        → Step 5: LLM Answer (Llama 3.1 + source grounding)
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 0: 規則式閒聊快速攔截（在 LLM Router 之前，零延遲）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    import re as _re
    # 去除所有 emoji 和特殊符號後，檢查是否還有實質內容
    stripped = _re.sub(r'[\U00010000-\U0010ffff\u2600-\u27BF\u2700-\u27BF\uFE00-\uFE0F\u200d❤♥☺✨⭐🌟💕😀-🙏]+', '', question, flags=_re.UNICODE).strip()
    common_greetings = {"你好", "哈囉", "嗨", "hi", "hello", "早安", "午安", "晚安", "謝謝", "感謝", "掰掰", "bye", "嗯", "ok", "好"}
    is_trivial = (
        len(stripped) == 0 or                       # 純 emoji
        len(stripped) <= 2 or                        # 太短沒意義（如「是」「好」）
        stripped.lower() in common_greetings         # 常見打招呼
    )
    if is_trivial:
        logger.info("  👋 規則式閒聊攔截：跳過 Router+Retriever，直接回應")
        from llm_answer import generate_chitchat_answer
        answer_obj = generate_chitchat_answer(question, memory)
        memory.add_user_message(question)
        memory.add_assistant_message(answer_obj.answer)
        return answer_obj.answer

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 1: 合併式 Router + Rewrite（單次 LLM 呼叫）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step_start = time.time()
    route_result, queries = route_and_rewrite(
        question=question,
        chat_history=memory.get_history(),
    )

    # 【學期預設】若使用者未指定學期，自動注入當前學期 filter
    if "semester" not in route_result.metadata_filters:
        route_result.metadata_filters["semester"] = str(config.CURRENT_SEMESTER)
    if "academic_year" not in route_result.metadata_filters:
        route_result.metadata_filters["academic_year"] = str(config.CURRENT_ACADEMIC_YEAR)

    print_step(1, f"Router+Rewrite → type={route_result.query_type}, filters={route_result.metadata_filters}, {len(queries)} queries", time.time() - step_start)

    # 【Agentic 攔截：閒聊短路】
    if route_result.query_type == "chitchat":
        logger.info("  [dim]👋 偵測到日常閒聊，跳過檢索直接回應[/dim]")
        from llm_answer import generate_chitchat_answer
        answer_obj = generate_chitchat_answer(question, memory)
        memory.add_user_message(question)
        memory.add_assistant_message(answer_obj.answer)
        return answer_obj.answer

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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 3: Hybrid Retriever
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
                c.node.metadata.get("course_name", "?")[:15],
                c.node.metadata.get("section", "?"),
                f"{c.vector_score:.3f}",
                f"{c.bm25_score:.3f}",
                f"{c.metadata_score:.3f}",
                f"{c.final_score:.3f}",
            )
        console.print(table)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 4: Reranker (精細重排序與斬草除根)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step_start = time.time()
    reranked_chunks = rerank(
        query=question,
        chunks=retrieved_chunks,
        top_n=config.RERANKER_TOP_N,
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5: LLM Answer Generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step_start = time.time()
    answer_result = generate_answer(
        query=question,
        chunks=reranked_chunks,
        memory=memory,
    )
    print_step(5, f"LLM Answer → {len(answer_result.answer)} 字", time.time() - step_start)

    total_time = time.time() - total_start
    console.print(f"\n  [dim]⏱️ 總耗時：{total_time:.2f}s[/dim]\n")

    # ── 更新對話記憶 ──
    memory.add_user_message(question)
    memory.add_assistant_message(answer_result.answer)

    # 【VRAM 防爆】每次 pipeline 結束後清理 GPU 殘留記憶體
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return answer_result.answer


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

    # ── 載入索引 ──
    console.print("[bold green]📂 載入資料與索引...[/bold green]\n")
    try:
        nodes, faiss_index, bm25_index = load_and_index()
        console.print(f"[green]✅ 索引載入完成！共 {len(nodes)} 個文件區段[/green]\n")
    except Exception as e:
        console.print(f"[red]❌ 索引載入失敗：{e}[/red]")
        console.print("[yellow]請確認：[/yellow]")
        console.print("  1. Ollama 服務是否運行（ollama serve）")
        console.print("  2. multilingual-e5-large 模型是否已下載")
        console.print(f"  3. 資料目錄是否存在：{config.DATA_DIR}")
        sys.exit(1)

    # ── 初始化對話記憶 ──
    memory = ConversationMemory()
    debug_mode = False

    # ── 互動主迴圈 ──
    console.print("[bold]💬 請輸入你的問題（輸入 /quit 退出）：[/bold]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]🧑‍🎓 你：[/bold cyan]").strip()

            if not user_input:
                continue

            # ── 指令處理 ──
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

            # ── 執行 RAG Pipeline ──
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
