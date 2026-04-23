# -*- coding: utf-8 -*-
"""
reranker.py — Reranker 模組
==============================
負責：
1. 使用 bge-reranker-large cross-encoder 進行精細重排序
2. 輸入 Top-30 → 輸出 Top-5
3. GPU batch 處理（適配 RTX 4060 8GB VRAM）
4. 自動偵測 GPU/CPU 並調整 batch size
"""

import logging
import math
import gc
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

import config
from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# =============================================================================
# 🔄 Reranker 單例
# =============================================================================

_reranker_model: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """
    取得 bge-reranker-large cross-encoder 模型（單例模式）。

    自動偵測 GPU，若有 CUDA 則使用 GPU，否則使用 CPU。
    RTX 4060 有 8GB VRAM，需要注意 batch size 控制。
    """
    global _reranker_model
    if _reranker_model is None:
        # GPU 偵測（防禦 CUDA 初始化失敗 — Ollama 佔用 GPU 時可能發生）
        device = "cpu"
        if torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                device = "cuda"
                logger.info(f"載入 Reranker 模型：{config.RERANKER_MODEL_NAME} (device=cuda, VRAM={vram_gb:.1f}GB)")
            except Exception as e:
                logger.warning(f"CUDA 裝置存取失敗，改用 CPU：{e}")

        if device == "cpu":
            logger.info(f"載入 Reranker 模型：{config.RERANKER_MODEL_NAME} (device=cpu)")

        model_kwargs = {}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        _reranker_model = CrossEncoder(
            config.RERANKER_MODEL_NAME,
            max_length=512,
            device=device,
            model_kwargs=model_kwargs
        )
            
        logger.info("Reranker 模型載入完成")

    return _reranker_model


# =============================================================================
# 🔄 Rerank 主函式
# =============================================================================

def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = None,
    batch_size: int = None,
    route_result = None,
) -> list[RetrievedChunk]:
    """
    使用 bge-reranker-large cross-encoder 對候選 chunks 進行精細重排序。

    Cross-encoder 會計算 (query, passage) pair 的相關性分數，
    比 bi-encoder（embedding）更準確，但速度較慢。

    Pipeline: Top-30 (from retriever) → Cross-encoder scoring → Top-5

    Args:
        query: 使用者原始問題
        chunks: 候選 RetrievedChunk 列表（通常 top-30）
        top_n: 重排後保留的前 N 個結果
        batch_size: GPU batch 大小（RTX 4060 建議 16）

    Returns:
        重排序後的 RetrievedChunk 列表（top_n 個）
    """
    if top_n is None:
        top_n = config.RERANKER_TOP_N
    if batch_size is None:
        batch_size = config.RERANKER_BATCH_SIZE

    if not chunks:
        logger.warning("Reranker：沒有候選 chunks")
        return []

    logger.info(f"🔄 Reranker：對 {len(chunks)} 個候選 chunks 進行重排序...")

    model = get_reranker()

    # ── 準備 (query, passage) pairs ──
    pairs = [(query, chunk.node.get_content()) for chunk in chunks]

    # ── Batch 處理（CrossEncoder 內建 batch，直接傳入即可） ──
    with torch.no_grad():
        all_scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    all_scores = all_scores.tolist() if hasattr(all_scores, 'tolist') else list(all_scores)

    # ── 更新分數並排序 ──
    for chunk, rerank_score in zip(chunks, all_scores):
        rerank_prob = 1.0 / (1.0 + math.exp(-float(rerank_score)))
        
        # 【關鍵修正】將原本的 metadata_score 加權回來，確保「課程名稱對的」絕對排在前面！
        # 0.7 給語意理解，0.3 留給實體屬性防禦
        chunk.final_score = (rerank_prob * 0.75) + (chunk.metadata_score * 0.25)

        # ── 特殊情境加分防護 (抵抗 Cross-Encoder 的語意盲點) ──
        section = chunk.node.metadata.get("section", "")
        info_type = chunk.node.metadata.get("info_type", "")
        
        # 情境 A: 詢問排課/時間/推薦 -> 保護 basic_info
        if section == "basic_info" and any(k in query for k in ["時間", "空堂", "星期", "禮拜", "節", "推薦"]):
            chunk.final_score += 0.10
            
        # 情境 B: 詢問教授領域/實驗室/專長 -> 強力保護 教學目標 與 課程綱要
        if section in ["objectives", "syllabus"] and any(k in query for k in ["領域", "實驗室", "專長", "能力", "研究", "教什麼"]):
            chunk.final_score += 0.20

        # 🆕 情境 C: 意圖為 professor_info → 教授資訊 chunk 獲得大幅加分
        # 確保問「教授相關資訊」時，教授資料頁面排在課程頁面前面
        if route_result and getattr(route_result, 'query_type', '') == 'professor_info':
            if info_type == "professor_info":
                chunk.final_score += 0.50  # 教授資訊 chunk 大幅加分

    # 依 reranker 分數排序
    chunks.sort(key=lambda c: c.final_score, reverse=True)

    # 【去重邏輯修正】同課程 + 同區段 + 同老師的多班級 chunk 只保留一個來源檔案
    # 允許同一份 source_file 的連續切片 (Sequential Chunks) 並存，防止長課表被腰斬
    seen_files_for_key = {}  # dedup_key -> 該課程被選中的 source_file
    deduped = []
    dup_count = 0
    _NON_COURSE_TYPES = ("professor_info", "dept_intro", "career_info", "student_union", "dept_news", "dept_general", "graduation_rules", "policy", "policy_rules")
    
    for chunk in chunks:
        meta = chunk.node.metadata
        info_type = meta.get("info_type", "")
        if info_type in _NON_COURSE_TYPES:
            deduped.append(chunk)
            continue
            
        dedup_key = (meta.get("course_name", ""), meta.get("section", ""), meta.get("teacher", ""))
        source_file = meta.get("source_file", "")
        
        if dedup_key not in seen_files_for_key:
            # 第一次看到這個組合，直接登記這份檔案為「合法來源」
            seen_files_for_key[dedup_key] = source_file
            deduped.append(chunk)
        else:
            chosen_file = seen_files_for_key[dedup_key]
            if source_file == chosen_file:
                # 這是同一份檔案被切出來的「下半段」，絕對不能刪除！
                deduped.append(chunk)
            else:
                # 這是另一份檔案（例如乙班），視為重複班級資料刪除
                dup_count += 1
                
    deduped.sort(key=lambda c: c.final_score, reverse=True)
    
    if dup_count > 0:
        logger.info(f"  🔄 去重：移除 {dup_count} 個重複 chunks（保留高分班級，允許同檔連續切片）")
    
    # ========================================================================
    # 【課程完整覆蓋保證】— 確保每門課至少出現一次
    #
    # 觸發條件：
    #   1. 課程相關的廣泛查詢（course_info）
    #   2. 特定過濾（教師、必選修）
    #   3. 一般查詢（general）但帶有系級過濾（通常是使用者透過 Profile 詢問「我有什麼課」）
    #   一律啟動防漏課機制，動態擴大 Top-N 容量。
    # ========================================================================
    
    is_course_query = bool(route_result and route_result.query_type == "course_info")
    has_teacher_filter = bool(route_result and route_result.metadata_filters.get("teacher"))
    has_req_filter = bool(route_result and route_result.metadata_filters.get("required_or_elective"))
    is_general_with_profile = bool(route_result and route_result.query_type == "general" and 
                                  (route_result.metadata_filters.get("dept_short") or route_result.metadata_filters.get("grade")))
    
    is_career_planning = bool(route_result and getattr(route_result, "is_career_planning", False))
    
    needs_coverage = is_course_query or has_teacher_filter or has_req_filter or is_general_with_profile
    
    if is_career_planning and route_result and route_result.query_type == "course_info":
        # 大範圍探索問題（僅限 course_info），直接放行所有去重後的結果
        results = deduped
        logger.info(f"  🌌 職涯探索模式：強制放行所有 {len(results)} 個不重複的 chunks 供 LLM 全景分析")
    elif needs_coverage:
        course_best = {}
        for chunk in deduped:
            meta = chunk.node.metadata
            info_type = meta.get("info_type", "")
            # 教授/系所資訊不參與「課程覆蓋」邏輯，直接當獨立項目保留（使用 node_id 確保不覆蓋同教授的多個 chunks）
            if info_type in _NON_COURSE_TYPES:
                unique_key = f"_info_{chunk.node.node_id}"
                course_best[unique_key] = chunk
                continue
            cname = meta.get("course_name", "")
            section = meta.get("section", "")
            if cname not in course_best:
                course_best[cname] = chunk
            # 優先保留 basic_info 或 syllabus，避免被 schedule_table 等區段取代
            elif course_best[cname].node.metadata.get("section", "") not in ("basic_info", "syllabus") and section in ("basic_info", "syllabus"):
                course_best[cname] = chunk
        
        guaranteed = list(course_best.values())
        
        # 讓保底課程之間依分數排序，避免顯示順序混亂
        guaranteed.sort(key=lambda c: c.final_score, reverse=True)
        
        guaranteed_ids = {c.node.node_id for c in guaranteed}
        remaining = [c for c in deduped if c.node.node_id not in guaranteed_ids]
        remaining.sort(key=lambda c: c.final_score, reverse=True)
        
        # 確保最終容量至少為 top_n，且至少能裝下所有保底課程再外加 5 個名額供進度表使用
        effective_top_n = max(top_n, len(guaranteed) + 5)
        fill_count = max(0, effective_top_n - len(guaranteed))
        
        # 將保底名單與剩餘名單合併，並【重新依分數排序】，確保整體顯示順序符合相關性
        results = guaranteed + remaining[:fill_count]
        results.sort(key=lambda c: c.final_score, reverse=True)
        
        logger.info(f"  📚 清單模式：保底 {len(guaranteed)} 門課程，實際輸出 Top-{len(results)}")
    else:
        results = deduped[:top_n]

    # 【Bug 2 修復】防止 results 為空時 IndexError
    if results:
        logger.info(
            f"✅ Reranker 完成：Top-{len(chunks)} → Top-{len(results)} "
            f"(最高={results[0].final_score:.4f}, "
            f"最低={results[-1].final_score:.4f})"
        )
        # 只印前 30 筆，避免 log 爆炸
        log_limit = min(30, len(results))
        for i, chunk in enumerate(results[:log_limit]):
            meta = chunk.node.metadata
            info_type = meta.get("info_type", "")
            if info_type:
                label = meta.get("professor_name", meta.get("category", info_type))
                logger.info(f"  #{i+1} [📋 {label}][{info_type}] score={chunk.final_score:.4f}")
            else:
                course = meta.get("course_name", "?")
                section = meta.get("section", "?")
                logger.info(f"  #{i+1} [{course}][{section}] score={chunk.final_score:.4f}")
        if len(results) > log_limit:
            logger.info(f"  ... 還有 {len(results) - log_limit} 個 chunks 未顯示")
    else:
        logger.warning("⚠️ Reranker 完成但結果為空，所有候選 chunks 均被過濾")

    # 【VRAM 防爆】強制釋放 Cross-Encoder 推理殘留的 GPU 記憶體
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Reranker VRAM GC 完成")

    return results


# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    config.setup_logging()

    # 測試 reranker 載入
    model = get_reranker()
    # 簡單測試
    test_query = "深度學習的老師是誰？"
    test_passages = [
        "課程名稱：深度學習。授課教師：馮玄明。學分數：3學分。",
        "課程名稱：資料結構。授課教師：馮玄明。學分數：3學分。",
        "今天天氣不錯，適合出去玩。",
    ]

    pairs = [(test_query, p) for p in test_passages]
    scores = model.predict(pairs)

    print(f"\n❓ Query: {test_query}")
    for p, s in zip(test_passages, scores):
        print(f"  Score={s:.4f} | {p[:60]}")
