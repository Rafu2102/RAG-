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
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # GPU VRAM 檢查（RTX 4060 = 8GB，bge-reranker-large ≈ 1.3GB）
        if device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"載入 Reranker 模型：{config.RERANKER_MODEL_NAME} (device=cuda, VRAM={vram_gb:.1f}GB)")
        else:
            logger.info(f"載入 Reranker 模型：{config.RERANKER_MODEL_NAME} (device={device})")

        _reranker_model = CrossEncoder(
            config.RERANKER_MODEL_NAME,
            max_length=512,
            device=device,
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

    # ── Batch 處理（控制 GPU 記憶體使用） ──
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]

        with torch.no_grad():
            batch_scores = model.predict(
                batch_pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )

        all_scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else list(batch_scores))

        logger.debug(f"  Rerank batch {i//batch_size + 1}：{len(batch_pairs)} pairs")

    # ── 更新分數並排序 ──
    for chunk, rerank_score in zip(chunks, all_scores):
        rerank_prob = 1.0 / (1.0 + math.exp(-float(rerank_score)))
        
        # 【關鍵修正】將原本的 metadata_score 加權回來，確保「課程名稱對的」絕對排在前面！
        # 0.7 給語意理解，0.3 留給實體屬性防禦
        chunk.final_score = (rerank_prob * 0.75) + (chunk.metadata_score * 0.25)

        # ── 特殊情境加分防護 (抵抗 Cross-Encoder 的語意盲點) ──
        section = chunk.node.metadata.get("section", "")
        
        # 情境 A: 詢問排課/時間/推薦 -> 保護 basic_info
        if section == "basic_info" and any(k in query for k in ["時間", "空堂", "星期", "禮拜", "節", "推薦"]):
            chunk.final_score += 0.10
            
        # 【新增】情境 B: 詢問教授領域/實驗室/專長 -> 強力保護 教學目標 與 課程綱要
        if section in ["objectives", "syllabus"] and any(k in query for k in ["領域", "實驗室", "專長", "能力", "研究", "教什麼"]):
            chunk.final_score += 0.20

    # 依 reranker 分數排序
    chunks.sort(key=lambda c: c.final_score, reverse=True)

    # 【甲乙去重】同課程 + 同區段 + 同老師的多班級 chunk 只保留一個
    # 優先保留甲班（或無班級標記的），除非使用者明確查詢乙班
    seen_keys = {}  # key -> chunk
    dup_count = 0
    for chunk in chunks:
        meta = chunk.node.metadata
        dedup_key = (meta.get("course_name", ""), meta.get("section", ""), meta.get("teacher", ""))
        cg = meta.get("class_group", "")
        
        if dedup_key in seen_keys:
            existing_cg = seen_keys[dedup_key].node.metadata.get("class_group", "")
            # 偏好甲班（甲 or 空）：如果現有的是乙而新的是甲，替換
            if existing_cg == "乙" and cg in ("甲", ""):
                seen_keys[dedup_key] = chunk
            dup_count += 1
            continue
        seen_keys[dedup_key] = chunk
    
    deduped = list(seen_keys.values())
    deduped.sort(key=lambda c: c.final_score, reverse=True)
    
    if dup_count > 0:
        logger.info(f"  🔄 去重：移除 {dup_count} 個重複 chunks（同課程+同區段+同老師，優先保留甲班）")
    
    # ========================================================================
    # 【課程完整覆蓋保證】— 教師查詢時確保每門課至少出現一次
    #
    # 問題：每門課有 ~6 個 section chunks，Top-10 只能裝 2-3 門課。
    #       若老師教 5 門課，有 2 門會被 Top-N 切掉。
    # 解法：先收集每門課的 basic_info（最精華），再用剩餘名額補高分 section。
    # ========================================================================
    
    has_teacher_filter = bool(route_result and route_result.metadata_filters.get("teacher"))
    
    if has_teacher_filter and len(deduped) > top_n:
        # 1. 收集每門課的最佳 chunk（優先 basic_info）
        course_best = {}  # course_name -> chunk
        for chunk in deduped:
            cname = chunk.node.metadata.get("course_name", "")
            section = chunk.node.metadata.get("section", "")
            if cname not in course_best:
                course_best[cname] = chunk
            elif section == "basic_info" and course_best[cname].node.metadata.get("section", "") != "basic_info":
                course_best[cname] = chunk  # basic_info 優先
        
        # 2. 保底：每門課至少 1 個 chunk
        guaranteed = list(course_best.values())
        guaranteed_ids = {c.node.node_id for c in guaranteed}
        
        # 3. 剩餘名額補高分 section
        remaining = [c for c in deduped if c.node.node_id not in guaranteed_ids]
        remaining.sort(key=lambda c: c.final_score, reverse=True)
        
        # 【關鍵優化】動態提升 top_n 容量
        # 原本硬限制 top_n(10)，導致老師若教 3 門課，每門課只能分到約 3 個 chunk，LLM 沒資料寫課綱
        # 現在改為：保障每門課「平均」有 4 個 chunk 的空間 (basic_info + syllabus + grading + schedule)
        # 上限不超過 RETRIEVER_TOP_K (30)，避免塞爆 Context Window
        effective_top_n = max(top_n, min(len(course_best) * 4, 30))
        
        fill_count = max(0, effective_top_n - len(guaranteed))
        
        results = guaranteed + remaining[:fill_count]
        results.sort(key=lambda c: c.final_score, reverse=True)
        
        logger.info(f"  📚 教師模式：開課數 {len(course_best)} 門，動態提升 Top-N 容量至 {effective_top_n}，已放入 {len(results)} 個區段")
    else:
        # 一般模式：直接 Top-N
        results = deduped[:top_n]

    logger.info(
        f"✅ Reranker 完成：Top-{len(chunks)} → Top-{len(results)} "
        f"(最高={results[0].final_score:.4f}, "
        f"最低={results[-1].final_score:.4f})"
    )

    for i, chunk in enumerate(results):
        course = chunk.node.metadata.get("course_name", "?")
        section = chunk.node.metadata.get("section", "?")
        logger.info(f"  #{i+1} [{course}][{section}] score={chunk.final_score:.4f}")

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
