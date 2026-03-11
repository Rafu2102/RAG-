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

    # 取 Top-N
    results = chunks[:top_n]

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
