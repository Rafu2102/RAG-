# -*- coding: utf-8 -*-
"""
retriever.py — Hybrid Retriever 模組
======================================
負責：
1. Vector Search（FAISS + Ollama embedding）
2. BM25 Keyword Search（jieba 分詞）
3. Metadata Filtering（department, grade, course_type, teacher, topic）
4. Hybrid Fusion 分數計算：final_score = α*vector + β*BM25 + γ*metadata
5. 支援 Multi-query 聚合

核心公式：
    final_score = α * normalized_vector_score
               + β * normalized_bm25_score
               + γ * metadata_match_score

α/β/γ 在 config.py 中可調整。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from llama_index.core.schema import TextNode

import json
import re
import config
from .data_loader import ollama_embed_query, tokenize_chinese
from .query_router import RouteResult
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# 📋 檢索結果資料類別
# =============================================================================

@dataclass
class RetrievedChunk:
    """單個檢索結果"""
    node: TextNode              # 原始 Node
    vector_score: float = 0.0   # Vector search 分數（cosine similarity）
    bm25_score: float = 0.0     # BM25 分數
    metadata_score: float = 0.0 # Metadata 匹配分數
    final_score: float = 0.0    # 融合後最終分數
    source: str = ""            # 來源標記（vector / bm25 / both）

    def __repr__(self):
        course = self.node.metadata.get("course_name", "?")
        section = self.node.metadata.get("section", "?")
        return (f"Chunk(course={course}, section={section}, "
                f"v={self.vector_score:.3f}, b={self.bm25_score:.3f}, "
                f"m={self.metadata_score:.3f}, final={self.final_score:.3f})")


# =============================================================================
# 🔍 Vector Search
# =============================================================================

def vector_search(
    query: str,
    faiss_index: faiss.IndexFlatIP,
    nodes: list[TextNode],
    top_k: int = None,
) -> list[RetrievedChunk]:
    """
    使用 FAISS 進行向量語意搜尋。

    Args:
        query: 搜尋查詢
        faiss_index: FAISS 索引
        nodes: 全部 Nodes 列表（與 FAISS 索引對應）
        top_k: 取前 k 個結果

    Returns:
        RetrievedChunk 列表（依 vector_score 排序）
    """
    if top_k is None:
        top_k = config.RETRIEVER_TOP_K

    # 產生 query embedding（使用 "query: " 前綴）
    query_embedding = ollama_embed_query(query)

    # L2 正規化
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm

    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

    # FAISS 搜尋
    scores, indices = faiss_index.search(query_embedding, min(top_k, len(nodes)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(nodes):
            continue
        results.append(RetrievedChunk(
            node=nodes[idx],
            vector_score=float(score),
            source="vector",
        ))

    logger.debug(f"Vector Search：查詢「{query[:30]}...」→ {len(results)} 結果")
    return results


def vector_search_with_embedding(
    query_embedding: list,  # Ollama API 回傳的是 list[float]，不是 np.ndarray
    faiss_index: faiss.IndexFlatIP,
    nodes: list[TextNode],
    top_k: int = None,
) -> list[RetrievedChunk]:
    """
    使用已計算好的 embedding 進行 FAISS 搜尋（避免重複呼叫 Ollama Embed API）。
    配合 embed_queries_parallel 使用，實現真正的並行搜尋。
    """
    if top_k is None:
        top_k = config.RETRIEVER_TOP_K

    # 【關鍵修復】將 Python list 轉換為 numpy array 才能呼叫 reshape
    query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    scores, indices = faiss_index.search(query_vec, min(top_k, len(nodes)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(nodes):
            continue
        results.append(RetrievedChunk(
            node=nodes[idx],
            vector_score=float(score),
            source="vector",
        ))
    return results


def embed_queries_parallel(queries: list[str]) -> list[np.ndarray]:
    """
    並行呼叫 Ollama 取得多個 query 的 embedding。
    真正的效能瓶頸在於 Embedding API 呼叫（每次 ~1秒），而不是 FAISS search（~0.01秒）。
    """
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        embeddings = list(executor.map(ollama_embed_query, queries))
    return embeddings


# =============================================================================
# 📚 BM25 Keyword Search
# =============================================================================

def bm25_search(
    query: str,
    bm25_index: BM25Okapi,
    nodes: list[TextNode],
    top_k: int = None,
) -> list[RetrievedChunk]:
    """
    使用 BM25 進行關鍵字搜尋。

    Args:
        query: 搜尋查詢
        bm25_index: BM25 索引
        nodes: 全部 Nodes 列表
        top_k: 取前 k 個結果

    Returns:
        RetrievedChunk 列表（依 bm25_score 排序）
    """
    if top_k is None:
        top_k = config.RETRIEVER_TOP_K

    # 中文分詞
    tokens = tokenize_chinese(query)
    if not tokens:
        logger.warning("BM25 Search：分詞結果為空")
        return []

    # BM25 計算分數
    scores = bm25_index.get_scores(tokens)

    # 取 top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # 只保留有分數的結果
            results.append(RetrievedChunk(
                node=nodes[idx],
                bm25_score=float(scores[idx]),
                source="bm25",
            ))

    logger.debug(f"BM25 Search：查詢「{query[:30]}...」→ {len(results)} 結果")
    return results


# =============================================================================
# 🏷️ Metadata Matching Score
# =============================================================================

def compute_metadata_score(node: TextNode, route_result: RouteResult) -> float:
    """
    計算單個 Node 與 query metadata 條件的匹配分數。

    對每個 metadata 欄位進行匹配檢查，匹配到的加分。
    分數範圍 [0.0, 1.0]。

    Args:
        node: TextNode
        route_result: Query Router 的結果（包含 metadata_filters）

    Returns:
        metadata 匹配分數（0.0 ~ 1.0）
    """
    if not route_result.metadata_filters:
        return 0.0

    filters = route_result.metadata_filters
    node_meta = node.metadata
    score = 0.0
    max_possible = 0.0

    def _check_match(filter_val, node_val, exact=False):
        if not filter_val: return False
        n_val = str(node_val) if node_val else ""
        
        # 兼容 LLM 可能回傳字串或列表 (e.g., ['未提及']) 的情況
        f_vals = filter_val if isinstance(filter_val, list) else [filter_val]
        for v in f_vals:
            sv = str(v)
            if exact:
                if sv == n_val: return True
            else:
                if sv in n_val: return True
        return False

    # ── 系所匹配 ──
    if "dept_short" in filters:
        max_possible += config.METADATA_MATCH_SCORES["department"]
        if _check_match(filters["dept_short"], node_meta.get("dept_short", ""), exact=False):
            score += config.METADATA_MATCH_SCORES["department"]

    # ── 年級匹配 ──
    if "grade" in filters:
        max_possible += config.METADATA_MATCH_SCORES["grade"]
        if _check_match(filters["grade"], node_meta.get("grade", ""), exact=True):
            score += config.METADATA_MATCH_SCORES["grade"]

    # ── 必修/選修匹配 ──
    if "required_or_elective" in filters:
        max_possible += config.METADATA_MATCH_SCORES["course_type"]
        if _check_match(filters["required_or_elective"], node_meta.get("required_or_elective", ""), exact=False):
            score += config.METADATA_MATCH_SCORES["course_type"]

    # ── 教師匹配 ──
    if "teacher" in filters:
        max_possible += config.METADATA_MATCH_SCORES["teacher"]
        if _check_match(filters["teacher"], node_meta.get("teacher", ""), exact=False):
            score += config.METADATA_MATCH_SCORES["teacher"]

    # ── 課程名稱關鍵字匹配 ──
    if "course_name_keyword" in filters:
        max_possible += 0.3
        if _check_match(filters["course_name_keyword"], node_meta.get("course_name", ""), exact=False):
            score += 0.3

    # ── 星期/時間匹配 (加權) ──
    if "day_of_week" in filters:
        max_possible += 0.3
        # 檢查該片段的內文是否有提到「星期X」
        if filters["day_of_week"] in node.get_content():
            score += 0.3

    # ── Section 類型匹配（根據 query_type）──
    section_type_map = {
        "course_info": "basic_info",
        "syllabus": "syllabus",
        "textbook": "textbooks",
        "grading": "grading",
        "schedule": "schedule_table",
    }
    if route_result.query_type in section_type_map:
        max_possible += 0.2
        expected_section = section_type_map[route_result.query_type]
        if node_meta.get("section", "") == expected_section:
            score += 0.2

    # 正規化到 [0, 1]
    if max_possible > 0:
        return score / max_possible
    return 0.0


# =============================================================================
# 🔗 Hybrid Fusion
# =============================================================================

def hybrid_fusion(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    route_result: RouteResult,
    top_k: int = None,
) -> list[RetrievedChunk]:
    """
    融合 Vector Search + BM25 Search 結果，計算最終分數。

    公式：final_score = α * norm_vector + β * norm_bm25 + γ * metadata_score

    Args:
        vector_results: Vector Search 結果
        bm25_results: BM25 Search 結果
        route_result: Query Router 結果（含 metadata filters）
        top_k: 最終取前 k 個

    Returns:
        融合排序後的 RetrievedChunk 列表
    """
    if top_k is None:
        top_k = config.RETRIEVER_TOP_K

    α = config.HYBRID_ALPHA
    β = config.HYBRID_BETA
    γ = config.HYBRID_GAMMA

    # ── 建立 node_id → RetrievedChunk 的映射（去重合併）──
    merged: dict[str, RetrievedChunk] = {}

    for chunk in vector_results:
        node_id = chunk.node.node_id
        if node_id not in merged:
            merged[node_id] = RetrievedChunk(node=chunk.node, source="vector")
        # 【關鍵修正】取最高分，而不是無腦覆蓋
        merged[node_id].vector_score = max(merged[node_id].vector_score, chunk.vector_score)

    for chunk in bm25_results:
        node_id = chunk.node.node_id
        if node_id not in merged:
            merged[node_id] = RetrievedChunk(node=chunk.node, source="bm25")
        else:
            merged[node_id].source = "both"
        # 【關鍵修正】取最高分
        merged[node_id].bm25_score = max(merged[node_id].bm25_score, chunk.bm25_score)

    # ── 計算 RRF (Reciprocal Rank Fusion) 分數 ──
    # 取代容易產生雜訊放大的 Min-Max 正規化
    all_chunks = list(merged.values())

    vector_sorted = sorted([c for c in all_chunks if c.vector_score > 0], key=lambda x: x.vector_score, reverse=True)
    bm25_sorted = sorted([c for c in all_chunks if c.bm25_score > 0], key=lambda x: x.bm25_score, reverse=True)

    v_rank = {c.node.node_id: rank + 1 for rank, c in enumerate(vector_sorted)}
    b_rank = {c.node.node_id: rank + 1 for rank, c in enumerate(bm25_sorted)}

    k_rrf = getattr(config, "RRF_K", 60)
    max_rrf_score = 1.0 / (k_rrf + 1)  # 理論上的最高分 (第 1 名)

    # 1. 先計算原始的 metadata_score
    for chunk in all_chunks:
        chunk.metadata_score = compute_metadata_score(chunk.node, route_result)

    # 2. 【關鍵改動：提前過濾！】在切 Top-50 之前，就對全部 150 個 chunk 進行時間與老師的降權
    if route_result.metadata_filters:
        all_chunks = _apply_hard_metadata_filter(all_chunks, route_result)

    # 3. 再結算最終的 final_score
    for chunk in all_chunks:
        node_id = chunk.node.node_id
        
        # 取得原始 RRF 分數
        raw_v_rrf = 1.0 / (k_rrf + v_rank.get(node_id, 1000)) if node_id in v_rank else 0.0
        raw_b_rrf = 1.0 / (k_rrf + b_rank.get(node_id, 1000)) if node_id in b_rank else 0.0
        
        # 將 RRF 正規化到 0.0 ~ 1.0 之間，才能跟 Metadata 平起平坐
        v_rrf_norm = raw_v_rrf / max_rrf_score
        b_rrf_norm = raw_b_rrf / max_rrf_score
        
        chunk.final_score = (
            α * v_rrf_norm +
            β * b_rrf_norm +
            γ * chunk.metadata_score
        )

    # 4. 【最後才排序切片】讓被扣 10 分的垃圾沉到谷底，真正的匹配課程浮上來
    all_chunks.sort(key=lambda c: c.final_score, reverse=True)
    results = all_chunks[:top_k]

    logger.info(
        f"🔗 Hybrid Fusion：{len(vector_results)} vector + {len(bm25_results)} bm25 "
        f"→ {len(merged)} 合併 → Top-{len(results)} "
        f"(α={α}, β={β}, γ={γ})"
    )

    return results


# =============================================================================
# 🚀 Hybrid Retrieve 主函式（支援 Multi-query）
# =============================================================================

def hybrid_retrieve(
    queries: list[str],
    route_result: RouteResult,
    faiss_index: faiss.IndexFlatIP,
    bm25_index: BM25Okapi,
    nodes: list[TextNode],
    top_k: int = None,
) -> list[RetrievedChunk]:
    """
    執行 Hybrid Retrieval（支援 Multi-query 聚合）。

    對每個 query 分別進行 Vector + BM25 搜尋，
    然後將所有結果融合排序。

    Pipeline:
        Multi-queries → Vector Search × N + BM25 Search × N
        → Merge & Deduplicate → Metadata Filter Scoring
        → Hybrid Fusion (α·V + β·B + γ·M)
        → Top-k

    Args:
        queries: 搜尋查詢列表（Multi-query）
        route_result: Query Router 結果
        faiss_index: FAISS 索引
        bm25_index: BM25 索引
        nodes: 全部 Nodes
        top_k: 最終取前 k 個

    Returns:
        融合排序後的 RetrievedChunk 列表
    """
    if top_k is None:
        top_k = config.RETRIEVER_TOP_K

    all_vector_results = []
    all_bm25_results = []

    # 【關鍵加速】並行取得所有 query 的 embedding（真正瓶頸在此！）
    logger.info(f"  🚀 並行 Embedding {len(queries)} 個查詢...")
    query_embeddings = embed_queries_parallel(queries)

    # 並行執行所有 FAISS + BM25 搜尋
    with ThreadPoolExecutor(max_workers=len(queries) * 2) as executor:
        v_futures = [
            executor.submit(vector_search_with_embedding, emb, faiss_index, nodes, top_k)
            for emb in query_embeddings
        ]
        b_futures = [
            executor.submit(bm25_search, q, bm25_index, nodes, top_k)
            for q in queries
        ]
        for f in v_futures:
            all_vector_results.extend(f.result())
        for f in b_futures:
            all_bm25_results.extend(f.result())

    # Hybrid Fusion（自動去重合併，內部已包含 Hard Filter 過濾排序與 Top-K 裁切）
    fused = hybrid_fusion(all_vector_results, all_bm25_results, route_result, top_k=top_k)

    logger.info(f"✅ Hybrid Retrieve 完成：{len(fused)} chunks")
    return fused


def _apply_hard_metadata_filter(
    chunks: list[RetrievedChunk],
    route_result: RouteResult,
) -> list[RetrievedChunk]:
    """
    對高信心的 metadata filter 進行 hard filtering。
    當 filter 條件很明確（如指定了特定教師或課程名稱），
    優先保留匹配的 chunks，將不匹配的排到後面。

    不會移除不匹配的 chunks（以防過濾太嚴導致無結果），
    而是將匹配的排在前面。
    """
    filters = route_result.metadata_filters

    # 【完整硬過濾名單】包含所有結構化 metadata 條件
    hard_keys = {"dept_short", "course_name_keyword", "teacher", "day_of_week", "grade", "class_group", "is_evening", "required_or_elective", "time_period", "semester", "academic_year"}
    active_hard = {k: v for k, v in filters.items() if k in hard_keys}

    if not active_hard:
        return chunks

    def matches_hard(chunk: RetrievedChunk) -> bool:
        meta = chunk.node.metadata
        content = chunk.node.get_content()  # 提取文本內容
        for key, val in active_hard.items():
            # 【防呆機制】確保 val 一律變成 list 格式，避免 TypeError
            v_list = val if isinstance(val, list) else [val]
            
            if key == "dept_short":
                meta_dept = meta.get("dept_short", "")
                if not any(v in meta_dept or meta_dept in v for v in v_list if v):
                    return False
            elif key == "course_name_keyword":
                # 只要 list 裡面有任何一個關鍵字對中即可
                if not any(v in meta.get("course_name", "") for v in v_list):
                    return False
            elif key == "teacher":
                # 【修復稱謂 Bug】自動移除「教授」、「老師」，確保能精準比對到名字
                clean_vals = [re.sub(r"(老師|教授)$", "", v) for v in v_list]
                if not any(cv in meta.get("teacher", "") for cv in clean_vals):
                    return False
            elif key == "day_of_week":
                if not any(v in content for v in v_list):
                    return False
            elif key == "grade":
                # 【正規化】大一→一、大二→二，確保 Router 輸出和 metadata 格式一致
                meta_grade = meta.get("grade", "")
                normalized = [re.sub(r"^大", "", v) for v in v_list]
                if not any(v == meta_grade for v in normalized):
                    return False
            elif key == "required_or_elective":
                meta_req = meta.get("required_or_elective", "")
                if not any(v in meta_req for v in v_list):
                    return False
            elif key == "time_period":
                # 從 schedule metadata 提取節次範圍，比對使用者要求的時段
                schedule = meta.get("schedule", "")
                sched_match = re.search(r"第(\d+)節[~～至到\-]第?(\d+)節", schedule)
                if sched_match:
                    sched_start = int(sched_match.group(1))
                    sched_end = int(sched_match.group(2))
                    # 解析 filter 的時段範圍（格式："2-4"）
                    parts = v_list[0].split("-")
                    filter_start, filter_end = int(parts[0]), int(parts[1])
                    # 檢查是否有重疊
                    if sched_start > filter_end or sched_end < filter_start:
                        return False  # 完全沒有重疊，排除
                else:
                    return False  # 無法解析 schedule，排除
            elif key == "semester":
                meta_sem = meta.get("semester", "")
                if not any(v == meta_sem for v in v_list):
                    return False
            elif key == "academic_year":
                meta_ay = meta.get("academic_year", "")
                if not any(v == meta_ay for v in v_list):
                    return False
            elif key == "class_group":
                meta_cg = meta.get("class_group", "")
                if not any(v == meta_cg for v in v_list):
                    return False
            elif key == "is_evening":
                meta_evening = meta.get("is_evening", False)
                if not meta_evening:
                    return False
        return True

    matched = [c for c in chunks if matches_hard(c)]
    unmatched = [c for c in chunks if not matches_hard(c)]

    # 【關鍵修改】嚴格丟棄所有不符合 Metadata 的 chunks (Hard Filter)
    # 如果使用者明確指定了學期、老師、星期等條件，不符合的資料絕不能讓 LLM 看到，否則會導致嚴重幻覺。
    result = matched
    logger.info(f"  🏷️ Hard Metadata Filter：{len(matched)} matched (保留) + {len(unmatched)} unmatched (嚴格捨棄)")
    
    return result


# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    config.setup_logging()
    from .data_loader import load_and_index

    print("载入索引...")
    nodes, faiss_idx, bm25_idx = load_and_index()

    # 測試路由
    from .query_router import route_query
    question = "深度學習的教科書是什麼？"
    route = route_query(question)
    print(f"\n❓ {question}")
    print(f"   路由：{route}")

    # 測試檢索
    results = hybrid_retrieve(
        queries=[question],
        route_result=route,
        faiss_index=faiss_idx,
        bm25_index=bm25_idx,
        nodes=nodes,
    )

    print(f"\n📋 檢索結果（Top-{len(results)}）：")
    for i, chunk in enumerate(results[:5]):
        print(f"\n--- #{i+1} ---")
        print(f"   {chunk}")
        print(f"   Text: {chunk.node.get_content()[:100]}...")
