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
from .data_loader import gemini_embed_query, gemini_embed
from nlp_utils import tokenize_chinese
from .query_router import RouteResult
from .metadata_filters import apply_hard_metadata_filter
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
# 🔍 模糊姓名比對（防止 LLM 寫錯相似字）
# =============================================================================

def _fuzzy_name_match(query_name: str, target_name: str) -> bool:
    """
    模糊比對教授姓名，容忍 1 個字的差異。
    例如：柯志恆 vs 柯志亨 → 2/3 字相同 → 視為匹配
    """
    # 先做精確匹配（快速路徑）
    if query_name in target_name or target_name in query_name:
        return True
    
    # 模糊比對：至少 2/3 字元相同
    if len(query_name) >= 2 and len(target_name) >= 2:
        overlap = sum(1 for c in query_name if c in target_name)
        threshold = max(2, int(len(query_name) * 0.67))
        return overlap >= threshold
    
    return False


# =============================================================================
# 🎯 Intent-Driven Injection（意圖驅動注入）
# =============================================================================

def intent_inject_chunks(
    intent: str,
    filters: dict,
    nodes: list[TextNode],
) -> list[RetrievedChunk]:
    """
    根據 AI Router 的意圖決策，直接從 nodes 池中撈出最匹配的 chunks，
    保證它們進入 Reranker 的候選池。

    這是解決「Top-K 稀釋效應」的核心機制：
    當稀有資料（教授、法規）在 FAISS/BM25 的 Top-120 中被數千筆課程資料淹沒時，
    這個函式直接繞過檢索階段，把 AI 認為該查的資料「保送」進候選池。
    Reranker 仍做最終裁決，不會因此錯殺。

    Args:
        intent: AI Router 判斷的意圖 (e.g. "professor_info")
        filters: AI Router 提取的 metadata filters
        nodes: 全部 TextNode 列表（記憶體中）

    Returns:
        注入的 RetrievedChunk 列表（高分保送）
    """
    injected = []

    if intent == "professor_info":
        # 從 filters 提取教授名稱（去除「教授」「老師」等後綴）
        raw_teacher = filters.get("teacher", "")
        teacher = re.sub(r"(老師|教授)$", "", raw_teacher).strip()
        
        if teacher:
            # 有明確教授名稱：精準注入該教授
            for node in nodes:
                prof_name = node.metadata.get("professor_name", "")
                info_type = node.metadata.get("info_type", "")
                
                # 模糊匹配教授名稱（防止 LLM 寫錯相似字，如 柯志恆 vs 柯志亨）
                if prof_name and _fuzzy_name_match(teacher, prof_name) and info_type == "professor_info":
                    injected.append(RetrievedChunk(
                        node=node,
                        vector_score=1.0,
                        bm25_score=1.0,
                        metadata_score=10.0,
                        source="intent_inject_exact",
                    ))
                # 同系所的系所簡介也一併帶入（提供背景脈絡）
                elif info_type in ("dept_intro", "career_info") and node.metadata.get("department", ""):
                    injected.append(RetrievedChunk(
                        node=node,
                        vector_score=0.5,
                        bm25_score=0.5,
                        metadata_score=5.0,
                        source="intent_inject",
                    ))
        else:
            # 【關鍵修復】使用者沒講老師名字 (e.g. "教 Linux 的老師")，但意圖是教授資訊
            # 由於全系只有不到 20 位教授，我們直接將「所有」教授簡歷注入候選池！
            # 讓強大的 Reranker (Cross-Encoder) 從這些簡歷中找出誰教 Linux。
            for node in nodes:
                info_type = node.metadata.get("info_type", "")
                if info_type == "professor_info":
                    injected.append(RetrievedChunk(
                        node=node,
                        vector_score=0.9,
                        bm25_score=0.9,
                        metadata_score=5.0,
                        source="intent_inject_all_profs",
                    ))
                # 同系所的系所簡介也一併帶入（提供背景脈絡）
                elif info_type in ("dept_intro", "career_info", "facility_info") and node.metadata.get("department", ""):
                    injected.append(RetrievedChunk(
                        node=node,
                        vector_score=0.5,
                        bm25_score=0.5,
                        metadata_score=5.0,
                        source="intent_inject",
                    ))

    elif intent == "policy_rules":
        # 如果是系所規定/未來出路/職涯發展，直接將所有相關的系所介紹與規章保送進候選池
        # 這樣才能確保「學生就業具體方向.txt」或「畢業門檻」不會被課程大綱擠掉
        target_types = (
            "graduation_rules", "policy", "policy_rules", "dept_intro", 
            "career_info", "student_union", "dept_news", "dept_general"
        )
        for node in nodes:
            if node.metadata.get("info_type", "") in target_types:
                injected.append(RetrievedChunk(
                    node=node,
                    vector_score=1.0,
                    bm25_score=1.0,
                    metadata_score=10.0,
                    source="intent_inject_policy",
                ))

    if injected:
        logger.info(f"  🎯 Intent Injection ({intent})：找到 {len(injected)} 個 chunks 直接注入候選池")

    return injected


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

    # 產生 query embedding（使用 Gemini task_type="RETRIEVAL_QUERY"）
    query_embedding = gemini_embed_query(query)

    # Gemini Embedding 2 已自動正規化，但為保險仍做 L2 正規化
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
    使用 Gemini batchEmbedContents 批次取得多個 query 的 embedding。
    比逐一呼叫更高效，單次 API 請求即可處理所有 queries。
    """
    # Gemini batch API：一次呼叫處理所有 queries
    all_embeddings = gemini_embed(queries, task_type="RETRIEVAL_QUERY")
    return [all_embeddings[i] for i in range(len(queries))]


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

    # ── 年級匹配（正規化阿拉伯↔中文數字）──
    if "grade" in filters:
        max_possible += config.METADATA_MATCH_SCORES["grade"]
        from .metadata_filters import _normalize_grade
        filter_grade = _normalize_grade(str(filters["grade"]))
        meta_grade = _normalize_grade(str(node_meta.get("grade", "")))
        if filter_grade and filter_grade == meta_grade:
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

    # ── 通識主題領域匹配 ──
    if "ge_domain" in filters:
        max_possible += 0.3
        meta_domain = node_meta.get("ge_domain", "")
        filter_domain = str(filters["ge_domain"])
        if meta_domain and (filter_domain in meta_domain or meta_domain in filter_domain):
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
        all_chunks = apply_hard_metadata_filter(all_chunks, route_result)

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

    # ========================================================================
    # 🌌 職涯大範圍探勘模式 (Full Curriculum Scan Mode)
    # 針對 Frontend/Backend/AI 等規劃，直接抽取科系之所有課程與就業資訊，無視檢索！
    # ========================================================================
    if getattr(route_result, "is_career_planning", False):
        target_depts = route_result.metadata_filters.get("dept_short", [])
        if isinstance(target_depts, str):
            target_depts = [target_depts]
            
        logger.info(f"  🌌 啟動大範圍職涯全境掃描模式 (Full Curriculum Scan Mode)")
        logger.info(f"  📌 目標系所：{target_depts if target_depts else '全校掃描'}")
        
        scanned_results = []
        for node in nodes:
            meta = node.metadata
            info_type = meta.get("info_type", "")
            node_dept = meta.get("dept_short", meta.get("department", ""))
            
            # 檢查系所是否吻合 (如果有指定科系的話)
            dept_match = True
            if target_depts:
                # 緊急：舊版索引可能只有 department='資訊工程學系' 沒標 'dept_short'='資工系'
                def _fuzzy_dept(t_dept, n_dept):
                    if t_dept in n_dept: return True
                    # 當 query 為 "資工" 而 node 為 "資訊工程" 時放行
                    if t_dept.replace("系", "") in n_dept.replace("資訊工程", "資工"): return True
                    return False
                dept_match = any(_fuzzy_dept(d, node_dept) for d in target_depts if d)
                
            if not dept_match:
                continue
                
            # 提取 1: 該系所有的課程基本資訊與教學目標
            # 注意：早期的課程 metadata 可能漏標 info_type="course_info"，因此用 course_name 欄位作為防呆判斷
            if info_type == "course_info" or "course_name" in meta:
                if meta.get("section", "") in ("basic_info", "objectives"):
                    scanned_results.append(RetrievedChunk(
                        node=node, vector_score=1.0, bm25_score=1.0, metadata_score=10.0, source="full_curriculum_scan"
                    ))
            
            # 提取 2: 該系專屬的就業方向、系所簡介
            elif info_type in ("career_info", "dept_intro"):
                scanned_results.append(RetrievedChunk(
                    node=node, vector_score=1.0, bm25_score=1.0, metadata_score=15.0, source="full_curriculum_scan"
                ))
                
        logger.info(f"  ✅ 全境掃描完成，共抽取 {len(scanned_results)} 個核心節點供 LLM 分析")
        return scanned_results

    # ========================================================================
    # 正常檢索模式
    # ========================================================================
    all_vector_results = []
    all_bm25_results = []

    # 【關鍵加速】並行取得所有 query 的 embedding
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

    # ========================================================================
    # 問題：當使用者問「E321有什麼課」或「柯志亨的課」時，vector/BM25 的 top-K 很容易遺漏
    #       語義上不相似但 metadata (如教室、老師) 完全匹配的課程。
    #
    # 解法：如果在 Query Router 中明確萃取出 高權重實體(教室, 老師, 課名) 或 核心條件(系級)，
    #       直接跨過向量庫限制，把符合條件的 basic_info chunks 暴力補進候選名單。
    # ========================================================================
    filters = route_result.metadata_filters
    
    # 判斷是否符合注入資格
    has_high_weight_entity = any(k in filters for k in ("teacher", "classroom", "course_name_keyword"))
    has_specific_dept_grade = ("dept_short" in filters and "grade" in filters)
    
    if has_high_weight_entity or has_specific_dept_grade:
        logger.info(f"  💉 觸發 Metadata-First 注入機制，補齊精確匹配的課程節點...")
        
        # 收集已經在結果中的 node_id
        existing_ids = set()
        for c in all_vector_results:
            existing_ids.add(c.node.node_id)
        for c in all_bm25_results:
            existing_ids.add(c.node.node_id)
        
        # 掃描所有 nodes，找出 metadata 完美匹配但被遺漏的
        injected_count = 0
        
        # 從 rag.metadata_filters 導入判斷邏輯
        from rag.metadata_filters import _match_teacher, _match_classroom, _match_course_name_keyword, _match_grade
        
        for node in nodes:
            if node.node_id in existing_ids:
                continue
            meta = node.metadata
            
            # 只注入 basic_info 區段（避免灌入大量課程大綱與進度表）
            if meta.get("section", "") != "basic_info":
                continue
            
            # 確認該節點是否符合使用者指定的嚴格條件
            all_match = True
            for k, v in filters.items():
                if not v: continue
                v_list = v if isinstance(v, list) else [v]
                
                # 只檢查關鍵的過濾條件，只要這幾個通過，就算符合注入標準
                if k == "dept_short":
                    meta_val = meta.get("dept_short", "")
                    if not any(fv in meta_val or meta_val in fv for fv in v_list if fv):
                        all_match = False
                        break
                elif k == "grade":
                    if not _match_grade(meta, v_list):
                        all_match = False
                        break
                elif k == "teacher":
                    if not _match_teacher(meta, v_list):
                        all_match = False
                        break
                elif k == "classroom":
                    if not _match_classroom(meta, v_list):
                        all_match = False
                        break
                elif k == "course_name_keyword":
                    if not _match_course_name_keyword(meta, v_list):
                        all_match = False
                        break
                elif k == "academic_year":
                    if not any(fv == meta.get("academic_year", "") for fv in v_list):
                        all_match = False
                        break
                elif k == "semester":
                    if not any(fv == meta.get("semester", "") for fv in v_list):
                        all_match = False
                        break
                        
                elif k == "day_of_week":
                    # 加入對上課時間的檢查
                    from rag.metadata_filters import _match_day_of_week
                    if not _match_day_of_week(meta, v_list):
                        all_match = False
                        break
            
            if all_match:
                # 以極低的 vector/bm25 分數注入，讓 hard filter 保留它
                injected = RetrievedChunk(node=node, source="metadata_inject")
                injected.vector_score = 0.001
                injected.bm25_score = 0.001
                all_vector_results.append(injected)
                existing_ids.add(node.node_id)
                injected_count += 1
        
        if injected_count > 0:
            logger.info(f"  🏷️ Metadata-First 注入：補充 {injected_count} 個被 top-k 遺漏的 basic_info chunks")

    # Hybrid Fusion（自動去重合併，內部已包含 Hard Filter 過濾排序與 Top-K 裁切）
    fused = hybrid_fusion(all_vector_results, all_bm25_results, route_result, top_k=top_k)

    logger.info(f"✅ Hybrid Retrieve 完成：{len(fused)} chunks")
    return fused



# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    config.setup_logging()
    from .index_manager import load_and_index

    print("载入索引...")
    nodes, faiss_idx, bm25_idx = load_and_index()

    # 測試路由
    from .query_router import route_and_rewrite
    question = "深度學習的教科書是什麼？"
    route = route_and_rewrite(question)
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
