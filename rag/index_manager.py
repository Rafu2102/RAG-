# -*- coding: utf-8 -*-
"""
index_manager.py — 索引管理模組
================================================
負責：
1. FAISS Vector Index 的 build / save / load
2. BM25 Keyword Index 的 build / save / load
3. Nodes 的 save / load
4. 資料變更偵測（自動重建索引用）
5. 一鍵載入或重建索引入口 (load_and_index)
"""

import os
import re
import pickle
import hashlib
import json
import logging
from typing import Optional

import numpy as np
import faiss  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore
from llama_index.core.schema import TextNode  # type: ignore

import config
from nlp_utils import build_ckip_dictionary, tokenize_texts_ckip
from .data_loader import gemini_embed, build_nodes_from_courses, build_nodes_from_dept_info, build_nodes_from_rules

logger = logging.getLogger(__name__)


# =============================================================================
# 🗂️ FAISS Vector Index
# =============================================================================

def build_faiss_index(nodes: list[TextNode]) -> faiss.IndexFlatIP:
    """
    為所有 Nodes 建立 FAISS 向量索引（Gemini Embedding 2 Preview，cosine similarity）。
    使用 task_type="RETRIEVAL_DOCUMENT" 最大化文件語意表示。
    """
    logger.info("開始建立 FAISS 向量索引（Gemini Embedding 2 Preview）...")

    texts = [node.get_content() for node in nodes]
    embeddings = gemini_embed(texts, task_type="RETRIEVAL_DOCUMENT")

    # Gemini Embedding 2 回傳的向量已經過正規化，但為安全起見仍做 L2 正規化
    # 確保 Inner Product 完全等同 Cosine Similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除以零
    embeddings = embeddings / norms

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    logger.info(f"FAISS 索引建立完成：{index.ntotal} 向量，維度 {dimension}")
    return index


def save_faiss_index(index: faiss.IndexFlatIP, path: str):
    """儲存 FAISS 索引到磁碟"""
    faiss.write_index(index, path)
    logger.info(f"FAISS 索引已儲存：{path}")


def load_faiss_index(path: str) -> Optional[faiss.IndexFlatIP]:
    """從磁碟載入 FAISS 索引"""
    if os.path.exists(path):
        index = faiss.read_index(path)
        logger.info(f"FAISS 索引已載入：{index.ntotal} 向量")
        return index
    return None


# =============================================================================
# 📚 BM25 Keyword Index
# =============================================================================

def build_bm25_index(nodes: list[TextNode]) -> BM25Okapi:
    """為所有 Nodes 建立 BM25 關鍵字索引（使用 CKIP Tagger 繁體中文分詞）。"""
    logger.info("  建構 BM25 Keyword 索引 (使用 CKIP Tagger)...")
    
    # 1. 收集領域專有名詞（老師與課程名稱）— 保護它們不被 CKIP 切碎
    domain_words = set()
    for node in nodes:
        meta = node.metadata
        teacher = meta.get("teacher", "")
        course = meta.get("course_name", "")
        
        # 加入老師名稱（處理多人共授）
        if teacher and teacher != "未知":
            for t in re.split(r"[、,，/]", teacher):
                t = t.strip()
                if len(t) >= 2:
                    domain_words.add(t)
                    
        # 加入課程名稱（全名 + 去括號簡稱）
        if course and course != "未知":
            domain_words.add(course.strip())
            short = re.sub(r"[（(][^）)]*[）)]", "", course).strip()
            if short and len(short) >= 2 and short != course:
                domain_words.add(short)

    # 2. 建立 CKIP 強制斷詞字典
    custom_dict = build_ckip_dictionary(list(domain_words))

    # 3. 提取所有內文
    texts = [node.get_content() for node in nodes]
    
    # 4. 【批次斷詞】一次送進 CKIP 處理所有文本
    logger.info(f"  🧠 正在進行深度學習批次斷詞 (共 {len(texts)} 筆區段)...")
    tokenized_corpus = tokenize_texts_ckip(texts, custom_dict=custom_dict)
    
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"  ✅ BM25 索引建立完成：{len(tokenized_corpus)} 文件")
    return bm25


def save_bm25_index(bm25: BM25Okapi, path: str):
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 索引已儲存：{path}")


def load_bm25_index(path: str) -> Optional[BM25Okapi]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            bm25 = pickle.load(f)
        logger.info("BM25 索引已載入")
        return bm25
    return None


# =============================================================================
# 💾 Nodes 存取
# =============================================================================

def save_nodes(nodes: list[TextNode], path: str):
    with open(path, "wb") as f:
        pickle.dump(nodes, f)
    logger.info(f"Nodes 已儲存：{len(nodes)} 個")


def load_nodes(path: str) -> Optional[list[TextNode]]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            nodes = pickle.load(f)
        logger.info(f"Nodes 已載入：{len(nodes)} 個")
        return nodes
    return None


# =============================================================================
# 🔍 資料變更偵測（自動重建索引用）
# =============================================================================

MANIFEST_PATH = os.path.join(config.INDEX_DIR, "data_manifest.json")


def _compute_file_hash(filepath: str) -> str:
    """計算檔案的 SHA256 hash"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_data_files(data_dir: str) -> dict:
    """掃描 data/ 目錄下所有 .txt 與 .json 檔案，回傳 {相對路徑: hash}"""
    file_hashes = {}
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".txt") or fname.endswith(".json"):
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, data_dir)
                file_hashes[rel_path] = _compute_file_hash(fpath)
    return file_hashes


def check_data_changes(data_dir: str = None) -> dict:
    """
    偵測 data/ 目錄中的課程檔案是否有變更。
    
    Returns:
        {
            "has_changes": bool,
            "new_files": list[str],
            "modified_files": list[str],
            "deleted_files": list[str],
        }
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    current_hashes = _scan_data_files(data_dir)
    
    # 讀取上次的 manifest
    old_hashes = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                old_hashes = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 讀取 data_manifest.json 失敗，視為全新：{e}")
    
    new_files = [f for f in current_hashes if f not in old_hashes]
    deleted_files = [f for f in old_hashes if f not in current_hashes]
    modified_files = [
        f for f in current_hashes 
        if f in old_hashes and current_hashes[f] != old_hashes[f]
    ]
    
    has_changes = bool(new_files or modified_files or deleted_files)
    
    if has_changes:
        logger.info(f"📂 資料變更偵測結果：新增 {len(new_files)} · 修改 {len(modified_files)} · 刪除 {len(deleted_files)}")
        for f in new_files:
            logger.info(f"  📄 [新增] {f}")
        for f in modified_files:
            logger.info(f"  ✏️ [修改] {f}")
        for f in deleted_files:
            logger.info(f"  🗑️ [刪除] {f}")
    else:
        logger.info("📂 資料無變更，索引可直接載入")
    
    return {
        "has_changes": has_changes,
        "new_files": new_files,
        "modified_files": modified_files,
        "deleted_files": deleted_files,
    }


def save_data_manifest(data_dir: str = None):
    """儲存當前 data/ 目錄的 hash manifest（在索引重建後呼叫）"""
    if data_dir is None:
        data_dir = config.DATA_DIR
    current_hashes = _scan_data_files(data_dir)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(current_hashes, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 已儲存 data_manifest.json（{len(current_hashes)} 個檔案）")


# =============================================================================
# 🚀 主要入口：載入或建立索引
# =============================================================================

def load_and_index(force_rebuild: bool = False):
    """
    載入既有索引或重新建立。

    Returns:
        tuple: (nodes, faiss_index, bm25_index)
    """
    config.ensure_dirs()

    # ── 嘗試載入既有索引 ──
    if not force_rebuild:
        nodes = load_nodes(config.NODES_STORE_PATH)
        faiss_index = load_faiss_index(config.FAISS_INDEX_PATH)
        bm25_index = load_bm25_index(config.BM25_INDEX_PATH)

        if nodes and faiss_index and bm25_index:
            logger.info("✅ 所有索引已從磁碟載入，無需重建")
            return nodes, faiss_index, bm25_index

    # ── 重新建立索引 ──
    logger.info("🔨 開始重新建立索引...")
    course_nodes = build_nodes_from_courses(config.COURSES_DIR)
    dept_nodes = build_nodes_from_dept_info(config.PROFESSORS_DIR, config.DEPT_INFO_DIR)
    rules_nodes = build_nodes_from_rules(config.RULES_DIR)
    nodes = course_nodes + dept_nodes + rules_nodes
    logger.info(f"📊 合計 Nodes：{len(course_nodes)} 課程 + {len(dept_nodes)} 系所資訊 + {len(rules_nodes)} 規則 = {len(nodes)} 總計")
    faiss_index = build_faiss_index(nodes)
    bm25_index = build_bm25_index(nodes)

    save_nodes(nodes, config.NODES_STORE_PATH)
    save_faiss_index(faiss_index, config.FAISS_INDEX_PATH)
    save_bm25_index(bm25_index, config.BM25_INDEX_PATH)
    save_data_manifest()  # 儲存檔案 hash，下次啟動可偵測變更

    logger.info("✅ 索引建立完成！")
    return nodes, faiss_index, bm25_index
