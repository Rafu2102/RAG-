# -*- coding: utf-8 -*-
"""
data_loader.py — 資料載入與索引建立模組
=========================================
負責：
1. 解析課程 TXT 檔案為結構化資料
2. 建立 LlamaIndex Document / Node
3. 建立 FAISS Vector Index（使用 Ollama multilingual-e5-large）
4. 建立 BM25 Keyword Index（jieba 中文分詞）
5. 支援索引持久化（存檔/載入）
"""

import os
import re
import pickle
import logging
from typing import Optional

import jieba
import numpy as np
import faiss
import requests
from rank_bm25 import BM25Okapi

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

import config

logger = logging.getLogger(__name__)


# =============================================================================
# 📝 課程 TXT 檔案解析器
# =============================================================================

def parse_course_file(filepath: str) -> dict:
    """
    解析單個課程 TXT 檔案，提取結構化欄位。

    Args:
        filepath: 課程 TXT 檔案的完整路徑

    Returns:
        包含課程資訊的 dict，包括 metadata 和各區段文字
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # ── 基本 metadata 提取 ──
    def extract(pattern: str, text: str, default: str = "未知") -> str:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else default

    course_name = extract(r"課程名稱：(.+?)(?:\r?\n)", content)
    department = extract(r"部別：(.+?)(?:\r?\n)", content)
    grade_class = extract(r"開課班級：(.+?)(?:\r?\n)", content)
    credits = extract(r"學分數：(.+?)(?:\r?\n)", content)
    teacher = extract(r"授課教師：(.+?)(?:\r?\n)", content)
    required = extract(r"必選修：(.+?)(?:\r?\n)", content)
    classroom = extract(r"教室：(.+?)(?:\r?\n)", content)
    schedule = extract(r"上課時間：(.+?)(?:\r?\n)", content)
    max_students = extract(r"修課上限人數：(.+?)(?:\r?\n)", content)

    # ── 學年度與學期提取 ──
    academic_year = extract(r"學年度：(\d+)學年度", content, "0")
    semester_raw = extract(r"學期：第(\d+)學期", content, "0")

    # ── 推斷年級（grade）──
    grade = "未知"
    grade_match = re.search(r"(碩[一二三]|[一二三四五])", grade_class)
    if grade_match:
        grade = grade_match.group(1)
    
    # ── 推斷班級分組（class_group）── 如 電機一甲 → "甲"
    class_group = ""
    group_match = re.search(r"[一二三四五][甲乙丙丁]", grade_class)
    if group_match:
        class_group = group_match.group(0)[-1]  # 取最後一個字：甲 / 乙
    
    # ── 推斷進修部 ──
    is_evening = "進修" in grade_class or "進修" in department

    # ── 推斷系所簡稱（從資料夾名稱或檔案路徑智慧偵測）──
    # 完整科系對照表：資料夾名稱關鍵字 → dept_short
    _DEPT_FOLDER_PATTERNS = {
        # 理工學院
        "資工": "資工系", "資訊工程": "資工系",
        "電機": "電機系", "電機工程": "電機系",
        "土木": "土木系", "工程管理": "土木系",
        "食品": "食品系", "食品科學": "食品系",
        # 管理學院
        "企管": "企管系", "企業管理": "企管系",
        "觀光": "觀光系", "觀光管理": "觀光系",
        "運休": "運休系", "運動與休閒": "運休系",
        "工管": "工管系", "工業工程": "工管系",
        # 人文社會學院
        "國際": "國際系", "大陸事務": "國際系",
        "建築": "建築系",
        "海邊": "海邊系", "海洋與邊境": "海邊系", "邊境管理": "海邊系",
        "應英": "應英系", "應用英語": "應英系",
        "華語": "華語系", "華語文": "華語系",
        "都景": "都景系", "都市計畫": "都景系", "景觀": "都景系",
        # 健康護理學院
        "護理": "護理系",
        "長照": "長照系", "長期照護": "長照系",
        "社工": "社工系", "社會工作": "社工系",
        # 通識
        "通識": "通識中心",
    }
    
    def _detect_dept_from_path(fpath: str) -> str:
        """從檔案路徑（含資料夾名稱）偵測科系簡稱"""
        # 優先用較長關鍵字匹配（避免「工」匹配到不相關的系）
        sorted_patterns = sorted(_DEPT_FOLDER_PATTERNS.keys(), key=len, reverse=True)
        for kw in sorted_patterns:
            if kw in fpath:
                return _DEPT_FOLDER_PATTERNS[kw]
        return "未知"
    
    dept_short = _detect_dept_from_path(filepath)
    if "研究所" in department or "碩士" in department:
        dept_short = dept_short.replace("系", "碩") if "系" in dept_short else dept_short + "碩"

    # ── 各區段文字提取 ──
    def extract_section(start_pattern: str, end_patterns: list, text: str) -> str:
        start = re.search(start_pattern, text)
        if not start:
            return ""
        text_after = text[start.end():]
        for end_pat in end_patterns:
            end = re.search(end_pat, text_after)
            if end:
                return text_after[:end.start()].strip()
        return text_after.strip()

    objectives = extract_section(
        r"課程教學目標：\s*\r?\n",
        [r"\r?\n\s*\r?\n\s*課程教學綱要", r"課程教學綱要"],
        content
    )

    syllabus = extract_section(
        r"課程教學綱要：\s*\r?\n",
        [r"\r?\n\s*\r?\n\s*教科書資料", r"教科書資料"],
        content
    )

    textbooks = extract_section(
        r"教科書資料：\s*\r?\n",
        [r"\r?\n\s*\r?\n\s*參考書資料", r"參考書資料"],
        content
    )

    references = extract_section(
        r"參考書資料：\s*\r?\n?",
        [r"\r?\n\s*\r?\n\s*(?:※|教學進度表)", r"※請遵守", r"教學進度表"],
        content
    )

    schedule_table = extract_section(
        r"教學進度表.+?：\s*\r?\n",
        [r"\r?\n\s*\r?\n\s*成績評定", r"成績評定"],
        content
    )

    grading = extract_section(
        r"成績評定方式：\s*\r?\n",
        [r"\r?\n\s*\r?\n\s*課堂要求", r"課堂要求"],
        content
    )

    requirements_text = extract_section(
        r"課堂要求：\s*\r?\n",
        [r"\r?\n\s*\r?\n\s*$"],
        content
    )

    metadata = {
        "course_name": course_name,
        "department": department,
        "dept_short": dept_short,
        "grade_class": grade_class,
        "grade": grade,
        "class_group": class_group,
        "is_evening": is_evening,
        "credits": credits,
        "teacher": teacher,
        "required_or_elective": required,
        "classroom": classroom,
        "schedule": schedule,
        "max_students": max_students,
        "academic_year": academic_year,
        "semester": semester_raw,
        "source_file": os.path.basename(filepath),
    }

    sections = {
        "basic_info": (
            f"# 課程基本資訊\n"
            f"## 【{course_name}】\n"
            f"- **部別**：{department}\n"
            f"- **開課班級**：{grade_class}\n"
            f"- **學分數**：{credits}\n"
            f"- **授課教師**：{teacher}\n"
            f"- **必選修**：{required}\n"
            f"- **教室**：{classroom}\n"
            f"- **上課時間**：{schedule}\n"
            f"- **修課上限人數**：{max_students}\n"
        ),
        "objectives": (
            f"# 課程教學目標\n"
            f"## 【{course_name}】\n"
            f"{objectives}\n"
        ) if objectives else "",
        "syllabus": (
            f"# 課程教學綱要\n"
            f"## 【{course_name}】\n"
            f"{syllabus}\n"
        ) if syllabus else "",
        "textbooks": (
            f"# 教材資訊\n"
            f"## 【{course_name}】\n"
            f"### 教科書\n{textbooks if textbooks else '無紀錄'}\n\n"
            f"### 參考書\n{references if references else '無紀錄'}\n"
        ) if (textbooks or references) else "",
        "schedule_table": (
            f"# 教學進度表\n"
            f"## 【{course_name}】\n"
            f"{schedule_table}\n"
        ) if schedule_table else "",
        "grading": (
            f"# 評量方式與課堂要求\n"
            f"## 【{course_name}】\n"
            f"### 成績評定\n{grading if grading else '未提供'}\n\n"
            f"### 課堂要求\n{requirements_text if requirements_text else '未提供'}\n"
        ) if (grading or requirements_text) else "",
    }

    return {"metadata": metadata, "sections": sections}


# =============================================================================
# 📦 建立 LlamaIndex Nodes
# =============================================================================

def build_nodes_from_courses(data_dir: str) -> list[TextNode]:
    """
    從資料目錄中所有課程 TXT 檔，建立 LlamaIndex TextNode 列表。

    每個課程按區段（基本資訊、教學目標、教學綱要等）分別建立 Node，
    每個 Node 附帶完整 metadata，方便後續 metadata filtering。

    Args:
        data_dir: 課程 TXT 檔案所在目錄

    Returns:
        TextNode 列表
    """
    all_nodes = []
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # 過專掃描 data_dir 及所有子目錄中的 .txt
    txt_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    logger.info(f"找到 {len(txt_files)} 個課程檔案（跨學期）")

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        try:
            course_data = parse_course_file(filepath)
            metadata = course_data["metadata"]
            sections = course_data["sections"]

            for section_name, section_text in sections.items():
                if not section_text.strip():
                    continue

                full_metadata = {**metadata, "section": section_name}
                
                # 【優化 1】上下文防遺失機制 — 每個 Node 前綴注入核心資訊
                course_name = metadata.get("course_name", "未知")
                teacher = metadata.get("teacher", "未知")
                grade = metadata.get("grade", "未知")
                req = metadata.get("required_or_elective", "未知")
                prefix = f"[課程：{course_name} | 教師：{teacher} | 年級：{grade}年級 | 屬性：{req} | 區段：{section_name}]\n"
                
                # 【優化 2】智慧區段感知 Chunking
                # 短區段（≤ CHUNK_SIZE）保持完整不切；只對超長區段（schedule_table）啟動 SentenceSplitter
                if len(section_text) <= config.CHUNK_SIZE:
                    # 區段完整性保護：直接作為單一 Node
                    node = TextNode(
                        text=prefix + section_text,
                        metadata=full_metadata,
                    )
                    node.excluded_embed_metadata_keys = list(full_metadata.keys())
                    all_nodes.append(node)
                else:
                    # 超長區段（如進度表）才使用 SentenceSplitter 切分
                    doc = Document(
                        text=section_text,
                        metadata={}, 
                    )
                    nodes = splitter.get_nodes_from_documents([doc])
                    for n in nodes:
                        n.set_content(prefix + n.get_content())
                        n.metadata = full_metadata
                        n.excluded_embed_metadata_keys = list(full_metadata.keys())
                    all_nodes.extend(nodes)

            logger.info(f"  ✅ {filename} → {sum(1 for s in sections.values() if s.strip())} 個區段")

        except Exception as e:
            logger.error(f"  ❌ 解析失敗：{filename} — {e}")

    logger.info(f"共建立 {len(all_nodes)} 個 Nodes")
    return all_nodes


# =============================================================================
# 🔤 Ollama Embedding 工具函式
# =============================================================================

def ollama_embed(texts: list[str], model: str = None, prefix: str = "passage: ") -> np.ndarray:
    """
    使用 Ollama API 產生 embedding 向量。

    multilingual-e5-large 需要加前綴：
    - 建索引時（passage）使用 "passage: "
    - 查詢時（query）使用 "query: "

    Args:
        texts: 文字列表
        model: Ollama 模型名稱
        prefix: 前綴字串（"passage: " 或 "query: "）

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    if model is None:
        model = config.EMBEDDING_MODEL_NAME

    url = f"{config.OLLAMA_BASE_URL}/api/embed"
    embeddings = []

    # 批次處理，避免一次傳送太多資料
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prefixed = [f"{prefix}{t}" for t in batch]

        response = requests.post(url, json={
            "model": model,
            "input": prefixed,
        }, timeout=config.OLLAMA_REQUEST_TIMEOUT)
        response.raise_for_status()

        result = response.json()
        batch_embeddings = result["embeddings"]
        embeddings.extend(batch_embeddings)

        logger.info(f"  Embedding 進度：{min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(embeddings, dtype=np.float32)


def ollama_embed_query(text: str, model: str = None) -> np.ndarray:
    """
    使用 Ollama 產生「查詢」用的 embedding（使用 "query: " 前綴）。

    Args:
        text: 查詢文字

    Returns:
        numpy array of shape (embedding_dim,)
    """
    result = ollama_embed([text], model=model, prefix="query: ")
    return result[0]


# =============================================================================
# 🗂️ FAISS Vector Index
# =============================================================================

def build_faiss_index(nodes: list[TextNode]) -> faiss.IndexFlatIP:
    """
    為所有 Nodes 建立 FAISS 向量索引（Inner Product，配合正規化等同 cosine similarity）。

    Args:
        nodes: TextNode 列表

    Returns:
        FAISS IndexFlatIP 索引
    """
    logger.info("開始建立 FAISS 向量索引（Ollama embedding）...")

    texts = [node.get_content() for node in nodes]
    embeddings = ollama_embed(texts, prefix="passage: ")

    # L2 正規化，使 Inner Product 等同 Cosine Similarity
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

def tokenize_chinese(text: str) -> list[str]:
    """
    使用 jieba 進行中文分詞，過濾停用詞和短詞。
    """
    stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一個", "上", "也", "很", "到", "說", "要", "去",
        "你", "會", "著", "沒有", "看", "好", "自己", "這", "他", "她",
        "們", "那", "個", "與", "及", "或", "但", "而", "將", "已",
        "為", "從", "以", "可", "能", "之", "於", "等", "被", "其",
        "把", "所", "對", "讓", "這個", "那個", "什麼", "怎麼", "哪",
        "\r", "\n", "\t", " ", "：", "，", "。", "、", "（", "）",
    }
    words = jieba.lcut(text)
    return [w.strip() for w in words if w.strip() and len(w.strip()) > 1 and w not in stopwords]


def build_bm25_index(nodes: list[TextNode]) -> BM25Okapi:
    """
    為所有 Nodes 建立 BM25 關鍵字索引（jieba 中文分詞）。
    """
    logger.info("開始建立 BM25 索引...")
    tokenized_corpus = [tokenize_chinese(node.get_content()) for node in nodes]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 索引建立完成：{len(tokenized_corpus)} 文件")
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

import hashlib
import json

MANIFEST_PATH = os.path.join(config.INDEX_STORE_DIR, "data_manifest.json")


def _compute_file_hash(filepath: str) -> str:
    """計算檔案的 SHA256 hash"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_data_files(data_dir: str) -> dict:
    """掃描 data/ 目錄下所有 .txt 課程檔，回傳 {相對路徑: hash}"""
    file_hashes = {}
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".txt"):
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
    nodes = build_nodes_from_courses(config.DATA_DIR)
    faiss_index = build_faiss_index(nodes)
    bm25_index = build_bm25_index(nodes)

    save_nodes(nodes, config.NODES_STORE_PATH)
    save_faiss_index(faiss_index, config.FAISS_INDEX_PATH)
    save_bm25_index(bm25_index, config.BM25_INDEX_PATH)
    save_data_manifest()  # 儲存檔案 hash，下次啟動可偵測變更

    logger.info("✅ 索引建立完成！")
    return nodes, faiss_index, bm25_index


# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    config.setup_logging()
    nodes, faiss_idx, bm25_idx = load_and_index(force_rebuild=True)
    print(f"\n📊 總結：")
    print(f"   Nodes 數量：{len(nodes)}")
    print(f"   FAISS 向量數：{faiss_idx.ntotal}")
    print(f"\n📝 前 3 個 Node 範例：")
    for i, node in enumerate(nodes[:3]):
        print(f"\n--- Node {i+1} ---")
        print(f"   Section: {node.metadata.get('section', 'N/A')}")
        print(f"   Course:  {node.metadata.get('course_name', 'N/A')}")
        print(f"   Text:    {node.get_content()[:100]}...")
