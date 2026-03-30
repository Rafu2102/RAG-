# -*- coding: utf-8 -*-
"""
data_loader.py — 資料載入與索引建立模組
=========================================
負責：
1. 解析課程 TXT 檔案為結構化資料
2. 建立 LlamaIndex Document / Node
3. 建立 FAISS Vector Index（使用 Gemini Embedding 2 Preview 雲端 API）
4. 建立 BM25 Keyword Index（CKIP Tagger 繁體中文分詞）
5. 支援索引持久化（存檔/載入）
"""

import os
import re
import pickle
import hashlib
import json
import logging
import time as _time
import threading
from typing import Optional

from nlp_utils import build_ckip_dictionary, tokenize_texts_ckip, tokenize_chinese
import numpy as np
import faiss
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rank_bm25 import BM25Okapi

# 【全域 HTTP 連線池】Gemini Cloud API 專用，TCP 連線復用 + 自動重試
_gemini_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
_gemini_session.mount('https://', HTTPAdapter(max_retries=_retries, pool_connections=20, pool_maxsize=20))

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
    note_raw = extract(r"(?:備註|通識主題領域)：(.+?)(?:\r?\n)", content, "")

    # ── 學年度與學期提取 ──
    academic_year = extract(r"學年度：(\d+)學年度", content, "0")
    semester_raw = extract(r"學期：第(\d+)學期", content, "0")

    # ── 推斷年級（grade）──
    grade = "未知"
    if "通識" in grade_class:
        grade = "通識"  # 通識課不適用年級概念
    else:
        grade_match = re.search(r"(碩[一二三]|[一二三四五])", grade_class)
        if grade_match:
            grade = grade_match.group(1)
    
    # ── 推斷班級分組（class_group）── 如 電機一甲 → "甲"
    class_group = ""
    group_match = re.search(r"[一二三四五][甲乙丙丁]", grade_class)
    if group_match:
        class_group = group_match.group(0)[-1]  # 取最後一個字：甲 / 乙

    # ── 通識主題領域（ge_domain）──
    # 只對通識課提取，從「備註」欄解析，格式如「社會科學—社會類」
    ge_domain = ""
    if "通識" in grade_class or "通識" in filepath:
        if "人文" in note_raw:
            ge_domain = "人文"
        elif "社會" in note_raw:
            ge_domain = "社會"
        elif "自然" in note_raw:
            ge_domain = "自然"
    
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
        text_after = str(text[start.end():])
        for end_pat in end_patterns:
            end = re.search(end_pat, text_after)
            if end:
                return str(text_after[:end.start()]).strip()
        return str(text_after).strip()

    _SECTION_MAPPINGS = {
        "objectives": (r"課程教學目標：\s*\r?\n", [r"\r?\n\s*\r?\n\s*課程教學綱要", r"課程教學綱要"]),
        "syllabus": (r"課程教學綱要：\s*\r?\n", [r"\r?\n\s*\r?\n\s*教科書資料", r"教科書資料"]),
        "textbooks": (r"教科書資料：\s*\r?\n", [r"\r?\n\s*\r?\n\s*參考書資料", r"參考書資料"]),
        "references": (r"參考書資料：\s*\r?\n?", [r"\r?\n\s*\r?\n\s*(?:※|教學進度表)", r"※請遵守", r"教學進度表"]),
        "schedule_table": (r"教學進度表.+?：\s*\r?\n", [r"\r?\n\s*\r?\n\s*成績評定", r"成績評定"]),
        "grading": (r"成績評定方式：\s*\r?\n", [r"\r?\n\s*\r?\n\s*課堂要求", r"課堂要求"]),
        "requirements_text": (r"課堂要求：\s*\r?\n", [r"\r?\n\s*\r?\n\s*$"])
    }

    sections_raw = {
        key: extract_section(start_pat, end_pats, content) 
        for key, (start_pat, end_pats) in _SECTION_MAPPINGS.items()
    }


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
        "ge_domain": ge_domain,  # 通識主題領域：人文/社會/自然/""
    }

    objectives = sections_raw.get("objectives")
    syllabus = sections_raw.get("syllabus")
    textbooks = sections_raw.get("textbooks")
    references = sections_raw.get("references")
    schedule_table = sections_raw.get("schedule_table")
    grading = sections_raw.get("grading")
    requirements_text = sections_raw.get("requirements_text")

    ge_domain_line = f"- **通識主題領域**：{ge_domain}\n" if ge_domain else ""
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
            f"{ge_domain_line}"
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

# ── 資料目錄分類器（保留作為工具函式） ──
def _classify_data_dir(dir_name: str) -> str:
    """
    根據目錄名稱分類（工具函式，供未來擴充用）。
    目前目錄結構已固定為 data/courses/, data/professors/, data/dept_info/
    """
    if re.search(r"\d+學年度.*課程", dir_name):
        return "course"
    if "教授" in dir_name:
        return "professor"
    if any(kw in dir_name for kw in ["所資訊", "簡介", "就業", "學會", "新聞", "最新"]):
        return "dept_info"
    if dir_name in ("rules", "data"):
        return "skip"
    return "unknown"

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

    # 直接掃描 config.COURSES_DIR 下所有子目錄的 .txt
    txt_files = []
    courses_dir = data_dir  # 相容舊呼叫
    if os.path.isdir(courses_dir):
        for root, dirs, files in os.walk(courses_dir):
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
                ge_dom = metadata.get("ge_domain", "")
                if ge_dom:
                    prefix = f"[課程：{course_name} | 教師：{teacher} | 通識領域：{ge_dom} | 屬性：{req} | 區段：{section_name}]\n"
                else:
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

    logger.info(f"共建立 {len(all_nodes)} 個課程 Nodes")
    return all_nodes


def build_nodes_from_dept_info(professors_dir: str, dept_info_dir: str) -> list[TextNode]:
    """
    從教授資訊和系所資訊目錄建立 Nodes。
    
    Args:
        professors_dir: data/professors/ 目錄路徑
        dept_info_dir:  data/dept_info/ 目錄路徑
    """
    all_nodes = []
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    # 掃描兩個目錄下的所有子資料夾
    target_dirs = []
    for base_dir, dtype in [(professors_dir, "professor"), (dept_info_dir, "dept_info")]:
        if not os.path.isdir(base_dir):
            continue
        # 掃描所有子目錄（例如 professors/資工系教授資訊/）
        for entry in os.listdir(base_dir):
            entry_path = os.path.join(base_dir, entry)
            if os.path.isdir(entry_path):
                target_dirs.append((entry, entry_path, dtype))
        # 也掃描 base_dir 本身有沒有直接放 .txt
        has_txt = any(f.endswith(".txt") for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)))
        if has_txt:
            target_dirs.append((os.path.basename(base_dir), base_dir, dtype))

    if not target_dirs:
        logger.info("未找到任何系所資訊/教授資訊目錄")
        return all_nodes

    logger.info(f"偵測到 {len(target_dirs)} 個系所資訊目錄：{[d[0] for d in target_dirs]}")

    for dir_name, dir_path, dtype in target_dirs:
        if not os.path.isdir(dir_path):
            continue

        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".txt"):
                continue
            filepath = os.path.join(dir_path, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # 自動判斷資料類型（先看父目錄類型，再看檔名）
                if dtype == "professor":
                    info_type = "professor_info"
                    category = "教授資訊"
                elif "簡介" in fname:
                    info_type = "dept_intro"
                    category = "系所簡介"
                elif "就業" in fname:
                    info_type = "career_info"
                    category = "就業方向"
                elif "學會" in fname:
                    info_type = "student_union"
                    category = "系學會"
                elif "新聞" in fname or "最新" in fname:
                    info_type = "dept_news"
                    category = "系所新聞"
                elif "師資" in fname:
                    # 師資陣容一覽表 → 當作教授資訊處理（按【教授姓名】分段）
                    info_type = "professor_info"
                    category = "教授資訊"
                elif "設備" in fname or "教室" in fname or "實驗室" in fname:
                    info_type = "facility_info"
                    category = "教學設備與教室"
                else:
                    info_type = "dept_general"
                    category = "系所資訊"

                # 教授資訊：按 --- 或 【教授姓名】 分隔每位教授
                if info_type == "professor_info":
                    # 支援兩種格式：
                    # 格式 A（professors/ 目錄）：用 --- 分隔
                    # 格式 B（師資陣容.txt）：用 【教授姓名】 分隔
                    if fname == "師資陣容.txt":
                        # 格式 B：按 【...】 分段（保留標題行）
                        sections = re.split(r"\n(?=【)", content)
                    else:
                        # 格式 A（professors/ 目錄）：用 --- 分段
                        sections = re.split(r"\n-{3,}\n", content)
                    for sec in sections:
                        sec = sec.strip()
                        if not sec or len(sec) < 20:
                            continue
                        # 提取教授姓名（支援 姓名：XXX 和 【XXX】 兩種格式）
                        name_match = re.search(r"姓名[：:]\s*(.+?)(?:\s*[（(]|\n)", sec)
                        if not name_match:
                            name_match = re.search(r"【(.+?)】", sec)
                        prof_name = name_match.group(1).strip() if name_match else "未知教授"

                        # 萃取聯絡資訊（避免切分後聯絡資訊遺失）
                        email_match = re.search(r"電子郵件[：:]\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", sec)
                        phone_match = re.search(r"聯絡電話[：:]\s*([0-9()#\-]+)", sec)
                        email_str = email_match.group(1).strip() if email_match else "未提供"
                        phone_str = phone_match.group(1).strip() if phone_match else "未提供"

                        metadata = {
                            "info_type": info_type,
                            "category": category,
                            "professor_name": prof_name,
                            "source_file": fname,
                            "department": "資訊工程學系",
                            "dept_short": "資工系",
                        }
                        prefix = f"[資工系教授資訊 | 教授：{prof_name} | Email：{email_str} | 電話：{phone_str}]\n"

                        if len(sec) <= config.CHUNK_SIZE:
                            node = TextNode(text=prefix + sec, metadata=metadata)
                            node.excluded_embed_metadata_keys = list(metadata.keys())
                            all_nodes.append(node)
                        else:
                            doc = Document(text=sec, metadata={})
                            chunks = splitter.get_nodes_from_documents([doc])
                            for c in chunks:
                                c.set_content(prefix + c.get_content())
                                c.metadata = metadata
                                c.excluded_embed_metadata_keys = list(metadata.keys())
                            all_nodes.extend(chunks)

                        logger.info(f"  👨‍🏫 {prof_name} → 已建立 Node")

                else:
                    # 一般系所資訊：整段或按大標題分段
                    metadata = {
                        "info_type": info_type,
                        "category": category,
                        "source_file": fname,
                        "department": "資訊工程學系",
                        "dept_short": "資工系",
                    }
                    prefix = f"[資工系{category}]\n"

                    # 嘗試按「【...】」或「一、二、三...」分段
                    if re.search(r"【.+?】", content):
                        sections = re.split(r"\n(?=【)", content)
                    else:
                        sections = re.split(r"\n(?=[一二三四五六七八九十]+、)", content)
                    for sec in sections:
                        sec = sec.strip()
                        if not sec or len(sec) < 10:
                            continue
                        if len(sec) <= config.CHUNK_SIZE:
                            node = TextNode(text=prefix + sec, metadata=metadata)
                            node.excluded_embed_metadata_keys = list(metadata.keys())
                            all_nodes.append(node)
                        else:
                            doc = Document(text=sec, metadata={})
                            chunks = splitter.get_nodes_from_documents([doc])
                            for c in chunks:
                                c.set_content(prefix + c.get_content())
                                c.metadata = metadata
                                c.excluded_embed_metadata_keys = list(metadata.keys())
                            all_nodes.extend(chunks)

                logger.info(f"  ✅ {fname} ({category}) → 完成")

            except Exception as e:
                logger.error(f"  ❌ 系所資訊解析失敗：{fname} — {e}")

    logger.info(f"共建立 {len(all_nodes)} 個系所資訊 Nodes")
    return all_nodes


# =============================================================================
# 📜 建立 獨立規則 JSON Nodes (如畢業門檻)
# =============================================================================

def build_nodes_from_rules(rules_dir: str) -> list[TextNode]:
    """將 data/rules 目錄下的 JSON 規則轉換為 Nodes"""
    all_nodes = []
    if not os.path.isdir(rules_dir):
        logger.info("未找到任何規則目錄")
        return all_nodes

    for fname in os.listdir(rules_dir):
        if not fname.endswith(".json"):
            continue
            
        filepath = os.path.join(rules_dir, fname)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # 將 JSON 轉為可讀的 Markdown 文本
            desc = data.get("_description", "系統規章")
            dept = data.get("department", "全校")
            total_req = data.get("total_required", 0)
            
            lines = [
                f"# {desc}",
                f"**適用系所**：{dept}",
                f"**畢業學分門檻**：{total_req} 學分\n",
            ]
            
            cats = data.get("categories", {})
            for cat_name, cat_data in cats.items():
                req_cred = cat_data.get("required_credits", 0)
                note = cat_data.get("note", "")
                
                cat_header = f"## {cat_name} ({req_cred} 學分)"
                if note:
                    cat_header += f" - {note}"
                lines.append(cat_header)
                
                # 處理 domains (通識)
                if "domains" in cat_data:
                    for d_name, d_data in cat_data["domains"].items():
                        lines.append(f"- {d_name}領域：至少 {d_data.get('min_credits', 0)} 學分")
                
                # 處理 courses (必修/選修)
                if "courses" in cat_data:
                    c_list = cat_data["courses"]
                    if c_list and isinstance(c_list[0], dict):
                        for c in c_list:
                            lines.append(f"- {c.get('name')} ({c.get('credits', '?')}學分)")
                    elif c_list and isinstance(c_list[0], str):
                        lines.append(f"- 可選課程：{', '.join(c_list)}")
                
                # 處理 tracks (專業選修分組)
                if "tracks" in cat_data:
                    for t_name, t_data in cat_data["tracks"].items():
                        lines.append(f"### {t_name}")
                        t_courses = t_data.get("courses", [])
                        lines.append(f"- 包含課程：{', '.join(t_courses)}\n")
                        
            # 畢業條件
            conds = data.get("graduation_conditions", [])
            if conds:
                lines.append("## 其他畢業條件")
                for c in conds:
                    lines.append(f"- {c}")

            text_content = "\n".join(lines)
            prefix = f"[{dept} {desc}]\n"
            
            is_grad = "graduation" in fname.lower() or "畢業" in fname
            metadata = {
                "info_type": "graduation_rules" if is_grad else "policy_rules",
                "category": "畢業門檻與修業規定" if is_grad else "系統規章",
                "department": dept,
                "source_file": fname
            }
            
            node = TextNode(text=prefix + text_content, metadata=metadata)
            node.excluded_embed_metadata_keys = list(metadata.keys())
            all_nodes.append(node)
            logger.info(f"  ✅ 規則檔 {fname} → 已建立 Node")
            
        except Exception as e:
            logger.error(f"  ❌ 規則檔解析失敗：{fname} — {e}")
            
    return all_nodes


# =============================================================================
# 🔤 Gemini Embedding 2 Preview — 雲端 Embedding API
# =============================================================================

class _GeminiRateLimiter:
    """
    Thread-safe Token-bucket 速率限制器。
    防止 Gemini API 429 Too Many Requests。
    根據使用者的 RPM (3000) 和 TPM (1M) 限制自動節流。
    """
    def __init__(self, rpm: int, tpm: int):
        self._rpm = rpm
        self._tpm = tpm
        self._request_times: list[float] = []
        self._token_log: list[tuple[float, int]] = []  # (timestamp, token_count)
        self._lock = threading.Lock()

    def wait_if_needed(self, estimated_tokens: int = 0):
        """若即將超限，自動 sleep 等待到安全窗口"""
        with self._lock:
            now = _time.time()
            window_start = now - 60.0  # 1 分鐘滑動窗口

            # 清除過期紀錄
            self._request_times = [t for t in self._request_times if t > window_start]
            self._token_log = [(t, c) for t, c in self._token_log if t > window_start]

            # RPM 檢查
            if len(self._request_times) >= self._rpm:
                sleep_until = self._request_times[0] + 60.0
                wait_sec = max(0, sleep_until - now)
                if wait_sec > 0:
                    logger.info(f"  ⏳ Rate Limiter: RPM 將達上限，等待 {wait_sec:.1f}s...")
                    _time.sleep(wait_sec)

            # TPM 檢查
            total_tokens = sum(c for _, c in self._token_log)
            if total_tokens + estimated_tokens >= self._tpm:
                sleep_until = self._token_log[0][0] + 60.0
                wait_sec = max(0, sleep_until - now)
                if wait_sec > 0:
                    logger.info(f"  ⏳ Rate Limiter: TPM 將達上限 ({total_tokens}/{self._tpm})，等待 {wait_sec:.1f}s...")
                    _time.sleep(wait_sec)

            # 記錄本次請求
            self._request_times.append(_time.time())
            if estimated_tokens > 0:
                self._token_log.append((_time.time(), estimated_tokens))


_rate_limiter = _GeminiRateLimiter(
    rpm=config.EMBEDDING_RATE_LIMIT_RPM,
    tpm=config.EMBEDDING_RATE_LIMIT_TPM,
)


def gemini_embed(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int | None = None,
) -> np.ndarray:
    """
    使用 Gemini Embedding 2 Preview API 產生 embedding 向量。

    原生支援 task_type 非對稱檢索優化：
    - 建索引時：task_type="RETRIEVAL_DOCUMENT"
    - 查詢時：task_type="RETRIEVAL_QUERY"

    Args:
        texts: 文字列表
        task_type: Gemini 任務類型
            - "RETRIEVAL_DOCUMENT"：建索引用（最大化文件語意表示）
            - "RETRIEVAL_QUERY"：查詢用（最大化查詢意圖理解）
            - "SEMANTIC_SIMILARITY"：語意相似度
            - "CLASSIFICATION"：分類
            - "CLUSTERING"：聚類
        output_dimensionality: 輸出向量維度（128~3072，None=使用 config 預設值）

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    if not texts:
        logger.warning("  ⚠️ gemini_embed 收到空列表，直接返回 empty array")
        return np.array([], dtype=np.float32).reshape(0, config.EMBEDDING_DIMENSION)

    if output_dimensionality is None:
        output_dimensionality = config.EMBEDDING_DIMENSION

    url = f"{config.GEMINI_EMBEDDING_API_URL}:batchEmbedContents?key={config.GEMINI_API_KEY}"
    batch_size = config.EMBEDDING_BATCH_SIZE  # Gemini batchEmbedContents 最大 100 筆
    model_name = f"models/{config.GEMINI_EMBEDDING_MODEL}"
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # 預估 token 數（粗略：中文 1 字 ≈ 2 tokens，英文 1 word ≈ 1 token）
        estimated_tokens = sum(len(t) * 2 for t in batch)
        _rate_limiter.wait_if_needed(estimated_tokens)

        # 建構 batchEmbedContents 請求
        requests_body = []
        for text in batch:
            # 【重要防呆】Gemini API 遇到純空字串 ("" 或只含空白) 會直接回傳 400 Bad Request，導致整個 batch 掛掉
            # 為了保證長度對齊 FAISS，遇到空字串時自動補上一個預設詞
            safe_text = text.strip()
            if not safe_text:
                safe_text = "[EMPTY_CONTENT]"

            req = {
                "model": model_name,
                "content": {"parts": [{"text": safe_text}]},
                "taskType": task_type,
                "outputDimensionality": output_dimensionality,
            }
            requests_body.append(req)

        payload = {"requests": requests_body}

        # 指數退避重試
        for attempt in range(config.EMBEDDING_MAX_RETRIES):
            try:
                response = _gemini_session.post(
                    url,
                    json=payload,
                    timeout=120.0,
                )

                if response.status_code == 429:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    logger.warning(f"  ⚠️ Gemini API 429 (Rate Limited)，等待 {wait}s 後重試 (attempt {attempt+1}/{config.EMBEDDING_MAX_RETRIES})...")
                    _time.sleep(wait)
                    continue

                response.raise_for_status()
                result = response.json()
                batch_embeddings = [item["values"] for item in result["embeddings"]]
                all_embeddings.extend(batch_embeddings)
                break  # 成功，跳出重試迴圈

            except requests.exceptions.RequestException as e:
                if attempt < config.EMBEDDING_MAX_RETRIES - 1:
                    wait = 2 ** attempt * 2  # 2s, 4s, 8s
                    logger.warning(f"  ⚠️ Gemini Embedding API 錯誤：{e}，{wait}s 後重試 (attempt {attempt+1})...")
                    _time.sleep(wait)
                else:
                    logger.error(f"  ❌ Gemini Embedding API 最終失敗（{config.EMBEDDING_MAX_RETRIES} 次重試後放棄）：{e}")
                    raise

        logger.info(f"  Embedding 進度：{min(i + batch_size, len(texts))}/{len(texts)} ({task_type})")

    return np.array(all_embeddings, dtype=np.float32)


def gemini_embed_query(text: str) -> np.ndarray:
    """
    使用 Gemini 產生「查詢」用的 embedding。
    自動使用 task_type="RETRIEVAL_QUERY" 最佳化查詢意圖理解。

    Args:
        text: 查詢文字

    Returns:
        numpy array of shape (embedding_dim,)
    """
    result = gemini_embed([text], task_type="RETRIEVAL_QUERY")
    return result[0]

# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    config.setup_logging()
    courses = build_nodes_from_courses(config.DATA_DIR)
    print(f"\n📊 總結：")
    print(f"   Nodes 數量：{len(courses)}")
    print(f"\n📝 前 3 個 Node 範例：")
    for i, node in enumerate(courses[:3]):
        print(f"\n--- Node {i+1} ---")
        print(f"   Section: {node.metadata.get('section', 'N/A')}")
        print(f"   Course:  {node.metadata.get('course_name', 'N/A')}")
        print(f"   Text:    {node.get_content()[:100]}...")
