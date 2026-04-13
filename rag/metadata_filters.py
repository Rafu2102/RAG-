# -*- coding: utf-8 -*-
"""
metadata_filters.py — Soft Metadata Scoring 模組
==================================================
負責根據路由結果的 metadata 篩選條件，對檢索結果進行**軟評分**。

v2 架構改進：
  - 不再「硬過濾丟棄」不匹配的 chunks
  - 匹配的 chunks 獲得加分 (+3.0)
  - 不匹配的 chunks 獲得減分 (-2.0)
  - 最終交由 Reranker 做最終裁決
  - 這樣即使 metadata 規則沒有完美覆蓋，Reranker 仍能救回相關結果
"""

import re
import logging
from typing import TYPE_CHECKING

from .query_router import RouteResult

if TYPE_CHECKING:
    from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# 各 filter 的加分/減分權重
MATCH_BONUS = 3.0     # 匹配到的 chunk 加分
MISMATCH_PENALTY = -2.0  # 不匹配的 chunk 減分


# =============================================================================
# 🔍 個別欄位匹配函式
# =============================================================================

def _match_dept_short(meta: dict, v_list: list, content: str = "") -> bool:
    meta_dept = meta.get("dept_short", meta.get("department", ""))
    if meta_dept == "全校":
        return True
    
    # 建立別名映射
    aliases = {
        "資工系": ["資訊工程學系", "資工", "CSIE"],
        "海巡系": ["海洋與邊境管理學系", "海巡"],
        "企管系": ["企業管理學系", "企管"],
    }
    
    for v in v_list:
        if not v: continue
        # 展開別名
        targets = [v] + aliases.get(v, [])
        for t in targets:
            if t in meta_dept or meta_dept in t:
                return True
    return False

def _match_course_name_keyword(meta: dict, v_list: list, content: str = "") -> bool:
    c_name = meta.get("course_name", "")
    c_name_short = re.sub(r"[（(][^）)]*[）)]", "", c_name).strip()
    return any(v == c_name_short or v == c_name or v in c_name for v in v_list)

def _match_teacher(meta: dict, v_list: list, content: str = "") -> bool:
    clean_vals = [re.sub(r"(老師|教授)$", "", v) for v in v_list]
    # 同時匹配課程的 teacher 欄位和教授資訊的 professor_name 欄位
    teacher = meta.get("teacher", "")
    prof_name = meta.get("professor_name", "")
    
    for cv in clean_vals:
        if not cv:
            continue
        # 精確匹配
        if cv in teacher or cv in prof_name:
            return True
        # 模糊匹配：容忍 1 字之差（如 柯志恆 vs 柯志亨）
        if len(cv) >= 2:
            for name in [teacher, prof_name]:
                if len(name) >= 2:
                    overlap = sum(1 for c in cv if c in name)
                    if overlap >= max(2, int(len(cv) * 0.67)):
                        return True
    return False

def _match_day_of_week(meta: dict, v_list: list, content: str = "") -> bool:
    sched = meta.get("schedule", "")
    if sched:
        return any(v in sched for v in v_list)
    return any(v in content for v in v_list)

# 年級正規化映射表（阿拉伯數字 ↔ 中文數字雙向）
_GRADE_NORMALIZE = {
    "1": "一", "2": "二", "3": "三", "4": "四", "5": "五",
    "一": "一", "二": "二", "三": "三", "四": "四", "五": "五",
    "碩一": "碩一", "碩二": "碩二", "碩三": "碩三",
}

def _normalize_grade(g: str) -> str:
    """將年級字串正規化為統一的中文格式"""
    g = re.sub(r"^大", "", g).strip()
    return _GRADE_NORMALIZE.get(g, g)

def _match_grade(meta: dict, v_list: list, content: str = "") -> bool:
    meta_grade = _normalize_grade(meta.get("grade", ""))
    return any(_normalize_grade(v) == meta_grade for v in v_list if v)

def _match_required_or_elective(meta: dict, v_list: list, content: str = "") -> bool:
    meta_req = meta.get("required_or_elective", "")
    return any(v in meta_req for v in v_list)

def _match_time_period(meta: dict, v_list: list, content: str = "") -> bool:
    schedule = meta.get("schedule", "")
    sched_match = re.search(r"第(\d+)節[~～至到\-]第?(\d+)節", schedule)
    if not sched_match: return False
    
    sched_start, sched_end = int(sched_match.group(1)), int(sched_match.group(2))
    parts = v_list[0].split("-")
    if len(parts) != 2: return False
    
    filter_start, filter_end = int(parts[0]), int(parts[1])
    return not (sched_start > filter_end or sched_end < filter_start)

def _match_semester(meta: dict, v_list: list, content: str = "") -> bool:
    return any(v == meta.get("semester", "") for v in v_list)

def _match_academic_year(meta: dict, v_list: list, content: str = "") -> bool:
    return any(v == meta.get("academic_year", "") for v in v_list)

def _match_class_group(meta: dict, v_list: list, content: str = "") -> bool:
    return any(v == meta.get("class_group", "") for v in v_list)

def _match_is_evening(meta: dict, v_list: list, content: str = "") -> bool:
    return meta.get("is_evening", False)

def _match_classroom(meta: dict, v_list: list, content: str = "") -> bool:
    meta_classroom = meta.get("classroom", "")
    return any(v in meta_classroom for v in v_list if v)

def _match_ge_domain(meta: dict, v_list: list, content: str = "") -> bool:
    """通識主題領域比對（人文/社會/自然）"""
    meta_domain = meta.get("ge_domain", "")
    if not meta_domain:
        return False
    return any(v in meta_domain or meta_domain in v for v in v_list if v)


# =============================================================================
# 📋 Handler 註冊表
# =============================================================================

_FILTER_HANDLERS = {
    "dept_short": _match_dept_short,
    "course_name_keyword": _match_course_name_keyword,
    "teacher": _match_teacher,
    "day_of_week": _match_day_of_week,
    "grade": _match_grade,
    "required_or_elective": _match_required_or_elective,
    "time_period": _match_time_period,
    "semester": _match_semester,
    "academic_year": _match_academic_year,
    "class_group": _match_class_group,
    "is_evening": _match_is_evening,
    "classroom": _match_classroom,
    "ge_domain": _match_ge_domain,
}

# 絕對排他性 filter（科系、年級、星期、節次、學期、學年），一旦不匹配絕對不可寬容（-100分直接斬殺）
# 因為使用者若指定 114-2 星期二，給出 114-1 的課是嚴重錯誤的跨學期幻覺
_EXCLUSIVE_KEYS = {"dept_short", "grade", "day_of_week", "time_period", "semester", "academic_year"}

# 高權重 filter（使用者明確指定的實體：老師、課程名、教室）
# 這些 filter 匹配時加更多分，不匹配時扣多一點分
_HIGH_WEIGHT_KEYS = {"teacher", "course_name_keyword", "classroom"}

# 低權重 filter（目前已全數升級，保留空集合以防報錯）
_LOW_WEIGHT_KEYS = set()


# =============================================================================
# 🏗️ Soft Metadata Scoring 主引擎
# =============================================================================

def apply_hard_metadata_filter(
    chunks: list,  # list[RetrievedChunk]
    route_result: RouteResult,
) -> list:  # list[RetrievedChunk]
    """
    對檢索結果進行 Soft Metadata 評分。

    v2 改進（不再硬過濾！）：
    - 匹配的 chunks 加分（boost），不匹配的 chunks 減分（penalty）
    - 使用者明確指定的 filter（老師、課程名、教室）加更多分
    - 系統自動注入的 filter（學期、年級）只微調
    - 最終排序交由 Reranker 做最終裁決
    """
    filters = route_result.metadata_filters

    # 過濾掉沒有 handler 的 key
    active_filters = {k: v for k, v in filters.items() if k in _FILTER_HANDLERS and v}

    if not active_filters:
        return chunks

    # 智慧豁免：精確實體查詢時，豁免 profile 注入的 grade/dept
    exempt_keys = set()
    if "teacher" in active_filters:
        exempt_keys.update(["grade", "dept_short"])
        logger.info(f"  🏷️ 教師查詢「{active_filters['teacher']}」→ 豁免 grade + dept_short 評分")
    if "course_name_keyword" in active_filters:
        # 【關鍵修復】不再對 course_name_keyword 自動豁免 dept_short 與 grade
        # 這樣當使用者「明確指定」資工系微積分時，電機系微積分才會被成功 -100 分斬殺！
        # 若使用者只是找微積分，Router 現在不會補上 dept_short，所以也不會被誤殺。
        logger.info(f"  🏷️ 課程名稱查詢「{active_filters['course_name_keyword']}」→ 嚴格套用系級評分機制")
    if "classroom" in active_filters:
        exempt_keys.update(["grade", "dept_short"])
        logger.info(f"  🏷️ 教室查詢「{active_filters['classroom']}」→ 豁免 grade + dept_short 評分")

    # 移除豁免的 keys
    scoring_filters = {k: v for k, v in active_filters.items() if k not in exempt_keys}

    if not scoring_filters:
        return chunks

    # 對每個 chunk 計算 metadata 加減分
    match_count = 0
    for chunk in chunks:
        meta = chunk.node.metadata
        content = chunk.node.get_content()
        score_delta = 0.0

        for key, val in scoring_filters.items():
            handler = _FILTER_HANDLERS.get(key)
            if not handler:
                continue
            
            # 動態判定排他性條件：若有指定特定課程名稱或教師，學期和學年不再是必須完美匹配的排他條件
            # 這樣即使課程是上學期開的，也能被找到，只是分數稍低（-2.0 而非 -100.0）
            is_exclusive = key in _EXCLUSIVE_KEYS
            if key in ["semester", "academic_year"] and ("course_name_keyword" in scoring_filters or "teacher" in scoring_filters):
                is_exclusive = False
            
            v_list = val if isinstance(val, list) else [val]
            is_match = handler(meta, v_list, content)

            if is_exclusive:
                # 排他性條件：匹配加分，不匹配直接斬殺（-100）
                score_delta += 2.0 if is_match else -100.0
            elif key in _HIGH_WEIGHT_KEYS:
                # 明確實體：匹配 +5, 不匹配 -3
                score_delta += 5.0 if is_match else -3.0
            elif key in _LOW_WEIGHT_KEYS:
                # 背景條件：匹配 +1, 不匹配 -0.5
                score_delta += 1.0 if is_match else -0.5
            else:
                # 一般條件：匹配 +3, 不匹配 -2
                score_delta += MATCH_BONUS if is_match else MISMATCH_PENALTY

        chunk.metadata_score += score_delta
        if score_delta > 0:
            match_count += 1

    # 按 metadata_score + final_score 重新排序
    chunks.sort(key=lambda c: c.metadata_score + c.final_score, reverse=True)

    logger.info(
        f"  🏷️ Soft Metadata Scoring：{match_count}/{len(chunks)} chunks 獲得加分 "
        f"(filters: {list(scoring_filters.keys())})"
    )
    
    return chunks
