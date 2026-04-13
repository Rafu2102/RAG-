# -*- coding: utf-8 -*-
"""
search_event_tool.py — 學校行事曆事件搜尋工具
================================================
使用 CKIP Tagger 萃取關鍵字來搜尋 events.json 中的學校行事曆事件。
支援同義詞擴充、日期鄰近事件自動聚合、連假摘要預計算。
"""

import json
import logging
from datetime import datetime, timedelta, date as _date
from pathlib import Path
from typing import Optional

from nlp_utils import tokenize_texts_ckip

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
EVENTS_PATH = BASE_DIR / "data" / "events.json"

# 【Bug 3 + Opt 5 修復】Lazy Loading — 避免模組載入時 events.json 不存在直接 crash
_events_cache: Optional[list] = None


def _get_events() -> list:
    """延遲載入 events.json，第一次呼叫才讀取"""
    global _events_cache
    if _events_cache is None:
        if EVENTS_PATH.exists():
            try:
                with EVENTS_PATH.open(encoding="utf-8") as f:
                    _events_cache = json.load(f)
                logger.info(f"📅 events.json 載入完成：{len(_events_cache)} 個事件")
            except Exception as e:
                logger.error(f"❌ 讀取 events.json 失敗：{e}")
                _events_cache = []
        else:
            logger.warning(f"⚠️ events.json 不存在：{EVENTS_PATH}，學校行事曆功能將不可用")
            _events_cache = []
    return _events_cache


# ═════════════════════════════════════════════════════════════════
# 同義詞對照表（學生口語 → 官方行事曆標題中的關鍵字）
# 值為「空格分隔的關鍵字組」，會直接當作搜尋關鍵字（不經過 CKIP 重新斷詞）
# ═════════════════════════════════════════════════════════════════
SYNONYM_MAP = {
    "退選": "停修",
    "加退選": "選課 停修",
    "加選": "選課",
    "選課": "選課",
    "宿舍": "宿舍",
    "住宿": "宿舍",
    "開學": "上課開始",
    "註冊": "繳費",
    "學費": "繳費",
    "期中": "期中考",
    "期末": "期末考",
    "暑假": "暑假",
    "放暑假": "暑假",
    # 國定假日同義詞（學生口語 → 官方行事曆名稱）
    "清明": "掃墓 兒童節",
    "清明節": "掃墓 兒童節",
    "掃墓": "掃墓",
    "連假": "放假 補假",
    "春假": "掃墓 兒童節",
    "228": "和平紀念日",
    "二二八": "和平紀念日",
    "和平紀念": "和平紀念日",
    "國慶": "國慶日",
    "雙十": "國慶日",
    "端午": "端午節",
    "中秋": "中秋節",
    "元旦": "元旦",
    "校慶": "校慶",
    "畢業": "畢業典禮",
    "畢典": "畢業典禮",
    "轉系": "轉系",
    "輔系": "輔系",
    "雙主修": "雙主修",
    "停修": "停修",
    "休學": "休學",
    "復學": "復學",
}

# CKIP 斷詞用的停用詞
_STOPWORDS = {
    "什麼時候", "幫我", "加到", "行事曆", "的", "請問", "日期",
    "時間", "是", "何時", "查詢", "有", "嗎", "我想", "放",
    "什麼", "甚麼", "學校", "金門大學", "國立", "時候",
    "什麼", "還有", "可以", "哪些", "告訴",
}

# 「放假類」事件的標題特徵（用於日期鄰近擴充）
_HOLIDAY_TITLE_PATTERNS = {"放假", "補假", "調整放假"}

# 星期幾（格式化用）
_WEEKDAY_NAMES = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}


def _parse_event_date(event: dict) -> Optional[datetime]:
    """安全解析事件的 start 日期"""
    try:
        return datetime.fromisoformat(event.get("start", ""))
    except (ValueError, TypeError):
        return None


def _fmt_date(iso_str: str) -> str:
    """ISO 日期 → X月X日(週X)"""
    try:
        dt = datetime.fromisoformat(iso_str)
        wd = _WEEKDAY_NAMES.get(dt.weekday(), "?")
        return f"{dt.month}月{dt.day}日(週{wd})"
    except (ValueError, TypeError):
        return iso_str.split("T")[0] if iso_str else "未知"


def _is_holiday_event(title: str) -> bool:
    """判斷事件標題是否為放假類事件"""
    return any(pat in title for pat in _HOLIDAY_TITLE_PATTERNS)


# ═════════════════════════════════════════════════════════════════
# 搜尋主函式
# ═════════════════════════════════════════════════════════════════

def search_academic_events(query: str) -> list[dict]:
    """
    使用 CKIP + 同義詞擴充搜尋學校行事曆事件。

    改進：
    1. 同義詞不經過 CKIP 重新斷詞（防止「補假」被拆成「補」+「假」）
    2. 找到核心匹配後，自動聚合日期鄰近的假日事件（連假完整覆蓋）
    3. 排除與核心匹配不在同一時間段的無關事件
    """
    events = _get_events()
    if not events:
        return []

    # ═══ Step 1: 從原始問題提取 CKIP 關鍵字 ═══
    ckip_results = tokenize_texts_ckip([query])
    ckip_keywords = ckip_results[0] if ckip_results else []
    valid_ckip = [kw for kw in ckip_keywords if kw not in _STOPWORDS and len(kw) >= 2]

    # ═══ Step 2: 同義詞擴充（直接加入，不經過 CKIP）═══
    synonym_terms = set()
    for slang, official in SYNONYM_MAP.items():
        if slang in query:
            for term in official.split():
                if len(term) >= 2:
                    synonym_terms.add(term)

    # 合併去重（CKIP 關鍵字 + 同義詞直接注入）
    all_keywords = list(set(valid_ckip) | synonym_terms)
    logger.info(f"  🔑 行事曆搜尋關鍵字：{all_keywords}")

    if not all_keywords:
        # 無法切出有效關鍵字，嘗試直接 substring 搜尋
        results = [e for e in events if query in e.get("title", "")]
        return results[:8]

    # ═══ Step 3: 雙層評分 ═══
    # 區分「主題關鍵字」和「泛用關鍵字」，避免「放假/補假」讓所有假日都命中
    generic_keywords = {"放假", "補假", "連假", "調整放假"}
    topic_keywords = [kw for kw in all_keywords if kw not in generic_keywords]
    generic_only = [kw for kw in all_keywords if kw in generic_keywords]

    scored_events = []
    for e in events:
        title = e.get("title", "")
        # 主題關鍵字得 3 分，泛用關鍵字得 1 分
        topic_score = sum(3 for kw in topic_keywords if kw in title)
        generic_score = sum(1 for kw in generic_only if kw in title)
        total_score = topic_score + generic_score
        if total_score > 0:
            scored_events.append((total_score, topic_score, e))

    scored_events.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if not scored_events:
        return []

    # ═══ Step 4: 智慧篩選 ═══
    # 如果有主題命中（topic_score > 0），只保留有主題命中的事件
    # 如果全部都是泛用命中（只有「放假」「補假」），才降級為全取
    has_topic_match = any(ts > 0 for _, ts, _ in scored_events)

    if has_topic_match:
        core_events = [e for _, ts, e in scored_events if ts > 0]
    else:
        # 純泛用查詢（如「什麼時候放假」）→ 取最近的 8 個放假事件
        core_events = [e for _, _, e in scored_events[:8]]

    # ═══ Step 5: 日期鄰近聚合（自動抓取連假完整區間）═══
    core_dates = set()
    for e in core_events:
        dt = _parse_event_date(e)
        if dt:
            core_dates.add(dt.date())
            # 也加入 end date
            try:
                end_dt = datetime.fromisoformat(e.get("end", ""))
                current = dt.date()
                while current <= end_dt.date():
                    core_dates.add(current)
                    current += timedelta(days=1)
            except (ValueError, TypeError):
                pass

    if not core_dates:
        return [e for _, _, e in scored_events[:8]]

    # 以核心日期為中心，向前後擴展 3 天抓取同一假期區塊的所有事件
    min_core = min(core_dates)
    max_core = max(core_dates)
    expand_start = min_core - timedelta(days=3)
    expand_end = max_core + timedelta(days=3)

    # 收集最終結果：核心匹配 + 日期鄰近的假日事件
    result_ids = set()
    final_results = []

    # 先加入核心匹配
    for e in core_events:
        eid = id(e)
        if eid not in result_ids:
            result_ids.add(eid)
            final_results.append(e)

    # 再擴充日期鄰近的假日事件
    for e in events:
        eid = id(e)
        if eid in result_ids:
            continue
        title = e.get("title", "")
        dt = _parse_event_date(e)
        if dt and expand_start <= dt.date() <= expand_end:
            # 只納入「放假/補假/調整放假」類型的事件
            if _is_holiday_event(title):
                result_ids.add(eid)
                final_results.append(e)

    # 按日期排序
    final_results.sort(key=lambda e: e.get("start", ""))

    logger.info(f"  📅 搜尋結果：{len(core_events)} 核心 + {len(final_results) - len(core_events)} 鄰近 = {len(final_results)} 總計")
    return final_results


# ═════════════════════════════════════════════════════════════════
# 預計算格式化（Python 負責所有日期運算，LLM 只管呈現）
# ═════════════════════════════════════════════════════════════════

def format_events_for_llm(events: list[dict]) -> dict:
    """
    將搜尋到的事件預先格式化，回傳結構化 dict 給 LLM prompt 使用。
    
    Returns:
        {
            "events_text": "格式化的事件清單文字",
            "holiday_summary": "連假摘要（含週末延伸計算），若非連假則為空字串",
            "event_count": 事件數量,
            "has_holidays": 是否包含放假事件,
        }
    """
    if not events:
        return {"events_text": "", "holiday_summary": "", "event_count": 0, "has_holidays": False}

    # ── 格式化每筆事件 ──
    formatted_lines = []
    all_dates: set[_date] = set()
    has_holidays = False

    for e in events:
        title = e.get("title", "未知事件")
        start_str = e.get("start", "")
        end_str = e.get("end", start_str)

        start_fmt = _fmt_date(start_str)
        end_fmt = _fmt_date(end_str)

        start_date_str = start_str.split("T")[0] if start_str else ""
        end_date_str = end_str.split("T")[0] if end_str else ""

        if start_date_str == end_date_str:
            formatted_lines.append(f"📌 {start_fmt}：{title}")
        else:
            formatted_lines.append(f"📌 {start_fmt} ～ {end_fmt}：{title}")

        # 收集放假日期（用於計算連假）
        if _is_holiday_event(title):
            has_holidays = True
            try:
                s_dt = datetime.fromisoformat(start_str).date()
                e_dt = datetime.fromisoformat(end_str).date()
                current = s_dt
                while current <= e_dt:
                    all_dates.add(current)
                    current += timedelta(days=1)
            except (ValueError, TypeError):
                pass

    events_text = "\n".join(formatted_lines)

    # ── 計算連假摘要（含週末延伸）──
    holiday_summary = ""
    if len(all_dates) >= 2:
        min_date = min(all_dates)
        max_date = max(all_dates)

        # 檢查是否為連續日期（允許中間有 1 天間隔，例如跨週末）
        sorted_dates = sorted(all_dates)
        is_consecutive = all(
            (sorted_dates[i + 1] - sorted_dates[i]).days <= 2
            for i in range(len(sorted_dates) - 1)
        )

        if is_consecutive:
            # 向前延伸：如果前一天是週末，也算連假
            while True:
                prev = min_date - timedelta(days=1)
                if prev.weekday() in (5, 6):  # 六=5, 日=6
                    min_date = prev
                else:
                    break

            # 向後延伸：如果後一天是週末，也算連假
            while True:
                nxt = max_date + timedelta(days=1)
                if nxt.weekday() in (5, 6):
                    max_date = nxt
                else:
                    break

            total_days = (max_date - min_date).days + 1
            min_fmt = f"{min_date.month}月{min_date.day}日(週{_WEEKDAY_NAMES[min_date.weekday()]})"
            max_fmt = f"{max_date.month}月{max_date.day}日(週{_WEEKDAY_NAMES[max_date.weekday()]})"
            holiday_summary = f"整段連假是 {min_fmt} ～ {max_fmt}，共 {total_days} 天"

    return {
        "events_text": events_text,
        "holiday_summary": holiday_summary,
        "event_count": len(events),
        "has_holidays": has_holidays,
    }