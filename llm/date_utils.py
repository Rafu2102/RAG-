# -*- coding: utf-8 -*-
"""
date_utils.py — 全字元中文+阿拉伯數字日期時間正規化器
=====================================================
將任意混合的中文/阿拉伯數字日期時間表達式正規化為純數字格式，
使下游的 regex 和 LLM 能正確處理。

支援格式示例：
  三月16號  → 3月16號
  3/十六    → 3/16
  三月十六日 → 3月16日
  九點半     → 9點半
  下午三點十分 → 下午3點10分
  十二月三十一日 → 12月31日
  二十三號   → 23號
"""

import re
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# 🔢 中文數字 → 阿拉伯數字轉換
# =============================================================================

# 基本中文數字對照
_CN_DIGIT_MAP = {
    "零": 0, "〇": 0,
    "一": 1, "壹": 1,
    "二": 2, "兩": 2, "貳": 2,
    "三": 3, "參": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陸": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
    "十": 10, "拾": 10,
}


def _cn_to_int(cn_str: str) -> int:
    """
    將中文數字字串轉為整數。
    支援 1~99 的範圍（足以覆蓋月/日/時/分/節次）。
    
    範例：
      "三"    → 3
      "十"    → 10
      "十二"  → 12
      "二十"  → 20
      "二十三" → 23
      "三十一" → 31
    """
    cn_str = cn_str.strip()
    if not cn_str:
        return 0
    
    # 純阿拉伯數字直接轉
    if cn_str.isdigit():
        return int(cn_str)
    
    # 單字中文數字
    if len(cn_str) == 1 and cn_str in _CN_DIGIT_MAP:
        return _CN_DIGIT_MAP[cn_str]
    
    # 處理「十X」「X十」「X十Y」格式
    if "十" in cn_str or "拾" in cn_str:
        ten_char = "十" if "十" in cn_str else "拾"
        parts = cn_str.split(ten_char)
        tens = _CN_DIGIT_MAP.get(parts[0], 1) if parts[0] else 1  # 「十二」→ tens=1
        ones = _CN_DIGIT_MAP.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    
    # 多字中文但沒有「十」→ 嘗試逐字轉換 (如「二三」→ 23)
    result = 0
    for ch in cn_str:
        if ch in _CN_DIGIT_MAP:
            result = result * 10 + _CN_DIGIT_MAP[ch]
    return result if result > 0 else 0


def _replace_cn_num(match: re.Match) -> str:
    """正則替換回呼：將匹配到的中文數字轉為阿拉伯數字字串"""
    return str(_cn_to_int(match.group(0)))


# =============================================================================
# 📅 日期時間正規化
# =============================================================================

# 中文數字字元集（用於 regex）
_CN_CHARS = r"[零〇一壹二兩貳三參四肆五伍六陸七柒八捌九玖十拾]"

# 匹配中文數字序列（包含「十」的組合，如 十二、二十三、三十一）
_CN_NUM_PATTERN = f"(?:{_CN_CHARS}+)"


def normalize_chinese_datetime(text: str) -> str:
    """
    將文本中的中文數字日期/時間表達式正規化為阿拉伯數字。
    
    轉換範例：
      "三月十六號"     → "3月16號"
      "3/十六"         → "3/16"
      "三月16日"       → "3月16日"
      "九點半"         → "9點半"
      "下午三點十分"   → "下午3點10分"
      "第五節到第七節" → "第5節到第7節"
      "明天十點"       → "明天10點"
    
    Args:
        text: 原始文本
    
    Returns:
        正規化後的文本（中文數字部分轉為阿拉伯數字）
    """
    if not text:
        return text
    
    result = text
    
    # ── 1. 日期格式正規化 ──
    
    # 模式：X月Y日/號（X, Y 可為中文或阿拉伯數字）
    # 例：三月十六號 → 3月16號，三月16日 → 3月16日
    def _norm_month_day(m):
        month_str = m.group(1)
        day_str = m.group(2)
        suffix = m.group(3) if m.group(3) else ""
        month = _cn_to_int(month_str) if not month_str.isdigit() else int(month_str)
        day = _cn_to_int(day_str) if not day_str.isdigit() else int(day_str)
        return f"{month}月{day}{suffix}"
    
    result = re.sub(
        rf"({_CN_NUM_PATTERN}|\d+)\s*月\s*({_CN_NUM_PATTERN}|\d+)\s*([日號])?",
        _norm_month_day,
        result
    )
    
    # 模式：X/Y（X, Y 可為中文或阿拉伯數字）
    # 例：3/十六 → 3/16
    def _norm_slash_date(m):
        month_str = m.group(1)
        day_str = m.group(2)
        month = _cn_to_int(month_str) if not month_str.isdigit() else int(month_str)
        day = _cn_to_int(day_str) if not day_str.isdigit() else int(day_str)
        return f"{month}/{day}"
    
    result = re.sub(
        rf"({_CN_NUM_PATTERN}|\d+)\s*/\s*({_CN_NUM_PATTERN}|\d+)",
        _norm_slash_date,
        result
    )
    
    # 模式：單獨的「X號/日」（前面沒有月份的情況）
    # 例：十六號 → 16號，二十三日 → 23日
    def _norm_standalone_day(m):
        day_str = m.group(1)
        suffix = m.group(2)
        day = _cn_to_int(day_str)
        return f"{day}{suffix}"
    
    result = re.sub(
        rf"({_CN_NUM_PATTERN})\s*([日號])",
        _norm_standalone_day,
        result
    )
    
    # ── 2. 時間格式正規化 ──
    
    # 模式：X點Y分 / X點半（X, Y 可為中文或阿拉伯數字）
    # 例：九點十分 → 9點10分，三點半 → 3點半
    def _norm_time(m):
        hour_str = m.group(1)
        minute_str = m.group(2) if m.group(2) else ""
        suffix = m.group(3) if m.group(3) else ""
        hour = _cn_to_int(hour_str) if not hour_str.isdigit() else int(hour_str)
        if minute_str:
            minute = _cn_to_int(minute_str) if not minute_str.isdigit() else int(minute_str)
            return f"{hour}點{minute}分{suffix}"
        return f"{hour}點{suffix}"
    
    result = re.sub(
        rf"({_CN_NUM_PATTERN}|\d+)\s*點\s*(?:({_CN_NUM_PATTERN}|\d+)\s*分?)?\s*(半)?",
        _norm_time,
        result
    )
    
    # ── 3. 節次正規化 ──
    # 例：第五節 → 第5節
    def _norm_period(m):
        prefix = m.group(1)
        num_str = m.group(2)
        num = _cn_to_int(num_str) if not num_str.isdigit() else int(num_str)
        return f"{prefix}{num}節"
    
    result = re.sub(
        rf"(第)\s*({_CN_NUM_PATTERN}|\d+)\s*節",
        _norm_period,
        result
    )
    
    # ── 4. 星期正規化 ──
    # 例：星期三/禮拜三/週三 → 不需轉數字，但「禮拜」→「星期」
    result = re.sub(r"禮拜", "星期", result)
    
    logger.debug(f"📅 日期正規化：「{text}」→「{result}」")
    return result
