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
    
    # ── 0. 相對日期 → 絕對日期轉換 ──
    # 在交給 LLM 之前，先把「後天」「大後天」「3天後」「下週三」等相對日期
    # 直接轉換成精確的 YYYY-MM-DD，徹底消除 LLM 日期推算幻覺。
    from datetime import datetime, timedelta
    _now = datetime.now()
    _today = _now.date()
    
    # 0a. 固定偏移詞：大大後天(+4) > 大後天(+3) > 後天(+2) > 明天(+1) > 今天(+0)
    # 注意匹配順序：長的先匹配，避免「大後天」被「後天」先吃掉
    _relative_map = [
        ("大大後天", 4),
        ("大後天", 3),
        ("後天", 2),
        ("明天", 1),
        ("今天", 0),
        ("昨天", -1),
        ("前天", -2),
    ]
    for word, offset in _relative_map:
        if word in result:
            target = (_today + timedelta(days=offset)).strftime("%Y-%m-%d")
            result = result.replace(word, target, 1)
            break  # 一句話通常只有一個相對日期詞
    
    # 0b. 動態偏移：「X天後」「X天之後」「X日後」（X 可為中文或阿拉伯數字）
    def _norm_days_later(m):
        num_str = m.group(1)
        n = _cn_to_int(num_str) if not num_str.isdigit() else int(num_str)
        target = (_today + timedelta(days=n)).strftime("%Y-%m-%d")
        return target
    
    result = re.sub(
        rf"({_CN_NUM_PATTERN}|\d+)\s*[天日]\s*[後后]",
        _norm_days_later,
        result
    )
    
    # 0c. 下週X / 下禮拜X / 下星期X → 計算下一個星期X的日期
    _weekday_cn = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    def _norm_next_week(m):
        day_cn = m.group(1)
        target_wd = _weekday_cn.get(day_cn, 0)
        current_wd = _today.weekday()
        days_ahead = (target_wd - current_wd) % 7 + 7  # 永遠跳到下週
        target = (_today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        return target
    
    result = re.sub(
        r"下\s*(?:週|禮拜|星期)\s*([一二三四五六日天])",
        _norm_next_week,
        result
    )
    
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
    
    # ── 2. 時間格式正規化 (自動轉 24 小時制，支援區間語境) ──
    
    # 模式：(上午/下午)X點[Y分][到/至/-](上午/下午)?Z點[W分] (X, Y, Z, W 可為中文或阿拉伯數字)
    # 例：下午八點到九點 → 20點到21點
    def _norm_time(m):
        ampm1_str = m.group(1) if m.group(1) else ""
        hour1_str = m.group(2)
        min1_str = m.group(3) if m.group(3) else ""
        half1 = m.group(4) if m.group(4) else ""
        
        sep = m.group(5) if m.group(5) else ""
        
        ampm2_str = m.group(6) if m.group(6) else ""
        hour2_str = m.group(7) if m.group(7) else ""
        min2_str = m.group(8) if m.group(8) else ""
        half2 = m.group(9) if m.group(9) else ""
        
        def _convert_hour(h_str, prev_ampm=""):
            h = _cn_to_int(h_str) if not h_str.isdigit() else int(h_str)
            
            # 使用當前 ampm 或繼承前面的 ampm (處理 "下午八點到九點")
            ampm = prev_ampm
            
            if ampm in ["下午", "晚上", "晚", "夜間"] and h < 12:
                h += 12
            elif ampm in ["上午", "早上", "早", "凌晨"] and h == 12:
                h = 0
            
            # 處理跨越 12 點 (如 下午11點到1點 → 23點到1點(隔天凌晨) -> 這裡保持 24 小時制即可)
            # 或 早上9點到1點 -> 9點到13點 (若 h1=9, h2=1，自動當作下午)
            return h

        h1 = _convert_hour(hour1_str, ampm1_str)
        t1 = f"{h1}點"
        if min1_str:
            t1 += f"{_cn_to_int(min1_str) if not min1_str.isdigit() else int(min1_str)}分"
        t1 += half1

        if sep and hour2_str:
            # 第二個時間，若沒有明確 AM/PM，繼承第一個的
            ampm2_effective = ampm2_str if ampm2_str else ampm1_str
            
            # 智慧判斷：如果早上9點到1點，1顯然是下午1點 (13點)
            h2 = _convert_hour(hour2_str, ampm2_effective)
            if h1 < 12 and h2 < h1 and not ampm2_str:
                h2 += 12 # 自動補正為下午
                
            t2 = f"{sep}{h2}點"
            if min2_str:
                t2 += f"{_cn_to_int(min2_str) if not min2_str.isdigit() else int(min2_str)}分"
            t2 += half2
            return t1 + t2
            
        return t1
    
    result = re.sub(
        rf"(上午|早上|早|凌晨|下午|晚上|晚|夜間)?\s*({_CN_NUM_PATTERN}|\d+)\s*點\s*(?:({_CN_NUM_PATTERN}|\d+)\s*分?)?\s*(半)?"
        rf"(?:\s*([到至~\-])\s*(上午|早上|早|凌晨|下午|晚上|晚|夜間)?\s*({_CN_NUM_PATTERN}|\d+)\s*點\s*(?:({_CN_NUM_PATTERN}|\d+)\s*分?)?\s*(半)?)?",
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
