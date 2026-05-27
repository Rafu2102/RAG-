# -*- coding: utf-8 -*-
"""
utils.py — 共用工具常式
======================
存放不牽涉業務邏輯核心流程的純工具函式。
"""

import json
import os
import re
from pathlib import Path


def smart_split_message(text: str, max_len: int = 1900) -> list[str]:
    """將過長的訊息智慧分段，在段落邊界擷取，不破壞排版。
    同時具備 Markdown 格式感知，若在程式碼塊、粗體等標籤中被截斷，會自動在段末閉合，並在下一段開頭重新開啟。
    """
    if len(text) <= max_len:
        return [text]
    
    chunks = []
    remaining = text
    
    # 追蹤未閉合狀態的 flags
    in_code_block = False
    in_bold = False
    in_italic = False
    in_inline_code = False
    
    while remaining:
        if len(remaining) <= max_len:
            # 最後一段，但如果之前有未閉合的狀態，需要在開頭補上
            prefix = ""
            if in_code_block:
                prefix += "```\n"
            if in_bold:
                prefix += "**"
            if in_italic:
                prefix += "*"
            if in_inline_code:
                prefix += "`"
            
            chunks.append(prefix + remaining)
            break
        
        # 尋找最佳切分點，在 max_len 以內
        # 優先找雙換行（段落間隔），其次是單換行，再次是空格
        split_at = remaining.rfind('\n\n', 0, max_len)
        if split_at == -1 or split_at < max_len // 3:
            split_at = remaining.rfind('\n', 0, max_len)
        if split_at == -1 or split_at < max_len // 3:
            split_at = remaining.rfind(' ', 0, max_len)
        if split_at == -1 or split_at < max_len // 3:
            split_at = max_len
            
        chunk = remaining[:split_at].rstrip()
        remaining = remaining[split_at:].lstrip()
        
        # 統計當前 chunk 裡的標籤數量
        code_blocks = chunk.count("```")
        bolds = chunk.count("**")
        # 為了避免將 ** 算進單星號，我們先將 ** 替換掉，再統計單星號
        temp_chunk = chunk.replace("**", "")
        # 【修正】排除 Markdown 無序清單中的星號 (如 '* 清單項目')，避免其被誤判為未閉合的斜體標籤
        temp_chunk = re.sub(r'(?:^|\n)\s*\*\s+', '\n', temp_chunk)
        italics = temp_chunk.count("*")
        # 行內程式碼，先替換掉 ``` 以免重複統計
        temp_chunk2 = chunk.replace("```", "")
        inline_codes = temp_chunk2.count("`")
        
        # 計算此 chunk 結束時的狀態
        next_in_code_block = in_code_block ^ (code_blocks % 2 == 1)
        next_in_bold = in_bold ^ (bolds % 2 == 1)
        next_in_italic = in_italic ^ (italics % 2 == 1)
        next_in_inline_code = in_inline_code ^ (inline_codes % 2 == 1)
        
        # 開始為當前 chunk 補上閉合標籤
        suffix = ""
        
        if next_in_inline_code:
            suffix += "`"
        if next_in_italic:
            suffix += "*"
        if next_in_bold:
            suffix += "**"
        if next_in_code_block:
            suffix += "\n```"
            
        # 繼承自上一段的開啟標籤，需要加在 chunk 的最前面
        current_prefix = ""
        if in_code_block:
            current_prefix += "```\n"
        if in_bold:
            current_prefix += "**"
        if in_italic:
            current_prefix += "*"
        if in_inline_code:
            current_prefix += "`"
            
        chunks.append(current_prefix + chunk + suffix)
        
        # 更新狀態供下一輪使用
        in_code_block = next_in_code_block
        in_bold = next_in_bold
        in_italic = next_in_italic
        in_inline_code = next_in_inline_code

    # 加上續傳標記（Discord 多段訊息友善提示）
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            chunks[i] += f"\n\n✅ *續讀下一則 ({i+2}/{len(chunks)})...*"
    
    return chunks


def format_spacing(text: str) -> str:
    """
    全域智慧 Spacing 格式化排版引擎。
    在純文字的中英數邊界自動補上一個半形空格，排除百分比 `%` 與度數 `°`，
    並使用 placeholder 機制保護所有代碼塊、行內代碼、網址及 Markdown 連結，確保排版完美不損毀。
    """
    if not text:
        return ""
        
    # 0. 表情符號與零寬度連接器 (ZWJ) 還原引擎，修正常見 LLM 生成與傳輸亂碼
    text = text.replace("&zwj;", "\u200d")
    text = text.replace("🏃♂️", "🏃‍♂️")
    text = text.replace("🏃♀️", "🏃‍♀️")
    text = text.replace("🙋♂️", "🙋‍♂️")
    text = text.replace("🙋♀️", "🙋‍♀️")
    text = text.replace("🤦♂️", "🤦‍♂️")
    text = text.replace("🤦♀️", "🤦‍♀️")
    text = text.replace("🤷♂️", "🤷‍♂️")
    text = text.replace("🤷♀️", "🤷‍♀️")

    placeholders = []
    
    # 1. 保護代碼塊 (Code Blocks)
    def repl_code_block(match):
        placeholder = f"__CODE_BLOCK_PLACEHOLDER_{len(placeholders)}__"
        placeholders.append(match.group(0))
        return placeholder
        
    # 2. 保護行內代碼 (Inline Code)
    def repl_inline_code(match):
        placeholder = f"__INLINE_CODE_PLACEHOLDER_{len(placeholders)}__"
        placeholders.append(match.group(0))
        return placeholder

    # 3. 保護 Markdown 圖片與連結 (Markdown Images & Links)
    def repl_markdown_link(match):
        placeholder = f"__MD_LINK_PLACEHOLDER_{len(placeholders)}__"
        placeholders.append(match.group(0))
        return placeholder

    # 4. 保護網址 (URLs)
    def repl_url(match):
        placeholder = f"__URL_PLACEHOLDER_{len(placeholders)}__"
        placeholders.append(match.group(0))
        return placeholder

    # 依序抽離保護
    text = re.sub(r"```[\s\S]*?```", repl_code_block, text)
    text = re.sub(r"`[^`\n]+?`", repl_inline_code, text)
    text = re.sub(r"!\[.*?\]\(.*?\)", repl_markdown_link, text)
    text = re.sub(r"\[.*?\]\(.*?\)", repl_markdown_link, text)
    text = re.sub(r"https?://[^\s()<>]+", repl_url, text)

    # 5. 進行中英數邊界 Spacing
    # 中文字元區間：\u4e00-\u9fff
    # 英數字區間：a-zA-Z0-9
    # 漢字 -> 英數字
    text = re.sub(r"([\u4e00-\u9fff])([a-zA-Z0-9])", r"\1 \2", text)
    # 英數字 -> 漢字
    text = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fff])", r"\1 \2", text)
    # 數字 -> 英文單位 (如 10Gbps -> 10 Gbps)，排除 % 與 ° (非英文字母)
    text = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", text)

    # 6. 逆序安全還原 placeholders
    for i in range(len(placeholders) - 1, -1, -1):
        text = text.replace(f"__CODE_BLOCK_PLACEHOLDER_{i}__", placeholders[i])
        text = text.replace(f"__INLINE_CODE_PLACEHOLDER_{i}__", placeholders[i])
        text = text.replace(f"__MD_LINK_PLACEHOLDER_{i}__", placeholders[i])
        text = text.replace(f"__URL_PLACEHOLDER_{i}__", placeholders[i])
        
    return text


def atomic_write_json(file_path: Path | str, data: dict, indent: int = 4):
    """
    原子性寫入 JSON 檔案，避免 Race Condition 或中途崩潰導致資料毀損。
    1. 將資料寫入同目錄之臨時檔案 (*.json.tmp)。
    2. 完全寫入關閉後，以 os.replace 原子性取代原檔案。
    """
    path = Path(file_path)
    # 確保父目錄存在
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        # Windows 上 os.replace 也是原子操作，可安全取代已存在的目標檔案
        os.replace(tmp_path, path)
    except Exception as e:
        # 清理臨時檔案
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise e

