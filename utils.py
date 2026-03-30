# -*- coding: utf-8 -*-
"""
utils.py — 共用工具常式
======================
存放不牽涉業務邏輯核心流程的純工具函式。
"""


def smart_split_message(text: str, max_len: int = 1900) -> list[str]:
    """將過長的訊息智慧分段，在段落邊界擷取，不破壞排版。適用於 Discord 或其他長度受限的發送端。"""
    if len(text) <= max_len:
        return [text]
    
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        
        # 在 max_len 以內找最後一個換行分割點
        split_at = text.rfind('\n', 0, max_len)
        if split_at == -1 or split_at < max_len // 2:
            # 找不到換行就找最後一個空白
            split_at = text.rfind(' ', 0, max_len)
        if split_at == -1 or split_at < max_len // 2:
            split_at = max_len
        
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    
    # 加上續傳標記（Discord 多段訊息友善提示）
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            chunks[i] += f"\n\n✅ *續讀下一則 ({i+2}/{len(chunks)})...*"
    
    return chunks
