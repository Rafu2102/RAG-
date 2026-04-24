# -*- coding: utf-8 -*-
"""
llm/coreference.py — 指代消解 (Coreference Resolution)
=======================================================
使用 Gemini Flash Lite 極速消解代名詞與隱式省略，
讓 Query Router 收到乾淨的完整句子。
"""

import logging

import config
from llm.gemini_client import call_gemini, GeminiAPIError

logger = logging.getLogger(__name__)


def resolve_coreference(question: str, chat_history: list[dict]) -> str:
    """分析最新提問的上下文依賴，補全代名詞或隱式省略的實體名稱。

    Args:
        question: 使用者的原始提問。
        chat_history: 對話歷史（list of {"role": ..., "content": ...}）。

    Returns:
        改寫後的完整句子。若無需改寫，回傳原始問題。
    """
    # 指代消解只需最近 2 輪對話（4 條訊息），避免被太久以前的主題帶偏
    recent = chat_history[-4:]
    history_lines = []
    for msg in recent:
        role = "使用者" if msg["role"] == "user" else "助理"
        history_lines.append(f"{role}：{msg['content']}")
    history_str = "\n".join(history_lines)

    prompt = f"""你是一個負責「上下文補全」與「指代消解」的 AI 代理。

【對話歷史】
{history_str}

【最新提問】
{question}

【任務規則】
1. 分析最新提問：是否包含代名詞（這、那、他、這門課），或是「隱式省略」（例如直接問「期中考呢？」、「那老師是誰？」、「第九週是幾號？」）。
2. 對照對話歷史：尋找【最接近當下】的討論主題（例如剛剛才提到的課程名稱、教授姓名）。
3. 補全提問：將省略的主詞、受詞或代名詞替換為明確的實體名稱。
4. ⚠️ 嚴禁畫蛇添足 (Over-resolution)：如果最新提問已經非常完整，或是明確的指令（例如「幫我把我行事曆裡面的第九週課程進度刪除」本身已經指定了目標「第九週課程進度」），請【直接照抄原句】。不要自作主張把歷史對話中的實體（如剛剛查過的課程名稱）強行塞進已完整的名詞中。
5. 嚴禁對話：你不是聊天機器人，請【只輸出】改寫後的那一句話，絕對不要輸出「改寫如下：」或任何解釋。

請輸出最終改寫的句子："""

    try:
        rewritten = call_gemini(
            prompt,
            model="fast",
            thinking="minimal",
            max_tokens=config.GEMINI_SHORT_MAX_TOKENS,
            timeout=10,
        )
        if rewritten and len(rewritten) >= 2:
            if rewritten != question:
                logger.info(f"🔗 指代消解：「{question}」→「{rewritten}」")
            return rewritten
    except GeminiAPIError as e:
        logger.warning(f"⚠️ 指代消解失敗，使用原始問題：{e}")

    return question
