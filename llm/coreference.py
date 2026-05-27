# -*- coding: utf-8 -*-
"""
llm/coreference.py — 指代消解 (Coreference Resolution)
=======================================================
使用 Gemini 3.5 Flash 極速消解代名詞與隱式省略，
讓 Query Router 收到乾淨的完整句子。
"""

import logging

import config
from llm.gemini_client import a_call_gemini, GeminiAPIError

logger = logging.getLogger(__name__)


async def resolve_coreference(question: str, chat_history: list[dict]) -> str:
    """分析最新提問的上下文依賴，補全代名詞或隱式省略的實體名稱。

    Args:
        question: 使用者的原始提問。
        chat_history: 對話歷史（list of {"role": ..., "content": ...}）。

    Returns:
        改寫後的完整句子。若無需改寫，回傳原始問題。
    """
    # ── 🛡️ Python 端短路安全閥 (Python Short-Circuit Valve) ──
    # 如果使用者提問已包含明確的科系關鍵字，且完全沒有任何代名詞，且長度足夠，直接跳過 LLM
    pronouns = {"這", "那", "他", "她", "牠", "它", "該", "此", "這門", "那門", "這些", "那些", "這堂", "那堂", "這老師", "那老師", "這個", "那個", "他教", "她教"}
    has_pronoun = any(p in question for p in pronouns)
    
    has_known_dept = False
    for dept_info in config.DEPT_REGISTRY.values():
        keywords = [dept_info["short_name"], dept_info["full_name"]] + dept_info.get("aliases", []) + dept_info.get("keywords", [])
        if any(kw in question for kw in keywords):
            has_known_dept = True
            break
            
    if has_known_dept and not has_pronoun and len(question) >= 5:
        logger.info(f"🔗 指代消解 (Python 安全閥短路)：問句「{question}」具備明確科系且無代名詞，直接跳過改寫。")
        return question

    # 指代消解只需最近 2 輪對話（4 條訊息），避免被太久以前的主題帶偏
    recent = chat_history[-4:]
    history_lines = []
    for msg in recent:
        role = "使用者" if msg["role"] == "user" else "助理"
        history_lines.append(f"{role}：{msg['content']}")
    history_str = "\n".join(history_lines)

    prompt = f"""你是一個負責「上下文補全」與「指代消解」的專業 AI 代理。

【對話歷史】
{history_str}

【最新提問】
{question}

【任務規則】
1. **判斷是否需要改寫（黃金法規）**：
   - 如果最新提問已經包含具體的實體名稱（例如明確的科系名如「長照系」、明確的課程名稱、人名），且語意完整、意圖清晰，**請絕對不要做任何修改，直接原封不動照抄回傳**。
   - 只有當最新提問中含有代名詞（如「這」、「那」、「他」、「此」、「該」、「這些」、「這門課」），或者有明顯的語意省略（例如上一句在問微積分，這句接著問「那期中考呢？」）時，才需要結合【對話歷史】進行補全。

2. **如何補全**：
   - 尋找【對話歷史】中最靠近當下的相關討論實體，將最新提問中的代名詞或省略部分替換為明確的實體名稱（例如將「那老師是誰？」補全為「[歷史對話中提到的課程]的老師是誰？」）。

3. **⚠️ 嚴禁過度改寫與誤改 (Prevent Over-resolution)**：
   - **實體保護**：若最新提問已有主體（例如：「長照系推薦課程」、「資工系大一必修」、「黃積淵老師的專長」），**絕對禁止**從歷史對話中抽取其他無關的科系、課名或人名來混淆或覆蓋最新提問的主體！
   - **主體變更**：若最新提問提及的科系或老師與歷史對話不同，說明使用者已切換主題，請直接照抄最新提問，不要與歷史對話混合。

4. **對比範例 (Few-Shot Examples)**：
   - 範例 1 (主體變更)：
     歷史：使用者問資工系的 Linux 課。最新提問：「長照系推薦課程」
     ❌ 錯誤改寫：長照系與資工系推薦課程 / 長照系 Linux 推薦課程
     ✅ 正確輸出（不改寫）：長照系推薦課程
   - 範例 2 (教師變更)：
     歷史：討論柯志亨老師的專長。最新提問：「李錫捷老師的信箱是什麼？」
     ❌ 錯誤改寫：李錫捷老師和柯志亨老師的信箱是什麼？
     ✅ 正確輸出（不改寫）：李錫捷老師的信箱是什麼？

5. **輸出限制**：
   - 請【僅輸出】改寫完成後的最終句子。
   - 絕對不要包含任何解釋、標點符號包裹或「改寫如下：」等廢話。如果不需要改寫，請直接輸出原始問題。

請輸出最終改寫的句子："""

    try:
        rewritten = await a_call_gemini(
            prompt,
            model="fast",
            thinking="low",
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
