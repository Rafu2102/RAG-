# -*- coding: utf-8 -*-
"""
llm/chitchat.py — 非 RAG 生成函式集合
========================================
將不需要 RAG 檢索的獨立生成函式從 llm_answer.py 中抽出：
  - generate_chitchat_answer()       — 日常閒聊
  - generate_personal_info_answer()  — 個人課表/成績包裝
  - generate_web_search_answer()     — 聯網搜尋
"""

import logging
from datetime import datetime

import config
from llm.gemini_client import call_gemini, call_gemini_with_fallback, GeminiAPIError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 💬 閒聊回應
# ═══════════════════════════════════════════════════════════════

def generate_chitchat_answer(question: str, memory, user_profile: dict | None = None):
    """處理日常閒聊，完全跳過 RAG 檢索。

    Args:
        question: 使用者問題。
        memory: ConversationMemory 物件。
        user_profile: 使用者身分資訊（可選）。

    Returns:
        AnswerResult 包含閒聊回答。
    """
    from llm.llm_answer import AnswerResult  # 避免循環匯入

    now_str = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S")

    user_context = ""
    if user_profile:
        dept = user_profile.get("department", "資工系")
        grade = user_profile.get("grade", "")
        grade_str = f"{grade}年級" if grade else ""
        user_context = f" 【系統隱藏資訊】這個跟你對話的學生是：{dept}{grade_str}的學生！如果他問到「我是誰/幾年級/什麼系」，請熱情且直接用這個資訊回答他！\n"

    chitchat_prompt = f"""你是一個有點宅、幽默但非常熱心助人的「國立金門大學（NQU）資工系學長/學姐助理」 🤓💻。
使用者正在跟你閒聊或打招呼。{user_context}

現在系統時間：{now_str}
（若使用者問現在幾點、今天星期幾，請根據此時間精準回答，切勿自己發明日期）

## 你的任務：
1. 用學長姐的幽默語氣、摻雜一點點資工系/電腦科學的冷笑話或宅梗來簡短回應（不超過 100 字）。如果是問時間/日期，請確實回報上方系統時間。
2. 適當使用黑客或學生的 Emoji（例如 💻、🔥、☕、🐛、Bug）。
3. 回應完之後，熱情地引導使用者詢問課程或教授相關的問題（例如：「遇到修課 Bug 了嗎？有什麼我可以幫你 query 的課程資訊嗎？」）。
4. 絕對不可以輸出「助理：」或「對話歷史」等標籤。
5. 【禁止捏造】：閒聊模式下你手上沒有任何課程/教授/校園資料，絕對不可以編造任何校園相關事實。如果使用者在閒聊中夾帶課程問題（例如「順便問一下微積分幾學分」），請引導他用正式提問讓系統幫他查詢。
6. 【語氣一致性】：保持金門在地大學生的輕鬆感，可以偶爾提到金門的風獅爺、高粱、或離島生活的梗。

## 對話歷史：
{memory.get_history()}

## 使用者說：
{question}
"""

    logger.info("🤖 呼叫 Gemini Flash Lite 進行閒聊回應...")

    try:
        answer_text = call_gemini(
            chitchat_prompt,
            model="fast",
            thinking="low",
            max_tokens=config.GEMINI_SHORT_MAX_TOKENS,
        )
    except GeminiAPIError as e:
        logger.error(f"❌ 閒聊回應生成失敗: {e}")
        answer_text = "哈囉！我現在 CPU 有點過載，有什麼我可以幫你查詢的課程或教授資訊嗎？💻"

    return AnswerResult(answer=answer_text, sources=[], query=question)


# ═══════════════════════════════════════════════════════════════
# 📋 個人資料包裝
# ═══════════════════════════════════════════════════════════════

def generate_personal_info_answer(question: str, raw_data: str, info_type: str) -> str:
    """將使用者的個人課表或成績單，以學長姐的口吻包裝成友善對話。

    Args:
        question: 使用者問題。
        raw_data: 從資料庫撈出的原始資料字串。
        info_type: 資料類型，如 "課表" 或 "成績單"。

    Returns:
        LLM 包裝後的友善回答，或 fallback 為原始資料。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S")

    prompt = f"""你是一個熱心助人的國立金門大學（NQU）專屬校園助理。
使用者詢問關於他自己的【{info_type}】資訊，請根據下方資料庫撈出的真實資料回答。

【系統時間參考】
現在時間是：{now_str}
如果使用者詢問「今天」、「明天」、「這禮拜X」，請根據上述時間進行推算。
【重要】星期判斷指引：
- 上方時間已包含英文星期（如 Monday=星期一）。請據此計算相對日期。
- 「下禮拜X」= 從下一個星期一算起的那個星期X
- 「這禮拜X」= 本週的星期X（若今天已過則仍是本週的）

【資料庫真實資料】
{raw_data}

【使用者的問題】
{question}

【回答要求】
1. 直接、準確地回答他的問題（例如他問星期幾有課，就只回答那天的課）。
2. 語氣像學長或學姐一樣親切友善，可以加一點 emoji 豐富排版。
3. 如果真實資料中沒有他問的特定細節，就老實說沒看到或沒課。
4. 【絕對禁止】生硬地把整個資料表原封不動印出來，除非他真的要求印出全部！
5. 如果使用者問的是「X 點有沒有課」或「X 點有空嗎」，請比對課表中各課程的上課節次與時段，精準判斷該時段是否有課。每節課約 50 分鐘，第一節 08:10 開始。
6. 【禁止捏造】：只能用上方的【資料庫真實資料】回答。如果資料中沒有某課的詳細時間或學分，絕對不可以自己編。
7. 如果使用者問成績相關問題（info_type=成績單），請特別注意：
   - 計算平均分數時必須精確，不可四捨五入到整數
   - 低於 60 分的科目要特別標注提醒
   - 如果使用者問「我被當幾科」，請精確列出所有不及格科目
"""
    try:
        return call_gemini(prompt, thinking="medium")
    except GeminiAPIError as e:
        logger.error(f"❌ Personal LLM Wrapper 發生錯誤: {e}")
        return raw_data  # 退回原始字串


# ═══════════════════════════════════════════════════════════════
# 🌐 聯網搜尋
# ═══════════════════════════════════════════════════════════════

def generate_web_search_answer(question: str, memory):
    """處理校外知識與即時資訊，啟動 Google Search Grounding 聯網搜尋。

    Args:
        question: 使用者問題。
        memory: ConversationMemory 物件。

    Returns:
        AnswerResult 包含聯網搜尋回答。
    """
    from llm.llm_answer import AnswerResult  # 避免循環匯入

    logger.info("🌐 啟動 Google Search Grounding 聯網搜尋 (Gemini 3.1 Pro)...")

    web_prompt = f"""你是博學多聞的「國立金門大學（NQU）資工系全能學長/學姐」。
使用者問了一個超出校園範圍的一般性、即時性或科技知識問題（例如新聞、美食、論文或專有名詞）。
請使用你的 Google 搜尋能力與深度思考找出前沿資訊，並綜合彙整後，用親切、專業、充滿熱情的學長姐語氣回答。

【對話歷史】：
{memory.get_formatted_history()}

【使用者問題】：
{question}
"""

    try:
        answer_text = call_gemini(
            web_prompt,
            thinking="high",
            tools=[{"google_search": {}}],
            timeout=600,
        )
        logger.info(f"✅ 聯網搜尋回答生成成功 ({len(answer_text)} 字)")
        return AnswerResult(answer=answer_text, sources=[], query=question)
    except GeminiAPIError as e:
        logger.error(f"❌ 聯網搜尋失敗: {e}")
        fallback_answer = "🌐 學長/學姐剛剛想上網幫你查資料，但是網路好像有點不穩，請稍後再試一次喔！"
        return AnswerResult(answer=fallback_answer, sources=[], query=question)
