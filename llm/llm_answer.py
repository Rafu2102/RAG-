# -*- coding: utf-8 -*-
"""
llm_answer.py — LLM 回答生成模組
===================================
負責：
1. 使用 Ollama Llama 3.1 8B 生成回答
2. Answer Grounding：回答必須引用資料來源
3. Conversation Memory：支援多輪對話（保留最近 N 輪）
4. 結構化的 prompt 模板，引導 LLM 產生高品質回答
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

import config
from rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# =============================================================================
# 📋 回答結果資料類別
# =============================================================================

@dataclass
class AnswerResult:
    """LLM 回答的結果"""
    answer: str                              # LLM 生成的回答
    sources: list[dict] = field(default_factory=list)  # 引用的來源
    query: str = ""                          # 原始問題


# =============================================================================
# 💬 Conversation Memory
# =============================================================================

class ConversationMemory:
    """
    簡單的對話記憶管理。
    使用 sliding window 保留最近 N 輪對話。
    """

    def __init__(self, window_size: int = None):
        self.window_size = window_size or config.MEMORY_WINDOW_SIZE
        self.history: list[dict] = []  # [{"role": "user/assistant", "content": "..."}]

    def add_user_message(self, content: str):
        """新增使用者訊息"""
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        """新增助理回覆"""
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> list[dict]:
        """取得對話歷史"""
        return self.history.copy()

    def clear(self):
        """清空對話記憶"""
        self.history.clear()

    def get_formatted_history(self) -> str:
        """取得格式化的對話歷史字串"""
        if not self.history:
            return "無先前對話"

        lines = []
        for msg in self.history:
            role = "使用者" if msg["role"] == "user" else "助理"
            # 截斷過長的訊息
            content = msg["content"][:300]
            if len(msg["content"]) > 300:
                content += "..."
            lines.append(f"{role}：{content}")
        return "\n".join(lines)

    def _trim(self):
        """裁剪到 window_size 輪（每輪 = 1 user + 1 assistant = 2 messages）"""
        max_messages = self.window_size * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]


# =============================================================================
# 📝 Prompt Templates（拆分為 System + User 兩段，防止 TAIDE 忽略 context）
# =============================================================================

SYSTEM_RULES_PROMPT = """你是國立金門大學（NQU）的智慧校園課程助理 🎓。
個性活潑親切，自然融入對話，適當使用 Emoji。不要自我介紹。

# 🛑 核心規則（絕對遵守）
1. **只用檢索資料**：你【只能使用】下方 [檢索到的課程資料] 中的資訊來回答。
2. **禁止捏造**：絕對禁止憑空發明課程、老師、教室、時間。如果資料中沒有，就不要列。
3. **查無資料**：如果檢索資料中完全沒有答案，坦白說：「目前系統資料庫中找不到相關資訊，要不要換個問法試試？」
4. **不漏答**：當使用者問清單類問題（如「有哪些課程」），必須列出檢索資料中所有符合的課，不可遺漏。
5. **只回答一次**：回答中絕對不可重複列出相同的課程。每門課只出現一次。

# 📝 回答格式
使用以下統一格式，清楚列出每門課程：

📘 **[課程名稱]** ([英文名])
- 👨‍🏫 授課教師：[老師] | 🎓 學分：[學分] | 🏷️ [必/選修]
- ⏰ 時間地點：[時間] @ [教室]
- 💡 [從教學目標或大綱中，一句話描述這堂課重點]

（如有多門課程，依上述格式逐一列出）

👉 最後可針對使用者問題補充建議或總結。

# 🧠 推理規則
- **教師專長推論**：從老師授課的課程名稱和教學綱要歸納其研究領域。
- **選修推薦**：依使用者興趣或年級，從檢索到的課中推薦最合適的，並說明理由。

【警告】禁止把每週教學進度表整本印出。禁止自我介紹。禁止重複列出同一門課。
"""

USER_CONTEXT_PROMPT = """[檢索到的課程資料]
==========
{context}
==========

[使用者問題]
{question}

📝 系統提示：本次檢索共找到 【{course_count}】 門不同的課程。
請只根據上方資料回答，不要增加資料中沒有的課程，也不要遺漏。

請給出最終回答："""

# =============================================================================
# 🔄 重複回答偵測與清除
# =============================================================================

def _remove_duplicate_blocks(answer: str) -> str:
    """
    偵測 Llama 3.1 8B 偶爾會把回答輸出兩遍的問題。
    策略：將回答切為上下兩半，若相似度 > 60% 則只保留前半。
    """
    if len(answer) < 200:
        return answer
    
    # 用段落分割找重複
    lines = answer.split('\n')
    total = len(lines)
    if total < 6:
        return answer
    
    # 嘗試在中間附近找到重複的標題行（如 📖、📘、以下為您）
    mid = total // 2
    first_lines_set = set()
    for line in lines[:mid]:
        stripped = line.strip()
        if stripped and len(stripped) > 5:
            first_lines_set.add(stripped)
    
    # 從中間開始找第一個重複出現的標題行
    for i in range(mid - 2, total):
        stripped = lines[i].strip()
        if stripped and len(stripped) > 10 and stripped in first_lines_set:
            # 找到重複起始點，檢查後面是否真的是重複內容
            remaining = '\n'.join(lines[i:]).strip()
            first_half = '\n'.join(lines[:i]).strip()
            
            # 簡單相似度：比較字元重疊率
            if len(remaining) > 0 and len(first_half) > 0:
                shorter = min(len(remaining), len(first_half))
                overlap = sum(1 for a, b in zip(remaining[:shorter], first_half[:shorter]) if a == b)
                similarity = overlap / shorter if shorter > 0 else 0
                
                if similarity > 0.4:
                    return first_half
    
    return answer


# =============================================================================
# 🤖 LLM Answer 主函式
# =============================================================================

def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    memory: ConversationMemory,
) -> AnswerResult:
    """
    使用 Ollama Llama 3.1 根據檢索結果生成回答。

    結合：
    - 檢索到的 Top-5 chunks 作為 context
    - 對話歷史作為上下文
    - Answer Grounding prompt 確保引用來源

    Args:
        query: 使用者原始問題
        chunks: Reranker 後的 Top-N RetrievedChunk
        memory: 對話記憶物件

    Returns:
        AnswerResult 包含回答和來源
    """
    # ── 將檢索結果組合為 context，並加上中文翻譯蒟蒻 ──
    context_items = []
    sources = []

    section_translator = {
        "basic_info": "基本資訊與上課時間",
        "objectives": "課程教學目標",
        "syllabus": "課程教學綱要",
        "textbooks": "教科書與參考書",
        "grading": "成績評定與課堂要求",
        "schedule_table": "每週教學進度表"
    }

    for i, chunk in enumerate(chunks):
        # 過濾負分 chunk：reranker 分數 < 0 表示明顯不相關，不應讓 LLM 看到
        if chunk.final_score < 0.0:
            logger.info(f"  ⏭️ 跳過低分 chunk：{chunk.node.metadata.get('course_name', '?')} (score={chunk.final_score:.4f})")
            continue

        meta = chunk.node.metadata
        course_name = meta.get("course_name", "未知課程")
        raw_section = meta.get("section", "未知區段")
        teacher = meta.get("teacher", "")

        ch_section = section_translator.get(raw_section, raw_section)

        context_items.append(
            f"---\n"
            f"📄 資料 {i+1}：【{course_name}】{ch_section}\n"
            f"老師: {teacher} | {meta.get('credits', '?')}學分 | {meta.get('required_or_elective', '?')} | {meta.get('schedule', '?')}\n"
            f"{chunk.node.get_content()}\n"
            f"---"
        )

        sources.append({
            "course_name": course_name,
            "section": ch_section,
            "teacher": teacher,
            "score": chunk.final_score,
            "source_file": meta.get("source_file", ""),
        })

    context = "\n\n".join(context_items) if context_items else "（無相關資料）"

    unique_courses = set()
    for chunk in chunks:
        if chunk.final_score >= 0.0:
            course_name = chunk.node.metadata.get("course_name")
            if course_name:
                unique_courses.add(course_name)
    course_count = len(unique_courses)

    # ── 組合 prompt（拆分為 System 規則 + User 資料，防止 TAIDE 忽略 context）──
    user_prompt = USER_CONTEXT_PROMPT.format(
        context=context,
        question=query,
        course_count=course_count
    )

    # ── 呼叫 Ollama API ──
    try:
        logger.info(f"🤖 呼叫 LLM 生成回答 (使用 /api/chat)...")

        # ── 將 messages 組裝（暫時停用歷史記憶以防幻覺） ──
        # 【關鍵修復】將 System 規則與 User 資料合併為單一 user message
        # 這是為了防止 Llama 3.1 8B (Ollama) 在某些硬體或設定下直接忽略 system role
        combined_prompt = f"{SYSTEM_RULES_PROMPT}\n\n====================\n\n{user_prompt}"
        
        messages = [
            {"role": "user", "content": combined_prompt}
        ]

        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": getattr(config, "OLLAMA_KEEP_ALIVE", -1),
                "options": {
                    "temperature": config.OLLAMA_TEMPERATURE,
                    "num_ctx": config.OLLAMA_CONTEXT_WINDOW,
                    "num_predict": 1500,
                },
            },
            timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        answer = response.json()["message"]["content"].strip()
        
        # 【防重複】Llama 3.1 8B 偶爾會把同一個回答輸出兩次（情況一 + 情況二格式混用）
        answer = _remove_duplicate_blocks(answer)
        
        logger.info(f"✅ LLM 回答生成完成（{len(answer)} 字）")

    except Exception as e:
        logger.error(f"❌ LLM 回答生成失敗：{e}")
        answer = f"抱歉，生成回答時發生錯誤：{str(e)}\n請確認 Ollama 服務是否正常運行。"

    return AnswerResult(
        answer=answer,
        sources=sources,
        query=query,
    )


def generate_chitchat_answer(question: str, memory: ConversationMemory) -> AnswerResult:
    """處理日常閒聊，完全跳過 RAG 檢索"""
    
    chitchat_prompt = f"""你是一個親切、活潑的國立金門大學（NQU）資工系智慧校園助理。
使用者正在跟你閒聊或打招呼。

## 你的任務：
1. 用自然、友善的語氣簡短回應（不超過 50 字）。
2. 可以適當使用 Emoji。
3. 回應完之後，主動引導使用者詢問課程相關的問題（例如：「有什麼我可以幫你查詢的課程資訊嗎？」）。
4. 絕對不可以輸出「助理：」或「對話歷史」等標籤。

## 對話歷史：
{memory.get_history()}

## 使用者說：
{question}
"""
    
    logger.info("🤖 呼叫 3B 快速模型進行閒聊回應...")
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": config.OLLAMA_FAST_MODEL,  # 用 3B 快速模型，不佔用 8B 的 VRAM
            "messages": [
                {"role": "system", "content": chitchat_prompt},
                {"role": "user", "content": question}
            ],
            "stream": False,
            "keep_alive": 0,  # [VRAM 防護] 閒聊完立即卸載
            "options": {
                "temperature": 0.6,
                "num_predict": 200,
            }
        },
        timeout=config.OLLAMA_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    answer_text = response.json()["message"]["content"].strip()
    
    return AnswerResult(answer=answer_text, sources=[], query=question)


def format_sources(sources: list[dict]) -> str:
    """
    格式化來源資訊為顯示用字串。

    Args:
        sources: 來源列表

    Returns:
        格式化的來源字串
    """
    if not sources:
        return ""

    seen = set()
    unique_sources = []
    for s in sources:
        key = f"{s['course_name']}_{s['section']}"
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    lines = ["\n📚 資料來源："]
    for i, s in enumerate(unique_sources, 1):
        teacher_str = f"（{s['teacher']}）" if s.get("teacher") else ""
        lines.append(f"  {i}. 【{s['course_name']}】{s['section']}{teacher_str}")

    return "\n".join(lines)


# =============================================================================
# 🧪 測試
# =============================================================================
if __name__ == "__main__":
    config.setup_logging()

    # 測試 ConversationMemory
    print("=== 測試 ConversationMemory ===")
    memory = ConversationMemory(window_size=3)
    memory.add_user_message("你好")
    memory.add_assistant_message("你好！我是校園課程助理。")
    memory.add_user_message("資工系有什麼課？")
    print(memory.get_formatted_history())

    # 測試 format_sources
    print("\n=== 測試 format_sources ===")
    test_sources = [
        {"course_name": "深度學習", "section": "基本資訊", "teacher": "馮玄明", "score": 0.95},
        {"course_name": "深度學習", "section": "教學綱要", "teacher": "馮玄明", "score": 0.88},
    ]
    print(format_sources(test_sources))
