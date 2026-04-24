# -*- coding: utf-8 -*-
"""
llm_answer.py — LLM 回答生成模組（核心 RAG 回答）
=====================================================
負責：
1. 使用 Gemini 3.1 Pro 生成 RAG 回答
2. Answer Grounding：回答必須引用資料來源
3. Conversation Memory：支援多輪對話（保留最近 N 輪）
4. Context 重組（教授 backfill、課程過濾、學期感知）

非 RAG 生成函式已拆分至：
  - llm/chitchat.py       — 閒聊、個人資料包裝、聯網搜尋
  - llm/coreference.py    — 指代消解
Prompt 模板已拆分至：
  - llm/prompts.py
"""

import re as _re
import logging
from dataclasses import dataclass, field

import config
from rag.retriever import RetrievedChunk
from llm.gemini_client import call_gemini_with_fallback, GeminiAPIError
from llm.prompts import (
    SYSTEM_RULES_PROMPT, USER_CONTEXT_PROMPT,
    SYSTEM_RULES_PROFESSOR_PROMPT, USER_CONTEXT_PROFESSOR_PROMPT,
)

logger = logging.getLogger(__name__)


 # =============================================================================.Value -replace '=', [char]0x2550 
# 📋 回答結果資料類別
 # =============================================================================.Value -replace '=', [char]0x2550 

@dataclass
class AnswerResult:
    """LLM 回答的結果"""
    answer: str                              # LLM 生成的回答
    sources: list[dict] = field(default_factory=list)  # 引用的來源
    query: str = ""                          # 原始問題


 # =============================================================================.Value -replace '=', [char]0x2550 
# 💬 Conversation Memory
 # =============================================================================.Value -replace '=', [char]0x2550 

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
            # content = msg["content"][:300]  # 暫時解除字數限制（外接 API 不吃本地記憶體）
            content = msg["content"]
            # if len(msg["content"]) > 300:  # 暫時解除（配合上方解除）
            #     content += "..."
            lines.append(f"{role}：{content}")
        return "\n".join(lines)

    def _trim(self):
        """裁剪到 window_size 輪（每輪 = 1 user + 1 assistant = 2 messages）"""
        # 【Bug 7 修復】window_size=0 時完全清除記憶（零記憶模式）
        if self.window_size <= 0:
            self.history.clear()
            return
        max_messages = self.window_size * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]




 # =============================================================================.Value -replace '=', [char]0x2550 
# 🔄 重複回答偵測與清除
 # =============================================================================.Value -replace '=', [char]0x2550 

def _remove_duplicate_blocks(answer: str) -> str:
    """
    偵測 LLM 偶爾會把回答輸出兩遍的問題。
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


 # =============================================================================.Value -replace '=', [char]0x2550 
# 📅 課表注入格式化工具
 # =============================================================================.Value -replace '=', [char]0x2550 

def _format_schedule_for_llm(discord_id: str) -> str:
    """從學生 Token JSON 讀取個人課表，格式化為 LLM 可理解的文字。"""
    if not discord_id:
        return ""

    try:
        from tools.schedule_manager import get_schedule, DAY_NAMES, PERIOD_TIME_MAP
        schedule = get_schedule(discord_id)
        if not schedule or not schedule.get("courses"):
            return ""

        courses = schedule.get("courses", [])
        timetable = schedule.get("timetable", {})
        free_periods = schedule.get("free_periods", {})
        total_credits = schedule.get("total_credits", 0)
        year = schedule.get("academic_year", "?")
        sem = schedule.get("semester", "?")

        lines = [f"📅【學生本學期個人課表】（{year}學年度 第{sem}學期，共 {total_credits} 學分）"]

        for day in range(1, 6):
            day_str = str(day)
            day_name = DAY_NAMES.get(day, f"星期{day}")
            day_courses = timetable.get(day_str, {})

            if not day_courses:
                lines.append(f"{day_name}：整天空堂 🎉")
                continue

            # 合併同一門課的連續節次
            course_groups = {}
            for period_str, name in sorted(day_courses.items(), key=lambda x: int(x[0])):
                period = int(period_str)
                if name not in course_groups:
                    course_groups[name] = []
                course_groups[name].append(period)

            parts = []
            for name, periods in course_groups.items():
                instructor = "?"
                for c in courses:
                    if c["name"] == name and c["day"] == day:
                        instructor = c.get("instructor", "?")
                        break
                time_start = PERIOD_TIME_MAP.get(periods[0], ("?", "?"))[0]
                time_end = PERIOD_TIME_MAP.get(periods[-1], ("?", "?"))[1]
                parts.append(f"第{periods[0]}-{periods[-1]}節({time_start}~{time_end}) {name} ({instructor})")

            lines.append(f"{day_name}：{'、'.join(parts)}")

            free = free_periods.get(day_str, [])
            if free:
                free_strs = [f"第{p}節" for p in free]
                lines.append(f"        ➡️ 空堂：{', '.join(free_strs)}")

        lines.append("")
        lines.append("🔑 推薦指導原則：在推薦課程時，務必交叉比對上方課表。")
        lines.append("   - 若推薦的課程時間與學生已有課程「完全衝突」，必須標記 ⚠️ 並說明衝突原因。")
        lines.append("   - 若推薦的課程落在學生的「空堂時段」，標記 ✅ 表示完美契合。")
        lines.append("   - 優先推薦「無衝突」的課程，衝突的課程放在最後作為參考。")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"⚠️ 讀取學生課表失敗（不影響回答）：{e}")
        return ""


 # =============================================================================.Value -replace '=', [char]0x2550 
# 🤖 LLM Answer 主函式
 # =============================================================================.Value -replace '=', [char]0x2550 

def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    memory: ConversationMemory,
    route_result=None,
    all_nodes=None,
    user_profile: dict = None,
    discord_id: str = None,
) -> AnswerResult:
    """
    使用 Gemini 3.1 Pro 根據檢索結果生成 RAG 回答。

    結合：
    - 檢索到的 Top-5 chunks 作為 context
    - 對話歷史作為上下文
    - Answer Grounding prompt 確保引用來源
    - 根據 AI Router 的意圖動態選擇 Prompt（課程 vs 教授 vs 通用）

    Args:
        query: 使用者原始問題
        chunks: Reranker 後的 Top-N RetrievedChunk
        memory: 對話記憶物件
        route_result: AI Router 的意圖判斷結果
        user_profile: 使用者身分資訊

    Returns:
        AnswerResult 包含回答和來源
    """
    # ── 防禦 Prompt Injection：限制長度並轉義 XML 標籤 ──
    # safe_query = query[:300].replace("<", "＜").replace(">", "＞")  # 暫時解除字數限制（外接 API 不吃本地記憶體）
    safe_query = query.replace("<", "＜").replace(">", "＞")

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

    # v3: 「相對」分數門檻（智慧版 — 不用絕對門檻，用 Top-1 分數的比例）
    is_career_planning = getattr(route_result, "is_career_planning", False) if route_result else False
    if chunks and not is_career_planning:
        max_score = max(c.final_score for c in chunks)
        if max_score > 0.5:
            # 精確課程查詢（有 course_name_keyword）用更嚴格的門檻，削減不相關課程
            has_course_name = bool(route_result and route_result.metadata_filters.get("course_name_keyword"))
            threshold_ratio = 0.40 if has_course_name else 0.25
            relative_threshold = max_score * threshold_ratio
            original_count = len(chunks)
            chunks = [c for c in chunks if c.final_score >= relative_threshold]
            if len(chunks) < original_count:
                logger.info(
                    f"  🎯 相對門檻過濾：max={max_score:.3f}, threshold={relative_threshold:.3f} (ratio={threshold_ratio}), "
                    f"保留 {len(chunks)}/{original_count} chunks"
                )
        elif max_score < -10:
            # 【關鍵修復】所有 chunks 都被 metadata 嚴重扣分（如學期/系所全部不匹配）
            # 這代表資料庫中很可能沒有符合條件的課程，但 Reranker 保底機制仍輸出了結果
            # 只保留 metadata_score 最高的前幾個，讓 LLM 有機會判斷「查無此課」
            original_count = len(chunks)
            # 按 metadata_score 排序，只取最不差的（metadata_score 最高代表最接近匹配）
            chunks_sorted = sorted(chunks, key=lambda c: c.metadata_score, reverse=True)
            # 只保留 metadata_score > 最差分的一半 的 chunks
            best_meta = chunks_sorted[0].metadata_score
            if best_meta < -50:
                # 全部都被嚴重懲罰，只保留前 3 個讓 LLM 判斷
                chunks = chunks_sorted[:3]
            else:
                chunks = [c for c in chunks_sorted if c.metadata_score >= best_meta * 0.5]
            if len(chunks) < original_count:
                logger.info(
                    f"  🎯 負分門檻過濾：max_score={max_score:.3f}, best_meta={best_meta:.3f}, "
                    f"保留 {len(chunks)}/{original_count} chunks"
                )

    # 🆕 意圖驅動 Context 重組：教授查詢時，教授資訊排最前、只保留相關課程
    intent = route_result.query_type if route_result else "course_info"
    if intent == "professor_info" and chunks:
        teacher_name = ""
        if route_result and route_result.metadata_filters.get("teacher"):
            teacher_name = _re.sub(r"(老師|教授)$", "", route_result.metadata_filters["teacher"]).strip()
        
        # 🔑 核心修復：分類收集 chunk，以便處理沒有老師名字的情況
        prof_chunks_by_name = {}  # { "柯志亨": [chunk1, chunk2...] }
        dept_chunks = []
        course_chunks = []
        
        for c in chunks:
            it = c.node.metadata.get("info_type", "")
            if it == "professor_info":
                pn = c.node.metadata.get("professor_name", "")
                if pn:
                    if pn not in prof_chunks_by_name:
                        prof_chunks_by_name[pn] = []
                    prof_chunks_by_name[pn].append(c)
            elif it in ("dept_intro", "career_info", "student_union", "dept_news", "dept_general"):
                dept_chunks.append(c)
            elif "course_name" in c.node.metadata:  # 課程資料沒有 info_type，是用 course_name 辨識
                if not teacher_name or teacher_name in c.node.metadata.get("teacher", ""):
                    course_chunks.append(c)
        
        # 🔑 【關鍵修復】從 query 或 expanded queries 補救 teacher_name (防幻覺或暱稱查詢填了 null)
        if not teacher_name:
            all_q = [query] + (route_result.search_queries if route_result else [])
            for pn in prof_chunks_by_name.keys():
                if pn != "未知教授" and any(pn in q for q in all_q):
                    teacher_name = pn
                    logger.info(f"  🔍 從查詢擴充補救 teacher_name: {teacher_name}")
                    break

        # 🔑 【通用修復】從課程資料反查教授名（處理「教我們XX的老師」這類間接查詢）
        # 如果 Reranker 已經直接命中了教授檔案（例如因為專長=物聯網），就不應該再去反查單一課程把目標綁死在一個人身上
        if not teacher_name and not prof_chunks_by_name:
            all_q_text = query + " " + " ".join(route_result.search_queries if route_result else [])
            course_kw = route_result.metadata_filters.get("course_name_keyword", "") if route_result else ""
            
            # 先掃 reranked chunks（已排序，優先級高）
            for c in chunks:
                cn = c.node.metadata.get("course_name", "")
                t = c.node.metadata.get("teacher", "")
                if cn and t and (course_kw and course_kw in cn):
                    teacher_name = _re.sub(r"(老師|教授)$", "", t).strip()
                    if "," in teacher_name:
                        teacher_name = teacher_name.split(",")[0].strip()
                    logger.info(f"  🔍 從 reranked chunk 課程「{cn}」反查到單一教授: {teacher_name}")
                    break
            
            # 若 reranked chunks 裡沒找到，掃 all_nodes
            if not teacher_name and all_nodes and course_kw:
                for n in all_nodes:
                    cn = n.metadata.get("course_name", "")
                    t = n.metadata.get("teacher", "")
                    if cn and t and n.metadata.get("section") == "basic_info" and course_kw in cn:
                        teacher_name = _re.sub(r"(老師|教授)$", "", t).strip()
                        if "," in teacher_name:
                            teacher_name = teacher_name.split(",")[0].strip()
                        logger.info(f"  🔍 從 all_nodes 課程「{cn}」反查到教授: {teacher_name}")
                        break

        _seen_node_ids = set()
        for pn_key, pn_chunks in prof_chunks_by_name.items():
            for c in pn_chunks:
                _seen_node_ids.add(c.node.node_id)

        # 🔑 【關鍵修復】Reranker Top-N 可能只選了 1 個教授 chunk，但該教授實際有 5~7 個 chunks（論文被切分）
        # 從完整 all_nodes 補齊同教授的所有 chunks，確保論文資料不遺漏
        if all_nodes and teacher_name:
            
            # DEBUG: 計算 all_nodes 中有多少教授 chunks
            _debug_prof_count = 0
            _debug_match_count = 0
            for n in all_nodes:
                if n.metadata.get("info_type") == "professor_info":
                    _debug_prof_count += 1
                    n_prof = n.metadata.get("professor_name", "")
                    if n_prof and teacher_name in n_prof:
                        _debug_match_count += 1
            logger.info(f"  🔍 Backfill DEBUG: all_nodes 共 {len(all_nodes)} 個, professor_info={_debug_prof_count}, 匹配'{teacher_name}'={_debug_match_count}, 已知 IDs={len(_seen_node_ids)}")
            
            for n in all_nodes:
                if n.node_id in _seen_node_ids:
                    continue
                if n.metadata.get("info_type") == "professor_info":
                    n_prof = n.metadata.get("professor_name", "")
                    if n_prof and teacher_name in n_prof:
                        injected_chunk = RetrievedChunk(
                            node=n, vector_score=1.0, bm25_score=1.0,
                            metadata_score=10.0, source="context_backfill",
                        )
                        if n_prof not in prof_chunks_by_name:
                            prof_chunks_by_name[n_prof] = []
                        prof_chunks_by_name[n_prof].append(injected_chunk)
                        _seen_node_ids.add(n.node_id)
            
            # 記錄補齊結果
            for pn_key in prof_chunks_by_name:
                if teacher_name in pn_key:
                    logger.info(f"  📚 教授 Context Backfill：{pn_key} 共 {len(prof_chunks_by_name[pn_key])} 個 chunks 準備送入 LLM")
        
        # 🔑 【關鍵修復】Reranker Top-N 可能只選了少數教授的 chunk（因單一教授論文佔據名額）
        # 如果使用者沒有明確問哪位老師，我們應該主動從 all_nodes 把系上所有老師的「所有塊」都補進來
        if all_nodes and not teacher_name:
            _seen_profs = set(prof_chunks_by_name.keys())
            for n in all_nodes:
                if n.node_id in _seen_node_ids:
                    continue
                info_type = n.metadata.get("info_type", "")
                if info_type == "professor_info":
                    n_prof = n.metadata.get("professor_name", "")
                    if n_prof and n_prof != "未知教授":
                        from rag.retriever import RetrievedChunk
                        injected_chunk = RetrievedChunk(
                            node=n, vector_score=1.0, bm25_score=1.0,
                            metadata_score=5.0, source="general_prof_backfill",
                        )
                        if n_prof not in prof_chunks_by_name:
                            prof_chunks_by_name[n_prof] = []
                            _seen_profs.add(n_prof)
                        prof_chunks_by_name[n_prof].append(injected_chunk)
                        _seen_node_ids.add(n.node_id)
                elif info_type == "facility_info":
                    from rag.retriever import RetrievedChunk
                    injected_chunk = RetrievedChunk(
                        node=n, vector_score=1.0, bm25_score=1.0,
                        metadata_score=5.0, source="general_facility_backfill",
                    )
                    fac_key = "教學設備與辦公室空間"
                    if fac_key not in prof_chunks_by_name:
                        prof_chunks_by_name[fac_key] = []
                    prof_chunks_by_name[fac_key].append(injected_chunk)
                    _seen_node_ids.add(n.node_id)
            logger.info(f"  📚 系所全體教授 Context Backfill：發現共 {len(_seen_profs)} 位教授 (已完整傾倒所有關聯資料塊，包含教學空間資訊)")

        context_items_override = []
        appended_prof_count = 0
        
        # 找出要送入 context 的目標教授（若有明確人名則送匹配的，否則送所有找到的教授）
        target_profs = []
        if teacher_name:
            for pn in prof_chunks_by_name:
                overlap = sum(1 for ch in teacher_name if ch in pn)
                if overlap >= max(2, int(len(teacher_name) * 0.67)):
                    target_profs.append(pn)
                    break
        
        if not target_profs:
            # 如果沒有指定教授，則將 Reranker 排名高的 + 剛剛全體補齊的教授全部送入
            target_profs = list(prof_chunks_by_name.keys())
            
        for pn in target_profs:
            prof_content_lines = []
            prof_meta = prof_chunks_by_name[pn][0].node.metadata
            
            # 放寬教授 chunk 限制，讓高產出教授（如柯志亨有 30+ 個 chunks）能將完整論文與實驗室資訊送入極大 Context 的 Gemini 裡
            for c in prof_chunks_by_name[pn][:100]:
                prof_content_lines.append(c.node.get_content())
                
            merged_text = "\n".join(prof_content_lines)
            
            # 去除重複的標頭語
            lines = merged_text.split("\n")
            seen_lines = set()
            deduped_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and stripped not in seen_lines:
                    seen_lines.add(stripped)
                    deduped_lines.append(line)
            clean_text = "\n".join(deduped_lines)
            dept = prof_meta.get("department", "")
            
            context_items_override.append(
                f"---\n"
                f"📄 資料 {appended_prof_count + 1}：👨‍🏫 {pn} 教授完整資訊\n"
                f"系所: {dept}\n"
                f"{clean_text}\n"
                f"---"
            )
            sources.append({
                "course": f"👨‍🏫 {pn}",
                "section": "professor_info",
                "teacher": pn,
                "score": 10.0 - appended_prof_count,
                "source_file": prof_meta.get("source_file", ""),
            })
            appended_prof_count += 1
        
        # 附加最多 1 個系所 chunk
        for c in dept_chunks[:1]:
            meta = c.node.metadata
            label = f"📋 {meta.get('category', '系所資訊')}"
            context_items_override.append(
                f"---\n📄 {label}\n{c.node.get_content()}\n---"
            )
        
        seen_courses = set()
        course_added = 0
        
        # 從 router 抽取出預期的年級與學期（預設為全域設定或 Router 提供的年份）
        filter_year = route_result.metadata_filters.get("academic_year") if route_result else None
        filter_sem = route_result.metadata_filters.get("semester") if route_result else None
        
        if all_nodes is not None and target_profs:
            # 直接從完整資料庫撈出這位(些)教授近期開授的「所有」課程 (限制最多顯示25門以防爆量)
            for n in all_nodes:
                if "course_name" in n.metadata and n.metadata.get("section") == "basic_info":
                    # 確保過濾過時或不是該學期的課程（例外：如果路由器明確指定 null/跨學期查詢，則不阻擋）
                    if filter_year and filter_year != "__EXPLICIT_NULL__" and n.metadata.get("academic_year") != filter_year:
                        continue
                    if filter_sem and filter_sem != "__EXPLICIT_NULL__" and n.metadata.get("semester") != filter_sem:
                        continue
                        
                    teacher = n.metadata.get("teacher", "")
                    for pn in target_profs:
                        if pn in teacher:
                            cn = n.metadata.get("course_name", "")
                            if cn not in seen_courses:
                                seen_courses.add(cn)
                                context_items_override.append(
                                    f"---\n📄 授課課程：{cn} (教室: {n.metadata.get('classroom', '未註明教室')} | 時間: {n.metadata.get('schedule', '未註明時間')} | 必選修: {n.metadata.get('required_or_elective', '?')})\n{n.get_content()}\n---"
                                )
                                sources.append({
                                    "course": cn, "section": "basic_info",
                                    "teacher": teacher, "score": 8.0,
                                    "source_file": n.metadata.get("source_file", ""),
                                })
                                course_added += 1
                                if course_added >= 25:
                                    break
                            break
                if course_added >= 25:
                    break
        else:
            # 兼容舊版邏輯 (只從 Reranker 過濾後的 chunks 中提取)
            for c in course_chunks:
                cn = c.node.metadata.get("course_name", "")
                sec = c.node.metadata.get("section", "")
                if cn not in seen_courses and sec == "basic_info":
                    seen_courses.add(cn)
                    teacher = c.node.metadata.get("teacher", "?")
                    context_items_override.append(
                        f"---\n📄 授課課程：{cn} (教室: {c.node.metadata.get('classroom', '未註明教室')} | 時間: {c.node.metadata.get('schedule', '未註明時間')} | 必選修: {c.node.metadata.get('required_or_elective', '?')})\n{c.node.get_content()}\n---"
                    )
                    sources.append({
                        "course": cn, "section": sec,
                        "teacher": teacher, "score": c.final_score,
                        "source_file": c.node.metadata.get("source_file", ""),
                    })
                    course_added += 1
                    if course_added >= 5:
                        break
        
        if context_items_override:
            context_items = context_items_override
            logger.info(f"  🎯 教授 Context 合併：打包了 {appended_prof_count} 位教授的資料 + {len(dept_chunks[:1])} 系所 + {len(seen_courses)} 課程")
            # 跳過下面的 for loop
            chunks = []

    for i, chunk in enumerate(chunks):
        meta = chunk.node.metadata
        info_type = meta.get("info_type", "")
        
        # 🆕 教授/系所資訊用專屬格式（不套用課程格式）
        if info_type in ("professor_info", "dept_intro", "career_info", "student_union", "dept_news", "dept_general", "facility_info"):
            category = meta.get("category", "系所資訊")
            prof_name = meta.get("professor_name", "")
            dept = meta.get("department", "")
            label = f"👨‍🏫 {prof_name} 教授資訊" if prof_name else f"📋 {category}"
            
            context_items.append(
                f"---\n"
                f"📄 資料 {i+1}：{label}\n"
                f"系所: {dept}\n"
                f"{chunk.node.get_content()}\n"
                f"---"
            )
            sources.append({
                "course_name": label,
                "section": category,
                "teacher": prof_name,
                "score": chunk.final_score,
                "source_file": meta.get("source_file", ""),
            })
            continue

        # 一般課程 chunk 格式
        course_name = meta.get("course_name", "未知課程")
        raw_section = meta.get("section", "未知區段")
        teacher = meta.get("teacher", "")

        ch_section = section_translator.get(raw_section, raw_section)
        classroom_info = meta.get("classroom", "未註明教室")

        context_items.append(
            f"---\n"
            f"📄 資料 {i+1}：【{course_name}】{ch_section}\n"
            f"學期: {meta.get('academic_year', '?')}-{meta.get('semester', '?')} | 教室: {classroom_info} | 老師: {teacher} | {meta.get('credits', '?')}學分 | {meta.get('required_or_elective', '?')} | 年級: {meta.get('grade', '?')} | {meta.get('schedule', '?')}\n"
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
        course_name = chunk.node.metadata.get("course_name")
        if course_name:
            unique_courses.add(course_name)
    course_count = len(unique_courses)

    # ── 根據意圖動態選擇 Prompt ──
    intent = route_result.query_type if route_result else "course_info"
    current_term = f"{config.CURRENT_ACADEMIC_YEAR}-{config.CURRENT_SEMESTER}"
    
    if intent == "professor_info":
        system_prompt = SYSTEM_RULES_PROFESSOR_PROMPT
        user_prompt = USER_CONTEXT_PROFESSOR_PROMPT.format(
            context=context,
            question=safe_query,
        )
    else:
        system_prompt = SYSTEM_RULES_PROMPT
        user_prompt = USER_CONTEXT_PROMPT.format(
            context=context,
            question=safe_query,
            course_count=course_count,
            current_term=current_term
        )

    # ── 提取並注入使用者身分（解決「不知道年級」的笨問題，並推動極致個人化） ──
    user_info_str = ""
    if user_profile:
        dept = user_profile.get("department", "未知系所")
        grade = user_profile.get("grade", "未知年級")
        if dept != "未知系所" or grade != "未知年級":
            user_info_str = f"👤【提問學生身分資料庫】：這位正在發問的學弟妹是【{dept}】的【{grade}年級】學生！\n✨ 隱藏指導原則：請務必在你的總結建議中，根據對方的科系與年級，給出強烈「個人化（Personalized）」的專屬修課建議、鼓勵或資源盤點！讓對話充滿溫度且像是為他量身打造！\n\n"

    # ── 注入學生個人課表（讓 LLM 能做時間衝突判斷與空堂推薦） ──
    if discord_id:
        schedule_text = _format_schedule_for_llm(discord_id)
        if schedule_text:
            user_info_str += f"{schedule_text}\n\n"
            logger.info("  📅 已注入學生個人課表至 LLM Prompt（啟用課表感知推薦）")

        # ── 注入學生個人成績與畢業學分進度（讓 LLM 能判斷畢業門檻與修課狀況） ──
        try:
            from tools.transcript_manager import get_transcript_context_for_llm
            transcript_text = get_transcript_context_for_llm(discord_id, query)
            if transcript_text:
                user_info_str += f"{transcript_text}\n\n"
                logger.info("  📊 已注入學生個人歷年成績進度至 LLM Prompt（啟用學分與畢業門檻感知）")
        except Exception as e:
            logger.warning(f"⚠️ 讀取學生成績單失敗（不影響回答）：{e}")

    if user_info_str:
        user_info_str += "====================\n\n"

    # ── 將 messages 組裝 ──
    history_str = ""
    if memory and memory.history:
        history_str = f"📚【前情提要 / 對話歷史】：\n{memory.get_formatted_history()}\n====================\n\n"

    # 【關鍵修復】將 System 規則與 User 資料合併為單一 user message
    # 這是為了防止模型在某些硬體或設定下直接忽略 system role
    combined_prompt = f"{system_prompt}\n\n====================\n\n{history_str}{user_info_str}{user_prompt}"

    # ── 動態 thinkingLevel：職涯規劃用 high，其餘用 medium ──
    thinking_level = "high" if is_career_planning else "medium"
    
    logger.info(f"  🧠 Thinking Level: {thinking_level} (intent={intent}, career={is_career_planning})")

    # ── 呼叫 Gemini API（含 429 自動降級） ──
    try:
        logger.info(f"🤖 呼叫 Gemini 生成回答 (傳送 {len(combined_prompt)} 字元)...")
        answer = call_gemini_with_fallback(
            combined_prompt,
            thinking=thinking_level,
        )
        # 【防重複】LLM 偶爾會把同一個回答輸出兩遍
        answer = _remove_duplicate_blocks(answer)
        logger.info(f"✅ LLM 回答生成完成（{len(answer)} 字）")

    except GeminiAPIError as e:
        logger.error(f"❌ LLM 回答生成失敗：{e}")
        answer = f"抱歉，生成回答時發生錯誤：{str(e)}"

    return AnswerResult(
        answer=answer,
        sources=sources,
        query=query,
    )


# ═══════════════════════════════════════════════════════════════
# 📦 向後相容 re-export（避免破壞現有 import 路徑）
# ═══════════════════════════════════════════════════════════════
# 以下函式已搬遷至 llm/chitchat.py，此處保留 re-export。
from llm.chitchat import generate_chitchat_answer  # noqa: F401
from llm.chitchat import generate_personal_info_answer  # noqa: F401
from llm.chitchat import generate_web_search_answer  # noqa: F401


# ═══════════════════════════════════════════════════════════════
# 📋 來源格式化
# ═══════════════════════════════════════════════════════════════

def format_sources(sources: list[dict]) -> str:
    """格式化來源資訊為顯示用字串。

    Args:
        sources: 來源列表。

    Returns:
        格式化的來源字串。
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

