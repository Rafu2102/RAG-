# -*- coding: utf-8 -*-
"""
config.py — 全域設定檔
========================
集中管理所有可調參數，方便統一修改。
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

# =============================================================================
# 📁 路徑設定
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
COURSES_DIR = os.path.join(DATA_DIR, "courses")       # 各系各學期課程 TXT
PROFESSORS_DIR = os.path.join(DATA_DIR, "professors")  # 各系教授資訊
DEPT_INFO_DIR = os.path.join(DATA_DIR, "dept_info")    # 系所簡介/就業/學會/新聞
RULES_DIR = os.path.join(DATA_DIR, "rules")            # 畢業規則 JSON
INDEX_DIR = os.path.join(PROJECT_ROOT, "index_store")
TIMEZONE = "Asia/Taipei"  # 台灣時區（calendar_tool.py 使用）
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
NODES_STORE_PATH = os.path.join(INDEX_DIR, "nodes_store.pkl")

# =============================================================================
# 📅 學期自動判斷（掃描 data/ 資料夾取最新學期）
# =============================================================================
def _detect_current_semester():
    """掃描 data/courses/ 目錄中的子資料夾，取出最新的學年度+學期。
    資料夾命名格式：資工系XXX學年度第Y學期課程資訊
    """
    import re as _re
    courses_path = os.path.join(PROJECT_ROOT, "data", "courses")
    best_year, best_sem = 0, 0
    if os.path.isdir(courses_path):
        for name in os.listdir(courses_path):
            m = _re.search(r"(\d+)學年度第(\d+)學期", name)
            if m:
                y, s = int(m.group(1)), int(m.group(2))
                if (y, s) > (best_year, best_sem):
                    best_year, best_sem = y, s
    if best_year == 0:
        # fallback: 從日期推算
        now = datetime.now()
        month = now.month
        year = now.year
        if 2 <= month <= 7:
            best_year, best_sem = year - 1911 - 1, 2
        elif month >= 8:
            best_year, best_sem = year - 1911, 1
        else:
            best_year, best_sem = year - 1911 - 1, 1
    return best_year, best_sem

CURRENT_ACADEMIC_YEAR, CURRENT_SEMESTER = _detect_current_semester()

# =============================================================================
# 🤖 Gemini / Ollama LLM 設定
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={GEMINI_API_KEY}"
GEMINI_FAST_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={GEMINI_API_KEY}"

# ── Gemini API Timeout 常數（秒）──
GEMINI_PRO_TIMEOUT = 60.0          # Pro 模型預設 timeout（含 thinking 首 token 延遲）
GEMINI_FLASH_TIMEOUT = 15.0        # Flash Lite 模型預設 timeout
GEMINI_OCR_TIMEOUT = 120.0         # OCR 多模態長耗時 timeout（課表/成績單）

# ── Gemini API maxOutputTokens 常數 ──
GEMINI_PRO_MAX_TOKENS = 8192       # Pro 模型：RAG 主回答、OCR 等長文輸出
GEMINI_FLASH_MAX_TOKENS = 2048     # Flash Lite：Router/Rewrite、意圖分類等中等輸出
GEMINI_SHORT_MAX_TOKENS = 1024      # 極短輸出：指代消解、閒聊（一句話）、行事曆包裝

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "Yu-Feng/Llama-3.1-TAIDE-LX-8B-Chat:Q4_K_M"
OLLAMA_FAST_MODEL = "llama3.2:latest"  # 負責輕量級的 JSON Router/Rewrite，極速運算
OLLAMA_TEMPERATURE = 0.1
OLLAMA_REQUEST_TIMEOUT = 300.0
OLLAMA_CONTEXT_WINDOW = 8192             # 嚴格封印 8k，100% GPU 滿血運算
OLLAMA_KEEP_ALIVE = "5m"                   # 改為 5 分鐘後卸載，避免 8B 和 3B 同時卡死 8GB VRAM

# =============================================================================
# 🔤 Embedding 模型設定（Gemini Cloud API）
# =============================================================================
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2-preview"
GEMINI_EMBEDDING_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview"
EMBEDDING_DIMENSION = 3072              # Gemini Embedding 2 最高維度（Matryoshka 支援 128~3072）
EMBEDDING_BATCH_SIZE = 100              # batchEmbedContents API 每次最大 100 筆
EMBEDDING_RATE_LIMIT_RPM = 3000         # Rate limit: 每分鐘最大請求數
EMBEDDING_RATE_LIMIT_TPM = 1_000_000    # Rate limit: 每分鐘最大 token 數
EMBEDDING_MAX_RETRIES = 3               # API 失敗自動重試次數

# =============================================================================
# 🔄 Reranker 模型設定
# =============================================================================
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
RERANKER_TOP_N = 10               # 精選 10 個高品質 chunk（星期二有 9 門課，需足夠容量）
RERANKER_BATCH_SIZE = 16          # [GPU 壓榨] bge-reranker-large ≈1.3GB，8GB VRAM 穩定跑 16 批次

# =============================================================================
# 📄 Chunking 設定
# =============================================================================
# 根據實際資料分析：5/6 區段 ≤512 字元，僅進度表 (avg 826) 需要切分
CHUNK_SIZE = 512                  # 升級：讓 basic_info/objectives/syllabus/grading/textbooks 100% 不被切分
CHUNK_OVERLAP = 50                # 增加重疊，確保進度表切分時保持完整的週次上下文

# =============================================================================
# 🔍 檢索設定
# =============================================================================
RETRIEVER_TOP_K = 30              # 59 門課（跨學期），但 semester filter 會先篩出 ~30，30 chunks 足夠

# Hybrid Fusion 權重：final_score = α*vector + β*BM25 + γ*metadata
HYBRID_ALPHA = 0.5                # α — Vector search 語意相似度權重
HYBRID_BETA = 0.3                 # β — BM25 關鍵字匹配權重
HYBRID_GAMMA = 0.2                # γ — Metadata 匹配獎勵權重

# Metadata 匹配獎勵分數（當 metadata 欄位與 query 匹配時加分）
METADATA_MATCH_SCORES = {
    "department": 0.3,            # 系所匹配
    "grade": 0.25,                # 年級匹配
    "course_type": 0.25,          # 必修/選修匹配
    "teacher": 0.2,               # 教師匹配
}

RRF_K = 60                        # Reciprocal Rank Fusion 常數

# =============================================================================
# 💬 對話記憶設定
# =============================================================================
MEMORY_WINDOW_SIZE = 0                # [暫時停用] 設為 0 停用對話記憶，避免跨題污染。恢復請改回 5

# =============================================================================
# 🔀 Multi-query 設定
# =============================================================================
MULTI_QUERY_COUNT = 3

# =============================================================================
# 📝 Logging 設定
# =============================================================================
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging():
    """初始化 logging 設定"""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


# =============================================================================
# 🎌 國定假日 / 補假日清單（從 events.json 動態載入）
# =============================================================================
def _load_holidays_from_events() -> list[str]:
    """
    從 events.json 中自動提取所有包含「放假」或「補假」關鍵字的日期。
    回傳格式：['2026-02-27', '2026-02-28', '2026-04-03', ...]
    """
    import json
    events_path = os.path.join(DATA_DIR, "events.json")
    holidays = set()
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        for ev in events:
            title = ev.get("title", "")
            if any(kw in title for kw in ["放假", "補假", "調整放假"]):
                start_str = ev.get("start", "")[:10]
                end_str = ev.get("end", "")[:10]
                if start_str:
                    holidays.add(start_str)
                # 若 end > start，代表連續多天放假，逐日展開
                if end_str and end_str > start_str:
                    from datetime import date, timedelta as _td
                    d = date.fromisoformat(start_str)
                    d_end = date.fromisoformat(end_str)
                    while d <= d_end:
                        holidays.add(d.isoformat())
                        d += _td(days=1)
    except Exception:
        pass  # 若檔案不存在或格式錯誤，靜默返回空清單
    return sorted(holidays)

HOLIDAYS: list[str] = _load_holidays_from_events()


def ensure_dirs():
    """確保必要的目錄存在"""
    os.makedirs(INDEX_DIR, exist_ok=True)
