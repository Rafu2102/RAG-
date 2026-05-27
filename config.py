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
    防呆設計：排除包含備份、臨時、backup 關鍵字的子資料夾，並確保為目錄。
    """
    import re as _re
    courses_path = os.path.join(PROJECT_ROOT, "data", "courses")
    best_year, best_sem = 0, 0
    if os.path.isdir(courses_path):
        for name in os.listdir(courses_path):
            # 確保是目錄
            full_path = os.path.join(courses_path, name)
            if not os.path.isdir(full_path):
                continue
            # 排除備份相關關鍵字
            lower_name = name.lower()
            if any(kw in lower_name for kw in ["backup", "bak", "old", "temp", "tmp", "備份", "複製"]):
                continue
            
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
# 🤖 Gemini LLM 設定（已遷移至 Google Cloud Vertex AI ADC 無密鑰安全授權）
# =============================================================================
# 已全面拔除 AI Studio 金鑰與 URL，防止對個人信用卡扣款。
GEMINI_API_KEY = ""  # 保留此空變數以維持 legacy 程式碼讀取時的相容性，防止 AttributeError

# ── Telegram Bot 設定 ──
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ── Gemini API Timeout 常數（秒）──
GEMINI_PRO_TIMEOUT = 240.0         # Pro 模型預設 timeout（大資料 + thinking 模式耗時較長）
GEMINI_FLASH_TIMEOUT = 120.0        # Gemini 3.5 Flash 模型預設 timeout
GEMINI_OCR_TIMEOUT = 240.0         # OCR 多模態長耗時 timeout（課表/成績單）

# ── Gemini API maxOutputTokens 常數 ──
GEMINI_PRO_MAX_TOKENS = 65536       # Pro 模型：RAG 主回答，完美對齊 Gemini 3.5 Flash 物理上限 64K tokens，為 HIGH 思考與詳盡解答釋放極致空間！
GEMINI_FLASH_MAX_TOKENS = 32768     # Gemini 3.5 Flash：Router/Rewrite 等中等輸出，擴大至 32K tokens 確保重寫不截斷
GEMINI_SHORT_MAX_TOKENS = 16384      # 短輸出：指代消解、閒聊、行事曆，擴大至 16K tokens 以備不時之需

# [LEGACY] Ollama 設定——專案已全面遷移至 Gemini API，以下常數僅保留以備未來回退。
# OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_MODEL = "Yu-Feng/Llama-3.1-TAIDE-LX-8B-Chat:Q4_K_M"
# OLLAMA_FAST_MODEL = "llama3.2:latest"
# OLLAMA_TEMPERATURE = 0.1
# OLLAMA_REQUEST_TIMEOUT = 300.0
# OLLAMA_CONTEXT_WINDOW = 8192
# OLLAMA_KEEP_ALIVE = "5m"

# =============================================================================
# 🔤 Embedding 模型設定 (已重構至 GCP Vertex AI)
# =============================================================================
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 3072              # 嚴格保持原本的 3072 維度設定，完全相容原有 FAISS 索引
EMBEDDING_BATCH_SIZE = 100              # batch predict 每次最大 100 筆
EMBEDDING_RATE_LIMIT_RPM = 3000         # Rate limit
EMBEDDING_RATE_LIMIT_TPM = 1_000_000    # Rate limit
EMBEDDING_MAX_RETRIES = 3               # API 失敗自動重試次數

# =============================================================================
# 🔄 Reranker 模型設定
# =============================================================================
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANKER_TOP_N = 10               # 精選 10 個高品質 chunk（星期二有 9 門課，需足夠容量）
RERANKER_BATCH_SIZE = 32          # [GPU 壓榨] bge-reranker-base ≈1.3GB，8GB VRAM 穩定跑 32 批次

# =============================================================================
# 📄 Chunking 設定
# =============================================================================
# 根據實際資料分析：進度表 (avg 826, max ~1200) 若被切分會導致後半段遺失上下文
CHUNK_SIZE = 1536                 # 升級：擴大 Chunk Size，確保 18 週進度表 100% 裝進同一個 Chunk 裡！
CHUNK_OVERLAP = 128               # 重疊字元

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
MEMORY_WINDOW_SIZE = 5                # 設定為 5 輪記憶，以利多輪連續提問（如：對此教授繼續深入追問）

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

# =============================================================================
# 🏢 統一科系與學分配置中心 (Unified Config Registry)
# =============================================================================
DEPT_REGISTRY = {
    "資工系": {
        "full_name": "資訊工程學系",
        "short_name": "資工系",
        "aliases": ["資工", "資訊工程", "CSIE", "資工系"],
        "keywords": ["資工", "資訊工程", "csie"]
    },
    "電機系": {
        "full_name": "電機工程學系",
        "short_name": "電機系",
        "aliases": ["電機", "電機工程", "電機系"],
        "keywords": ["電機", "電機工程"]
    },
    "電子系": {
        "full_name": "電子工程學系",
        "short_name": "電子系",
        "aliases": ["電子", "電子工程", "電子系"],
        "keywords": ["電子", "電子工程"]
    },
    "土木系": {
        "full_name": "土木與工程管理學系",
        "short_name": "土木系",
        "aliases": ["土木", "工程管理", "土木系"],
        "keywords": ["土木", "工程管理"]
    },
    "食品系": {
        "full_name": "食品科學系",
        "short_name": "食品系",
        "aliases": ["食品", "食品科學", "食品系"],
        "keywords": ["食品", "食品科學"]
    },
    "企管系": {
        "full_name": "企業管理學系",
        "short_name": "企管系",
        "aliases": ["企管", "企業管理", "企管系"],
        "keywords": ["企管", "企業管理"]
    },
    "觀光系": {
        "full_name": "觀光管理學系",
        "short_name": "觀光系",
        "aliases": ["觀光", "觀光管理", "觀光系"],
        "keywords": ["觀光", "觀光管理"]
    },
    "運休系": {
        "full_name": "運動與休閒學系",
        "short_name": "運休系",
        "aliases": ["運休", "運動與休閒", "運休系"],
        "keywords": ["運休", "運動與休閒"]
    },
    "工管系": {
        "full_name": "工業工程與管理學系",
        "short_name": "工管系",
        "aliases": ["工管", "工業工程", "工管系"],
        "keywords": ["工管", "工業工程"]
    },
    "資管系": {
        "full_name": "資訊管理學系",
        "short_name": "資管系",
        "aliases": ["資管", "資訊管理", "資管系"],
        "keywords": ["資管", "資訊管理"]
    },
    "國際系": {
        "full_name": "國際暨大陸事務學系",
        "short_name": "國際系",
        "aliases": ["國際", "大陸事務", "國際系"],
        "keywords": ["國際", "大陸事務"]
    },
    "建築系": {
        "full_name": "建築學系",
        "short_name": "建築系",
        "aliases": ["建築", "建築系"],
        "keywords": ["建築"]
    },
    "海邊系": {
        "full_name": "海洋與邊境管理學系",
        "short_name": "海邊系",
        "aliases": ["海邊", "海洋與邊境", "邊境管理", "海邊系", "海巡"],
        "keywords": ["海邊", "海洋與邊境", "邊境管理", "海巡"]
    },
    "應英系": {
        "full_name": "應用英語學系",
        "short_name": "應英系",
        "aliases": ["應英", "應用英語", "應英系"],
        "keywords": ["應英", "應用英語"]
    },
    "華語系": {
        "full_name": "華語文學系",
        "short_name": "華語系",
        "aliases": ["華語", "華語文", "華語系"],
        "keywords": ["華語", "華語文"]
    },
    "都景系": {
        "full_name": "都市計畫與景觀學系",
        "short_name": "都景系",
        "aliases": ["都景", "都市計畫", "景觀", "都景系"],
        "keywords": ["都景", "都市計畫", "景觀"]
    },
    "護理系": {
        "full_name": "護理學系",
        "short_name": "護理系",
        "aliases": ["護理", "護理系"],
        "keywords": ["護理"]
    },
    "長照系": {
        "full_name": "長期照護學系",
        "short_name": "長照系",
        "aliases": ["長照", "長期照護", "長照系"],
        "keywords": ["長照", "長期照護"]
    },
    "社工系": {
        "full_name": "社會工作學系",
        "short_name": "社工系",
        "aliases": ["社工", "社會工作", "社工系"],
        "keywords": ["社工", "社會工作"]
    },
    "通識中心": {
        "full_name": "通識教育中心",
        "short_name": "通識中心",
        "aliases": ["通識", "通識中心"],
        "keywords": ["通識"]
    },
    "日大學體育": {
        "full_name": "體育室",
        "short_name": "日大學體育",
        "aliases": ["體育室", "日大學體育", "共同體育", "體育課", "體育"],
        "keywords": ["日大學體育", "體育室"]
    },
    "機械系": {
        "full_name": "機械工程學系",
        "short_name": "機械系",
        "aliases": ["機械", "機械工程", "機械系"],
        "keywords": ["機械", "機械工程"]
    },
    "化工系": {
        "full_name": "化學工程學系",
        "short_name": "化工系",
        "aliases": ["化工", "化學工程", "化工系"],
        "keywords": ["化工", "化學工程"]
    },
    "網媒所": {
        "full_name": "資訊網路與多媒體研究所",
        "short_name": "網媒所",
        "aliases": ["網媒", "網媒所"],
        "keywords": ["網媒"]
    },
    "語創碩": {
        "full_name": "語文與人文創意應用碩士學位學程",
        "short_name": "語創碩",
        "aliases": ["語文與人文創意", "語創碩", "語創學程"],
        "keywords": ["語文與人文", "語創"]
    }
}

def load_dept_registry():
    """從 data/rules/dept_configs.json 動態熱載入科系配置。
    若檔案存在且格式正確，則會將其與內建的 DEPT_REGISTRY 合併；
    若檔案不存在或損壞，則 fallback 到內置的完整配置。
    """
    import json
    global DEPT_REGISTRY
    json_path = os.path.join(DATA_DIR, "rules", "dept_configs.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                external_config = json.load(f)
            if isinstance(external_config, dict):
                for k, v in external_config.items():
                    if isinstance(v, dict) and "full_name" in v and "short_name" in v:
                        DEPT_REGISTRY[k] = v
                logging.info("⚙️ 成功從 dept_configs.json 熱載入科系配置")
        except Exception as e:
            logging.error(f"⚠️ 載入外部科系配置失敗，已回退至預設配置：{e}")

# 初始化
ensure_dirs()
load_dept_registry()
