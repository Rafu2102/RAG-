# 🎓 NQU 校園課程助理機器人

> **國立金門大學資訊工程學系 · Hybrid RAG 智慧問答系統**

基於 **Hybrid RAG（Retrieval-Augmented Generation）** 架構的校園課程助理機器人，使用 Vector Search + BM25 關鍵字搜尋 + Cross-Encoder Reranker，搭配本地端 Llama 3.1 8B 語言模型提供智慧問答服務。

---

## 📋 目錄

- [系統特色](#-系統特色)
- [系統架構](#-系統架構)
- [技術棧](#-技術棧)
- [環境需求](#-環境需求)
- [安裝步驟](#-安裝步驟)
- [使用方法](#-使用方法)
- [模組詳細說明](#-模組詳細說明)
- [Pipeline 流程詳解](#-pipeline-流程詳解)
- [範例問答](#-範例問答)
- [設定調整](#-設定調整)
- [未來擴展方向](#-未來擴展方向)
- [專案結構](#-專案結構)

---

## ✨ 系統特色

| 功能 | 說明 |
|------|------|
| 🔀 **Multi-query RAG** | 一個問題生成 3 個搜尋查詢，提高檢索覆蓋率 |
| 🔗 **Hybrid Search** | Vector（語意）+ BM25（關鍵字）雙重搜尋，**並行 Embedding + 並行檢索** |
| 🏷️ **Metadata Filtering** | **五維度嚴格匹配**：系所/年級/教師/課程類型/學年度學期 |
| 🛡️ **Zero-Hit 嚴格防爆** | Hard Filter 物理捨棄不符條件資料，若無資料直接攔截，**實現零幻覺** |
| 📊 **RRF 融合公式** | `final_score = α·RRF(Vector) + β·RRF(BM25) + γ·Metadata` |
| 🔄 **Cross-Encoder Reranker** | bge-reranker-large 精細重排，Top-30 → Top-8，GPU batch=16 |
| 📅 **多學期動態支援** | 自動檢測最新學期，支援 `114上`、`114年第1學期` 等口語化時間查詢 |
| 🔗 **One-shot Router+Rewrite** | 合併路由分類與查詢改寫為**單次 LLM 呼叫**，省去重複載入開銷 |
| 📅 **Google Calendar Agent** | Llama 3 意圖轉譯，支援排課 (自動建立 18 週循環)、自訂事件 (生日/開會) 定位與學校既定行程寫入 |
| 🛡️ **安全行事曆防呆** | 賦予 Agent 行事曆**移除權限**，並透過嚴格的所有權 (Ownership) `source` 標籤比對，確保**絕對不誤刪**使用者私人事件 |
| 🤖 **Agentic Bypass 高速通道** | 偵測為閒聊、刪除事件、或自訂行程時，直接從主流程**短路攔截 (Bypass)**，省去神經網路檢索運算，回應速度小於 5 秒 |
| 🧠 **VRAM 死亡交叉防護** | 3B `keep_alive=0` 卸載 + Pipeline 後 `gc.collect()` + `torch.cuda.empty_cache()` |
| 📜 **XML 嚴格輸出鎖定** | 採 `<example_format>` 強制模板束縛，絕對拒絕 LLM 說廢話及亂印課表 |
| 🧩 **智慧區段感知 Chunking** | 短區段（≤512 字）保持完整不切；僅超長區段啟動 SentenceSplitter |
| ⚡ **GPU 加速 (CUDA)** | 自動偵測 GPU (PyTorch)，Reranker batch=16 壓榨 8GB VRAM |
| 🇹🇼 **繁中在地化與同義詞拓撲** | Regex 解碼器 + 口語翻譯蒟蒻 (禮拜二→星期二，加退選→停修)，另於 System Prompt 動態硬性注入絕對台灣時區與星期，使 相對時間 (如:下週二) 推算 100% 精準 |

---

## 🏗️ 系統架構

```
使用者提問
    │
    ▼
┌───────────────────────────────┐
│  Step 1: Router + Rewrite     │  ← 合併式單次 LLM 呼叫 (3B)
│  (query_router.py)            │     JSON Schema 強制格式
│  · 問題分類 + Metadata 提取   │     keep_alive=0 VRAM 防護
│  · Multi-query 改寫 (×3)      │
│  · 閒聊/行事曆短路判定        │
└──────────┬───────────┬────────┘
           │           │
           │           ▼
           │  ┌───────────────────────────────┐
           │  │  Agentic Bypass 高速通道      │
           │  │  (llm_calendar.py / answer)   │
           │  │  · 閒聊直接對答 (<1s)         │
           │  │  · 建立自訂/學校日曆 (<5s)    │
           │  │  · 刪除日曆事件 (<5s)         │
           │  └───────────────┬───────────────┘
           ▼                  │
┌─────────────────────────┐   │
│  Step 2: Hybrid Retrieve│   │
│  (retriever.py)         │   │
│  · Vector + BM25 並行   │   │
│  · RRF Fusion + 嚴格過濾│   │
└──────────┬──────────────┘   │
           ▼                  │
┌─────────────────────────┐   │
│  Step 3: Reranker       │   │
│  (reranker.py)          │   │
│  · Cross-Encoder GPU    │   │
└──────────┬──────────────┘   │
           ▼                  │
┌─────────────────────────┐   │
│  Step 4: LLM Action     │   │
│  (llm_answer/calendar)  │   │
│  · Llama 3.1 8B 生成    │   │
│  · Agent 建立排課 (18週)│   │
└──────────┬──────────────┘   │
           ▼                  ▼
    回答 + 來源標註 / 建立/刪除行事曆事件
```

---

## 🛠️ 技術棧

| 層級 | 技術 | 說明 |
|------|------|------|
| **生成大腦** | Llama 3.1 8B (TAIDE-LX Q4_K_M) | 繁中特化，處理進階推理與雙軌排版 |
| **路由小腦** | Llama 3.2 3B (Ollama) | One-shot 分類+改寫，JSON Schema 強制格式，`keep_alive=0` |
| **自動化代理** | Google Calendar API | CRUD 增刪改查、18週週期排課、所有權防呆機制 |
| **Embedding** | multilingual-e5-large (1024 維) | 並行 Embedding (`ThreadPoolExecutor`) |
| **Reranker** | BAAI/bge-reranker-large | Cross-Encoder，batch=16，推理後 VRAM GC |
| **Vector Store** | FAISS (IndexFlatIP) | 高效向量相似度搜尋 |
| **Keyword Search** | BM25Okapi + jieba | 中文關鍵字搜尋，並行執行 |
| **防禦機制** | Agentic Bypass / VRAM GC | 低延遲短路攔截 + VRAM 穩定釋放 |
| **介面** | Rich CLI / Discord.py | 終端機除錯介面與非同步 Discord 機器人 |

---

## 💻 環境需求

### 硬體需求

| 組件 | 最低需求 | 推薦配置 |
|------|---------|---------|
| **GPU** | GTX 1080 (8GB) | RTX 4060 (8GB) |
| **RAM** | 16GB | 32GB |
| **儲存空間** | 10GB | 20GB |

### 軟體需求

- **Python** 3.10+
- **Ollama** (已安裝並載入自訂的 `llama3.1-8k` 模型)
- **CUDA** 11.8+ / 12.4 (需安裝 PyTorch 2.6 CUDA 版本以啟用 GPU 加速)

---

## 📦 安裝步驟

### 1. 安裝 Ollama 模型

```bash
# 安裝 Llama 3.1 8B（生成模型）
ollama pull llama3.1:8b

# 安裝 multilingual-e5-large（Embedding 模型）
ollama pull multilingual-e5-large

# 確認模型已安裝
ollama list
```

### 2. 建立 Python 環境

```bash
# 建立虛擬環境
python -m venv venv

# 啟用虛擬環境（Windows）
.\venv\Scripts\activate

# 啟用虛擬環境（Linux/Mac）
source venv/bin/activate
```

### 3. 安裝 Python 套件

```bash
pip install -r requirements.txt
```

> **注意**：首次啟用行事曆功能時，請確保你有將 Google Cloud 產生的 `credentials.json` 放置於 `tools/` 目錄中。系統首次執行行事曆動作時將引導你進行網頁授權並產生 `token.json`。
> **注意 2**：`sentence-transformers` 會在首次執行時自動從 HuggingFace 下載 `bge-reranker-large` 模型（約 1.3GB）。

### 4. 確認 Ollama 服務運行

```bash
# 啟動 Ollama 服務
ollama serve

# （另開終端）測試 Ollama 是否正常
curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b", "prompt": "Hello"}'
```

---

## 🚀 使用方法

### 方式一：啟動終端機互動介面 (本地預覽測試)

```bash
cd "d:\AI HYBRID"
python main.py
```

| CLI 指令 | 說明 |
|------|------|
| `/quit` | 退出程式 |
| `/rebuild` | 重建索引（資料更新後使用） |
| `/clear` | 清除對話歷史 |
| `/debug` | 切換 Debug 模式（顯示詳細檢索資訊） |

### 方式二：啟動 Discord 機器人 (對外服務)

```bash
cd "d:\AI HYBRID"
# 請確保 .env 檔案中已填寫正確的 DISCORD_BOT_TOKEN
python discord_bot.py
```

機器人啟動後，在 Discord 頻道中可以：
1. 直接 Tag 機器人發問：`@NQU_Agent 深度學習是誰教的？`
2. 透過私人訊息 (DM) 與機器人無干擾對話。
3. 同步斜線指令：`!sync` (僅限管理員)。

### 首次執行

首次執行時，程式會：
1. 解析 `data/` 目錄下的所有課程 TXT 檔案
2. 使用 Ollama embedding 建立 FAISS 向量索引
3. 使用 jieba 分詞建立 BM25 關鍵字索引
4. 將索引持久化到 `index_store/` 目錄

後續執行會自動載入已建立的索引，無需重新建立。

---

## 📁 模組詳細說明

### `config.py` — 全域設定

集中管理所有可調參數：
- Ollama 連線設定（URL, TAIDE-LX 8B model, temperature）
- Embedding / Reranker 模型名稱
- Chunk 設定（size=512, overlap=50）
- Hybrid Fusion 權重（α=0.5, β=0.3, γ=0.2）
- Metadata 匹配獎勵分數
- 檢索 / Reranker top-k 設定 (RETRIEVER_TOP_K=30, RERANKER_TOP_N=8, BATCH=16)

### `data_loader.py` — 資料載入與索引

1. **解析課程 TXT**：正則表達式提取結構化欄位
2. **智慧區段感知 Chunking**：短區段（≤512 字）保持完整不切，僅超長區段啓動 SentenceSplitter
3. **上下文防遺失**：每個 Node 前綴注入課程名稱/教師/年級等核心資訊
4. **Ollama Embedding**：multilingual-e5-large + `passage:/query:` 前綴，batch=64
5. **FAISS 索引**：L2 正規化 + Inner Product = Cosine Similarity
6. **BM25 索引**：jieba 中文分詞 + 停用詞過濾

### `query_rewrite.py` — 查詢改寫（備用）

- **Multi-query RAG**：將一個問題改寫為 3 個不同角度的搜尋查詢
- 已被合併式 `route_and_rewrite()` 取代，作為獨立弌 fallback 保留

### `query_router.py` — 合併式 Router + Rewrite

- **One-shot Router+Rewrite**：單次 3B 模型呼叫同時完成路由分類 + 查詢改寫
- **JSON Schema 強制鎖定**：Ollama Structured Outputs 保證 100% 格式正確
- **閒聊語意陷阱防護**：「打招呼 + 課程問題」強制 `is_chitchat=false`
- **VRAM 死亡交叉防護**：`keep_alive=0` 用完立即卸載 3B 模型
- **規則式路由加強版**：支援口語化翻譯 (大二→二) + 正則表達式自動攔截 `114年第1學期`、`114上` 等時間片語
- **LLM 幻覺防線**：驗證 LLM 提取的 filter 是否真的出現在問題中

### `retriever.py` — Hybrid Retriever + 嚴格過濾防護

核心檢索模組，融合公式：

```
final_score = α × RRF_norm(vector_rank)     （語意相似度）
            + β × RRF_norm(bm25_rank)       （關鍵字匹配）
            + γ × metadata_match_score      （metadata 匹配獎勵）
```

預設權重：α=0.5, β=0.3, γ=0.2

**效能優化**：
- 並行 Embedding：`ThreadPoolExecutor` 同時對 3 個 query 呼叫 Ollama Embed API
- 並行搜尋：FAISS + BM25 同時執行，透過 `vector_search_with_embedding()` 避免重複 API 呼叫

**Zero-Hit 嚴格防護網**：當指定特定教師、學期或課程名稱時，不匹配的 chunk 會遭到**物理刪除**不送入後續流程。若刪除後結果為空，立即中斷流程，**徹底解決 RAG 最常見的無關資料通靈幻覺問題**。

### `reranker.py` — Cross-Encoder Reranker

- 使用 `BAAI/bge-reranker-large` cross-encoder，batch=16
- 分數融合：`final = sigmoid(rerank) × 0.75 + metadata × 0.25`
- 情境加分：時間查詢→basic_info +0.10，專長查詢→objectives/syllabus +0.20
- **VRAM GC**：推理後 `gc.collect()` + `torch.cuda.empty_cache()`

### `llm_answer.py` — 精簡化生成大腦

- **精簡 Prompt**：從 ~1200 字縮減至 ~700 字，節約 ~500 tokens 給實際資料
- **雙軌排版**：自動根據給定數據多寡，切換為詳細介紹或精簡列表
- **閒聊旁路**：使用 3B 快速模型（~1 秒），`keep_alive=0` 不佔 VRAM
- **進階推理授權**：允許從課程內容推斷教授專長與實驗室能力需求

### `llm_calendar.py` & `tools/calendar_tool.py` — 智慧行事曆代理

- **三重意圖分流**：將行事曆意圖精細分為 `weekly_course` (每週課程)、`academic_event` (學校行程) 與 `custom_event` (自訂事件)。
- **18 週自動擴充**：針對課程，自動生成 `RRULE:FREQ=WEEKLY;COUNT=18` 將一整學期課表建置完畢。
- **Agentic 刪除權限**：具備解析並執行 `remove` 動作的能力，搭配 `extendedProperties.private.source == NQU_agent` 所有權驗證，提供最高防護層級的防呆刪除。

### `main.py` — 主 Pipeline

CLI 互動介面，串接所有模組：
1. 載入索引 → 2. 接收問題 → 3. **合併 Router+Rewrite** →
   *(Agentic Bypass：閒聊 / 非發散型行事曆事件直接攔截回應跳出)*
4. 並行 Hybrid Retrieve → 5. Rerank → 6. LLM Answer/Calendar Action → 7. 顯示回答

每次 pipeline 結束後強制 VRAM GC，每個步驟都有計時器。

---

## 🔄 Pipeline 流程詳解

### Step 1: Router + Rewrite（單次 LLM 呼叫）

```
輸入：「深度學習是誰教的？」
輸出：
  type = course_info
  filters = {course_name_keyword: "深度學習"}
  search_queries = ["深度學習是誰教的", "深度學習 課程 授課教師", "深度學習 教授 資工系"]
  is_chitchat = false
```

### Step 2: Hybrid Retriever（並行）

```
並行 Embedding (3 queries → ThreadPool)
  → 並行 FAISS Vector Search (Top-30) + BM25 Search (Top-30)
  → 合併去重
  → Metadata Scoring（department ✓ +0.3, grade ✓ +0.25, ...）
  → RRF Fusion: score = 0.5×V + 0.3×B + 0.2×M
  → Hard Filter（不匹配 -3.0 沈底）
  → Top-30 候選
```

### Step 3: Reranker

```
Top-30 候選
  → bge-reranker-large (GPU batch=16)
  → 分數融合: sigmoid(rerank)×0.75 + metadata×0.25
  → VRAM GC (gc + empty_cache)
  → Top-8 最終結果
```

### Step 4: LLM Answer

```
Top-8 chunks + 對話歷史 + 問題
  → Llama 3.1 8B (TAIDE-LX, 8k Context)
  → 精簡 Prompt (~700 字)
  → 回答 + 來源標註 + VRAM GC
```

---

## 💬 範例問答

### 範例 1：課程資訊查詢

```
🧑‍🎓 你：深度學習是誰教的？

🤖 助理：這門「深度學習」是由馮玄明教授開設的選修課喔！這門課主要是帶領研究生深入了解最新的 AI 模型與實作。如果你有興趣上這門課，我可以再幫你查查它需要什麼先備知識，想了解看看嗎？ (來源：【深度學習】基本資訊)
```

### 範例 2：多課程查詢

```
🧑‍🎓 你：資工系二年級有哪些必修課？

🤖 助理：資工二的必修課總共有以下這幾門喔：
1. **資料結構** (3學分) — 授課老師是馮玄明教授。(來源：【資料結構】基本資訊)  
2. **系統程式** (3學分) — 授課老師是陳鍾誠教授。(來源：【系統程式】基本資訊)
...
```

### 範例 3：Agentic Calendar 行事曆智慧建置 (18週)

```
🧑‍🎓 你：幫我把微積分加到日曆
🤖 助理：📅 意圖判定：weekly_course (目標：微積分)
...(自動 RAG 檢索確認微積分是禮拜一第 2~4 節)...
✅ 已成功將課程加入 Google Calendar，為您設定為**每週重複 (共18週)**！
📌 課程：微積分
🕒 首堂開始：{'dateTime': '2026-03-16T09:10:00'}
🕒 首堂結束：{'dateTime': '2026-03-16T12:00:00'}
🔗 事件連結：https://www.google.com/calendar/event?eid=...
```

### 範例 4：行事曆 RAG 短路攔截 (Bypass)

```
🧑‍🎓 你：2026年5月29是我生日請幫我加到行事曆
🤖 助理：⚡ 行事曆意圖為 custom_event (add)，直接攔截跳過 RAG 檢索！(耗時 < 5s)
✅ 已新增到 Google Calendar
📌 標題：生日
🕒 開始：{'date': '2026-05-29'}
🕒 結束：{'date': '2026-05-29'}
🔗 事件連結：https://...
```

### 範例 5：多輪對話與閒聊

```
🧑‍🎓 你：嗨你好
🤖 助理：哈囉你好呀！近來課業還順利嗎？我是金大資工專屬的校園課程助理。不管你想找好過的選修還是各年級必修都能問我喔！今天想查點什麼呢？

🧑‍🎓 你：那資料結構用什麼教科書？
🤖 助理：我看了一下這門課的資料，資料結構的主要教科書是《Data Structures: A Pseudocode Approach with C》(Gilberg / Forouzan 著) 喔！需要幫你查這門課的配分方式嗎？
```

---

## ⚙️ 設定調整

### Hybrid Fusion 權重

在 `config.py` 中調整：

```python
HYBRID_ALPHA = 0.5   # α — Vector 語意相似度（增大提升語意理解）
HYBRID_BETA = 0.3    # β — BM25 關鍵字（增大提升精確匹配）
HYBRID_GAMMA = 0.2   # γ — Metadata 匹配（增大提升結構化過濾）
```

### Retriever / Reranker 參數

```python
RETRIEVER_TOP_K = 30      # 31門課×6區段≈186 chunks，30 已足夠且降低雜訊
RERANKER_TOP_N = 8        # 精選 8 個高品質 chunk 進入 LLM
RERANKER_BATCH_SIZE = 16  # GPU batch size（8GB VRAM 穩定運行）
```

### Chunk 設定

```python
CHUNK_SIZE = 512          # 讓 5/6 區段保持完整不被切分
CHUNK_OVERLAP = 50        # 超長區段切分時保持上下文連貫
```

---

## 🔮 未來擴展方向

1. **全校課程資料**：擴展到其他科系、通識課程
2. **校園新聞 / 公告**：新增新聞查詢功能（增加 news type filter）
3. **Web UI 前端**：使用 Gradio 或 Streamlit 建立網頁介面
4. **先修課程推薦**：根據知識圖譜推薦修課順序
5. **GraphRAG 整合**：加入 Graph Context Expansion
6. **Fine-tuning**：針對校園場景微調 Llama 模型
7. **API 服務化**：包裝為 FastAPI REST API
8. **多語言支援**：支援英文問答

---

## 📂 專案結構

```
d:\AI HYBRID\
├── config.py               # 全域設定（α/β/γ 權重、模型設定、自動偵測最新學期等）
├── main.py                 # 主 Pipeline（短路防爆攔截 + VRAM GC）
├── discord_bot.py          # Discord Bot 非同步介面 + 實時 sync 指令
├── requirements.txt        # Python 依賴套件
├── README.md               # 本文件
├── .env                    # DISCORD_BOT_TOKEN
├── rag/
│   ├── data_loader.py      # 資料解析 + 多學期遞迴掃描 + FAISS/BM25 索引
│   ├── query_router.py     # 合併式 Router+Rewrite + 時間 Regex 攔截 + VRAM 防護
│   ├── retriever.py        # Hybrid Retriever（嚴格過濾防護 + 零幻覺攔截 + RRF）
│   └── reranker.py         # Cross-Encoder Reranker（batch=16 + VRAM GC）
├── llm/
│   ├── llm_answer.py       # XML 強制模板 + 智慧推薦引擎 + 防進度表洗版
│   └── llm_calendar.py     # Llama 3 意圖轉譯與口語時間精準擷取（Agentic Calendar）
├── tools/
│   ├── calendar_tool.py    # Google Calendar API (CRUD) 與安全所有權檢查
│   ├── search_event_tool.py # 學校行事曆檢索與學生俗稱同義詞拓撲 (events.json)
│   ├── credentials.json    # Google API 憑證 (請自行放上以便授權)
│   ├── token.json          # Google API 授權 Token (自動產生)
│   └── events.json         # 學校行事曆靜態檔
├── index_store/            # 索引持久化目錄（自動生成）
└── data/
    ├── 資工系113學年度第2學期課程資訊/
    │   ├── 深度學習 (Deep Learning).txt
    │   └── ...
    └── 資工系114學年度第1學期課程資訊/
        ├── 程式設計 (Programming).txt
        └── ...
```

---

## 📄 授權

本專案為國立金門大學資訊工程學系專題研究使用。

---

<p align="center">
  Built with ❤️ using Llama 3.1 + LlamaIndex + FAISS + BM25
</p>
