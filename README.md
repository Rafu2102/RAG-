# 🎓 NQU 校園智慧助理機器人 (Campus AI Assistant)

> **國立金門大學資訊工程學系 · Ultimate Agentic RAG 智慧問答系統**
> *最後更新時間：2026-03-31*

基於 **Agentic Intent-Driven Hybrid RAG（代理式意圖驅動混合檢索增強生成）** 架構的超級校園生態系助理。全面升級採用最新的 **Gemini 3.1 Pro (主大腦)** 與 **Gemini Flash Lite (路由/決策)** 雙備援架構。搭載獨創的 **5-Step 自我修正思考鏈 (CoT)**、無上限的**全景上下文回填 (Context Backfill)** 技術，以及**原生語意職涯跨域探索**能力，提供低延遲、絕對零幻覺、高資安防護的校園問答與 Google 行事曆代管服務。

---

## 📋 目錄

- [系統特色](#-系統特色)
- [核心亮點：Agentic RAG 重大突破](#-核心亮點agentic-rag-重大突破)
- [資安與效能防護](#-資安與效能防護)
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
| 🧠 **Agentic 雙腦架構** | **Gemini 3.1 Pro** 負責極深度的長文本分析與回答，**Flash Lite** 負責快決策、5 步 CoT 路由規劃、意圖判斷與防禦攔截。 |
| 🔮 **5-Step CoT 路由** | 產生搜尋策略前，強制五階段自我詰問（意圖判定 → 條件盤點 → 跨域特判 → 防幻覺警告 → Step-Back 擴充），消滅搜尋誤判。 |
| 🔀 **全景 Context Backfill** | 突破 RAG Top-K 限制！涉及系所級別查詢時，繞過向量檢索，將高達 225KB+ 的全系教授簡歷、實驗室與設備地圖一次性「硬派注入」主大腦。 |
| 🌟 **Native Semantic Routing**| 棄用死板關鍵字，交由 LLM 透過語意原生輸出 `is_career_planning` 布林值，實現真正的智慧「跨域課程」探索解鎖。 |
| 🔀 **Intent-Driven Lookup** | 首創「意圖驅動注入」，當偵測為特定意圖查詢時，直接繞過傳統 RAG 的稀釋效應，並行載入完整背景設定，保證最高精準度。 |
| 🔀 **Multi-query RAG** | 一個問題生成 3 個搜尋查詢，提高檢索覆蓋率 |
| 🔗 **Hybrid Search** | Vector（語意）+ BM25（關鍵字）雙重搜尋，**並行 Embedding + 並行檢索** |
| 🏷️ **Metadata Filtering** | **六維度嚴格匹配**：系所/年級/教師/課程類型/學年度學期/必選修 |
| 🛡️ **Zero-Hit 嚴格防爆** | Hard Filter 物理捨棄不符條件資料（含系所嚴格匹配），若無資料直接攔截，**實現零幻覺** |
| 📊 **RRF 融合公式** | `final_score = α·RRF(Vector) + β·RRF(BM25) + γ·Metadata` |
| 🔄 **Cross-Encoder Reranker** | bge-reranker-large 精細重排 + **甲乙班去重偏好**，Top-30 → Top-8，GPU batch=16 |
| 📅 **多學期動態支援** | 自動檢測最新學期，支援 `114上`、`114年第1學期` 等口語化時間查詢 |
| 🔗 **One-shot Router+Rewrite** | 合併路由分類與查詢改寫為**單次 LLM 呼叫**，省去重複載入開銷 |
| 📅 **Google Calendar Agent** | 完整 CRUD 行事曆代理 — 支援新增/刪除/列出/修改，含週期排課、自訂事件、學校行事曆、時間定位搜尋 |
| 📸 **Gemini Vision 課表萃取** | 支援學生直接將「選課系統截圖」發送給機器人，由 Gemini 視覺模型一秒解析複雜的 NQU 節次、授課教師與教室，並且自動轉為 JSON 實體直接循環排入 Google 日曆。 |
| 🛡️ **安全行事曆防呆** | 賦予 Agent 行事曆**移除權限**，並透過嚴格的所有權 (Ownership) `source` 標籤比對，確保**絕對不誤刪**使用者私人事件 |
| 🤖 **Agentic Bypass 高速通道** | 偵測為閒聊、刪除事件、或自訂行程時，直接從主流程**短路攔截 (Bypass)**，省去神經網路檢索運算，回應速度小於 5 秒 |
| 🧠 **VRAM 死亡交叉防護** | 8B `keep_alive=0` 意圖解析後立即卸載 + 3B 輕量任務 + Pipeline 後 `gc.collect()` + `torch.cuda.empty_cache()` |
| 📜 **統一格式輸出** | 單一課程列表格式，杜絕 LLM 重複輸出和幻覺課程 |
| 🧩 **智慧區段感知 Chunking** | 短區段（≤512 字）保持完整不切；僅超長區段啟動 SentenceSplitter |
| ⚡ **GPU 加速 (CUDA)** | 自動偵測 GPU (PyTorch)，Reranker batch=16 壓榨 8GB VRAM |
| 🇹🇼 **繁中在地化與同義詞拓撲** | Regex 解碼器 + 口語翻譯蒟蒻 (禮拜二→星期二，加退選→停修)，另於 System Prompt 動態硬性注入絕對台灣時區與星期，使 相對時間 (如:下週二) 推算 100% 精準 |
| 🔒 **必選修智慧過濾** | 自動偵測疑問句（「是必修嗎？」），避免誤設篩選條件 |

---

## 🌟 核心亮點：Agentic RAG 重大突破

在最新世代的架構中，本系統從傳統硬編碼 (Hard-coded) 的 RAG 過渡到了**具有能動性 (Agentic) 的 LLM 驅動系統**：

1. **雙重 Few-Shot 陷阱防護**：賦予 LLM 解析「實體豁免陷阱（單查教授但不限科系）」與「跨域探索陷阱（商管學生想學寫程式）」的精準判斷力，自動決定何時應收緊過濾器，何時應放寬到全校搜尋。
2. **終極設備與師資注入 (Facility & Professor Injection)**：打破傳統 RAG「相關度低就被拋棄」的缺點。對教授的研究室、專業等廣泛查詢，系統會直接召喚系所底層的 `facility_info`（教學設備與空間文件）連同所有教授的履歷，全數傾倒入百萬 Token 級別的語境中盲測分析。
3. **消除 Python-Side 判斷瓶頸**：大幅刪減 Python 端的 `any(kw in ...)` 單詞命中漏洞，將所有決策權利與護城河驗證上繳至 Gemini 核心，達到「真正的降維打擊」。

---

## 🛡️ 資安與效能防護

本專案於核心層面實作了企業級的安全防護與資源管理機制：

1. **記憶體防護 (Memory Leak Prevention)**：實作手寫 `TTLMemoryCache` (LRU Cache)，頻道在預定時間無活躍後會自動釋放對話記憶體，解決無限增長的 RAM 問題。
2. **併發控制防呆 (Race Condition Fixes)**：GPU Semaphore 等待佇列引入嚴謹的 `try...finally` 搭配 Atomic 自增機制，確保排隊數字永不同步錯誤。
3. **安全限速 (Rate Limiting)**：為 `/ask`, `/add_calendar` 等指令加上了 `@app_commands.checks.cooldown(1, 10)`，限制每 10 秒只能送出一次請求，防禦洗版與 DoS。
4. **Prompt Injection 防範**：於字串層面轉義 XML 標籤 `< >`，並將使用者輸入框定於 `<user_question>` 標籤內，強制 LLM 忽略越權指令。
5. **例外處理遮罩 (Exception Masking)**：阻擋 Raw Stack Trace 噴出至 Discord 頻道，將內部 API 與資料庫報錯轉化為友善的「系統忙碌」提示，詳細錯誤僅存於伺服器 Log。

---

## 🏗️ 系統架構

```text
使用者提問
    │
    ▼
┌───────────────────────────────┐
│  Step 1: 5-Step Agentic Router│  ← 路由式思考鏈 (Gemini Flash Lite)
│  (query_router.py)            │     JSON Schema 強制輸出格式
│  · 五步思考 (CoT) 防呆護城河  │     is_career_planning 原生偵測
│  · Multi-query 擴充改寫 (×3)  │     實體豁免與職涯跨域陷阱判斷
│  · 閒聊/行事曆旁路短路判定    │
└──────────┬───────────┬────────┘
           │           │
           │           ▼
           │  ┌───────────────────────────────┐
           │  │  Agentic Bypass 高速通道      │
           │  │  (llm_calendar.py / answer)   │
           │  │  · 閒聊極速對答 (<1.5s)       │
           │  │  · 行事曆意圖精準解析         │
           │  │  · N-Type/夜間課表時間定位    │
           │  │  · 未註冊用戶防爆安全攔截     │
           │  └───────────────┬───────────────┘
           ▼                  │
┌─────────────────────────┐   │
│  Step 2: Hybrid Retrieve│   │
│  (retriever.py)         │   │
│  · Vector + BM25 並行   │   │
│  · RRF Fusion + 嚴格過濾│   │
│  · dept_short Hard Filter│   │
└──────────┬──────────────┘   │
           ▼                  │
┌─────────────────────────┐   │
│  Step 3: Intent Inject  │   │
│  & Context Backfill     │   │
│  · 偵測教授/系所大哉問意圖│   │
│  · 掛載 225KB+ 教學設備地圖│   │
│    與全系師資履歷入候選池  │   │
└──────────┬──────────────┘   │
           ▼                  │
┌─────────────────────────┐   │
│  Step 4: Reranker       │   │
│  (reranker.py)          │   │
│  · Cross-Encoder GPU    │   │
│  · 甲乙班去重偏好       │   │
└──────────┬──────────────┘   │
           ▼                  │
┌─────────────────────────┐   │
│  Step 5: Ultimate Answer│   │
│  (llm_answer.py)        │   │
│  · Gemini 3.1 Pro 主大腦│   │
│  · 強推理與超大視窗盲測分析│   │
│  · 統一格式 Markdown 輸出  │   │
└──────────┬──────────────┘   │
           ▼                  ▼
  零幻覺深度回答 / 行事曆 CRUD 執行
```

---

## 🛠️ 技術棧

| 層級 | 技術 | 說明 |
|------|------|------|
| **生成大腦** | **Gemini 3.1 Pro** | 全新升級主核心，負責複雜課程對答推理、225KB+ Context Backfill，以及 **Vision 高精度圖像解構** |
| **路由小腦** | **Gemini Flash Lite** | One-shot CoT 分類、意圖解構、原生職涯規劃偵測與 Schema 強制輸出 |
| **自動化代理** | Google Calendar API | 完整 CRUD、專武級 NQU N-Type (夜間部A/B/C) 時間定位、所有權防呆機制 |
| **Embedding** | multilingual-e5-large (1024 維) | 取代輕量級，提升檢索語意深度，支援並行 ThreadPool (可透過 Ollama 本地化佈署) |
| **Reranker** | BAAI/bge-reranker-large | Cross-Encoder，batch=16，推理後 VRAM GC + **甲乙班去重硬偏好** |
| **Vector Store** | FAISS (IndexFlatIP) | 餘弦相似度 (Cosine Similarity) 快速過濾 |
| **Keyword Search** | BM25Okapi + CKIP Tagger | 深度學習繁中分詞，領域專有名詞保護 (強制教師斷詞) |
| **防禦機制** | Agentic Bypass / TTL Cache / Cooldowns | 低延遲短路攔截 + 防記憶體流失 + 防洗版速率限制 |
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
- **Ollama** (僅作為本地 Embedding 橋樑使用)
- **CUDA** 11.8+ / 12.4 (需安裝 PyTorch 2.6 CUDA 版本以啟用 GPU 加速)

---

## 📦 安裝步驟

### 1. 安裝 Ollama 模型 (針對 Embedding)

即使大語言模型核心已升級為 Gemini API，本地向量計算 (Embedding) 仍依賴 `multilingual-e5-large` 以保護知識庫隱私並省下 API 費用：

```bash
# 安裝 multilingual-e5-large（核心 Embedding 模型）
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

> **注意**：首次啟用行事曆功能時，請確保你有將 Google Cloud 產生的 `credentials.json` 放置於 `tools/data/` 目錄中。系統首次執行行事曆動作時將引導你進行網頁授權並自動在 `tools/data/tokens/` 下產生每位使用者獨立的 Token 檔案。
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

機器人啟動後支援以下互動方式：

| 類型 | 操作 |
|------|------|
| **@Tag 問答** | `@NQU_Agent 深度學習是誰教的？` |
| **私訊 (DM)** | 直接私訊機器人，無干擾對話 |
| **`!ask`** | `!ask 星期二有什麼課？` |

#### Discord 斜線指令

| 指令 | 說明 | 權限 |
|------|------|------|
| `/ask` | 🤖 課程問答 | 所有人 |
| `/add_calendar` | 📅 加入行事曆 | 所有人 |
| `/identity_login` | 🔑 身分綁定 + Google 授權 | 所有人 |
| `/join_group` | 🏷️ 透過邀請碼加入群組 | 所有人 |
| `/dcard_search` | 🔍 搜尋 Dcard 教授評價 | 所有人 |
| `/clear` | 🗑️ 清除頻道對話記憶 | 所有人 |
| `/rebuild` | 🔄 重建索引 | 管理員 |
| `/admin_broadcast` | 📢 群體廣播公告 | 管理員 |
| `/admin_dm` | 💬 私訊特定使用者 | 管理員 |
| `/admin_invite` | 🏷️ 批量群組邀請 | 管理員 |
| `/admin_invite_code` | 🔗 產生群組邀請碼 | 管理員 |
| `/upload_schedule` | 📅 上傳課表 JSON | 所有人 |
| `/my_schedule` | 📅 查詢個人課表 | 所有人 |
| `/my_free` | 🆓 查詢空堂節次 | 所有人 |
| `/my_credits` | 📊 查詢本學期學分 | 所有人 |
| `/upload_transcript` | 📜 上傳歷年成績單 JSON | 所有人 |
| `/my_credits_total` | 🎓 查詢畢業學分進度 | 所有人 |
| `/my_gpa` | 📈 查詢歷年 GPA 與排名 | 所有人 |
| `/my_failed` | ⚠️ 查詢不及格或未完成課程 | 所有人 |
| `!sync` | ⚙️ 強制同步指令 | 管理員 |

### 首次執行

首次執行時，程式會：
1. 解析 `data/` 目錄下的所有課程 TXT 檔案
2. 使用 Ollama embedding 建立 FAISS 向量索引
3. 使用 CKIP Tagger 深度學習分詞建立 BM25 關鍵字索引
4. 將索引持久化到 `index_store/` 目錄

後續執行會自動載入已建立的索引，無需重新建立。

---

## 📁 模組詳細說明

### `config.py` — 全域設定

集中管理所有可調參數：
- Gemini API 設定（API 金鑰, 雙腦備援模型變數）
- Ollama 連線設定（URL, Embedding 專用模型名稱）
- Embedding / Reranker 模型名稱
- Chunk 設定（size=512, overlap=50）
- Hybrid Fusion 權重（α=0.5, β=0.3, γ=0.2）
- Metadata 匹配獎勵分數
- 檢索 / Reranker top-k 設定 (RETRIEVER_TOP_K=30, RERANKER_TOP_N=8, BATCH=16)

---

### `bot/` — Discord Bot 模組化架構

機器人被拆分為 7 個獨立模組，`discord_bot.py` 僅作為 32 行的啟動入口。

| 模組 | 職責 |
|------|------|
| `bot/__init__.py` | 共用 client、tree、全域狀態、科系對照表、GPU 佇列 |
| `bot/audit.py` | 監控審計 Log → 伺服器 `#bot_modify` 頻道 |
| `bot/cmd_identity.py` | OAuth 身分註冊流程 UI + `/identity_login` |
| `bot/cmd_admin.py` | `/rebuild`、`/clear`、`/admin_broadcast`、`/admin_dm`、`/admin_invite`、`/admin_invite_code` |
| `bot/cmd_groups.py` | 群組邀請 View (接受/拒絕按鈕) + `/join_group` |
| `bot/cmd_ask.py` | `/ask`、`/add_calendar`、`/dcard_search` |
| `bot/cmd_schedule.py` | `/upload_schedule`、`/my_schedule`、`/my_free`、`/my_credits` |
| `bot/cmd_transcript.py` | `/upload_transcript`、`/my_credits_total`、`/my_gpa`、`/my_failed` |
| `bot/events.py` | `on_ready`（索引載入+指令同步）、`on_message`（@tag/!ask/DM 問答+審計） |

---

### `rag/` — 檢索增強生成模組

#### `rag/data_loader.py` — 資料解析與 Embedding

1. **解析課程 TXT**：正則表達式提取結構化欄位
2. **智慧區段感知 Chunking**：短區段（≤512 字）保持完整不切，僅超長區段啓動 SentenceSplitter
3. **上下文防遺失**：每個 Node 前綴注入課程名稱/教師/年級等核心資訊
4. **Ollama Embedding**：multilingual-e5-large + `passage:/query:` 前綴，batch=64

#### `rag/index_manager.py` — 索引管理中心

- **FAISS 索引**：L2 正規化 + Inner Product = Cosine Similarity，build / save / load
- **BM25 索引**：CKIP Tagger 深度學習繁中分詞 + 領域專有名詞保護（教師名/課程名強制斷詞）
- **Nodes 存取**：pickle 序列化 / 反序列化
- **資料變更偵測**：SHA256 hash manifest，開機自動偵測新增/修改/刪除，觸發自動重建
- **`load_and_index()`**：一鍵載入或重建索引入口

#### `rag/query_router.py` — Agentic 5-Step Router

本系統的最高決策樞紐，負責全域路由分配：
- **5-Step CoT 分析**：強制要求 LLM 輸出 `1_intent_analysis`、`2_condition_check` 等 5 步推理過程，消滅幻覺可能。
- **Native Career 偵測**：讓 LLM 原生輸出 `is_career_planning` 布林值，智能解開科系與學期綁定。
- **雙重 Few-Shot 護城河**：完美教育 LLM 避免掉入「跨域探索陷阱」與「實體豁免陷阱」。
- **JSON Schema 強制鎖定**：保證 100% 格式正確。
- **閒聊短路攔截**：將無意義意圖高速導向 Bypass 旁路。

#### `rag/retriever.py` — Hybrid Retriever + 嚴格過濾防護

核心檢索模組，融合公式：

```
final_score = α × RRF_norm(vector_rank)     （語意相似度）
            + β × RRF_norm(bm25_rank)       （關鍵字匹配）
            + γ × metadata_match_score      （metadata 匹配獎勵）
```

預設權重：α=0.5, β=0.3, γ=0.2

**效能優化**：
- 並行 Embedding：`ThreadPoolExecutor` 同時對 3 個 query 呼叫 Ollama Embed API
- 並行搜尋：FAISS + BM25 同時執行

**Zero-Hit 嚴格防護網**：`dept_short` 加入 Hard Filter 嚴格匹配系所，不匹配的 chunk 遭**物理刪除**。若刪除後結果為空，立即中斷流程。

#### `rag/metadata_filters.py` — Metadata 過濾引擎

- **11 個匹配器**：dept_short、grade、teacher、course_name、section 等獨立匹配函式
- **Handler Registry**：統一註冊表，`{filter_key: handler_fn}` 架構
- **智慧豁免系統**：特定查詢情境自動跳過部分過濾條件
- **Soft-Fallback**：Hard Filter 過濾後若結果為零，自動回退到原始結果

#### `rag/reranker.py` — Cross-Encoder Reranker

- 使用 `BAAI/bge-reranker-large` cross-encoder，batch=16
- 分數融合：`final = sigmoid(rerank) × 0.75 + metadata × 0.25`
- **甲乙班去重偏好**：相同課程的甲/乙班只保留最高分者，預設偏好甲班
- **VRAM GC**：推理後 `gc.collect()` + `torch.cuda.empty_cache()`

---

### `llm/` — 語言模型模組

#### `llm/llm_answer.py` — 終極 Context Backfill 黑科技

- **Context Backfill (上下文全景回填)**：系統極大亮點。偵測為教授/系所大哉問時，無條件把所有的老師履歷、教學設備地圖 `facility_info` 全部灌入 Gemini 3.1 Pro 超大 Context，暴力盲測，破除 RAG Top-K 殘缺限制。
- **統一格式 Prompt**：嚴謹的人設與排版格式，杜絕重複輸出與虛構課程。
- **極速旁路閒聊**：判定為閒聊時 Bypass，1 秒回覆不佔算力。

#### `llm/llm_calendar.py` — NQU 專武行事曆代理

專為金門大學高度客製化的時間翻譯引擎：
- 完全相容 `N`、`A`、`B`、`C` 等夜間部與碩專班節次對照表。

| intent_type | 說明 | 範例 |
|---|---|---|
| `custom_event` | 使用者已給時間 | 「明天九點開會」 |
| `weekly_course` | 每週課程排課 | 「把線性代數加到日曆」 |
| `course_schedule_event` | 課程進度表活動 | 「微積分期中考加到行事曆」 |
| `academic_event` | 學校行政事件 | 「加退選加到行事曆」 |

---

### `tools/` — 工具模組

| 模組 | 職責 |
|------|------|
| `tools/auth.py` | Google OAuth 授權、Token 管理、使用者身分檔案、廣播名單篩選 |
| `tools/calendar_api.py` | Google Calendar CRUD (新增/刪除/修改/列出)、18 週排課、時間定位搜尋 |
| `tools/group_manager.py` | 群組標籤操作、`groups.json` 資料庫、邀請碼產生 |
| `tools/calendar_tool.py` | 相容性 shim — 重新匯出上述三個模組的函式 |
| `tools/search_event_tool.py` | 學校行事曆檢索與同義詞拓撲 |
| `tools/dcard_search_tool.py` | Dcard 金門大學版教授評價搜尋 |
| `tools/schedule_manager.py` | 個人課表資料存取、空堂計算、學期學分統計 |
| `tools/transcript_manager.py` | 歷年成績單解析、GPA 計算、畢業學分比對與不及格警告 |
| `tools/data/` | 資料目錄 — `credentials.json`、`groups.json`、`tokens/` |

### `main.py` — 主 Pipeline

CLI 互動介面，串接所有模組：
1. 載入索引 → 2. 接收問題 → 3. **合併 Router+Rewrite** →
   *(Agentic Bypass：閒聊 / 非發散型行事曆事件直接攔截回應跳出)*
4. 並行 Hybrid Retrieve → 5. Rerank → 6. LLM Answer/Calendar Action → 7. 顯示回答

每次 pipeline 結束後強制 VRAM GC，每個步驟都有計時器。

---

## 🔄 Pipeline 流程詳解

### Step 1: Agentic 5-Step Router (Gemini Flash Lite)

```
輸入：「教我們線代的老師實驗室在哪？」
五步推理 (CoT)：
  1. 意圖判定: 教授研究室查詢 (professor_info)。
  2. 條件盤點: 未明指姓名，透過「線性代數」追尋。
  ...
  5. 擴充建議: "線性代數 授課教師 教學設備 研究室位置"
輸出：
  type = professor_info
  filters = {course_name_keyword: "線性代數"}
  is_chitchat = false
```

### Step 2: Hybrid Retriever (並行檢索)

```
並行 Embedding (3 queries → ThreadPool)
  → 並行 FAISS Vector Search (Top-30) + BM25 Search (Top-30)
  → RRF Fusion + Metadata 絕對命中加權
  → Hard Filter（必選修/系所強制剃除）
```

### Step 3: Intent Injection & Context Backfill (全景回填)

```
偵測為「全系教授與資源普查」意圖：
  → 強勢提取所有 `professor_info`
  → 追加提取系所地圖 `facility_info` (教學設備與空間文件)
  → 打包 >225KB 的精純資料進入保送名單，無視原有 30 名額。
```

### Step 4: Reranker 重排序

```
Top-N 候選陣列
  → bge-reranker-large (GPU batch=16)
  → 分數融合 + 甲乙班去重偏好
  → VRAM GC 釋放
```

### Step 5: Ultimate LLM Answer (Gemini 3.1 Pro)

```
打包的龐大教授群資訊集 + Rerank 後的精簡課程快取 + 使用者發問
  → 進入 Gemini 3.1 Pro 深思視窗
  → 進行跨節點統整（如：將「線性代數教師」比對至「該教師辦公室位置」）
  → 輸出精準回答
```

---

## 💬 範例問答 (Agentic 展示)

### 範例 1：精準課程檢索與實體豁免

```text
🧑‍🎓 你：深度學習是誰教的？

🤖 助理：
📘 **深度學習** (Deep Learning)
- 👨‍🏫 授課教師：馮玄明 | 🎓 學分：3 | 🏷️ 選修
- ⏰ 時間地點：星期三 5~7節 @ E302
- 💡 帶領研究生深入了解最新的 AI 模型與實作
```

### 範例 2：跨域職涯探索 (Native Career Plan)

```text
🧑‍🎓 你：我大一剛進來，但以後其實想走商管理財，學校有什麼課適合我？

🤖 助理 (內部 CoT 狀態)：
  [✓] 觸發 is_career_planning = true，解除 `資工系` 與 `大一` 的硬核枷鎖。
  [✓] 擴大跨域搜尋：商管、投資、理財、通識。

🤖 助理：
雖然你是資工系的新生，但學校有非常多跨領域的資源可以利用喔！以下是為您推薦的商管相關課程：

📘 **投資理財與生活** (通識)
- 👨‍🏫 授課教師：李某某 | 🎓 學分：2 | 🏷️ 選修
- 💡 建立基礎財務知識與投資觀念，非常適合非商科學生入門！

📘 **經濟學原理** (企管系)
- 👨‍🏫 授課教師：王某某 | 🎓 學分：3 | 🏷️ 必修
- 💡 若你有興趣，也可以考慮去企管系旁聽或選修這門基礎課程。
```

### 範例 3：全景 Context Backfill (空間與師資對應)

```text
🧑‍🎓 你：教我們線代的老師實驗室在哪？有聯絡方式嗎？

🤖 助理 (內部運算)：
  [✓] 擷取到 `教授/研究室` 意圖。
  [✓] 啟動 Context Backfill：滿血載入 225KB「全系教授履歷」+「教學設備與空間.txt」。
  [✓] 交由 Gemini 3.1 Pro 盲測比對。

🤖 助理：
教你們**線性代數**的老師是**趙于翔**教授！
📍 **研究室位置**：工程大樓 E316 (巨量資料與醫學影像處理實驗室)
📧 **電子信箱**：yxc@nqu.edu.tw
📞 **聯絡電話**：082-313500 轉 3500

如果需要找教授討論，建議可以先發 Email 預約時間喔！
```

### 範例 4：NQU 特化行事曆 (N/A/B/C 夜間部支援)

```text
🧑‍🎓 你：幫我把微積分加到日曆，每個禮拜二的 N, A, B 節！

🤖 助理 (內部解析)：
  [✓] 意圖判定：weekly_course
  [✓] 時間定位器：辨識為夜間部 N節 (17:30)、A節 (18:25)、B節 (19:20)
  
🤖 助理：
✅ 已成功將課程加入 Google Calendar，為您設定為每週重複 (共18週)！
📌 標題：微積分 | 🕒 逢週二 17:30 ~ 20:10
```

### 範例 5：Gemini Vision 影像辨識排課 (超實用功能🔥)

```text
🧑‍🎓 你：[上傳了一張金大選課系統的圖片截圖] 幫我把這些課都加到行事曆！

🤖 助理 (內部解析)：
  [✓] 偵測到 Discord 圖片附加檔案。
  [✓] 提交圖片給 Gemini 3.1 Pro Vision 模型，執行 Zero-shot 萃取任務。
  [✓] 解析出四門課程的完整 JSON 結構（含名稱、教師、星期、節次）。

🤖 助理：
📝 正在分析您的課表圖片... 已萃取出 4 門課：
1. 資料結構 (馮玄明) - 星期一 2~4 節
2. 系統程式 (陳鍾誠) - 星期三 2~4 節
3. 體育 (李老師) - 星期二 5~6 節
4. 機器學習 (馮玄明) - 星期四 7~9 節

✅ 已經為您全部建立 18 週循環排課至 Google Calendar！
```

### 範例 6：極速閒聊 Bypass

```text
🧑‍🎓 你：謝謝你！你真的很厲害！

🤖 助理 (內部狀態)：
  [✓] Flash Lite 偵測無檢索需求，直接觸發 Bypass (< 1.5s)。
  
🤖 助理：
謝謝你的誇獎！身為金大的校園助理，這是我應該做的！如果有遇到選課、找教室或是排行事曆的問題，隨時可以 Tag 我喔！😎
```

---

## ⚙️ 設定調整

### Gemini API 模型配置

```python
# config.py
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-3.1-pro"          # 主力：複雜解答、Context Backfill
GEMINI_FAST_MODEL = "gemini-3.1-flash-lite"# 高速：五步 CoT 路由、閒聊 Bypass
```

**架構派工分配**：

| 模組 | 模型等級 | 負責任務 |
|------|------|------|
| `llm_answer.py` | 🔴 Pro | 分析多達 200KB 的回填資料、回答艱深選課策略 |
| `llm_calendar.py` | 🔴 Pro | 日曆增/刪/改查指令的 JSON 結構提取 |
| `query_router.py` | 🟢 Flash Lite | 高速運算 5-Step CoT 分析、擴充搜尋字串 |
| `llm_calendar.py`| 🟢 Flash Lite | 極速解析相對時間（如：下週五 N 節） |
| `llm_answer.py` | 🟢 Flash Lite | 閒聊 Bypass 答覆 |

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
├── config.py                   # 全域設定（α/β/γ 權重、Gemini 雙模配置、自動偵測最新學期）
├── main.py                     # 主 Pipeline（CLI 介面 + 短路防爆攔截 + VRAM GC）
├── discord_bot.py              # Discord Bot 啟動入口（32 行，匯入 bot/ 子模組）
├── utils.py                    # 共用工具函式（smart_split_message 等）
├── nlp_utils.py                # CKIP Tagger 繁中分詞工具
├── requirements.txt            # Python 依賴套件
├── .env                        # DISCORD_BOT_TOKEN + ADMIN_DISCORD_IDS
│
├── bot/                        # 🤖 Discord Bot 模組化架構
│   ├── __init__.py             # 共用 client、tree、全域狀態、科系對照表
│   ├── audit.py                # 監控審計 Log → #bot_modify 頻道
│   ├── cmd_identity.py         # OAuth 身分註冊 + /identity_login
│   ├── cmd_admin.py            # /rebuild, /clear, /admin_broadcast, /admin_dm, /admin_invite*
│   ├── cmd_groups.py           # GroupInviteView + /join_group
│   ├── cmd_ask.py              # /ask, /add_calendar, /dcard_search
│   ├── cmd_schedule.py         # 課表指令 (/upload_schedule, /my_schedule...)
│   ├── cmd_transcript.py       # 成績單與畢業進度指令 (/upload_transcript, /my_gpa...)
│   └── events.py               # on_ready (索引載入+同步) + on_message (問答+審計)
│
├── rag/                        # 🔍 Agentic 檢索增強生成核心
│   ├── data_loader.py          # 課程/師資 TXT 解析 + Chunking + Ollama Embedding
│   ├── index_manager.py        # FAISS/BM25 雙引擎管理與自動 Reload 機制
│   ├── metadata_filters.py     # 11 組 Hard Filter 特判與實體豁免邏輯
│   ├── query_router.py         # Gemini 5-Step CoT 路由樞紐 (Native Career 判定)
│   ├── retriever.py            # Hybrid RRF 檢索與 Zero-Hit 零幻覺攔截網
│   └── reranker.py             # Cross-Encoder (甲乙班去重偏好) 與 GC
│
├── llm/                        # 🧠 大腦生成模組
│   ├── llm_answer.py           # 終極大腦：處理 225KB+ Context Backfill 與防幻覺輸出
│   ├── llm_calendar.py         # NQU 特化行事曆引擎 (完備支援夜間部 N-Type 節次)
│   └── date_utils.py           # 基層時間轉換與對標物件
│
├── tools/                      # 🛠️ 工具模組
│   ├── auth.py                 # Google OAuth 授權 + Token 管理 + 使用者身分
│   ├── calendar_api.py         # Google Calendar CRUD + 時間定位搜尋 + 18 週排課
│   ├── group_manager.py        # 群組標籤操作 + groups.json 管理 + 邀請碼
│   ├── calendar_tool.py        # 相容性 shim（重新匯出上述三模組）
│   ├── search_event_tool.py    # 學校行事曆檢索 + 同義詞拓撲
│   ├── dcard_search_tool.py    # Dcard 教授評價搜尋
│   ├── schedule_manager.py     # 課表存取與排課解析
│   ├── transcript_manager.py   # 成績單存取與畢業進度驗證
│   ├── events.json             # 學校行事曆靜態檔
│   └── data/                   # 📁 資料目錄
│       ├── credentials.json    # Google API 憑證
│       ├── groups.json         # 群組資料庫
│       ├── users/              # 存放使用者課表 (.json) 與成績單 (_transcript.json) 等個人文件
│       └── tokens/             # 各用戶 OAuth Token (per-user, 自動產生)
│
├── index_store/                # 索引持久化目錄（自動生成）
└── data/                       # 系統 RAG 本地知識庫
    ├── courses/                # 課程資料 (依系所學期分類)
    ├── professors/             # 教授百科目錄
    ├── dept_info/              # 系所資訊目錄
    └── rules/                  # 畢業門檻等獨立規則 (graduation_rules.json)
```

---

## 📄 授權

本專案為國立金門大學資訊工程學系專題研究使用。

---

<p align="center">
  Built with ❤️ using Gemini 3.1 Pro + FAISS + Agentic RAG
</p>
