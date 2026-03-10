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
| 🤖 **Agentic RAG Bypass** | 3B 小模型辨識意圖 (Chitchat Bypass)，閒聊 ~1 秒直通不佔 RAG 資源 |
| 🧠 **VRAM 死亡交叉防護** | 3B `keep_alive=0` 卸載 + Pipeline 後 `gc.collect()` + `torch.cuda.empty_cache()` |
| 📜 **XML 嚴格輸出鎖定** | 採 `<example_format>` 強制模板束縛，絕對拒絕 LLM 說廢話及亂印課表 |
| 🧩 **智慧區段感知 Chunking** | 短區段（≤512 字）保持完整不切；僅超長區段啟動 SentenceSplitter |
| ⚡ **GPU 加速 (CUDA)** | 自動偵測 GPU (PyTorch)，Reranker batch=16 壓榨 8GB VRAM |
| 🇹🇼 **繁體中文最佳化** | Regex 解碼器 + 口語翻譯蒟蒻 (禮拜二→星期二，grading→成績評定) |

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
│  · 閒聊短路判定               │
└──────────┬────────────────────┘
           ▼
┌────────────────────────────────────┐
│  Step 2: Hybrid Retriever           │
│  (retriever.py)                     │
│                                     │
│  ┌─ 並行 Embedding (ThreadPool) ─┐ │
│  │  Q1, Q2, Q3 → Ollama Embed   │ │
│  └───────────┬───────────────────┘ │
│              ▼                      │
│  ┌──────────┐  ┌──────────┐        │
│  │ Vector   │  │ BM25     │  ← 並行│
│  │ (FAISS)  │  │ (jieba)  │        │
│  └────┬─────┘  └────┬─────┘        │
│       ▼              ▼              │
│  ┌─────────────────────────┐       │
│  │ RRF Fusion               │       │
│  │ score = α·V + β·B + γ·M │       │
│  │ + Hard Metadata Filter   │       │
│  └─────────────────────────┘       │
└──────────┬─────────────────────────┘
           ▼
┌──────────────────────┐
│  Step 3: Reranker     │  ← bge-reranker-large (cross-encoder)
│  (reranker.py)        │     Top-30 → Top-8, GPU batch=16
│                       │     + VRAM GC (gc + empty_cache)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Step 4: LLM Answer   │  ← Llama 3.1 8B (Ollama - 8k Context)
│  (llm_answer.py)      │     精簡 Prompt + 口語化問答
└──────────┬───────────┘
           ▼
    回答 + 來源標註
```

---

## 🛠️ 技術棧

| 層級 | 技術 | 說明 |
|------|------|------|
| **生成大腦** | Llama 3.1 8B (TAIDE-LX Q4_K_M) | 繁中特化，處理進階推理與雙軌排版 |
| **路由小腦** | Llama 3.2 3B (Ollama) | One-shot 分類+改寫，JSON Schema 強制格式，`keep_alive=0` |
| **Embedding** | multilingual-e5-large (1024 維) | 並行 Embedding (`ThreadPoolExecutor`) |
| **Reranker** | BAAI/bge-reranker-large | Cross-Encoder，batch=16，推理後 VRAM GC |
| **Vector Store** | FAISS (IndexFlatIP) | 高效向量相似度搜尋 |
| **Keyword Search** | BM25Okapi + jieba | 中文關鍵字搜尋，並行執行 |
| **防禦機制** | JSON Schema / VRAM GC / 翻譯蒟蒻 | 100% JSON 成功率 + VRAM 穩定 |
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

> **注意**：`sentence-transformers` 會在首次執行時自動從 HuggingFace 下載 `bge-reranker-large` 模型（約 1.3GB）。

### 4. 確認 Ollama 服務運行

```bash
# 啟動 Ollama 服務
ollama serve

# （另開終端）測試 Ollama 是否正常
curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b", "prompt": "Hello"}'
```

---

## 🚀 使用方法

### 啟動程式

```bash
cd "d:\AI HYBRID"
python main.py
```

### CLI 指令

| 指令 | 說明 |
|------|------|
| `/quit` | 退出程式 |
| `/rebuild` | 重建索引（資料更新後使用） |
| `/clear` | 清除對話歷史 |
| `/debug` | 切換 Debug 模式（顯示詳細檢索資訊） |

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

### `main.py` — 主 Pipeline

CLI 互動介面，串接所有模組：
1. 載入索引 → 2. 接收問題 → 3. **合併 Router+Rewrite** →
4. 並行 Hybrid Retrieve → 5. Rerank → 6. LLM Answer → 7. 顯示回答

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

### 範例 3：多輪對話與閒聊

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
├── data_loader.py           # 資料解析 + 多學期遞迴掃描 + FAISS/BM25 索引
├── query_rewrite.py         # Query Rewrite + Multi-query RAG（備用 fallback）
├── query_router.py          # 合併式 Router+Rewrite + 時間 Regex 攔截 + VRAM 防護
├── retriever.py             # Hybrid Retriever（嚴格過濾防護 + 零幻覺攔截 + RRF）
├── reranker.py              # Cross-Encoder Reranker（batch=16 + VRAM GC）
├── llm_answer.py            # XML 強制模板 + 智慧推薦引擎 + 防進度表洗版
├── main.py                  # 主 Pipeline（短路防爆攔截 + VRAM GC）
├── discord_bot.py           # Discord Bot 非同步介面 + 實時 sync 指令
├── requirements.txt         # Python 依賴套件
├── README.md                # 本文件
├── .env                     # DISCORD_BOT_TOKEN
├── index_store/             # 索引持久化目錄（自動生成）
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
