# -*- coding: utf-8 -*-
import os
import re

readme_path = r"d:\AI HYBRID\README.md"

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

# ── 1. Update Last Update Date ──
target_date = "> **國立金門大學資訊工程學系 · Ultimate Agentic RAG 智慧問答系統**\n> *最後更新時間：2026-04-13*"
replacement_date = "> **國立金門大學資訊工程學系 · Ultimate Agentic RAG 智慧問答系統**\n> *最後更新時間：2026-04-13 (2026-05-27 最新核心功能全面升級)*"

if replacement_date not in content:
    if target_date in content:
        content = content.replace(target_date, replacement_date)
    else:
        print("WARNING: target_date not found!")

# ── 2. Add features to table ──
target_feature = "| 🔒 **必選修智慧過濾** | 自動偵測疑問句（「是必修嗎？」），避免誤設篩選條件 |"
replacement_features = """| 🔒 **必選修智慧過濾** | 自動偵測疑問句（「是必修嗎？」），避免誤設篩選條件 |
| 🌐 **Vertex AI 企業版無金鑰 ADC** | 全面升級至 Google Cloud Vertex AI REST 企業級架構，改用應用程式預設憑證 (Application Default Credentials, ADC) 無密鑰授權，徹底消除金鑰洩露風險。 |
| 🛡️ **連線池與退避重試** | 實作非同步 HTTP 連線池單例，優化 TCP 連續建連開銷，並內建帶有隨機抖動 (Jitter) 的指數退避重試引擎以因應 429 / 5xx 配額限制。 |
| 🔀 **Dcard 搜尋指令與提示詞分流** | 解決 Perplexity 搜尋詞受 prompt 污染導致 0 筆搜尋結果之問題，將格式、過濾條件、保底指示及快取防禦皆移入 `system` 角色，`user` 僅保留極簡純淨搜尋詞。 |
| 🏷️ **Dcard 搜尋口語化與詞域智慧擴充** | 搜尋時自動識別 `資工`、`電機`、`英文` 等單詞並自動擴充同義詞，提升搜尋引擎召回率；快取防禦時間戳記移入 `system` 避免干擾搜尋召回。 |
| 🔌 **Windows 背景子進程與環境變數繼承** | 針對 Windows 環境下 `venv` 子進程 `CWD` 漂移與 `.env` 未繼承問題，重構進程啟動機制，顯式傳遞工作目錄與全域環境變數。 |
| 📱 **Telegram HTML 連結防污染解析** | 改善 Telegram HTML 超連結解析，動態去除 Discord 專屬防預覽角括號 `<>` 與轉義 `&lt;&gt;` 字元，防止 Telegram 用戶端點擊失效。 |
| 🎛️ **Reranker 非阻塞執行緒併發隔離** | 將本地 Cross-Encoder 推理 offload 至背景執行緒並以 `threading.Semaphore(3)` 限制併發，避免高計算量造成 asyncio 主執行緒心跳超時或中斷。 |
| 📅 **安全 OAuth 行事曆所有權校驗** | 行事曆刪除與覆寫操作嚴格限制於機器人建立之事件，並新增 DB 覆蓋警告與所有權 (Ownership) 比對安全閥，避免誤刪私人事件。 |
| 🔌 **FastMCP 伺服器整合** | 將核心 RAG 與 LLM 查詢、歷年課表與成績單解析曝露為 MCP 工具，啟動後支援 stdio 通訊，可掛載於 Cursor, Claude Desktop 等 MCP 用戶端。 |"""

if "🌐 **Vertex AI 企業版無金鑰 ADC**" not in content:
    if target_feature in content:
        content = content.replace(target_feature, replacement_features)
    else:
        print("WARNING: target_feature not found!")

# ── 3. Add point 5 in Agentic RAG section ──
target_rag_point = "4. **消除 Python-Side 判斷瓶頸**：大幅刪減 Python 端的 `any(kw in ...)` 單詞命中漏洞，將所有決策權利與護城河驗證上繳至 Gemini 核心，達到「真正的降維打擊」。"
replacement_rag_point = """4. **消除 Python-Side 判斷瓶頸**：大幅刪減 Python 端的 `any(kw in ...)` 單詞命中漏洞，將所有決策權利與護城河驗證上繳至 Gemini 核心，達到「真正的降維打擊」。
5. **Perplexity 意圖與指令分流 (Instruction-Query Separation)**：為了解決 Perplexity Sonar Pro 會將 prompt 中的格式指令誤判為檢索關鍵字的問題，首創「搜尋詞與指令解耦」架構。`user` 訊息只傳遞純粹的 `site:dcard.tw/f/nqu {query}` 條件，而所有格式表格、篩選標準與保底文字全數收歸 `system` 角色管理，實現 100% 準確搜尋。"""

if "Perplexity 意圖與指令分流" not in content:
    if target_rag_point in content:
        content = content.replace(target_rag_point, replacement_rag_point)
    else:
        print("WARNING: target_rag_point not found!")

# ── 4. Add security points ──
target_security = "6. **跨平台 IPC 稽核防護 (Cross-Process Audit)**：無論是從 CLI、Discord 還是 Telegram 進來的查詢，系統都會透過 IPC Server (Port 50505) 將所有互動記錄與除錯資訊發送至管理員的 `#bot_modify` 頻道。"
replacement_security = """6. **跨平台 IPC 稽核防護 (Cross-Process Audit)**：無論是從 CLI、Discord 還是 Telegram 進來的查詢，系統都會透過 IPC Server (Port 50505) 將所有互動記錄與除錯資訊發送至管理員的 `#bot_modify` 頻道。
7. **Vertex AI 企業級 ADC 無密鑰授權**：捨棄傳統寫死於程式碼或設定檔的 API Key，全站導入 Google Cloud Vertex AI 原生憑證授權（Application Default Credentials），大幅降低資安威脅。
8. **Reranker 執行緒併發隔離與 VRAM 釋放**：Cross-Encoder 推理在背景執行緒中非阻塞運行，配合核心 Semaphore 佇列進行硬體隔離，且於排序完畢後強制呼叫 `gc.collect()` 與 `torch.cuda.empty_cache()` 徹底釋放 VRAM。
9. **Windows 平台子進程安全防禦**：針對背景子進程啟動時工作目錄 (CWD) 可能漂移至直譯器目錄以及 `.env` 變數無法在非 Windows cmd 下正確繼承的漏洞，顯式傳遞工作目錄與全域環境變數，保證服務穩定與金鑰安全載入。
10. **Telegram HTML 超連結轉義清理**：自動修復 Telegram 因轉義造成的 `<URL>` 或 `&lt;URL&gt;` 語法錯誤，防止點擊網址失效或受到釣魚警告。"""

if "Vertex AI 企業級 ADC 無密鑰授權" not in content:
    if target_security in content:
        content = content.replace(target_security, replacement_security)
    else:
        print("WARNING: target_security not found!")

# ── 5. Update Technology Stack ──
target_tech_brain = '| **生成大腦** | **Gemini 3.5 Flash** (High Thinking) | 全新升級主核心，啟用 `"high"` 思考層級負責複雜課程對答推理、225KB+ Context Backfill、個人成績深度解析，以及 **Vision 高精度圖像解構** |'
replacement_tech_brain = '| **生成大腦** | **Gemini 3.5 Flash** (High Thinking) | 全新升級主核心，啟用 `"high"` 思考層級負責複雜課程對答推理、225 KB+ Context Backfill、個人成績深度解析，以及 **Vision 高精度圖像解構**。全面採用 **Vertex AI 企業端點與 ADC 無金鑰認證**。 |'

target_tech_brain2 = '| **路由小腦** | **Gemini 3.5 Flash** (Low Thinking) | One-shot CoT 分類、意圖解構、原生職涯規劃偵測與 Schema 強制輸出 |'
replacement_tech_brain2 = '| **路由小腦** | **Gemini 3.5 Flash** (Low Thinking) | One-shot CoT 分類、意圖解構、原生職涯規劃偵測與 Schema 強制輸出。全面採用 **Vertex AI 企業端點與 ADC 無金鑰認證**。 |'

target_tech_rerank = '| **Reranker** | BAAI/bge-reranker-base | Cross-Encoder，batch=32，推理後 VRAM GC + **甲乙班去重硬偏好** |'
replacement_tech_rerank = '| **Reranker** | BAAI/bge-reranker-base | Cross-Encoder，batch=32，推理後 VRAM GC + **甲乙班去重硬偏好**。在 RAG Pipeline 中**以背景非同步執行緒 (to_thread) 隔離運作**，防止 Heartbeat 逾時。 |'

if "Vertex AI 企業端點與 ADC 無金鑰認證" not in content:
    content = content.replace(target_tech_brain, replacement_tech_brain)
    content = content.replace(target_tech_brain2, replacement_tech_brain2)
    content = content.replace(target_tech_rerank, replacement_tech_rerank)

# ── 6. Update Module Detailed Descriptions ──
target_mod_config = "- Gemini API 設定（API 金鑰, 雙腦備援模型變數，含 Timeout 與 maxOutputTokens 常數）"
replacement_mod_config = "- Gemini API 設定（全面遷移至 Google Cloud Vertex AI ADC 無密鑰安全授權，已移除 legacy API 金鑰與硬編碼 URL 端點，含 Timeout 與 maxOutputTokens 常數）"
if "全面遷移至 Google Cloud Vertex AI ADC 無密鑰安全授權" not in content:
    content = content.replace(target_mod_config, replacement_mod_config)

target_mod_tg = "| `tg_events.py` | 文字訊息攔截、照片解析（Gemini Vision 聯動）與檔案上傳處理。 |"
replacement_mod_tg = "| `tg_events.py` | 文字訊息攔截、照片解析（Gemini Vision 聯動）與檔案上傳處理，並包含 Telegram HTML 連結防污染清理（自動移除角括號 `<>` 與轉義 `&lt;&gt;`）。 |"
if "Telegram HTML 連結防污染清理" not in content:
    content = content.replace(target_mod_tg, replacement_mod_tg)

target_mod_dcard = "| `tools/dcard_search_tool.py` | Dcard 金門大學版教授評價搜尋 |"
replacement_mod_dcard = "| `tools/dcard_search_tool.py` | Dcard 教授評價搜尋。全面改裝為 Perplexity 提示詞/查詢分流架構，將表格輸出格式與保底文案移入 system 提示詞，user 僅提供 site:dcard.tw/f/nqu精準檢索語句，防範搜尋引擎受到指令詞污染；支援關鍵字詞域智慧擴充（資工/電機/英文等）。 |"
if "Perplexity 提示詞/查詢分流架構" not in content:
    content = content.replace(target_mod_dcard, replacement_mod_dcard)

target_mod_run_all = "├── run_all.py                  # 🚀 雙平台統一啟動器（Beta，Discord + Telegram 同時運行）"
replacement_mod_run_all = "├── run_all.py                  # 🚀 雙平台統一啟動器。重構 Windows 子行程啟動機制，顯式指定 CWD=PROJECT_ROOT 並主動傳遞親代環境變數（含 OPENROUTER_API_KEY 等），徹底避免 CWD 漂移至虛擬環境內部目錄與環境變數遺漏問題。"
if "親代環境變數" not in content:
    content = content.replace(target_mod_run_all, replacement_mod_run_all)

target_mod_reranker = "| `rag/reranker.py`             # Cross-Encoder (甲乙班去重偏好) 與 GC"
replacement_mod_reranker = "| `rag/reranker.py`             # Cross-Encoder (甲乙班去重偏好) 與 GC。使用 BAAI/bge-reranker-base，新增非同步背景執行緒隔離與並行硬體 Semaphore(3) 排隊閥，解決 GPU 在高負載下造成 Discord 閘道器心跳超時的系統性風險。"
if "非同步背景執行緒隔離" not in content:
    content = content.replace(target_mod_reranker, replacement_mod_reranker)

target_mod_rerank_detail = """- 使用 `BAAI/bge-reranker-base` cross-encoder，batch=32
- 分數融合：`final = sigmoid(rerank) × 0.75 + metadata × 0.25`
- **情境加分防護**：根據查詢意圖動態加分 basic_info、objectives、教授資訊等區段
- **甲乙班去重偏好**：相同課程的甲/乙班只保留最高分者，預設偏好甲班
- **課程完整覆蓋保證**：確保每門課至少出現一次，動態擴大 Top-N容量
- **VRAM GC**：推理後 `gc.collect()` + `torch.cuda.empty_cache()`"""

replacement_mod_rerank_detail = """- 使用 `BAAI/bge-reranker-base` cross-encoder，batch=32
- 分數融合：`final = sigmoid(rerank) × 0.75 + metadata × 0.25`
- **情境加分防護**：根據查詢意圖動態加分 basic_info、objectives、教授資訊等區段
- **甲乙班去重偏好**：相同課程的甲/乙班只保留最高分者，預設偏好甲班
- **課程完整覆蓋保證**：確保每門課至少出現一次，動態擴大 Top-N容量
- **非阻塞執行緒與硬體限制**：透過 `asyncio.to_thread` 背景隔離執行，並引入 `threading.Semaphore(3)` 併發控制，避免高載下 GPU 推理阻塞 asyncio 事件循環，引發 Discord Gateway 心跳斷開
- **VRAM GC**：推理後 `gc.collect()` + `torch.cuda.empty_cache()`"""
if "非阻塞執行緒與硬體限制" not in content:
    content = content.replace(target_mod_rerank_detail, replacement_mod_rerank_detail)

# ── 7. Update Configuration Adjustment Block ──
target_config_adjust = """```python
# config.py
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 全面升級至 Gemini 3.5 Flash 端點（透過 thinkingLevel 調優效能與推理）
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
GEMINI_FAST_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
```"""

replacement_config_adjust = """```python
# config.py
# =============================================================================
# 🤖 Gemini LLM 設定（已遷移至 Google Cloud Vertex AI ADC 無密鑰安全授權）
# =============================================================================
# 已全面拔除 AI Studio 金鑰與 URL，防止對個人信用卡扣款。
GEMINI_API_KEY = ""  # 保留以維護舊代碼相容性

# llm/gemini_client.py 於執行階段自動加載並刷新 GCP ADC token，動態組裝 Vertex REST URL：
# url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model_name}:generateContent"
```"""
if "GCP Vertex AI ADC 無密鑰安全授權" not in content:
    content = content.replace(target_config_adjust, replacement_config_adjust)

target_config_chunk = """### Chunk 設定

```python
CHUNK_SIZE = 512          # 讓 5/6 區段保持完整不被切分
CHUNK_OVERLAP = 50        # 超長區段切分時保持上下文連貫
```"""

replacement_config_chunk = """### Chunk 設定

```python
CHUNK_SIZE = 1536         # (最新升級：原為 512，現已擴大至 1536 以確保 18 週進度表完整包含)
CHUNK_OVERLAP = 128       # (最新升級：原為 50，現已擴大至 128 以保持上下文完美連續)
```"""
if "(最新升級：原為 512" not in content:
    content = content.replace(target_config_chunk, replacement_config_chunk)

# ── 7.5 Insert MCP Server under Usage Methods ──
target_usage_mcp = """### 方式三：單獨啟動 Discord / Telegram

```bash
cd "d:\\AI HYBRID"
# 僅啟動 Discord
python discord_bot.py

# 僅啟動 Telegram
python telegram_bot.py
```

機器人啟動後支援以下互動方式："""

replacement_usage_mcp = """### 方式三：單獨啟動 Discord / Telegram

```bash
cd "d:\\AI HYBRID"
# 僅啟動 Discord
python discord_bot.py

# 僅啟動 Telegram
python telegram_bot.py
```

### 方式四：啟動 MCP 伺服器 (FastMCP)

本系統支援 Model Context Protocol (MCP)，將核心的 RAG 問答、個人成績與課表查詢曝露為 MCP 工具，讓您的 AI 編輯器（如 Cursor、Claude Desktop）直接調用：

```bash
cd "d:\\AI HYBRID"
# 啟動 MCP stdio 服務
python mcp_server.py
```

#### Cursor / Claude Desktop 設定範例：
```json
{
  "mcpServers": {
    "nqu-campus-assistant": {
      "command": "python",
      "args": ["d:/AI HYBRID/mcp_server.py"],
      "transport": "stdio"
    }
  }
}
```

機器人啟動後支援以下互動方式："""

if "方式四：啟動 MCP 伺服器" not in content:
    content = content.replace(target_usage_mcp, replacement_usage_mcp)

# ── 8. Typesetting space adjustments ──
content = content.replace("225KB", "225 KB")
content = content.replace("200KB", "200 KB")
content = content.replace("8GB", "8 GB")
content = content.replace("16GB", "16 GB")
content = content.replace("32GB", "32 GB")
content = content.replace("10GB", "10 GB")
content = content.replace("20GB", "20 GB")
content = content.replace("1GB", "1 GB")
content = content.replace("1.1GB", "1.1 GB")
content = content.replace("18週", "18 週")
content = content.replace("共18 週", "共 18 週")
content = content.replace("共18週", "共 18 週")
content = content.replace("59門課", "59 門課")
content = content.replace("114上", "114 上")
content = content.replace("114年", "114 年")
content = content.replace("第1學期", "第 1 學期")
content = content.replace("5~7節", "5~7 節")
content = content.replace("N節", "N 節")
content = content.replace("A節", "A 節")
content = content.replace("B節", "B 節")
content = content.replace("nqu精準", "nqu 精準")

# Also replace model mentions from 3.1 Pro -> 3.5 Flash
content = content.replace("Gemini 3.1 Pro", "Gemini 3.5 Flash")
content = content.replace("Flash Lite", "Gemini 3.5 Flash")
content = content.replace("Gemini Gemini", "Gemini")

# ── Write Content to README.md ──
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

print("README.md successfully updated!")
