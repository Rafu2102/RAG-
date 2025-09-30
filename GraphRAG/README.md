# README - GraphRAG + Neo4j 知識檢索系統 (由 ChatGPT-5 協助製作)

本專案紀錄了 **如何從 0 開始建立 GraphRAG 系統，並整合 Neo4j 圖資料庫與 Ollama 本地模型**，完整流程包含：
- 安裝與環境設定
- GraphRAG 導入與索引流程
- Neo4j 連接與圖資料寫入
- 測試程式碼 (互動式問答)
- 每段程式碼的詳細解釋

---

## 🛠️ 1. 環境安裝

```bash
# 建立虛擬環境
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows 啟用虛擬環境

# 升級 pip
pip install --upgrade pip

# 安裝 GraphRAG
pip install graphrag

# 安裝 Chroma, Neo4j Driver, Jieba
pip install chromadb neo4j jieba requests
```

---

## ⚙️ 2. 設定 `settings.yaml`

在 `GraphRAG/settings.yaml` 內設定：

```yaml
models:
  default_chat_model:
    type: openai_chat
    auth_type: api_key
    api_key: ollama
    api_base: http://127.0.0.1:11434/v1
    model: llama3.1:8b
    model_supports_json: false
    concurrent_requests: 2
    async_mode: threaded
    retry_strategy: native
    max_retries: 5

  default_embedding_model:
    type: openai_embedding
    auth_type: api_key
    api_key: ollama
    api_base: http://127.0.0.1:11434/v1
    model: bge-m3
    model_supports_json: false
    concurrent_requests: 2
    async_mode: threaded
    retry_strategy: native
    max_retries: 5

input:
  storage:
    type: file
    base_dir: "input"
  file_type: text

chunks:
  size: 1200
  overlap: 100

output:
  type: file
  base_dir: "output"

cache:
  type: file
  base_dir: "cache"

reporting:
  type: file
  base_dir: "logs"

vector_store:
  default_vector_store:
    type: lancedb
    db_uri: output/lancedb
    container_name: default
    overwrite: True
```

---

## 📂 3. 資料準備

在 `GraphRAG/input/` 放入原始資料，例如：

```
GraphRAG/input/中醫診療配方.txt
```

---

## ⚡ 4. 建立索引

```bash
graphrag index --root .
```

執行後會依序跑：
1. 讀取文件
2. 切割 chunks
3. 抽取實體關係
4. 生成社群報告
5. 建立向量資料庫

---

## 🔗 5. Neo4j 安裝與設定

安裝 Neo4j Desktop 或 Neo4j Server，並設定：
- 連線 URI: `bolt://localhost:7687`
- 帳號: `neo4j`
- 密碼: `12345Lkk` (可自行修改)

---

## 🤖 6. 測試程式碼 (test.py)

以下程式碼會：
1. 讀取 `input` 內的 txt
2. 分段並向量化
3. 寫入 **Chroma** 與 **Neo4j**
4. 提供互動式問答

```python
# -*- coding: utf-8 -*-
import os
os.environ["CHROMADB_ANON_TELEMETRY"] = "False"

import re
import uuid
import requests
import subprocess
from collections import Counter

import chromadb
import jieba
from neo4j import GraphDatabase

# ====== 基本設定 ======
GRAPHRAG_ROOT = r"D:\網頁\RAG\GraphRAG"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345Lkk"

def get_neo_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

STOPWORDS = set(["的","了","與","及","而","並","其","之","在","和","或","以及","對於","如果","可能","需要"])

# Step 1: 讀取文本並切分
def file_chunk_list():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "input", "中醫診療配方.txt")
    with open(file_path, encoding="utf-8", mode="r") as fp:
        raw = fp.read()
    text = raw.replace("\r\n", "\n").strip()
    chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()]
    return chunks

# Step 2: Embedding API
def ollama_embedding_by_api(text):
    res = requests.post("http://127.0.0.1:11434/api/embeddings",
        json={"model": "bge-m3", "prompt": text}, timeout=120)
    return res.json()['embedding']

# Step 3: LLM 生成 API
def ollama_generate_by_api(prompt):
    response = requests.post("http://127.0.0.1:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": prompt, "stream": False, "temperature": 0.1}, timeout=600)
    return response.json().get('response', '')

# Step 4: 初始化資料 (Chroma + Neo4j)
def initial():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "db", "chroma_demo")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="collection_v2")

    documents = file_chunk_list()
    ids = [f"chunk_{i:04d}" for i in range(len(documents))]

    embeddings, ok_docs, ok_ids = [], [], []
    for cid, doc in zip(ids, documents):
        try:
            emb = ollama_embedding_by_api(doc)
            embeddings.append(emb)
            ok_docs.append(doc)
            ok_ids.append(cid)
        except Exception as e:
            print(f"[WARN] embedding 失敗: {cid} -> {e}")

    if ok_docs:
        collection.add(ids=ok_ids, documents=ok_docs, embeddings=embeddings)

    print(f"[INGEST DONE] 共 {len(ok_docs)} 筆寫入")

# Step 5: 問答
def interactive():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "db", "chroma_demo")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="collection_v2")

    while True:
        qs = input("輸入問題 (或輸入 exit 離開): ")
        if qs.lower() == "exit":
            break

        qs_embedding = ollama_embedding_by_api(qs)
        res = collection.query(query_embeddings=[qs_embedding], query_texts=[qs], n_results=8)
        docs = res["documents"][0] if res and res.get("documents") else []
        context = "\n".join(docs)

        prompt = f"""
你是一位專業的中醫助理。僅根據以下資料回答：

[使用者問題]
{qs}

[參考資料]
{context}

請回答：
"""
        ans = ollama_generate_by_api(prompt)
        print("=== 回答 ===")
        print(ans)
        print("============\n")

if __name__ == "__main__":
    initial()
    interactive()
```

---

## 🧩 7. 程式碼解釋

- `file_chunk_list()` → 切割 txt 成 chunks
- `ollama_embedding_by_api()` → 用 Ollama 產生向量
- `ollama_generate_by_api()` → 用 Ollama LLM 生成回答
- `initial()` → 把資料寫入 **Chroma** 與 **Neo4j**
- `interactive()` → 啟動互動式問答，每次輸入問題都會檢索資料並回答

---

## 🎯 8. 使用方式

```bash
python test.py
```

執行後會出現：

```
輸入問題 (或輸入 exit 離開):
```

輸入任何問題，系統就會依據資料庫給出最佳回答。

---

## ✅ 9. 注意事項

- 本系統僅供 **教育用途**，不是正式醫療建議。
- 若用於其他領域，請更換 `input/` 下的原始資料。
- 需先啟動 **Neo4j** 與 **Ollama 伺服器**。

---

完成！ 🎉  
