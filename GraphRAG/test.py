# -*- coding: utf-8 -*-
import os
os.environ["CHROMADB_ANON_TELEMETRY"] = "False"  # 關掉 Chroma 遙測雜訊

import re
import uuid
import requests
import subprocess
from collections import Counter

import chromadb
import jieba

# ====== GraphRAG 專案根路徑（如未用可留著） ======
GRAPHRAG_ROOT = r"D:\網頁\RAG\GraphRAG"

# ====== Neo4j 連線設定 ======
from neo4j import GraphDatabase
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345Lkk"   # ← 改成你的密碼


def get_neo_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


STOPWORDS = set(["的","了","與","及","而","並","其","之","在","和","或","以及","對於","如果","可能","需要"])


# ========================================================
# Step 1: 讀取文本文件並分段（確保完整吃到每一段）
# ========================================================
def file_chunk_list():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "input", "中醫診療配方.txt")
    with open(file_path, encoding="utf-8", mode="r") as fp:
        raw = fp.read()

    text = raw.replace("\r\n", "\n").strip()
    chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()]

    # 去重（避免相同段落重複）
    seen, uniq = set(), []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


# ========================================================
# Step 2: 向量化文本（Ollama embeddings）
# ========================================================
def ollama_embedding_by_api(text):
    res = requests.post(
        url="http://127.0.0.1:11434/api/embeddings",
        json={"model": "bge-m3", "prompt": text},
        timeout=120
    )
    res.raise_for_status()
    return res.json()['embedding']


# ========================================================
# Step 3: 生成推理模型回覆（Ollama LLM）
# ========================================================
def ollama_generate_by_api(prompt):
    response = requests.post(
        url="http://127.0.0.1:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": prompt, "stream": False, "temperature": 0.1},
        timeout=600
    )
    response.raise_for_status()
    return response.json().get('response', '')


# ====== GraphRAG CLI（可選用；這版主目的是全量讀取） ======
def graphrag_query(query_text, method="local", root=GRAPHRAG_ROOT, timeout=300):
    try:
        completed = subprocess.run(
            ["graphrag", "query", "--root", root, "--method", method, "--query", query_text],
            capture_output=True, text=True, timeout=timeout
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def dedup_lines(blocks):
    seen, out = set(), []
    for b in blocks:
        if not b:
            continue
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def extract_keywords(text, topn=8):
    words = [w.strip() for w in jieba.lcut(text) if w.strip()]
    words = [w for w in words if w not in STOPWORDS and len(w) >= 2]
    return [w for w, _ in Counter(words).most_common(topn)]


# ====== Neo4j：把 chunks 與實體寫進圖上（index 時呼叫） ======
def neo_upsert_chunks_and_entities(ids, documents, source="中醫診療配方.txt"):
    driver = get_neo_driver()
    cy_chunk = """
    MERGE (c:DocChunk {id:$cid})
      ON CREATE SET c.text=$text, c.source=$source
      ON MATCH  SET c.text=$text, c.source=$source
    WITH c, $entities AS ents
    UNWIND ents AS en
      MERGE (e:Entity {name:en})
      MERGE (c)-[:MENTIONS]->(e)
    """
    cy_co = """
    UNWIND $entities AS a
    UNWIND $entities AS b
    WITH a,b WHERE a<b
    MERGE (ea:Entity {name:a})
    MERGE (eb:Entity {name:b})
    MERGE (ea)-[r:CO_OCCURS]->(eb)
      ON CREATE SET r.w=1
      ON MATCH  SET r.w=r.w+1
    """
    with driver.session() as sess:
        for cid, text in zip(ids, documents):
            ents = extract_keywords(text, topn=8)
            sess.run(cy_chunk, cid=cid, text=text, source=source, entities=ents)
            if ents:
                sess.run(cy_co, entities=ents)
    driver.close()


def neo_count_chunks():
    driver = get_neo_driver()
    with driver.session() as sess:
        rec = sess.run("MATCH (c:DocChunk) RETURN count(c) AS n").single()
    driver.close()
    return rec["n"] if rec else 0


# ====== Neo4j：回傳「全部」chunks（確保完整讀取） ======
def neo_all_chunks(limit=None):
    """
    直接取出圖上所有 DocChunk 的 text。
    若給 limit 會限制數量；不給就取全量。
    """
    driver = get_neo_driver()
    cy = "MATCH (c:DocChunk) RETURN c.text AS text ORDER BY c.id"
    if limit:
        cy += " LIMIT $limit"
    texts = []
    with driver.session() as sess:
        res = sess.run(cy, limit=limit)
        for r in res:
            texts.append(r["text"])
    driver.close()
    return texts


# ========================================================
# Step 4: 初始化資料庫（向量 + 圖）並對帳
# ========================================================
def initial():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "db", "chroma_demo")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="collection_v2")

    documents = file_chunk_list()
    total = len(documents)
    ids = [f"chunk_{i:04d}" for i in range(total)]

    # 生成 embeddings（逐筆；若要更快可自行加快取策略）
    embeddings, ok_docs, ok_ids = [], [], []
    for cid, doc in zip(ids, documents):
        try:
            emb = ollama_embedding_by_api(doc)
            embeddings.append(emb)
            ok_docs.append(doc)
            ok_ids.append(cid)
        except Exception as e:
            print(f"[WARN] embedding 失敗: {cid} -> {e}")

    # 寫入 Chroma
    if ok_docs:
        collection.add(ids=ok_ids, documents=ok_docs, embeddings=embeddings)

    # 寫入 Neo4j
    neo_upsert_chunks_and_entities(ok_ids, ok_docs, source="中醫診療配方.txt")

    # 對帳
    chroma_count = collection.count()
    neo_total = neo_count_chunks()
    print(f"[INGEST DONE] 原始:{total}  寫入Chroma:{chroma_count}  Neo4j:{neo_total}")
    if chroma_count != total or neo_total != total:
        print("[WARN] 數量不一致，請執行 verify_ingestion() 看明細")


# ========================================================
# 對帳：列出哪幾段沒進去（可選）
# ========================================================
def verify_ingestion():
    docs = file_chunk_list()
    ids_expected = [f"chunk_{i:04d}" for i in range(len(docs))]

    # Chroma 端
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "db", "chroma_demo")
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_or_create_collection("collection_v2")

    # 取全部資料（注意：不同版本 API 行為略有差異）
    all_docs = col.get(ids=None)
    ids_in_chroma = set(all_docs.get("ids", []))

    # Neo4j 端
    driver = get_neo_driver()
    with driver.session() as sess:
        ids_in_neo = {r["id"] for r in sess.run("MATCH (c:DocChunk) RETURN c.id AS id")}
    driver.close()

    miss_chroma = [i for i in ids_expected if i not in ids_in_chroma]
    miss_neo = [i for i in ids_expected if i not in ids_in_neo]

    print(f"總 chunks: {len(ids_expected)}")
    print(f"Chroma 缺 {len(miss_chroma)} -> {miss_chroma[:10]} ...")
    print(f"Neo4j  缺 {len(miss_neo)}    -> {miss_neo[:10]} ...")

def neo_related_chunks(query_text, max_chunks=12, hop=1):
    """
    依查詢詞做中文關鍵詞切分 → 找到對應 Entity → 取 hop 跳內鄰居
    → 回傳被提及次數最高的 DocChunk 文字
    """
    driver = get_neo_driver()
    q_ents = extract_keywords(query_text, topn=5)
    if not q_ents:
        driver.close()
        return []

    cypher = """
    // 1) 查詢實體
    MATCH (e:Entity)
    WHERE e.name IN $qents

    // 2) 取 hop 跳以內的鄰居（可為空）
    OPTIONAL MATCH (e)-[:CO_OCCURS*..HOP]->(nbr)

    // 3) 分別蒐集 e 與 nbr，然後合併成 allents
    WITH collect(DISTINCT e) AS es, collect(DISTINCT nbr) AS ns
    WITH [x IN es WHERE x IS NOT NULL] + [x IN ns WHERE x IS NOT NULL] AS allents

    // 4) 由這些實體回到被提及的 chunk，依提及次數排序
    UNWIND allents AS ae
    MATCH (c:DocChunk)-[:MENTIONS]->(ae)
    WITH c, count(*) AS score
    ORDER BY score DESC
    LIMIT $limit
    RETURN c.text AS text
    """.replace("HOP", str(hop))

    texts = []
    with driver.session() as sess:
        res = sess.run(cypher, qents=q_ents, limit=max_chunks)
        for r in res:
            texts.append(r["text"])
    driver.close()
    return texts



# ========================================================
# Step 5: 檢索 + 生成回答（確保「完整讀取」）
# ========================================================
def run():
    # 使用者問題（你要什麼就填什麼）
    qs = "我肚子好痛。"

    # ---- 1) 向量檢索：取前 k 個（聚焦，而不是全量）----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "db", "chroma_demo")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="collection_v2")

    total_count = collection.count()
    k = min(12, total_count)  # 先抓 12 段作為候選（可調 8~20）
    qs_embedding = ollama_embedding_by_api(qs)

    res = collection.query(
        query_embeddings=[qs_embedding],
        query_texts=[qs],
        n_results=k
    )
    vec_docs = res["documents"][0] if res and res.get("documents") else []

    # ---- 1.1) 關鍵詞加權重排（把含「腹/肚/胃/痛/絞痛/脹痛」的放前面）----
    kw = {"腹","肚","胃","痛","絞痛","脹痛","寒","冷","溫","嘔","瀉","便","泄"}
    def score(d):
        s = 0
        for w in kw:
            if w in d:
                s += 1
        return (s, len(d) * -0.0001)  # 關鍵詞多者優先，太長的稍微降權
    vec_docs = sorted(vec_docs, key=score, reverse=True)
    vec_context = "\n".join(vec_docs)

    # ---- 2) GraphRAG（local）可加可不加：用來補充語境，但不要太長 ----
    graph_local = graphrag_query(qs, method="local", root=GRAPHRAG_ROOT) or ""
    graph_local = "\n".join(graph_local.splitlines()[:80])  # 只留前 80 行防暴衝

    # ---- 3) Neo4j：只擴展相關的 chunks（不要全量），hop=1，取 12 段 ----
    neo_docs = neo_related_chunks(qs, max_chunks=12, hop=1)
    neo_context = "\n".join(neo_docs)

    # ---- 4) 合併、去重、再輕微截斷（避免 LLM 爆 context）----
    merged_blocks = dedup_lines([vec_context, neo_context, graph_local])
    context = "\n\n-----\n\n".join([b for b in merged_blocks if b])

    MAX_CTX_CHARS = 8000   # 視你的模型 context 設定，8k~12k 自行調
    if len(context) > MAX_CTX_CHARS:
        context = context[:MAX_CTX_CHARS] + "…"

    # ---- 5) Prompt：明確要求「僅依據參考資訊」且聚焦症狀 ----
    prompt = f"""
你是一位專業的中醫資訊助理。請僅根據「參考資訊」回答使用者的症狀問題，條理清楚、使用條列：
- 僅供教育用途，非醫療診斷與處方；若資訊不足，請直接說明並提醒就醫。
- 儘可能引用關鍵片段（用引號標示），並指出辨證要點與常見配伍思路。
- 嚴禁討論與問題無關的內容。

[使用者問題]
{qs}

[參考資訊（向量檢索 Top-K + Neo4j 關聯 + GraphRAG 摘要）]
{context}

請據此作答：
""".strip()

    result = ollama_generate_by_api(prompt)
    print(result)
    return result

# ========================================================
# Step 6: 主程式
# ========================================================
if __name__ == '__main__':
    # 一次性導入（完整）
    initial()

    # （可選）核對是否真的「完整放進去」
    verify_ingestion()

    # 查詢（把全庫內容都餵進模型）
    run()
