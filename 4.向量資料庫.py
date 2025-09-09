import chromadb
import uuid
import requests

def ollama_embedding_by_api(text):
    res = requests.post(
        url="http://127.0.0.1:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    embedding = res.json()['embedding']
    return embedding

client = chromadb.PersistentClient(path="db/chroma_demo")   #資料庫 類似等於 = 資料夾
collection = client.get_or_create_collection(name="collection_v1")   #集合 類似等於 = 表格

#搭建資料
documents = ["風寒感冒","風熱感冒","胃痛"] 
ids = [str(uuid.uuid4()) for _ in documents]
embeddings = [ollama_embedding_by_api(text) for text in documents]

#插入資料
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
)

#搜索關鍵字
qs = "風熱感冒"
qs_embedding = ollama_embedding_by_api(qs)
res = collection.query(query_embeddings=[qs_embedding], query_texts=[qs], n_results=2)
print(res)