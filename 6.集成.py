import chromadb
import uuid
import requests

def file_chunk_list():

# ========================================================
# Step 1: 讀取文本文件並分段
# ========================================================

    # 讀取整個文件
    with open("RAG\knowledge\文件.txt" , encoding='utf-8' , mode='r') as fp:
        data=fp.read()

    # 根據空行分段
    chunk_list = data.split("\n\n")
    return [chunk for chunk in chunk_list if chunk]

# ========================================================
# Step 2: 向量化文本
# ========================================================

def ollama_embedding_by_api(text):
    res = requests.post(
        url="http://127.0.0.1:11434/api/embeddings",
        json={
            "model": "bge-m3",
            "prompt": text
        }
    )
    embedding = res.json()['embedding']
    return embedding

# ========================================================
# Step 3: 生成推理模型回覆
# ========================================================

def ollama_generate_by_api(prompt):
    response=requests.post(
        url="http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream":False,
            'temperature':0.1
        }
    )

    res = response.json()['response']
    print(res)

# ========================================================
# Step 4: 初始化資料庫，存入向量化文件
# ========================================================

def initial():
    client = chromadb.PersistentClient(path="db/chroma_demo")

    #創建集合
    #client.delete_collection("collection_v2") #防止丟入同一個資料先刪除舊的集合
    collection = client.get_or_create_collection(name="collection_v2")

    #搭建資料
    documents = file_chunk_list()  
    ids = [str(uuid.uuid4()) for _ in range (len(documents))]
    embeddings = [ollama_embedding_by_api(text) for text in documents]

    #插入資料
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

# ========================================================
# Step 5: 搜尋資料並生成回答
# ========================================================

def run():

    # 查詢關鍵字
    qs = "風寒感冒"

    # 把關鍵字也向量化
    qs_embedding = ollama_embedding_by_api(qs)

    # 連接資料庫
    client = chromadb.PersistentClient(path="db/chroma_demo")
    collection = client.get_or_create_collection(name="collection_v2")

    # 檢索最相似的段落
    res = collection.query(query_embeddings=[qs_embedding, ], query_texts=[qs], n_results=2)
    result = res["documents"][0]
    context = "\n".join(result)

    # 提供給大語言模型的提示詞
    prompt = f"""
    你是一位專業的中醫師
    你的工作是根據參考資訊回答使用者問題
    回答格式分層清楚，條理分明。
    如果參考資訊不足以回答使用者問題請回答不知道，不要亂回答瞎猜。
    用參考資訊:{context}，來回答問題:{qs},
    """

    # 生成回答
    result = ollama_generate_by_api(prompt)
    print(result)

# ========================================================
# Step 6: 主程式執行順序
# ========================================================

if __name__ == '__main__':
    initial()   # 初始化資料庫 (生成向量並存入)
    run()   # 搜尋並生成回答