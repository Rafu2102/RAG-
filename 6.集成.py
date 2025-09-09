import chromadb
import uuid
import requests
import shutil

def file_chunk_list():

# ========================================================
# Step 1: 讀取文本文件並分段
# ========================================================

    # 讀取整個文件
    with open("knowledge/文件.txt" , encoding='utf-8' , mode='r') as fp:
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
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    embedding = res.json()['embedding']
    return embedding

# ========================================================
# Step 3: 生成推理模型回覆
# ========================================================

def ollama_generate_by_api(prompt):
    try:
        response = requests.post(
            url="http://127.0.0.1:11434/api/generate",
            json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False,
                'temperature': 0.1
            }
        )
        res_json = response.json()
        # 用 get 避免 KeyError
        return res_json.get('response', None)
    except Exception as e:
        print("生成 API 發生錯誤:", e)
        return None

# ========================================================
# Step 4: 初始化資料庫，存入向量化文件
# ========================================================

def initial():
    
    #創建集合
    shutil.rmtree("db/chroma_demo", ignore_errors=True) #防止丟入同一個資料先刪除舊的集合
    client = chromadb.PersistentClient(path="db/chroma_demo")
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
    qs = "我感覺看東西的時候視線很模糊"

    # 把關鍵字也向量化
    qs_embedding = ollama_embedding_by_api(qs)

     # 連接資料庫
    client = chromadb.PersistentClient(path="db/chroma_demo")
    collection = client.get_or_create_collection(name="collection_v2")

    # 檢索最相似的段落
    res = collection.query(query_embeddings=[qs_embedding, ], query_texts=[qs], n_results=3)
    result = res["documents"][0]
    context = "\n".join(result)

    #print(context)

    #exit()

    # 提供給大語言模型的提示詞
    prompt = f"""
    你是一位專業的中醫師，對中醫經典方劑與臨床辨證有深入了解。  
    你的任務是根據參考資訊回答使用者的中醫問題，**只使用中醫理論和方劑知識**，不要提及西醫概念或英文混雜的解釋。  
    回答要求：
    1. 條理分明，分層描述，說出可能的症狀。
    2. 根據症狀提供可能的方劑、組成、功效、用法與注意事項。
    3. 如果參考資訊不足以回答問題，請直接回答「不知道」，不要猜測。
    4. 僅使用中醫術語與標準中文，不夾雜英文或西醫概念。

    參考資訊: {context}, 
    使用者問題: {qs},
    請根據以上資訊回答：
    """

    # 生成回答
    result = ollama_generate_by_api(prompt)
    if result:
        print(result)
    else:
        print("生成回答失敗")

# ========================================================
# Step 6: 主程式執行順序
# ========================================================

if __name__ == '__main__':
    initial()   # 初始化資料庫 (生成向量並存入)
    run()   # 搜尋並生成回答