# RAG(Retrieval-Augmented Generation)檢索增強生成
把 知識檢索 和 大語言模型（LLM） 結合在一起

## 第一步 -- 準備知識庫把資料餵給大語言模型
 
### 自訂需要的文檔和內容

## 第二步 -- 把資料和文字進行分段和向量化

在RAG資料庫中會把資料和文字向量化

然後利用向量進行搜索 --> 每個資料轉換成向量之後根據每筆資料向量的角度和距離去做相似度搜索

### 1、分段處理

* 1、讀取文件內容:

        with open(r"knowledge\中醫診療配方.txt", encoding='utf-8' , mode='r') as fp:

            data=fp.read()

* 2、根據換行分段

        chunk_list = data.split("\n\n")

        chunk_list = [chunk for chunk in chunk_list if chunk]

        print(chunk_list)

### 2、將文字進行向量化

* 1、安裝向量化模型:

在電腦上安裝向量化模型ollama進行調用

也可使用像GPT提供的開源模型

* 2、本地下載bge-m3模型

        ollama pull bge-m3

* 3、啟動ollama進行向量化

        pip install requests
  
    ###

        import requests

        text = '測試'

        res = requests.post(
            url="http://127.0.0.1:11434/api/embeddings",
            json={
                "model": "bge-m3",
                "prompt": text
            }
        )

        embedding_list = res.json()['embedding']

        print(text)
        print(len(embedding_list) , embedding_list)

### 3、把分段結合向量化

    import requests
    import functools

    def file_chunk_list():

    # 1、讀取文件內容

        with open(r"knowledge\中醫診療配方.txt" , encoding='utf-8' , mode='r') as fp:
            data=fp.read()

    # 2、根據換行分段

        chunk_list = data.split("\n\n")
        return [chunk for chunk in chunk_list if chunk]

    # 3、丟給ollama

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

    #4、整理段落，生成向量並處理

    def run():
        chunk_list = file_chunk_list()
        for chunk in chunk_list:
            vector = ollama_embedding_by_api(chunk)
            print(chunk)
            print(vector)

    #5、程式入口：當此檔案被直接執行時，執行 run() 函式

    if __name__ == '__main__':
        run()

## 第三步 -- 搭建向量資料庫

向量資料庫有:chromadb、Faiss、Qdrant等等......

以chromadb做向量的儲存資料庫

### 1、安裝

    pip install chromadb

### 2、搭建資料庫

    import chromadb
    import uuid
    inport requests

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
        embeddings=embeddings
    )

    #搜索關鍵字
    qs = "風熱感冒"
    qs_embedding = ollama_embedding_by_api(qs)
    res = collection.query(query_embeddings=[qs_embedding], query_texts=[qs], n_results=2)
    print(res)

## 第四步 -- 文本推理模型

這邊使用ollama作為推理模型

來整理資料庫的資料

### 1、安裝

    ollama pull llama3.1:8b

### 2、使用

    import requests

    prompt = "" #提示詞使用關鍵字和向量資料庫相關的內容做推理

    response=response.post(
        url="http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream":False
        }
    )

    res = response.json()['response']
    print(res)

## 第五步 -- 集成

把所有程式整合在一起

    import chromadb
    import uuid
    import requests

    def file_chunk_list():

    # ========================================================
    # Step 1: 讀取文本文件並分段
    # ========================================================

        # 讀取整個文件
        with open(r"knowledge\中醫診療配方.txt", encoding='utf-8' , mode='r') as fp:
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