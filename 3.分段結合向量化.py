import requests
import functools

def file_chunk_list():

    # 1、讀取文件內容

    with open("knowledge/文件.txt" , encoding='utf-8' , mode='r') as fp:
        data=fp.read()

    # 2、根據換行分段

    chunk_list = data.split("\n\n")
    return [chunk for chunk in chunk_list if chunk]

    # 3、丟給ollama

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