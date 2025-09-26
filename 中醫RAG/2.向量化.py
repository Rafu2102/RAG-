import requests

text = '測試'

res = requests.post(
            url="http://127.0.0.1:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )

embedding_list = res.json()['embedding']

print(text)
print(embedding_list)