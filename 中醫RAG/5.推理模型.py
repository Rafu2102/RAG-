import requests

prompt = ""

response=requests.post(
    url="http://127.0.0.1:11434/api/generate",
    json={
        "model": "deepseek-r1:70b",
        "prompt": prompt,
        "stream":False
    }
)

res = response.json()['response']
print(res)