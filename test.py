from google import genai
from google.genai.types import HttpOptions

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model='gemini-3.5-flash',
    contents='你是誰?你是什麼模型?你的型號試什麼?',
)
print(response.text)