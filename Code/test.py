import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://api.together.xyz/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
    "Content-Type": "application/json"
}
payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0.7,
    "max_tokens": 100
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
