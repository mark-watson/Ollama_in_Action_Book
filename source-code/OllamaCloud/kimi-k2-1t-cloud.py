from ollama import Client
import os

client = Client(
    host="https://ollama.com",
    headers={'Authorization': os.environ.get("OLLAMA_API_KEY")}
)

messages = [
  {
    'role': 'user',
    'content': "Say 'hello' in three languages. Identify which language.",
  },
]

response = client.chat('kimi-k2:1t-cloud', messages=messages, stream=False)
print(f"{response['message']['content']}")