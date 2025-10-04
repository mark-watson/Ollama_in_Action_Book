from ollama import Client
import os

client = Client(
    host="https://ollama.com",
    headers={'Authorization': os.environ.get("OLLAMA_API_KEY")}
)

def generate(text):
    messages = [
      {
        'role': 'user',
        'content': text,
      },
    ]

    response = client.chat('gpt-oss:20b', messages=messages, stream=False)
    return response['message']['content']

print(generate("Say 'hello' in three languages."))