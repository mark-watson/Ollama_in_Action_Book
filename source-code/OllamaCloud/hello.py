import os
import sys
from pathlib import Path
from ollama import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

def generate(text):
    messages = [
      {
        'role': 'user',
        'content': text,
      },
    ]

    response = client.chat(get_model(), messages=messages, stream=False)
    return response['message']['content']

print(generate("Say 'hello' in three languages."))