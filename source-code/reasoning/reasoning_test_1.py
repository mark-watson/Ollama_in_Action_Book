import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model
from langchain_ollama.chat_models import ChatOllama

# Note: ChatOllama connects to the local Ollama service only.
# If CLOUD is set, this example still uses local Ollama since
# LangChain's ChatOllama does not support Ollama Cloud authentication.
# For cloud-based LLM calls, use the native ollama Client or OpenAI compat layer.

# chat_model = ChatOllama(model="deepseek-r1:latest", reasoning=True)
chat_model = ChatOllama(model=get_model(), reasoning=True)

result = chat_model.invoke("how many odd integers are greater than 0 and less than 10? Please be concise and state the final answer in JSON")

print(result)