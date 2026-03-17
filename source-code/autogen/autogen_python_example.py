import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama import Client, ChatResponse

from ollama_config import get_model

# Requirements:
# python3.11 -m venv venv
# pip install ollama yfinance matplotlib
# source venv/bin/activate

_model_name = get_model()

if os.environ.get("CLOUD"):
    api_key = os.environ.get("OLLAMA_API_KEY", "fakekey")
    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"}
    )
else:
    client = Client(host="http://localhost:11434")

response: ChatResponse = client.chat(
    model=_model_name,
    messages=[
        {
            "role": "user",
            "content": "Plot a chart of NVDA and TESLA stock price change YTD.",
        }
    ]
)
print(response["message"]["content"])
