from autogen import AssistantAgent, UserProxyAgent
import sys
from pathlib import Path

# Requirements:
# pip install ag2 ollama fix_busted_json yfinance matplotlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model

config_list = [
 {
  "model": get_model(), # Choose a model that supports tool calling
  "api_type": "ollama", # Specify Ollama as the API type
  "client_host": "http://localhost:11434", # local Ollama server
  "api_key": "fakekey",
  "native_tool_calls": True # Enable native tool calling
 }
]

# Create the AssistantAgent using the local model config
assistant = AssistantAgent("assistant",
                           llm_config={"config_list": config_list})
# Create the UserProxyAgent; adjust code_execution_config as needed.
user_proxy = UserProxyAgent("user_proxy",
                            code_execution_config={"work_dir": "coding",
                                                   "use_docker": False})

# Initiate an automated chat between the agents.
user_proxy.initiate_chat(assistant,
                         message="Plot a chart of NVDA and TESLA stock price change YTD.")
