from autogen import AssistantAgent, UserProxyAgent

# Requirements:
# python3.11 -m venv venv
# Using the forked version of Autogen on Pypi: https://github.com/ag2ai/ag2
# pip install ag2==0.7.3 ollama fix_busted_json yfinance matplotlib
# source venv/bin/activate

# Define the local model configuration
config_list = [
 {
    "model": "qwen2.5:14b",   # Choose a model that supports tool calling
    "api_type": "ollama",     # Specify Ollama as the API type
    "client_host": "http://localhost:11434",  # local Ollama server
    "api_key": "fakekey",
    "native_tool_calls": True # Enable native tool calling 
 }
]

# Create the AssistantAgent using the local model config
assistant = AssistantAgent("assistant",
                           llm_config={"config_list": config_list})

# Create the UserProxyAgent; adjust code_execution_config as needed.
user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

# Initiate an automated chat between the agents.
user_proxy.initiate_chat(
    assistant,
    message="Plot a chart of NVDA and TESLA stock price change YTD."
)