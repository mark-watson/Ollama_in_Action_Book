import smolagents_compat  # noqa: F401

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smolagents.agents import CodeAgent
from smolagents import LiteLLMModel

from smolagents_tools import sa_list_directory
from smolagents_tools import sa_summarize_directory
from smolagents_tools import sa_read_file_contents

from ollama_config import get_model

_model_name = get_model()

if os.environ.get("CLOUD"):
    api_base = "https://ollama.com"
    api_key = os.environ.get("OLLAMA_API_KEY", "")
else:
    api_base = "http://localhost:11434"
    api_key = "your-api-key"

model = LiteLLMModel(
    model_id=f"ollama_chat/{_model_name}",
    api_base=api_base,
    api_key=api_key,
)
model.set_verbose=True

agent = CodeAgent(tools=[sa_list_directory,
                                sa_summarize_directory,
                                sa_read_file_contents],
                         model=model,
                         #planning_interval=2, # enables pre-planning for multiple steps
                         max_steps=3, # maximum number of iterations
                         add_base_tools=False) # set to True to use built-in tools

#agent.write_inner_memory_from_logs() # creates an inner memory of the agent's logs for the LLM to view

#print(agent.run("List the Python programs in the current directory, and then tell me which Python programs in the current directory evaluate the performance of LLMs?\n\n"))
print(agent.run("What are the files in the current directory? Describe the current directory"))
#print(agent.run("Read the text in the file 'data/economics.txt' file and then summarize this text."))
