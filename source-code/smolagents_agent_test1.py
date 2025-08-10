from smolagents.agents import ToolCallingAgent
from smolagents import LiteLLMModel

from smolagents_tools import sa_list_directory
from smolagents_tools import sa_summarize_directory
from smolagents_tools import sa_read_file_contents


model = LiteLLMModel(
    model_id="ollama_chat/llama3.2:latest",
    api_base="http://localhost:11434", # replace with remote open-ai compatible server if necessary
    api_key="your-api-key" # replace with API key if necessary
)
model.set_verbose=True

agent = ToolCallingAgent(tools=[sa_list_directory,
                                sa_summarize_directory,
                                sa_read_file_contents],
                         model=model,
                         #planning_interval=2, # enables pre-planning for multiple steps
                         max_iterations=3, # maximum number of iterations
                         add_base_tools=False) # set to True to use built-in tools

#agent.write_inner_memory_from_logs() # creates an inner memory of the agent's logs for the LLM to view

#print(agent.run("List the Python programs in the current directory, and then tell me which Python programs in the current directory evaluate the performance of LLMs?\n\n"))
print(agent.run("What are the files in the current directory? Describe the current directory"))
#print(agent.run("Read the text in the file 'data/economics.txt' file and then summarize this text."))
