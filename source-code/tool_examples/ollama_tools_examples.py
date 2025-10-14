import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.tool_file_dir import list_directory
from tools.tool_file_contents import read_file_contents
from tools.tool_web_search import uri_to_markdown

# print(list_directory())
# print(read_file_contents('requirements.txt'))
# print(uri_to_markdown('https://markwatson.com'))

import ollama

# Map function names to function objects
available_functions = {
    "list_directory": list_directory,
    "read_file_contents": read_file_contents,
    "uri_to_markdown": uri_to_markdown,
}

# User prompt
user_prompt = "Please list the contents of the current directory, read the 'pyproject.toml' file, and convert 'https://markwatson.com' to markdown."

# Initiate chat with the model
response = ollama.chat(
    model="llama3.2:latest",
    messages=[{"role": "user", "content": user_prompt}],
    tools=[list_directory, read_file_contents, uri_to_markdown],
)

pprint(response)

# Process the model's response
for tool_call in response.message.tool_calls or []:
    function_to_call = available_functions.get(tool_call.function.name)
    print(f"{function_to_call=}")
    if function_to_call:
        result = function_to_call(**tool_call.function.arguments)
        print(f"\n\n** Output of {tool_call.function.name}: {result}")
    else:
        print(f"\n\n** Function {tool_call.function.name} not found.")
