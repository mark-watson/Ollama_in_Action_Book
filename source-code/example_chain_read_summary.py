from tool_file_dir import list_directory
from tool_file_contents import read_file_contents
from tool_web_search import uri_to_markdown
from tool_summarize_text import summarize_text

from pprint import pprint

# print(list_directory())
# print(read_file_contents('requirements.txt'))
# print(uri_to_markdown('https://markwatson.com'))

import ollama

# Map function names to function objects
available_functions = {
    "list_directory": list_directory,
    "read_file_contents": read_file_contents,
    "uri_to_markdown": uri_to_markdown,
    "summarize_text": summarize_text,
}

memory_context = ""

# User prompt
user_prompt = "Read the text in the file 'data/economics.txt' file and then summarize this text."

# Initiate chat with the model
response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {"role": "system", "content": f"Current conversation memory: {memory_context}"},
        {"role": "user", "content": user_prompt},
    ],
    tools=[read_file_contents, summarize_text],
)

print(f"{response.message.content=}")

# Process the model's response

for tool_call in response.message.tool_calls or []:
    function_to_call = available_functions.get(tool_call.function.name)
    print(
        f"\n***** {function_to_call=}\n\nmemory_context:\n\n{memory_context}\n\n*****\n"
    )
    if function_to_call:
        print()
        if len(memory_context) > 10:
            tool_call.function.arguments["context"] = memory_context
        print("\n* * tool_call.function.arguments:\n")
        pprint(tool_call.function.arguments)
        print(f"Arguments for {function_to_call.__name__}: {tool_call.function.arguments}")
        result = function_to_call(**tool_call.function.arguments)  # , memory_context)
        print(f"\n\n** Output of {tool_call.function.name}: {result}")
        memory_context = memory_context + "\n\n" + result
    else:
        print(f"\n\n** Function {tool_call.function.name} not found.")
