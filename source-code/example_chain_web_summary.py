from tool_web_search import uri_to_markdown
from tool_summarize_text import summarize_text

from pprint import pprint

# print(list_directory())
# print(read_file_contents('requirements.txt'))
# print(uri_to_markdown('https://markwatson.com'))

import ollama

# Map function names to function objects
available_functions = {
    "uri_to_markdown": uri_to_markdown,
    "summarize_text": summarize_text,
}

memory_context = ""
# User prompt
user_prompt = "Get the text of 'https://knowledgebooks.com' and then summarize the text from this web site."

# Initiate chat with the model
response = ollama.chat(
    model='llama3.2:latest',
    messages=[{"role": "user", "content": user_prompt}],
    tools=[uri_to_markdown, summarize_text],
)

# Process the model's response

pprint(response.message.tool_calls)

for tool_call in response.message.tool_calls or []:
    function_to_call = available_functions.get(tool_call.function.name)
    print(
        f"\n***** {function_to_call=}\n\nmemory_context[:70]:\n\n{memory_context[:70]}\n\n*****\n"
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
