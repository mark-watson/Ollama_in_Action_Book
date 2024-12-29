# LLM Tool Calling with Ollama

TBD update for file layout for examples

I wrote a separate Python library that contains the tools used in this book. You can clone the repository for my Ollama book examples using:

```
git clone https://github.com/mark-watson/OllamaEx.git
```

You can put the following in your project's **requirements.txt** file to use this library:

```
git+https://github.com/mark-watson/OllamaEx.git
```

Currently this library supplies the following tools:

- tool_file_contents.py
- tool_file_dir.py
- tool_sqlite.py
- tool_web_search.py
- tool_summarize.py
- tool_judge_results.py
- tool_llm_eval.py

The last two tools listed will be discusses in the next chapter **Automatic Evaluation of LLM Results**

You should also clone the GitHub repository that contains most examples in this book:

```
git clone https://github.com/mark-watson/Ollama-book-examples.git
```

The source file **ollama_tools_examples.py** contains simple examples of using these tools. In this source file we first import these tools:

```python
from ollama_tools.file_dir import list_directory
from ollama_tools.file_contents import read_file_contents
from ollama_tools.web_search import uri_to_markdown

# Map function names to function objects
available_functions = {
    'list_directory': list_directory,
    'read_file_contents': read_file_contents,
    'uri_to_markdown': uri_to_markdown,
}
```

We will now look at examples using these tools.

```python
# User prompt
user_prompt = "Please list the contents of the current directory, read the 'requirements.txt' file, and convert 'https://markwatson.com' to markdown."

# Initiate chat with the model
response = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': user_prompt}],
    tools=[list_directory, read_file_contents, uri_to_markdown],
)

# Process the model's response
for tool_call in response.message.tool_calls or []:
    function_to_call = available_functions.get(tool_call.function.name)
    print(f"{function_to_call=}")
    if function_to_call:
        result = function_to_call(**tool_call.function.arguments)
        print(f"Output of {tool_call.function.name}: {result}")
    else:
        print(f"Function {tool_call.function.name} not found.")
```

Here is sample output from using these three tools (most output removed for brevity and blank lines added for clarity):

```
$ python ollama_tools_examples.py

function_to_call=<function read_file_contents at 0x104fac9a0>

Output of read_file_contents: {'content': 'git+https://github.com/mark-watson/Ollama_Tools.git\nrequests\nbeautifulsoup4\naisuite[ollama]\n\n', 'size': 93, 'exists': True, 'error': None}

function_to_call=<function list_directory at 0x1050389a0>
Output of list_directory: {'files': ['.git', '.gitignore', 'LICENSE', 'Makefile', 'README.md', 'ollama_tools_examples.py', 'requirements.txt', 'venv'], 'count': 8, 'current_dir': '/Users/markw/GITHUB/Ollama-book-examples', 'error': None}

function_to_call=<function uri_to_markdown at 0x105038c20>

Output of uri_to_markdown: {'content': 'Read My Blog on Blogspot\n\nRead My Blog on Substack\n\nConsulting\n\nFree Mentoring\n\nFun stuff\n\nMy Books\n\nOpen Source\n\n Privacy Policy\n\n# Mark Watson AI Practitioner and Consultant Specializing in Large Language Models, LangChain/Llama-Index Integrations, Deep Learning, and the Semantic Web\n\n# I am the author of 20+ books on Artificial Intelligence, Python, Common Lisp, Deep Learning, Haskell, Clojure, Java, Ruby, Hy language, and the Semantic Web. I have 55 US Patents.\n\nMy customer list includes: Google, Capital One, Babylist, Olive AI, CompassLabs, Mind AI, Disney, SAIC, Americast, PacBell, CastTV, Lutris Technology, Arctan Group, Sitescout.com, Embed.ly, and Webmind Corporation.

 ...

 # Fun stuff\n\nIn addition to programming and writing my hobbies are cooking,\n photography, hiking, travel, and playing the following musical instruments: guitar, didgeridoo, and American Indian flute:\n\nMy guitar playing: a boogie riff\n\nMy didgeridoo playing\n\nMy Spanish guitar riff\n\nPlaying with George (didgeridoo), Carol and Crystal (drums and percussion) and Mark (Indian flute)\n\n# Open Source\n\nMy Open Source projects are hosted on my github account so please check that out!

 ...

Hosted on Cloudflare Pages\n\n © Mark Watson 1994-2024\n\nPrivacy Policy', 'title': 'Mark Watson: AI Practitioner and Author of 20+ AI Books | Mark Watson', 'error': None}
```

Please not that the text extracted from a web page is almost plain text. Section heads are maintained but the format is changed to markdown format. In the last (edited for brevity) listing, the HTML *H1* element with the text **Fun Stuff** is converted to markdown:

```
# Fun stuff

In addition to programming and writing my hobbies are cooking,
photography, hiking, travel, and playing the following musical
instruments: guitar, didgeridoo, and American Indian flute ...
```


