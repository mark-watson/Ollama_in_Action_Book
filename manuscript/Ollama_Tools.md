# LLM Tool Calling with Ollama

There are several example Python tool utilities we will use for function calling that start with the “tool” prefix:

```bash
OllamaExamples $ ls tool*
tool_file_contents.py	tool_llm_eval.py	tool_web_search.py
tool_file_dir.py	tool_sqlite.py
tool_judge_results.py	tool_summarize_text.py
```

We postpone using the tools **tool_llm_eval.py** and **tool_judge_results.py** until the next chapter **Automatic Evaluation of LLM Results**

If you have not done so yet, please clone the repository for my Ollama book examples using:

```
git clone https://github.com/mark-watson/OllamaExamples.git
```

The source file **ollama_tools_examples.py** contains simple examples of using these tools. We will look at example code using the tools, then at the implementation of the tools. In this examples source file we first import these tools:

```python
from tool_file_dir import list_directory
from tool_file_contents import read_file_contents
from tool_web_search import uri_to_markdown

import ollama

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

Please note that the text extracted from a web page is mostly plain text. Section heads are maintained but the format is changed to markdown format. In the last (edited for brevity) listing, the HTML *H1* element with the text **Fun Stuff** is converted to markdown:

```
# Fun stuff

In addition to programming and writing my hobbies are cooking,
photography, hiking, travel, and playing the following musical
instruments: guitar, didgeridoo, and American Indian flute ...
```

## Implementing Tools for Function Calling with Ollama

You have now looked at example tool use, now we will implement the tools examples for this book. We will look at the first tool for reading and writing files in fine detail and then more briefly discuss the other tools in the [https://github.com/mark-watson/OllamaExamples](https://github.com/mark-watson/OllamaExamples) repository.

### Tool for Reading and Writing File Contents

Here is the contents of tool utility **tool_file_contents.py**:

```python
"""
Provides functions for reading and writing file contents with proper error handling
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json


def read_file_contents(file_path: str, encoding: str = "utf-8") -> str:
    """
    Reads contents from a file and returns the text

    Args:
        file_path (str): Path to the file to read
        encoding (str): File encoding to use (default: utf-8)

    Returns:
        Contents of the file as a string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"

        with path.open("r", encoding=encoding) as f:
            content = f.read()
            return f"Contents of file '{file_path}' is:\n{content}\n"

    except Exception as e:
        return f"Error reading file '{file_path}' is: {str(e)}"


def write_file_contents(
        file_path: str, content: str,
        encoding: str = "utf-8",
        mode: str = "w") -> str:
    """
    Writes content to a file and returns operation status

    Args:
        file_path (str): Path to the file to write
        content (str): Content to write to the file
        encoding (str): File encoding to use (default: utf-8)
        mode (str): Write mode ('w' for write, 'a' for append)

    Returns:
        a message string
    """
    try:
        path = Path(file_path)

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode, encoding=encoding) as f:
            bytes_written = f.write(content)

        return f"File '{file_path}' written OK."

    except Exception as e:
        return f"Error writing file '{file_path}': {str(e)}"


# Function metadata for Ollama integration
read_file_contents.metadata = {
    "name": "read_file_contents",
    "description": "Reads contents from a file and returns the content as a string",
    "parameters": {"file_path": "Path to the file to read"},
}

write_file_contents.metadata = {
    "name": "write_file_contents",
    "description": "Writes content to a file and returns operation status",
    "parameters": {
        "file_path": "Path to the file to write",
        "content": "Content to write to the file",
        "encoding": "File encoding (default: utf-8)",
        "mode": 'Write mode ("w" for write, "a" for append)',
    },
}

# Export the functions
__all__ = ["read_file_contents", "write_file_contents"]
```

This is basic non-LLM specific code:

**read_file_contents**

This function provides file reading capabilities with robust error handling with parameters:

- file_path (str): Path to the file to read
- encoding (str, optional): File encoding (defaults to "utf-8")

Features:

- Uses pathlib.Path for cross-platform path handling
- Checks file existence before attempting to read
- Returns file contents with descriptive message
- Comprehensive error handling

LLM Integration:

- Includes metadata for function discovery
- Returns descriptive string responses instead of raising exceptions

**write_file_contents**

This function handles file writing operations with built-in safety features. The parameters are:

- file_path (str): Path to the file to write
- content (str): Content to write to the file
- encoding (str, optional): File encoding (defaults to "utf-8")
- mode (str, optional): Write mode ('w' for write, 'a' for append)


Features:

- Automatically creates parent directories
- Supports write and append modes
- Uses context managers for safe file handling
- Returns operation status messages


LLM Integration:

- Includes detailed metadata for function calling
- Provides clear feedback about operations

Common Features of both functions:

- Type hints for better code clarity
- Detailed **docstrings** that **are used at runtime in the tool/function calling code**. The text in the doc strings is supplied as context to the LLM currently in use.
- Proper error handling
- UTF-8 default encoding
- Context managers for file operations
- Metadata for LLM function discovery

Design Benefits for LLM Integration: the utilities are optimized for LLM function calling by:

- Returning descriptive string responses
- Including metadata for function discovery
- Handling errors gracefully
- Providing clear operation feedback
- Using consistent parameter patterns

The code is exported via __all__ list, making it clear which functions are intended for external use.

### Tool for Getting File Directory Contents

This tool is similar to the last tool so here we just list the worker function:

```python
def list_directory(pattern: str = "*", list_dots=None) -> Dict[str, Any]:
    """
    Lists files and directories in the current working directory

    Args:
        pattern (str): Glob pattern for filtering files (default: "*")

    Returns:
        string with directory name, followed by list of files in the directory
    """
    try:
        current_dir = Path.cwd()
        files = list(current_dir.glob(pattern))

        # Convert Path objects to strings and sort
        file_list = sorted([str(f.name) for f in files])

        file_list = [file for file in file_list if not file.endswith("~")]
        if not list_dots:
            file_list = [file for file in file_list if not file.startswith(".")]

        return f"Contents of current directory: [{', '.join(file_list)}]"

    except Exception as e:
        return f"Error listing directory: {str(e)}"
```

### Tool for Accessing SQLite Databases

The example file **tool_sqlite.py** serves two purposes here:

- Test and example code: utility function **_create_sample_data** creates several database tables and the function **main** serves as an example program.
- The Python class definitions **SQLiteTool** and **OllamaFunctionCaller** are meant to be copied and used in your applications.
