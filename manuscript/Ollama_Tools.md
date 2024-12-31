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

```python
import sqlite3
import json
from typing import Dict, Any, List, Optional
import ollama
from functools import wraps
import re
from contextlib import contextmanager
from textwrap import dedent # for multi-line string literals

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass


def _create_sample_data(cursor):  # Helper function to create sample data
    """Create sample data for tables"""
    sample_data = {
        'example': [
            ('Example 1', 10.5),
            ('Example 2', 25.0)
        ],
        'users': [
            ('Bob', 'bob@example.com'),
            ('Susan', 'susan@test.net')
        ],
        'products': [
            ('Laptop', 1200.00),
            ('Keyboard', 75.50)
        ]
    }

    for table, data in sample_data.items():
        for record in data:
            if table == 'example':
                cursor.execute(
                    "INSERT INTO example (name, value) VALUES (?, ?) ON CONFLICT DO NOTHING",
                    record
                )
            elif table == 'users':
                cursor.execute(
                    "INSERT INTO users (name, email) VALUES (?, ?) ON CONFLICT DO NOTHING",
                    record
                )
            elif table == 'products':
                cursor.execute(
                    "INSERT INTO products (product_name, price) VALUES (?, ?) ON CONFLICT DO NOTHING",
                    record
                )


class SQLiteTool:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(SQLiteTool, cls).__new__(cls)
        return cls._instance

    def __init__(self, default_db: str = "test.db"):
        if hasattr(self, 'default_db'):  # Skip initialization if already done
            return
        self.default_db = default_db
        self._initialize_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.default_db)
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self):
        """Initialize database with tables"""
        tables = {
            'example': """
                CREATE TABLE IF NOT EXISTS example (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL
                );
            """,
            'users': """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT UNIQUE
                );
            """,
            'products': """
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    product_name TEXT,
                    price REAL
                );
            """
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for table_sql in tables.values():
                cursor.execute(table_sql)
            conn.commit()
            _create_sample_data(cursor)
            conn.commit()

    def get_tables(self) -> List[str]:
        """Get list of tables in the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> List[tuple]:
        """Get schema for a specific table"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            return cursor.fetchall()

    def execute_query(self, query: str) -> List[tuple]:
        """Execute a SQL query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                return cursor.fetchall()
            except sqlite3.Error as e:
                raise DatabaseError(f"Query execution failed: {str(e)}")

class OllamaFunctionCaller:
    def __init__(self, model: str = "llama3.2:latest"):
        self.model = model
        self.sqlite_tool = SQLiteTool()
        self.function_definitions = self._get_function_definitions()

    def _get_function_definitions(self) -> Dict:
        return {
            "query_database": {
                "description": "Execute a SQL query on the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }
            },
            "list_tables": {
                "description": "List all tables in the database",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }

    def _generate_prompt(self, user_input: str) -> str:
        prompt = dedent(f"""
            You are a SQL assistant. Based on the user's request, generate a JSON response that calls the appropriate function.
            Available functions: {json.dumps(self.function_definitions, indent=2)}

            User request: {user_input}

            Respond with a JSON object containing:
            - "function": The function name to call
            - "parameters": The parameters for the function

            Response:
        """).strip()
        return prompt

    def _parse_ollama_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")

    def process_request(self, user_input: str) -> Any:
        try:
            response = ollama.generate(model=self.model, prompt=self._generate_prompt(user_input))
            function_call = self._parse_ollama_response(response.response)

            if function_call["function"] == "query_database":
                return self.sqlite_tool.execute_query(function_call["parameters"]["query"])
            elif function_call["function"] == "list_tables":
                return self.sqlite_tool.get_tables()
            else:
                raise ValueError(f"Unknown function: {function_call['function']}")
        except Exception as e:
            raise RuntimeError(f"Request processing failed: {str(e)}")

def main():
    function_caller = OllamaFunctionCaller()
    queries = [
        "Show me all tables in the database",
        "Get all users from the users table",
        "What are the top 5 products by price?"
    ]

    for query in queries:
        try:
            print(f"\nQuery: {query}")
            result = function_caller.process_request(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
```

This code provides a natural language interface for interacting with an SQLite database. It uses a combination of Python classes, SQLite, and using Ollama for running a language model to interpret user queries and execute corresponding database operations. Below is a breakdown of the code:

- Database Setup and Error Handling: a custom exception class, DatabaseError, is defined to handle database-specific errors.
The database is initialized with three tables: example, users, and products. These tables are populated with sample data for demonstration purposes.
- SQLiteTool Class: the SQLiteTool class is a singleton that manages all SQLite database operations. Key features include:
-- Singleton Pattern: Ensures only one instance of the class is created.
-- Database Initialization: Creates tables (example, users, products) if they do not already exist.
-- Sample Data: Populates the tables with predefined sample data.
-- Context Manager: Safely manages database connections using a context manager.

**Utility Methods:**

- get_tables: Retrieves a list of all tables in the database.
- get_table_schema: Retrieves the schema of a specific table.
- execute_query: Executes a given SQL query and returns the results.

**Sample Data Creation:**

A helper function, _create_sample_data, is used to populate the database with sample data. It inserts records into the example, users, and products tables. This ensures the database has some initial data for testing and demonstration.

**OllamaFunctionCaller Class:**

The **OllamaFunctionCaller** class acts as the interface between natural language queries and database operations. Key components include:

- Integration with Ollama LLM: Uses the Ollama language model to interpret natural language queries.
- Function Definitions: Defines two main functions:
-- query_database: Executes SQL queries on the database.
-- list_tables: Lists all tables in the database.
- Prompt Generation: Converts user input into a structured prompt for the language model.
- Response Parsing: Parses the language model's response into a JSON object that specifies the function to call and its parameters.
- Request Processing: Executes the appropriate database operation based on the parsed response.

Function Definitions:

The **OllamaFunctionCaller** class defines two main functions that can be called based on user input:

- query_database: Executes a SQL query provided by the user and returns the results of the query.
- list_tables: Lists all tables in the database and is useful for understanding the database structure.

Request Processing Workflow:

The **process_request** method handles the entire workflow of processing a user query:

- Input: Takes a natural language query from the user.
- Prompt Generation: Converts the query into a structured prompt for the Ollama language model.
- Response Parsing: Parses the language model's response into a JSON object.
- Function Execution: Calls the appropriate function (query_database or list_tables) based on the parsed response.
- Output: Returns the results of the database operation.

Main test/example function:

The main function demonstrates how the system works with sample queries. It initializes the **OllamaFunctionCaller** and processes a list of example queries, such as:

- “Show me all tables in the database."
- “Get all users from the users table."
- “What are the top 5 products by price?"

For each query, the system interprets the natural language input, executes the corresponding database operation, and prints the results.

Summary:

This code creates a natural language interface for interacting with an SQLite database. It works as follows:

- Database Management: The SQLiteTool class handles all database operations, including initialization, querying, and schema inspection.
- Natural Language Processing: The OllamaFunctionCaller uses the Ollama language model to interpret user queries and map them to database functions.
- Execution: The system executes the appropriate database operation and returns the results to the user.

This approach allows users to interact with the database using natural language instead of writing SQL queries directly, making it more user-friendly and accessible.

The output looks like this:

```bash
python /Users/markw/GITHUB/OllamaExamples/tool_sqlite.py 

Query: Show me all tables in the database
Result: ['example', 'users', 'products']

Query: Get all users from the users table
Result: [(1, 'Bob', 'bob@example.com'), (2, 'Susan', 'susan@test.net')]

Query: What are the top 5 products by price?
Result: [(1, 'Laptop', 1200.0), (3, 'Laptop', 1200.0), (2, 'Keyboard', 75.5), (4, 'Keyboard', 75.5)]
```

### Tool for Summarizing Text

Tools that are identified as useful by LLMs can themselves also use LLMs. The tool defined in the file **tool_summarize_text.py** might be triggered by a user prompt such as “summarize the text in local file test1.txt” of “summarize text from web page https://markwatson.com” where it is used by other tools like reading a local file contents, fetching a web page, etc.

We will start by looking at the file **tool_summarize_text.py** and then look at an example in **example_chain_web_summary.py**.

```python
"""
Summarize text
"""

from ollama import ChatResponse
from ollama import chat


def summarize_text(text: str, context: str = "") -> str:
    """
    Summarizes text

    Parameters:
        text (str): text to summarize
        context (str): another tool's output can at the application layer can be used set the context for this tool.

    Returns:
        a string of summarized text

    """
    prompt = "Summarize this text (and be concise), returning only the summary with NO OTHER COMMENTS:\n\n"
    if len(text.strip()) < 50:
        text = context
    elif len(context) > 50:
        prompt = f"Given this context:\n\n{context}\n\n" + prompt

    summary: ChatResponse = chat(
        model="llama3.2:latest",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    return summary["message"]["content"]


# Function metadata for Ollama integration
summarize_text.metadata = {
    "name": "summarize_text",
    "description": "Summarizes input text",
    "parameters": {"text": "string of text to summarize",
                   "context": "optional context string"},
}

# Export the functions
__all__ = ["summarize_text"]
```

This Python code implements a text summarization tool using the Ollama chat model. The core function **summarize_text** takes two parameters: the main text to summarize and an optional context string. The function operates by constructing a prompt that instructs the model to provide a concise summary without additional commentary. It includes an interesting logic where if the input text is very short (less than 50 characters), it defaults to using the context parameter instead. Additionally, if there's substantial context provided (more than 50 characters), it prepends this context to the prompt. The function utilizes the Ollama chat model "llama3.2:latest" to generate the summary, structuring the request with a system message containing the prompt and a user message containing the text to be summarized. The program includes metadata for Ollama integration, specifying the function name, description, and parameter details, and exports the summarize_text function through __all__.

Here is an example of using this tool that you can find in the file **example_chain_web_summary.py**. Please note that this example also uses the web search tool that is discussed in the next section.

```python
from tool_web_search import uri_to_markdown
from tool_summarize_text import summarize_text

from pprint import pprint

import ollama

# Map function names to function objects
available_functions = {
    "uri_to_markdown": uri_to_markdown,
    "summarize_text": summarize_text,
}

memory_context = ""
# User prompt
user_prompt = "Get the text of 'https://knowledgebooks.com' and then summarize the text."

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
```

### Tool for Web Search and Fetching Web Pages

This code provides a set of functions for web searching and HTML content processing, with the main functions being **uri_to_markdown**, **search_web**, **brave_search_summaries**, and **brave_search_text**. The **uri_to_markdown** function fetches content from a given URI and converts HTML to markdown-style text, handling various edge cases and cleaning up the text by removing multiple blank lines and spaces while converting HTML entities. The **search_web** function is a placeholder that's meant to be implemented with a preferred search API, while brave_search_summaries implements actual web searching using the Brave Search API, requiring an API key from the environment variables and returning structured results including titles, URLs, and descriptions. The **brave_search_text** function builds upon brave_search_summaries by fetching search results and then using **uri_to_markdown** to convert the content of each result URL to text, followed by summarizing the content using a separate **summarize_text** function. The code also includes utility functions like **replace_html_tags_with_text** which uses BeautifulSoup to strip HTML tags and return plain text, and includes proper error handling, logging, and type hints throughout. The module is designed to be integrated with Ollama and exports **uri_to_markdown** and **search_web** as its primary interfaces.

```python
"""
Provides functions for web searching and HTML to Markdown conversion
and for returning the contents of a URI as plain text (with minimal markdown)
"""

from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import html
from ollama import chat
import json
from tool_summarize_text import summarize_text

import requests
import os
import logging
from pprint import pprint
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
if not api_key:
    raise ValueError(
        "API key not found. Set 'BRAVE_SEARCH_API_KEY' environment variable."
    )


def replace_html_tags_with_text(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    return soup.get_text()


def uri_to_markdown(a_uri: str) -> Dict[str, Any]:
    """
    Fetches content from a URI and converts HTML to markdown-style text

    Args:
        a_uri (str): URI to fetch and convert

    Returns:
        web page text converted converted markdown content
    """
    try:
        # Validate URI
        parsed = urlparse(a_uri)
        if not all([parsed.scheme, parsed.netloc]):
            return f"Invalid URI: {a_uri}"

        # Fetch content
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(a_uri, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Get title
        title = soup.title.string if soup.title else ""

        # Get text and clean up
        text = soup.get_text()

        # Clean up the text
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Remove multiple blank lines
        text = re.sub(r" +", " ", text)  # Remove multiple spaces
        text = html.unescape(text)  # Convert HTML entities
        text = text.strip()

        return f"Contents of URI {a_uri} is:\n# {title}\n\n{text}\n"

    except requests.RequestException as e:
        return f"Network error: {str(e)}"

    except Exception as e:
        return f"Error processing URI: {str(e)}"


def search_web(query: str, max_results: int = 5) -> str:
    """
    Performs a web search and returns results
    Note: This is a placeholder. Implement with your preferred search API.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'results': List of search results
            - 'count': Number of results found
            - 'error': Error message if any, None otherwise
    """

    # Placeholder for search implementation
    return {
        "results": [],
        "count": 0,
        "error": "Web search not implemented. Please implement with your preferred search API.",
    }


def brave_search_summaries(
    query,
    num_results=3,
    url="https://api.search.brave.com/res/v1/web/search",
    api_key=api_key,
):
    headers = {"X-Subscription-Token": api_key, "Content-Type": "application/json"}
    params = {"q": query, "count": num_results}

    response = requests.get(url, headers=headers, params=params)
    ret = []

    if response.status_code == 200:
        search_results = response.json()
        ret = [
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "description": replace_html_tags_with_text(result.get("description")),
            }
            for result in search_results.get("web", {}).get("results", [])
        ]
        logging.info("Successfully retrieved results.")
    else:
        try:
            error_info = response.json()
            logging.error(f"Error {response.status_code}: {error_info.get('message')}")
        except json.JSONDecodeError:
            logging.error(f"Error {response.status_code}: {response.text}")

    return ret

def brave_search_text(query, num_results=3):
    summaries = brave_search_summaries(query, num_results)
    ret = ""
    for s in summaries:
        url = s["url"]
        text = uri_to_markdown(url)
        summary = summarize_text(
            f"Given the query:\n\n{query}\n\nthen, summarize text removing all material that is not relevant to the query and then be very concise for a very short summary:\n\n{text}\n"
        )
        ret += ret + summary
    print("\n\n-----------------------------------")
    return ret

# Function metadata for Ollama integration
uri_to_markdown.metadata = {
    "name": "uri_to_markdown",
    "description": "Converts web page content to markdown-style text",
    "parameters": {"a_uri": "URI of the web page to convert"},
}

search_web.metadata = {
    "name": "search_web",
    "description": "Performs a web search and returns results",
    "parameters": {
        "query": "Search query",
        "max_results": "Maximum number of results to return",
    },
}

# Export the functions
__all__ = ["uri_to_markdown", "search_web"]
```

## Tools Wrap Up

We have looked at the implementations and examples uses for several tools. In the next chapter we continue our study of tool use with the application of judging the accuracy of output generated of LLMs: basically LLMs judging the accuracy of other LLMs to reduce hallucinations, inaccurate output, etc.