"""
Wrapper for book example tools for smloagents compatibility
"""
from pathlib import Path

from smolagents import tool
from typing import Optional
from pprint import pprint

from tool_file_dir import list_directory
from tool_file_contents import read_file_contents

import ollama

@tool
def sa_list_directory() -> str:
    """
    Lists files and directories in the current working directory

    Args:
        None

    Returns:
        string with directory name, followed by list of files in the directory
    """
    lst = list_directory()
    pprint(lst)
    return lst

@tool
def sa_read_file_contents(file_path: str) -> str:
    """
    Reads contents from a file and returns the text

    Args:
        file_path: Path to the file to read

    Returns:
        Contents of the file as a string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"

        with path.open("r", encoding="utf-8") as f:
            content = f.read()
            return f"Contents of file '{file_path}' is:\n{content}\n"

    except Exception as e:
        return f"Error reading file '{file_path}' is: {str(e)}"

@tool
def sa_summarize_directory() -> str:
    """
    Summarizes the files and directories in the current working directory

    Args:
        None

    Returns:
        string with directory name, followed by summary of files in the directory and by
        a summary of what the software in the current directory does.
    """
    lst = list_directory()
    print(f"{lst=}")
    response = ollama.chat(
        model="llama3.2:latest",
        messages=[
            {"role": "system", "content": f"Consider the contents of the current directory: {lst}"},
            {"role": "user", "content": "Summarize the contents of the current directory. Make an educated guess as to what the major purposes of each file is, given the file name."},
        ],
        #tools=[read_file_contents],
    )

    return f"Summary of directory:{response.message.content}\n"
