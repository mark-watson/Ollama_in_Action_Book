"""
File Directory Module
Provides functions for listing files in the current directory
"""

from typing import Dict, List, Any
from typing import Optional
from pathlib import Path

import os


def list_directory() -> Dict[str, Any]:
    """
    Lists files and directories in the current working directory

    Args:
        None

    Returns:
        string containing the current directory name, followed by list of files in the directory
    """

    try:
        current_dir = Path.cwd()
        files = list(current_dir.glob("*"))

        # Convert Path objects to strings and sort
        file_list = sorted([str(f.name) for f in files])

        file_list = [file for file in file_list if not file.endswith("~")]
        file_list = [file for file in file_list if not file.startswith(".")]

        return f"Contents of current directory {current_dir} is: [{', '.join(file_list)}]"

    except Exception as e:
        return f"Error listing directory: {str(e)}"


# Function metadata for Ollama integration
list_directory.metadata = {
    "name": "list_directory",
    "description": "Lists files and directories in the current working directory",
    "parameters": {},
}

# Export the function
__all__ = ["list_directory"]
