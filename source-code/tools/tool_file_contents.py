"""
Provides functions for reading and writing file contents with proper error handling
"""

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
