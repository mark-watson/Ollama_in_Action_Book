# Setting Up Your Computing Environment for Using Ollama and Using Book Example Programs


There is a GitHub repository that I have prepared for you, dear reader, to both support working through the examples for this book as well as hopefully provide utilities for your own projects.

You need to **git clone** the following repository:

[https://github.com/mark-watson/OllamaExamples](https://github.com/mark-watson/OllamaExamples) that contains tools I have written in Python that you can use with Ollama as well as utilities I wrote to avoid repeated code in the book examples. There are also application level example files that have the string “example” in the file names. Tool library files begin with “tool” and files starting with “Agent” contain one of several approaches to writing Agents.

## Python Build Tools

The **requirements.txt** file contains the library requirements for all code developed in this book. My preference is to use **venv** and maintain a separate Python environment for each of the few hundred Python projects I have on my laptop. I keep a personal directory **~/bin** on my **PATH** and I use the following script **venv_setup.sh** in the **~/bin** directory to use a **requirements.txt** file to set up a virtual environment:

```bash
#!/bin/zsh

# Check if the directory has a requirements.txt file
if [ ! -f "requirements.txt" ]; then
    echo "No requirements.txt file found in the current directory."
    exit 1
fi

# Create a virtual environment in the venv directory
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
pip3 install --upgrade pip

# Install dependencies from requirements.txt
pip3 install -r requirements.txt

# Display installed packages
pip3 list

echo "Virtual environment setup complete. Reactivate it with:"
echo "source venv/bin/activate"
echo ""
```

I sometimes like to use the much faster **uv** build and package management tool:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run ollama_tools_examples.py
```
 
 There are many other good options like Anaconda, miniconda, poetry, etc.
 
 