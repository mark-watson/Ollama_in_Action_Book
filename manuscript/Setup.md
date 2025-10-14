# Setting Up Your Computing Environment for Using Ollama and Using Book Example Programs


There is a GitHub repository that I have prepared for you, dear reader, to both support working through the examples for this book as well as hopefully provide utilities for your own projects.

You need to **git clone** the following repository:

[https://github.com/mark-watson/Ollama_in_Action_Book](https://github.com/mark-watson/Ollama_in_Action_Book) that contains tools I have written in Python that you can use with Ollama as well as utilities I wrote to avoid repeated code in the book examples. There are also application level example files that have the string “example” in the file names. Tool library files begin with “tool” and files starting with “Agent” contain one of several approaches to writing Agents.

#### Note: the source code repository changed August 10, 2025. If you cloned the old repo please archive it and re-clone  https://github.com/mark-watson/Ollama_in_Action_Book

#### Note: Starting August 10, 2025 the GitHub Repo https://github.com/mark-watson/Ollama_in_Action_Book now contains the book's manuscript files as well as the source code for the examples.


## Python Build Tools

Starting October 2025 I exclusively use **uv** to manage dependencies and run the example programs. Change directory to and source code sub-directory and use **uv** to run an example; for example:

```bash
cd judges
uv run example_judge.py
```
 
 There are many other good options like Anaconda, miniconda, poetry, etc.
 
 