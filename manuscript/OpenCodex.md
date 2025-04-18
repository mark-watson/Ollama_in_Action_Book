# Using the Open Codex Command Line Interface Coding Agent

[Open Codex](https://github.com/ymichael/open-codex) is a fork of OpenAI’s [codex project](https://github.com/openai/codex) that can be used with most LLMs, local using Ollama as well as commercial APIs). Here we look at setting and using Open Codex with Ollama.

OpenAI Codex CLI and the clone Open Codex are a lightweight, open-source coding agents that runs locally in the terminal, integrating AI models with local code and computing tasks. Both tools support multimodal reasoning by allowing users to pass inputs like screenshots or sketches to the model, enhancing its ability to understand and generate code based on various inputs. Both tools are minimal and transparent, providing straightforward interfaces for developers to interact with AI models directly from the command line . ￼

The open-codex clone offers flexibility by allowing users to swap out models from different vendors easily. For instance, users can change the model by adjusting a simple command-line parameter, enabling the use of models like Claude or local models using Ollama. This adaptability makes it suitable for developers who wish to experiment with various AI models without being tied to a specific provider.

Both Codex CLI and open-codex aim to streamline the coding process by integrating AI capabilities directly into the developer’s workflow. 

You need to have a recent version of **npm** installed. You can then install the command line tool for Open Codex globally on your laptop using:

    npm i -g open-codex

I use the **qwen2.5:14b** with Ollama so make sure this model is installed locally:

    ollama pull qwen2.5:14b
    ollama serve

Create the directory **~/.codex** if it does not already exist and edit the file **~/.codex/config.json** to contain:

```json
{
    "provider": "ollama",
    "model": "qwen2.5:14b",
}
```

## Example Use Cases

I use Open Codex with Python, Common Lisp and Haskell projects.

### Explain the Codebase in the Current Directory

    open-codex "Describe the code in this directory"

### Modify Code and create New Source Files

    open-codex "Copy the functions summarize and answer-question from ollamaOLD.lisp to ollama.lisp, making any required changes. Create a new test file test.lisp that contains examples of using the functions defined in ollama.lisp"

