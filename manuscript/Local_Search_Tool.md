# Ollama Tool Use: Local Meilisearch Web Index Search



## How it works

### 1. Tool definition

Python fucntion **local_search** is a normal Python function that talks to your MeiliSearch instance.

The function is added to AVAILABLE_TOOLS so the script can look it up by name when the model asks for it.

### 2. Expose the function

We set up: **ollama.chat(..., tools=[local_search])** to tell the Ollama server to generate a JSON schema for local_search and make it available for the model to call (function‑calling / tool usage).

### 3. Tool call handling – After the first model response we inspect response.message.tool_calls. For each call we:

- Invoke the matching Python function.
- Print the raw result.
- Append a assistant‑tool‑call message and a tool message containing the result to the messages list.
- Send a second request so the model can incorporate the result into its answer.

### 4. Conversation loop – The script keeps a list of messages (messages); each user turn and each assistant/tool turn are appended, giving the model a full chat history.

```python

```

