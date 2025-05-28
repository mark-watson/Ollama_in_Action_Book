# Long Term Persistence Using Mem0 and Chroma

Something important that we haven’t covered yet: building a persistent memory for LLM applications. Here we use two libraries:

- Mem0: persistent memory for AI Agents and LLM applications. GitHub: [https://github.com/mem0ai/mem0](https://github.com/mem0ai/mem0).
- Chroma: AI-native open-source vector database that simplifies building LLM apps by providing tools for storing, embedding, and searching embeddings.

The example in this chapter is simple and can be copied and modified for multiple applications; for example:

- Code advice agent for Python
- Store thoughts and ideas


## Code Example Using Mem0 and Chroma

This Python script demonstrates how to create a persistent memory for an AI assistant using the **mem0ai** library, **chromadb** for vector storage, and ollama for interacting with LLMs. Designed to be run repeatedly, each execution processes a user prompt, leverages past interactions stored in a local ChromaDB database, and then generates a concise, relevant response using a local Gemma model. The core idea is that the **mem0ai** library facilitates storing conversation snippets and retrieving them based on the semantic similarity to the current query. This retrieved context, referred to as "memories," is then injected into the LLM's system prompt, allowing the AI to maintain a coherent and context-aware conversation across multiple, independent runs. By persisting these memories locally, the system effectively builds a long-term conversational understanding, enabling the AI to recall and utilize previously discussed information to provide more informed and relevant answers over time, even when the script is executed as a fresh process each time.

The Chroma vector store database is stored under the file path **./db_local** and until you delete this directory, memories of old interactions are maintained.

One parameter you may want to change is the number of memories matched in the Chroma database. This can be set in the line of code **m.search(query=args.prompt, limit=5, ...)**.


```python
# Run this script repeatedly to build a persistent memory:
#
#   python mem0_persistence.py "What color is the sky?"
#   python mem0_persistence.py "What is the last color we talked about?"

# pip install mem0ai chromadb ollama

import argparse
from mem0 import Memory
from ollama import chat
from ollama import ChatResponse
           
USER_ID = "123"

config = {
  "user_id": USER_ID,
  "vector_store": {
    "provider": "chroma",
    "config": { "path": "db_local" }
  },
  "llm": {
    "provider": "ollama",
    "config": {
      "model": "gemma3:4b-it-qat",
      "temperature": 0.1,
      "max_tokens": 5000
    }
  },
}

def call_ollama_chat(model: str, messages: list[dict]) -> str:
  """
  Send a chat request to Ollama and return the assistant's reply.
  """
  response: ChatResponse = chat(
      model=model,
      messages=messages
  )
  return response.message.content
  
def main():
  p = argparse.ArgumentParser()
  p.add_argument("prompt", help="Your question")
  args = p.parse_args()

  m = Memory.from_config(config)
  print(f"User: {args.prompt}")

  rel = m.search(query=args.prompt, limit=5, user_id=USER_ID)
  mems = "\n".join(f"- {e['memory']}" for e in rel["results"])
  print("Memories:\n", mems)

  system0 = "You are a helpful assistant who answers with concise, short answers."
  system = f"{system0}\nPrevious user memories:\n{mems}"
  
  msgs = [
    {"role":"system","content":system},
    {"role":"user","content":args.prompt}
  ]
  
  reply = call_ollama_chat("gemma3:4b-it-qat", msgs)

  convo = {"role":"assistant",
           "content":
           f"QUERY: {args.prompt}\n\nANSWER:\n{reply}\n"}
  m.add(convo, user_id=USER_ID, infer=False)
    
  print(f"\n\n** convo:\n{convo}\n\n")

  print("Assistant:", reply)

if __name__=="__main__":
  main()
```

In the line **m.add(...)** set **infer=True** if you want to use the configured LLM in Ollama to filter out inserts. I almost always set this to **False** to store all questions and answers in the Chroma vector database.

## Example Output

The following output has been lightly edited, removing library deprecation warnings and extra blank lines in the output.

The first time we run the test script the vector database is empty so the user query “Name two Physical laws” does not match any previous memories stored in the Chroma vector database:

```text
 $ python mem0_persistence.py "Name two Physical laws"

User: Name two Physical laws
Memories:


** convo:
{'role': 'assistant', 'content': "QUERY: Name two Physical laws\n\nANSWER:\n1.  Newton's First Law\n2.  Law of Conservation of Energy\n"}

Assistant: 1.  Newton's First Law
2.  Law of Conservation of Energy
```

Now the Chroma data store contains one memory:

```text
$ python mem0_persistence.py "Name another different Physical law"

User: Name another different Physical law
Memories:
 - QUERY: Name two Physical laws

ANSWER:
1.  Newton's First Law
2.  Law of Conservation of Energy

** convo:
{'role': 'assistant', 'content': "QUERY: Name another different Physical law\n\nANSWER:\n1.  Newton's Third Law\n"}

Assistant: 1.  Newton's Third Law
```

Here we ask a question in a different subject domain:

```text
$ python mem0_persistence.py "What color is the sky?" 

User: What color is the sky?
Memories:
 - QUERY: Name another different Physical law

ANSWER:
1.  Newton's Third Law

- QUERY: Name two Physical laws

ANSWER:
1.  Newton's First Law
2.  Law of Conservation of Energy

** convo:
{'role': 'assistant', 'content': 'QUERY: What color is the sky?\n\nANSWER:\nBlue.\n'}

Assistant: Blue.
```

We check persistence:

```text
$ python mem0_persistence.py "What is the last color we talked about?" 

User: What is the last color we talked about?
Memories:
 - QUERY: What color is the sky?

ANSWER:
Blue.

- QUERY: Name two Physical laws

ANSWER:
1.  Newton's First Law
2.  Law of Conservation of Energy

- QUERY: Name another different Physical law

ANSWER:
1.  Newton's Third Law

** convo:
{'role': 'assistant', 'content': 'QUERY: What is the last color we talked about?\n\nANSWER:\nBlue.\n'}

Assistant: Blue.
```



