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
  print("Memories:\n", mems)  # 

  system0 = "You are a helpful assistant who answers with very concise, short answers."
  system = f"{system0}\nPrevious user memories:\n{mems}"
  
  msgs = [
    {"role":"system","content":system},
    {"role":"user","content":args.prompt}
  ]
  
  reply = call_ollama_chat("gemma3:4b-it-qat", msgs) # "gemma3:27b-it-qat", msgs)

  convo = {"role":"assistant","content":f"QUERY: {args.prompt}\n\nANSWER:\n{reply}\n"}
  m.add(convo, user_id=USER_ID, infer=False) # set to True to use LLM to infer good inserts
    
  print(f"\n\n** convo:\n{convo}\n\n")

  print("Assistant:", reply)

if __name__=="__main__":
  main()
