# Ensure OLLAMA_API_KEY is set  in ENV

# Note: the answer.contents is nicely converted from HTML to Markdown.

import ollama
from ollama import Client
import os
from pprint import pprint

client = Client(
    host="https://ollama.com",
    headers={'Authorization': os.environ.get("OLLAMA_API_KEY")}
)

def generate(text):
    messages = [
      {
        'role': 'user',
        'content': text,
      },
    ]

    response = client.chat('gpt-oss:20b', messages=messages, stream=False)
    return response['message']['content']

P = "Summarize the following Markdown text, returning only plain text. Markdown text:\n\n"

def clean_web_query(query, max_results=2):
    ret = []
    response = ollama.web_search(query)
    for answer in response['results']:
        markdown = answer.content
        clean = generate(P + markdown)
        #print(f"\n\n{answer.url}\n\n{answer.content}\n\n")
        #pprint(answer)
        ret.append(clean)
    return ret
   
test = clean_web_query("AI Consultant Mark Watson works with AI, Lisp, Java, Clojure, and the Semantic Web.", max_results=1)

for clean in test:
    print(f"{clean}\n\n\n")