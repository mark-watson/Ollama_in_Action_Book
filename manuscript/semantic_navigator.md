# Semantic Navigator App Using Gradio

TBD

## Overview or Semantic Web and Linked Data

The Semantic Web, often referred to as Web 3.0 or the Web of Data, represents an ambitious vision originally proposed by Tim Berners-Lee, the inventor of the World Wide Web. While the traditional web is a collection of documents designed for human consumption—where computers essentially act as mailmen delivering pages they cannot "understand"—the Semantic Web aims to make the underlying data machine-readable. By providing a common framework that allows data to be shared and reused across application, enterprise, and community boundaries, it transforms the internet from a library of isolated silos into a massive, interconnected global database.

At the heart of this transformation is a specific technology stack designed to categorize and link information. The Resource Description Framework (RDF) serves as the standard data model, breaking down information into "triples" (subject, predicate, and object) that describe relationships between entities. To ensure these entities are unique and discoverable, they are identified using Uniform Resource Identifiers (URIs), which function like permanent, global addresses for concepts rather than just web pages. This layer is often visualized as a "Layer Cake," progressing from basic syntax to complex logic and trust protocols.

Linked Data provides the practical set of best practices required to realize this vision. It is governed by four core principles: using URIs as names for things, using HTTP URIs so people can look up those names, providing useful information via standards like RDF or SPARQL (a semantic query language), and including links to other URIs to facilitate discovery. When data is published according to these rules, it becomes part of the Linked Open Data (LOD) Cloud, a vast network of interlinked datasets, such as DBpedia or GeoNames, that allows machines to traverse the web of knowledge much like humans browse a web of pages.

In other books I have written there are examples of transforming text into RDF data (e.g., [https://leanpub.com/lovinglisp](https://leanpub.com/lovinglisp) and [https://leanpub.com/racket-ai](https://leanpub.com/racket-ai)). In this chapter we identify entities and JSON, storing this extracted data in JSON rather than RDF.


## Design Goals for the Semantic Navigator App

We develop an example web app that allows a user to paste in large blocks of text and extracts entities and relations between the identified entities.

Before looking at the code, here is what the *finished product* looks like:

![Screenshot of Semantic Navigator App](images/sm_screenshot.jpg)

## Implementation of the Semantic Navigator App Using Gradio

We will use the Gradio toolkit for creating interactive web apps. You can find detailed documentation here: [https://www.gradio.app/docs](https://www.gradio.app/docs).

The following program demonstrates the construction of a "Semantic Navigator," a web application built with Gradio that leverages Large Language Models (LLMs) to transform unstructured prose into structured knowledge. By integrating the ollama Python client, the application connects to a high-performance model to perform two distinct natural language processing tasks: named entity recognition (NER) and relationship extraction. The code implements a dual-stage workflow where users first submit raw text for analysis—triggering a system prompt that enforces a strict JSON schema for identifying persons, places, and organizations—and then interact with that data through a context-aware chatbot. This implementation showcases critical modern AI patterns, including the handling of structured LLM outputs, state management within a reactive UI, and the use of RAG-lite (Retrieval-Augmented Generation) techniques to constrain assistant responses to a specific, extracted dataset.

```python
import gradio as gr
import os
import json
from ollama import Client

client = Client(
  host="https://ollama.com",
  headers={'Authorization': os.environ.get("OLLAMA_API_KEY")}
)

MODEL = "gpt-oss:20b-cloud"

def extract_entities_and_links(text: str):
  """Uses an LLM to extract entities and links from text."""
  print("\n* Entered extract_entities_and_links function *\n")

  system_message = (
    "You are an expert in information extraction. From the given "
    "text, extract entities of type 'person', 'place', and "
    "'organization'. Also, identify links between these entities. "
    "Output the result as a single JSON object with two keys: "
    "'entities' and 'links'.\n"
    "- 'entities': list of objects, each with 'name' and 'type'.\n"
    "- 'links': list of objects, each with 'source', 'target', "
    "and 'relationship'.\n"
    "Example output format:\n"
    "{\n"
    "  \"entities\": [{\"name\": \"A\", \"type\": \"person\"}],\n"
    "  \"links\": [{\"source\": \"A\", \"target\": \"B\", "
    "\"relationship\": \"works for\"}]\n"
    "}"
  )
  
  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": text}
  ]

  try:
    response = client.chat(MODEL, messages=messages, stream=False)
    content = response['message']['content'].strip()
    
    # Strip Markdown formatting if present
    if content.startswith("```json"):
      content = content[7:-3].strip()
    elif content.startswith("```"):
      content = content[3:-3].strip()

    print("\n* Raw model output: *\n", content)
    data = json.loads(content)
    entities = data.get("entities", [])
    links = data.get("links", [])
    
    return entities, links, entities, links

  except Exception as e:
    print(f"Error during extraction: {e}")
    raise gr.Error(f"Extraction failed. Details: {e}")

def chat_responder(message, history, entities, links):
  """Streaming chatbot using extracted entities as context."""
  print("\n* Entered chat_responder function *\n")

  if not entities and not links:
    system_message = (
      "You are a helpful-but-skeptical assistant. The user has "
      "not extracted any information yet. Politely ask them to "
      "paste text above and click 'Extract' first."
    )
  else:
    system_message = (
      "You are a helpful assistant. Use ONLY the following extracted "
      "entities and links to answer questions. If the answer is "
      "not in this data, state that clearly.\n"
      f"Entities: {json.dumps(entities, indent=2)}\n"
      f"Links: {json.dumps(links, indent=2)}"
    )

  messages = [{"role": "system", "content": system_message}]
  messages.extend(history)
  messages.append({"role": "user", "content": message})

  response_text = ""
  new_history = history + [
    {"role": "user", "content": message},
    {"role": "assistant", "content": ""}
  ]

  yield new_history, ""

  for part in client.chat(MODEL, messages=messages, stream=True):
    if 'message' in part and 'content' in part['message']:
      token = part['message']['content']
      response_text += token
      new_history[-1]["content"] = response_text
      yield new_history, ""

# --- Gradio UI ---
with gr.Blocks(fill_height=True) as demo:
  gr.Markdown(
    "# Semantic Navigator\n"
    "Paste text, extract relationships, and chat with your data."
  )
  
  with gr.Row(scale=1):
    with gr.Column(scale=1):
      text_input = gr.Textbox(
        scale=1, lines=15, label="Text for Analysis", 
        placeholder="Paste paragraphs of text here to analyze..."
      )
      extract_button = gr.Button("Extract Entities & Links", 
                                 variant="primary")
    with gr.Column(scale=1):
      entities_out = gr.JSON(label="Entities", max_height="20vh")
      links_out = gr.JSON(label="Links", max_height="20vh")

  gr.Markdown("---")

  with gr.Column(scale=1):
    chatbot_display = gr.Chatbot(label="Chat Context", 
                                 max_height="20vh")
    chat_input = gr.Textbox(
      show_label=False, lines=1,
      placeholder="Ask a question about the extracted data..."
    )

  entity_state = gr.State()
  link_state = gr.State()

  extract_button.click(
    fn=extract_entities_and_links,
    inputs=[text_input],
    outputs=[entities_out, links_out, entity_state, link_state]
  )

  chat_input.submit(
    fn=chat_responder,
    inputs=[chat_input, chatbot_display, entity_state, link_state],
    outputs=[chatbot_display, chat_input]
  )

if __name__ == "__main__":
  demo.launch()
```

The core logic of this application resides in the **extract_entities_and_links** function, which serves as the bridge between raw text and structured data. It utilizes a system message to "program" the LLM to act as an information extraction expert, ensuring that the response is returned as a JSON object containing distinct lists for entities and their corresponding relationships. To ensure robustness, this function includes logic to strip common Markdown code block wrappers that models often include in their responses, and it employs internal state management via gr.State to persist this structured data across multiple user interactions without cluttering the visible interface.

The interactivity is rounded out by the **chat_responder** function which demonstrates a streaming chatbot implementation that utilizes the previously extracted entities as its primary source of truth. By dynamically injecting the JSON data into the system prompt, the assistant is constrained to answer questions based strictly on the provided context, effectively preventing hallucinations of outside information. The Gradio layout organizes these complex interactions into a clean, two-column interface, utilizing a combination of gr.Blocks, gr.Row, and gr.Column to provide a professional user experience that balances data input, structured visualization, and conversational exploration. This web app is responsive and adjusts for mobiel web browsers.