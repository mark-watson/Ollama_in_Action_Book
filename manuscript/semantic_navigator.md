# Semantic Navigator App Using Gradio

TBD

## Overview or Semantic Web and Linked Data

## Design Goals for the Semantic Navigator App

TBD

![Screenshot of Semantic Navigator App](images/sm_screenshot.jpg)

## Implementation of the Semantic Navigator App Using Gradio

TBD

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