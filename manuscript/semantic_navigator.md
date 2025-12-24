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
  """Uses an LLM to extract entities and links."""
  print("\n* Entering extraction function *\n")

  system_message = (
    "You are an expert in information extraction. From the "
    "given text, extract entities of type 'person', 'place', "
    "and 'organization'. Also, identify links between these "
    "entities. Output as a single JSON object with two keys: "
    "'entities' and 'links'.\n"
    "- 'entities': list of {name, type}\n"
    "- 'links': list of {source, target, relationship}\n"
    "Example:\n"
    "{\"entities\": [{\"name\": \"A\", \"type\": \"person\"}], "
    "\"links\": [{\"source\": \"A\", \"target\": \"B\", "
    "\"relationship\": \"works for\"}]}"
  )

  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": text}
  ]

  try:
    res = client.chat(MODEL, messages=messages, stream=False)
    content = res['message']['content'].strip()
    
    # Strip markdown code blocks
    if content.startswith("```json"):
      content = content[7:-3].strip()
    elif content.startswith("```"):
      content = content[3:-3].strip()

    print("\n* Raw output: *\n", content)
    data = json.loads(content)
    ents = data.get("entities", [])
    lnks = data.get("links", [])
    
    return ents, lnks, ents, lnks

  except Exception as e:
    print(f"Error: {e}")
    raise gr.Error(f"Extraction failed: {e}")

def chat_responder(message, history, entities, links):
  """Answers questions based on extracted data."""
  print("\n* Entering chat function *\n")

  if not entities and not links:
    sys_msg = (
      "You are a helpful assistant. The user has not "
      "extracted info yet. Ask them to paste text and "
      "click 'Extract' first."
    )
  else:
    sys_msg = (
      "Use ONLY this info to answer. If not found, say so.\n"
      f"Entities: {json.dumps(entities, indent=2)}\n"
      f"Links: {json.dumps(links, indent=2)}"
    )

  msgs = [{"role": "system", "content": sys_msg}]
  msgs.extend(history)
  msgs.append({"role": "user", "content": message})

  response_text = ""
  new_hist = history + [
    {"role": "user", "content": message},
    {"role": "assistant", "content": ""}
  ]

  yield new_hist, ""

  for part in client.chat(MODEL, messages=msgs, stream=True):
    if 'message' in part and 'content' in part['message']:
      token = part['message']['content']
      response_text += token
      new_hist[-1]["content"] = response_text
      yield new_hist, ""

# --- Gradio UI ---
with gr.Blocks(fill_height=True) as demo:
  gr.Markdown(
    "# Semantic Navigator\n"
    "Extract entities and chat with your data."
  )
  
  with gr.Row(scale=1):
    with gr.Column(scale=1):
      text_input = gr.Textbox(
        scale=1,
        lines=10,
        label="Input Text", 
        placeholder="Paste text here..."
      )
      extract_btn = gr.Button("Extract", variant="primary")
    with gr.Column(scale=1):
      ents_out = gr.JSON(label="Entities", max_height="20vh")
      lnks_out = gr.JSON(label="Links", max_height="20vh")

  gr.Markdown("---")

  with gr.Column(scale=1):
    chat_disp = gr.Chatbot(label="Chat", max_height="20vh")
    chat_in = gr.Textbox(
      show_label=False, 
      placeholder="Ask a question...", 
      lines=1
    )

  e_state = gr.State()
  l_state = gr.State()

  extract_btn.click(
    fn=extract_entities_and_links,
    inputs=[text_input],
    outputs=[ents_out, lnks_out, e_state, l_state]
  )

  chat_in.submit(
    fn=chat_responder,
    inputs=[chat_in, chat_disp, e_state, l_state],
    outputs=[chat_disp, chat_in]
  )

if __name__ == "__main__":
  demo.launch()
```