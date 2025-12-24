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
    """
    Uses an LLM to extract entities and links from the user-provided text.
    """
    print("\n* Entered extract_entities_and_links function *\n")

    system_message = """You are an expert in information extraction. From the given text, extract entities of type 'person', 'place', and 'organization'. Also, identify links between these entities. Output the result as a single JSON object with two keys: 'entities' and 'links'.
- 'entities' should be a list of objects, each with 'name' and 'type'.
- 'links' should be a list of objects, each with 'source' (entity name), 'target' (entity name), and 'relationship'.
Example output format:
{
  "entities": [
    {"name": "John Doe", "type": "person"},
    {"name": "New York", "type": "place"},
    {"name": "Acme Corp", "type": "organization"}
  ],
  "links": [
    {"source": "John Doe", "target": "Acme Corp", "relationship": "works for"},
    {"source": "John Doe", "target": "New York", "relationship": "lives in"}
  ]
}
"""
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": text}]

    try:
        # Use stream=False for a single, complete JSON response
        response = client.chat(MODEL, messages=messages, stream=False)
        content = response['message']['content']
        
        # The model might wrap the JSON in ```json ... ```, so strip that.
        if content.strip().startswith("```json"):
            content = content.strip()[7:-3].strip()
        elif content.strip().startswith("```"):
             content = content.strip()[3:-3].strip()


        print("\n* Raw model output for extraction: *\n", content)
        data = json.loads(content)
        
        entities = data.get("entities", [])
        links = data.get("links", [])
        
        # Return data for both the JSON components and the State components
        return entities, links, entities, links

    except Exception as e:
        print(f"Error during extraction: {e}")
        raise gr.Error(f"Failed to extract information. The model may have returned an invalid format. Details: {e}")

def chat_responder(
    message: str,
    history: list[dict],
    entities: list[dict],
    links: list[dict]
):
    """
    A streaming chatbot that answers questions based on the extracted entities and links.
    """
    print("\n* Entered chat_responder function *\n")

    if not entities and not links:
        system_message = "You are a helpful-but-skeptical assistant. The user has not extracted any information from their text yet. Politely ask them to paste text in the box above and click the 'Extract' button before asking questions."
    else:
        system_message = f"""You are a helpful assistant. The user has provided a text, and the following entities and links have been extracted from it. Use ONLY this information to answer the user's questions. If the answer cannot be found in the provided information, state that clearly.
Extracted Entities:
{json.dumps(entities, indent=2)}
Extracted Links:
{json.dumps(links, indent=2)}
"""

    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history
    messages.extend(history)
    
    # Add the current user message
    messages.append({"role": "user", "content": message})

    response_text = ""
    
    # The history component needs to be updated with the user's new message immediately.
    # Then, we stream the bot's response.
    new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]

    # Yield the history to update the chatbot UI with the user's message
    yield new_history, ""

    # Stream the model's response
    for part in client.chat(
        MODEL,
        messages=messages,
        stream=True,
    ):
        if 'message' in part and 'content' in part['message']:
            token = part['message']['content']
            response_text += token
            new_history[-1]["content"] = response_text
            # Yield the updated history and an empty string to clear the input box
            yield new_history, ""

# --- Gradio UI ---
with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# Semantic Navigator\nPaste text, extract entities and relationships, and then chat with your data.")
    
    # Upper section: Text input and extraction output
    with gr.Row(scale=1):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                scale=1,
                lines=15,
                label="Text for Analysis", 
                placeholder="Paste a block of text here to analyze. The model works best with a few paragraphs of text."
            )
            extract_button = gr.Button("Extract Entities & Links", variant="primary")
        with gr.Column(scale=1):
            entities_output = gr.JSON(label="Extracted Entities", scale=1, max_height="20vh")
            links_output = gr.JSON(label="Extracted Links", scale=1, max_height="20vh")

    gr.Markdown("---")

    # Lower section: Chat interface
    with gr.Column(scale=1):
        chatbot_display = gr.Chatbot(label="Chat About Your Text", scale=1, max_height="15vh")
        chat_input = gr.Textbox(
            show_label=False, 
            placeholder="Ask a question about the extracted information...", 
            lines=1
        )

    # Hidden state to hold the extracted data for the chat function
    entity_state = gr.State()
    link_state = gr.State()

    # --- Event Handlers ---
    extract_button.click(
        fn=extract_entities_and_links,
        inputs=[text_input],
        outputs=[entities_output, links_output, entity_state, link_state],
        api_name="extract"
    )

    chat_input.submit(
        fn=chat_responder,
        inputs=[chat_input, chatbot_display, entity_state, link_state],
        outputs=[chatbot_display, chat_input],
        api_name="chat"
    )

if __name__ == "__main__":
    demo.launch()
