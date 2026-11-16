import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_client, get_model
from parallel_tools import search_web


def chat_with_search(user_message: str, model: str | None = None) -> str:
    client = get_client()
    active_model = model or get_model()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant. Use search_web "
                "for current information. Cite sources with URLs."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    available = {"search_web": search_web}

    # Loop until the model produces a final answer without requesting more tool calls.
    while True:
        response = client.chat(
            model=active_model,
            messages=messages,
            tools=[search_web],
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            fn = available.get(call.function.name)
            if fn is None:
                result = {"error": f"Unknown tool: {call.function.name}"}
            else:
                print(
                    f"[Tool Call] {call.function.name}("
                    f"{json.dumps(call.function.arguments)})"
                )
                result = fn(**call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": json.dumps(result),
                }
            )


if __name__ == "__main__":
    prompt = "What is new with Parallel Web Systems and its Search API?"
    print(f"Prompt: {prompt}\n")
    answer = chat_with_search(prompt)
    print("\nResponse:\n")
    print(answer)
