import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_client, get_model
from parallel_tools import research_topic, search_web


def run_research_assistant(user_message: str, model: str | None = None) -> str:
    client = get_client()
    active_model = model or get_model()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant with two tools: "
                "use search_web for fast factual lookups, and use research_topic "
                "when the user asks for an in-depth report with confidence scores "
                "and verified citations. Always include source URLs in your answer."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    available = {
        "search_web": search_web,
        "research_topic": research_topic,
    }

    while True:
        response = client.chat(
            model=active_model,
            messages=messages,
            tools=[search_web, research_topic],
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
    prompt = (
        "Run a deep research report on recent open-weight small language models "
        "that support native tool calling, and summarize the findings with citations."
    )
    print(f"Prompt: {prompt}\n")
    answer = run_research_assistant(prompt)
    print("\nResponse:\n")
    print(answer)
