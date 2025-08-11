#!/usr/bin/env python3
"""
Demo script that lets a local Ollama model (e.g. llama3.1, gemma2, …) call
a custom Python tool named ``local_search``.  
The tool uses the same MeiliSearch code you supplied to look up a query in a
local web‑page index called ``web-pages``.

The script is a single, self‑contained file.  Install the required packages
once and then run it:

    pip install -U ollama meilisearch
    python ollama_local_search.py

After the script starts you can type any question.  If the model thinks a
search is useful it will call the ``local_search`` function; the script will
run the search and feed the result back to the model, which can then continue
the conversation.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import ollama
import meilisearch
from pprint import pprint

# --------------------------------------------------------------------------- #
# 1️⃣  The tool that will be exposed to the LLM
# --------------------------------------------------------------------------- #
def local_search(query: str) -> Dict[str, Any]:
    """
    Perform a MeiliSearch query against the ``web-pages`` index.

    Parameters
    ----------
    query: str
        The search string entered by the LLM.

    Returns
    -------
    dict
        The raw JSON response that MeiliSearch returns.  It contains keys such
        as ``hits``, ``estimatedTotalHits`` and ``processingTimeMs``.
    """
    # These values are exactly the ones you used in your working example.
    client = meilisearch.Client(
        "http://127.0.0.1:7700",
        "EEYV--rnIciRMpGdee2pmVfltPGRp1fDwUTnU1xDtPc",
    )
    index = client.index("web-pages")
    # ``search`` returns a Python ``dict`` – we just forward it.
    return index.search(query)


# Mapping from the function name that Ollama will send to the real Python
# implementation.  The key must match the function's ``__name__``.
AVAILABLE_TOOLS = {
    "local_search": local_search,
}

# --------------------------------------------------------------------------- #
# 2️⃣  Helper that adds a tool result to the message list
# --------------------------------------------------------------------------- #
def _add_tool_result(
    messages: List[Dict[str, Any]],
    tool_call: Any,
    result: Any,
) -> None:
    """
    Append the assistant‑tool call and the subsequent tool result to the
    ``messages`` list so that the next model request can see the output.

    The schema follows the OpenAI/ChatML convention used by Ollama.
    """
    # 1️⃣  The assistant message that *declares* the tool usage.
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.dumps(tool_call.function.arguments),
                    },
                }
            ],
        }
    )
    # 2️⃣  The tool’s response.
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
            "tool_call_id": tool_call.id,
        }
    )


# --------------------------------------------------------------------------- #
# 3️⃣  Main chat loop
# --------------------------------------------------------------------------- #
def main() -> None:
    # Choose the model you have locally.  Change this if you prefer another.
    model_name = "llama3.1"

    print(
        f"\n=== Ollama + local_search demo (model: {model_name}) ===\n"
        "Type a question and press ENTER.  Ctrl‑C to quit.\n"
    )

    # The conversation starts with an empty list of messages.
    messages: List[Dict[str, Any]] = []

    try:
        while True:
            # -----------------------------------------------------------------
            # 3.1️⃣  Get user input
            # -----------------------------------------------------------------
            user_input = input("\n🧑‍🦱 You: ").strip()
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            # -----------------------------------------------------------------
            # 3.2️⃣  First request – give the model the chance to call tools
            # -----------------------------------------------------------------
            response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=[local_search],  # <-- expose the function as a tool
                stream=False,         # we want the whole response at once
            )

            # The assistant's textual reply (may be empty if it only returned a tool)
            assistant_msg = response.message
            if assistant_msg.content:
                print("\n🤖 Ollama:", assistant_msg.content)

            # -----------------------------------------------------------------
            # 3.3️⃣  Did the model request a tool call?
            # -----------------------------------------------------------------
            if assistant_msg.tool_calls:
                for tool in assistant_msg.tool_calls:
                    func_name = tool.function.name
                    args = tool.function.arguments  # already a dict

                    print(f"\n🔧 Model requested tool → {func_name}({args})")

                    func = AVAILABLE_TOOLS.get(func_name)
                    if not func:
                        print(f"⚠️  No Python implementation for tool '{func_name}'")
                        continue

                    try:
                        # Call the actual Python function
                        tool_result = func(**args)  # type: ignore[arg-type]
                    except Exception as exc:  # pragma: no cover – safety net
                        tool_result = {"error": str(exc)}
                        print(f"❌ Tool execution failed: {exc}")

                    # Show the raw result (useful for debugging)
                    print("\n📄 Tool result (raw JSON):")
                    pprint(tool_result)

                    # Feed the result back to the model so it can continue.
                    _add_tool_result(messages, tool, tool_result)

                # -----------------------------------------------------------------
                # 3.4️⃣  Second request – now the model sees the tool output
                # -----------------------------------------------------------------
                follow_up = ollama.chat(
                    model=model_name,
                    messages=messages,
                    tools=[local_search],
                    stream=False,
                )
                follow_msg = follow_up.message
                if follow_msg.content:
                    print("\n🤖 Ollama (after tool):", follow_msg.content)
                # Append the assistant's follow‑up answer to the conversation.
                messages.append({"role": "assistant", "content": follow_msg.content})
            else:
                # No tool call → just store the assistant's reply.
                messages.append({"role": "assistant", "content": assistant_msg.content})

    except KeyboardInterrupt:
        print("\n👋 Bye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
