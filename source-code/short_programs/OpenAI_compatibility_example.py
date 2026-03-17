import os
import sys
from pathlib import Path
import openai
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model

DEFAULT_MODEL = get_model()


class OllamaClient:
    def __init__(self):
        if os.environ.get("CLOUD"):
            # Use Ollama Cloud via OpenAI-compatible endpoint
            api_key = os.environ.get("OLLAMA_API_KEY", "")
            self.client = openai.OpenAI(
                base_url="https://ollama.com/v1",
                api_key=api_key,
            )
        else:
            self.client = openai.OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="fake-key"  # Ollama doesn't require authentication locally
            )

    def chat_with_context(
        self,
        system_context: str,
        user_prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7
    ) -> str:
        try:
            messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_prompt}
            ]

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

    def chat_conversation(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

def main():
    # Initialize the client
    client = OllamaClient()

    # Example 1: Single interaction with context
    system_context = """You are a helpful AI assistant with expertise in 
    programming and technology. Provide clear, concise answers."""

    user_prompt = "Explain the concept of recursion in programming."

    response = client.chat_with_context(
        system_context=system_context,
        user_prompt=user_prompt,
        model=DEFAULT_MODEL,
        temperature=0.7
    )

    print("Response with context:")
    print(response)
    print("\n" + "="*50 + "\n")

    # Example 2: Multi-turn conversation
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."},
        {"role": "user", "content": "Can you give me a simple example?"}
    ]

    response = client.chat_conversation(
        messages=conversation,
        model=DEFAULT_MODEL
    )

    print("Conversation response:")
    print(response)

if __name__ == "__main__":
    main()
