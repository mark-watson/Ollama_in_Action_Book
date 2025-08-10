### NOTE: this does not work well (yet) with local models ###

# npx playwright install

from langchain_ollama import ChatOllama
from browser_use import Agent
#from pydantic import SecretStr
from playwright.async_api import async_playwright

import asyncio

# Requirements:
# python3.11 -m venv venv
# source venv/bin/activate
# pip install langchain langchain_ollama pydantic browser_use
# playwright install

# Initialize the model
llm=ChatOllama(model="qwen2.5:14b", num_ctx=32000)

async def main():
    # Create agent with the model
    agent = Agent(
        task="Consultant Mark Watson has written Common Lisp and AI books. What musical instruments does he play?",
        llm=llm,
        save_conversation_path="logs/conversation.json"
    )
    result = await agent.run()
    print(result)

asyncio.run(main())    