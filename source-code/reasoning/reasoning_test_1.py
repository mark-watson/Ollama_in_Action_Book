import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to sys.path to ensure local imports like ollama_config work
# This assumes the directory structure:
# root/
#   ollama_config.py
#   reasoning/
#     reasoning_test_1.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import ollama_config with a fallback for robustness
try:
    from ollama_config import get_model
except ImportError:
    print("Warning: ollama_config not found. Falling back to default model.")
    def get_model() -> str:
        return "deepseek-r1:latest"

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

def initialize_model(model_name: Optional[str] = None) -> ChatOllama:
    """
    Initializes the ChatOllama model.
    
    Note: ChatOllama connects to the local Ollama service only.
    LangChain's ChatOllama does not currently support Ollama Cloud authentication.
    """
    model = model_name or get_model()
    return ChatOllama(model=model, reasoning=True, temperature=0)

def run_reasoning_query(model: ChatOllama, query: str) -> Dict[str, Any]:
    """
    Executes a query and extracts both the reasoning process and the final answer.
    """
    prompt = ChatPromptTemplate.from_template("{query}")
    chain = prompt | model
    
    try:
        response: AIMessage = chain.invoke({"query": query})
        
        # reasoning=True puts thinking process in additional_kwargs['reasoning_content']
        # for compatible models (like DeepSeek-R1) via langchain-ollama.
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        
        # Fallback check for response_metadata
        if not reasoning:
            reasoning = response.response_metadata.get("reasoning", "")

        return {
            "reasoning": reasoning,
            "answer": response.content,
            "success": True
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

def main():
    # Define the query
    query = (
        "How many odd integers are greater than 0 and less than 10? "
        "Please be concise and state the final answer in JSON format."
    )
    
    # 1. Initialize
    print(f"Initializing model...")
    chat_model = initialize_model()
    
    # 2. Execute
    print(f"Invoking model with query: {query}\n")
    result = run_reasoning_query(chat_model, query)
    
    if not result["success"]:
        print(f"Error: {result.get('error')}")
        return

    # 3. Display Reasoning
    if result["reasoning"]:
        print("=== Reasoning Process ===")
        print(result["reasoning"].strip())
        print("=========================\n")
    
    # 4. Display Final Answer
    print("=== Final Answer ===")
    print(result["answer"].strip())
    
    # 5. Attempt JSON Parsing
    try:
        content = result["answer"].strip()
        # Handle potential markdown code blocks in the output
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(content)
        print("\n=== Parsed JSON Result ===")
        print(json.dumps(data, indent=2))
    except (json.JSONDecodeError, IndexError):
        print("\nNote: Answer was not in a standard JSON format or could not be parsed.")

if __name__ == "__main__":
    main()
