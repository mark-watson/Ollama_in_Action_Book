"""
Provides functions detecting hallucinations by other LLMs
"""

from typing import Optional, Dict, Any
from pathlib import Path
from pprint import pprint
import json
from ollama import ChatResponse
from ollama import chat

def read_anti_hallucination_template() -> str:
    """
    Reads the anti-hallucination template file and returns the content
    """
    template_path = Path(__file__).parent / "templates" / "anti_hallucination.txt"
    with template_path.open("r", encoding="utf-8") as f:
        content = f.read()
        return content

TEMPLATE = read_anti_hallucination_template()

def detect_hallucination(user_input: str, context: str, output: str) -> str:
    """
    Given user input, context, and LLM output, detect hallucination

    Args:
        user_input (str): User's input text prompt
        context (str): Context text for LLM
        output (str): LLM's output text that is to be evaluated as being a hallucination)

    Returns: JSON data:
     {
       "score": <your score between 0.0 and 1.0>,
       "reason": [
         <list your reasoning as bullet points>
       ]
     }
    """
    prompt = TEMPLATE.format(input=user_input, context=context, output=output)
    response: ChatResponse = chat(
        model="llama3.2:latest",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": output},
        ],
    )
    try:
        return json.loads(response.message.content)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {response.message.content}")
    return {"score": 0.0, "reason": ["Error decoding JSON"]}


# Export the functions
__all__ = ["detect_hallucination"]

## Test only code:

def main():
    def separator(title: str):
        """Prints a section separator"""
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)

    # Test file writing
    separator("Detect hallucination from a LLM")

    test_prompt = "Sally is 55, John is 18, and Mary is 31. What are pairwise combinations of the absolute value of age differences?"
    test_context = "Double check all math results."
    test_output = "Sally and John:  55 - 18 = 31. Sally and Mary:  55 - 31 = 24. John and Mary:  31 - 18 = 10."
    judgement = detect_hallucination(test_prompt, test_context, test_output)
    print(f"\n** JUDGEMENT ***\n")
    pprint(judgement)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
