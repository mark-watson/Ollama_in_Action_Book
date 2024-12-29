# Automatic Evaluation of LLM Results

TBD

## Tool For Judging LLM Results

TBD

The following listing sows the tool **tool_judge_results.py**:

```python
"""
Judge results from LLM generation from prompts
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json
import re
from pprint import pprint

import ollama

client = ollama.Client()

def judge_results(original_prompt: str, llm_gen_results: str) -> Dict[str, str]:
    """
    Takes an original prompt to a LLM and the output results

    Args:
        original_prompt (str): original prompt to a LLM
        llm_gen_results (str): output from the LLM that this function judges for accuracy

    Returns:
        result: str: string that is one character with one of these values:
            - 'B': Bad result
            - 'G': A Good result
    """
    try:
        messages = [
            {"role": "system", "content": "Always judge this output for correctness."},
            {"role": "user", "content": f"Evaluate this output:\n\n{llm_gen_results}\n\nfor this prompt:\n\n{original_prompt}\n\nDouble check your work and explain your thinking in a few sentences. End your output with a Y or N answer"},
        ]

        response = client.chat(
            model="qwen2.5-coder:14b", # "llama3.2:latest",
            messages=messages,
        )

        r = response.message.content.strip()
        print(f"\n\noriginal COT response:\n\n{r}\n\n")

        # look at the end of the response for the Y or N judgement
        s = r.lower()
        # remove all non-alphabetic characters:
        s = re.sub(r'[^a-zA-Z]', '', s).strip()

        return {'judgement': s[-1].upper(), 'reasoning': r[1:].strip()}

    except Exception as e:
        print(f"\n\n***** {e=}\n\n")
        return {'judgement': 'E', 'reasoning': str(e)}  # on any error, assign 'E' result
```

This Python code defines a function judge_results that takes an original prompt sent to a Large Language Model (LLM) and the generated response from the LLM, then attempts to judge the accuracy of the response.

Here's a breakdown of the code:

The main function **judge_results** takes two parameters:

- original_prompt: The initial prompt sent to an LLM
- llm_gen_results: The output from the LLM that needs evaluation

The function **judge_results** returns a dictionary with two keys:

- judgement: Single character ('B' for Bad, 'G' for Good, 'E' for Error)
- reasoning: Detailed explanation of the judgment

The evaluation process is:

- Creates a conversation with two messages:
-- System message: Sets the context for evaluation
-- User message: Combines the original prompt and results for evaluation
- Uses the Qwen 2.5 Coder (14B parameter) model through Ollama
- Expects a Y/N response at the end of the evaluation

## Sample output

```
$ cd OllamaEx
$ python example_judge.py 

==================================================
 Judge output from a LLM
==================================================

==================================================
 First test: should be Y, or good
==================================================


original COT response:

The given output correctly calculates the absolute value of age differences for each pair:

- Sally (55) and John (18): \( |55 - 18| = 37 \)
- Sally (55) and Mary (31): \( |55 - 31| = 24 \)
- John (18) and Mary (31): \( |31 - 18| = 13 \)

These calculations are accurate, matching the prompt's requirements. Therefore, the answer is Y.



** JUDGEMENT ***

judgement={'judgement': 'Y', 'reasoning': "The given output correctly calculates the absolute value of age differences for each pair:\n\n- Sally (55) and John (18): \\( |55 - 18| = 37 \\)\n- Sally (55) and Mary (31): \\( |55 - 31| = 24 \\)\n- John (18) and Mary (31): \\( |31 - 18| = 13 \\)\n\nThese calculations are accurate, matching the prompt's requirements. Therefore, the answer is Y."}

==================================================
 Second test: should be N, or bad
==================================================


original COT response:

Let's evaluate the given calculations step by step:

1. Sally (55) - John (18) = 37. The difference is calculated as 55 - 18, which equals 37.
2. Sally (55) - Mary (31) = 24. The difference is calculated as 55 - 31, which equals 24.
3. John (18) - Mary (31) = -13. However, the absolute value of this difference is |18 - 31| = 13.

The given output shows:
- Sally and John: 55 - 18 = 31. This should be 37.
- Sally and Mary: 55 - 31 = 24. This is correct.
- John and Mary: 31 - 18 = 10. This should be 13.

The output contains errors in the first and third calculations. Therefore, the answer is:

N



** JUDGEMENT ***

judgement={'judgement': 'N', 'reasoning': "et's evaluate the given calculations step by step:\n\n1. Sally (55) - John (18) = 37. The difference is calculated as 55 - 18, which equals 37.\n2. Sally (55) - Mary (31) = 24. The difference is calculated as 55 - 31, which equals 24.\n3. John (18) - Mary (31) = -13. However, the absolute value of this difference is |18 - 31| = 13.\n\nThe given output shows:\n- Sally and John: 55 - 18 = 31. This should be 37.\n- Sally and Mary: 55 - 31 = 24. This is correct.\n- John and Mary: 31 - 18 = 10. This should be 13.\n\nThe output contains errors in the first and third calculations. Therefore, the answer is:\n\nN"}
```


## Evaluating LLM Responses Given a Chat History

TBD

The following listing shows the tool utility **tool_llm_results**:

```python
import json
from typing import List, Dict, Optional, Iterator
import ollama
from ollama import GenerateResponse


def clean_json_response(response: str) -> str:
    """
    Cleans the response string by removing markdown code blocks and other formatting
    """
    # Remove markdown code block indicators
    response = response.replace("json", "").replace("```", "")
    # Strip whitespace
    response = response.strip()
    return response

def evaluate_llm_conversation(
    chat_history: List[Dict[str, str]],
    evaluation_criteria: Optional[List[str]] = None,
    model: str = "llama3.1" # older model that is very good at generating JSON
) -> Dict[str, any]:
    """
    Evaluates a chat history using Ollama to run the evaluation model.

    Args:
        chat_history: List of dictionaries containing the conversation
        evaluation_criteria: Optional list of specific criteria to evaluate
        model: Ollama model to use for evaluation

    Returns:
        Dictionary containing evaluation results
    """
    if evaluation_criteria is None:
        evaluation_criteria = [
            "Response accuracy",
            "Coherence and clarity",
            "Helpfulness",
            "Task completion",
            "Natural conversation flow"
        ]

    # Format chat history for evaluation
    formatted_chat = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in chat_history
    ])

    # Create evaluation prompt
    evaluation_prompt = f"""
    Please evaluate the following conversation between a user and an AI assistant.
    Focus on these criteria: {', '.join(evaluation_criteria)}

    Conversation:
    {formatted_chat}

    Provide a structured evaluation with:
    1. Scores (1-10) for each criterion
    2. Brief explanation for each score
    3. Overall assessment
    4. Suggestions for improvement

    Format your response as JSON.
    """

    try:
        # Get evaluation from Ollama
        response: GenerateResponse | Iterator[GenerateResponse] = ollama.generate(
            model=model,
            prompt=evaluation_prompt,
            system="You are an expert AI evaluator. Provide detailed, objective assessments in JSON format."
        )

        response_clean: str = clean_json_response(response['response'])

        # Parse the response to extract JSON
        try:
            evaluation_result = json.loads(response_clean)
        except json.JSONDecodeError:
            # Fallback if response isn't proper JSON
            evaluation_result = {
                "error": "Could not parse evaluation as JSON",
                "raw_response": response_clean
            }

        return evaluation_result

    except Exception as e:
        return {
            "error": f"Evaluation failed: {str(e)}",
            "status": "failed"
        }

# Example usage
if __name__ == "__main__":
    # Sample chat history
    sample_chat = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about it."},
        {"role": "assistant", "content": "Paris is the largest city in France and serves as the country's political, economic, and cultural center. It's known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."}
    ]

    # Run evaluation
    result = evaluate_llm_conversation(sample_chat)
    print(json.dumps(result, indent=2))
```

We will use these five evaluation criteria:

- Response accuracy
- Coherence and clarity
- Helpfulness
- Task completion
- Natural conversation flow

The main function **evaluate_llm_conversation** uses these steps:

- Receives chat history and optional parameters
- Formats the conversation into a readable string
- Creates a detailed evaluation prompt
- Sends prompt to Ollama for evaluation
- Cleans and parses the response
- Returns structured evaluation results

### Sample Output

```
$ cd OllamaEx 
$ python tool_llm_eval.py 
{
  "evaluation": {
    "responseAccuracy": {
      "score": 9,
      "explanation": "The assistant correctly answered the user's question about the capital of France, and provided accurate information when the user asked for more details."
    },
    "coherenceAndClarity": {
      "score": 8,
      "explanation": "The assistant's responses were clear and easy to understand. However, there was a slight shift in tone from a simple answer to a more formal description."
    },
    "helpfulness": {
      "score": 9,
      "explanation": "The assistant provided relevant information that helped the user gain a better understanding of Paris. The response was thorough and answered the user's follow-up question."
    },
    "taskCompletion": {
      "score": 10,
      "explanation": "The assistant completed both tasks: providing the capital of France and elaborating on it with additional context."
    },
    "naturalConversationFlow": {
      "score": 7,
      "explanation": "While the responses were clear, they felt a bit abrupt. The assistant could have maintained a more conversational tone or encouraged further discussion."
    }
  },
  "overallAssessment": {
    "score": 8.5,
    "explanation": "The assistant demonstrated strong technical knowledge and was able to provide accurate information on demand. However, there were some minor lapses in natural conversation flow and coherence."
  },
  "suggestionsForImprovement": [
    {
      "improvementArea": "NaturalConversationFlow",
      "description": "Consider using more conversational language or prompts to engage users further."
    },
    {
      "improvementArea": "CoherenceAndClarity",
      "description": "Use transitional phrases and maintain a consistent tone throughout the conversation."
    }
  ]
}
```
