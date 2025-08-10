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