# Short Examples

Here we look at a few short examples before later using libraries we develop and longer application style example programs with Ollama to solve more difficult problems.

## Using The Ollama Python SDK With Image and Text Prompts

We saw an example of image processing in the last chapter using Ollama command line mode. Here we do the same thing using a short Python script that you can find in the file ** short_programs/Ollama_sdk_image_example.py**:

```python
import ollama
import base64

def analyze_image(image_path: str, prompt: str) -> str:
    # Read and encode the image
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')

    try:
        # Create a stream of responses using the Ollama SDK
        stream = ollama.generate(
            model='llava:7b',
            prompt=prompt,
            images=[image_data],
            stream=True
        )

        # Accumulate the response
        full_response = ""
        for chunk in stream:
            if 'response' in chunk:
                full_response += chunk['response']

        return full_response

    except Exception as e:
        return f"Error processing image: {str(e)}"

def main():
    image_path = "data/sample.jpg"
    prompt = "Please describe this image in detail, focusing on the actions of people in the picture."

    result = analyze_image(image_path, prompt)
    print("Analysis Result:")
    print(result)

if __name__ == "__main__":
    main()
```

The output may look like the following when you run this example:

```text
Analysis Result:
 The image appears to be a photograph taken inside a room that serves as a meeting or gaming space and capturing an indoor scene where five individuals are engaged in playing a tabletop card game. In the foreground, there is a table with a green surface and multiple items on it, including what looks like playing cards spread out in front of the people seated around it.

The room has a comfortable and homely feel, with elements like a potted plant in the background on the left, which suggests that this might be a living room or a similar space repurposed for a group activity.
```

## Using the OpenAI Compatibility APIs with Local Models Running on Ollama

If you frequently use the OpenAI APIs for either your own LLM projects or work projects, you might want to simply use the same SDK library from OpenAI but specify a local Ollama REST endpoint:


```python
import openai
from typing import List, Dict

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="fake-key"  # Ollama doesn't require authentication locally
        )

    def chat_with_context(
        self,
        system_context: str,
        user_prompt: str,
        model: str = "llama3.2:latest",
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
        model: str = "llama2"
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
        model="llama3.2:latest",
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
        model="llama3.2:latest"
    )

    print("Conversation response:")
    print(response)

if __name__ == "__main__":
    main()
```

The output might look like (following listing is edited for brevity):

```text
Response with context:
Recursion is a fundamental concept in programming that allows a function or method to call itself repeatedly until it reaches a base case that stops the recursion.

**What is Recursion?**

In simple terms, recursion is a programming technique where a function invokes itself as a sub-procedure, repeating the same steps until it solves a problem ...

**Key Characteristics of Recursion:**

1. **Base case**: A trivial case that stops the recursion.
2. **Recursive call**: The function calls itself with new input or parameters.
3. **Termination condition**: When the base case is reached, the recursion terminates.

**How Recursion Works:**

Here's an example to illustrate recursion:

Imagine you have a recursive function `factorial(n)` that calculates the factorial of a number `n`. The function works as follows:

1. If `n` is 0 or 1 (base case), return 1.
2. Otherwise, call itself with `n-1` as input and multiply the result by `n`.
3. Repeat step 2 until `n` reaches 0 or 1.

Here's a simple recursive implementation in Python ...

**Benefits of Recursion:**

Recursion offers several advantages:

* **Divide and Conquer**: Break down complex problems into smaller, more manageable sub-problems.
* **Elegant Code**: Recursive solutions can be concise and easy to understand.
* **Efficient**: Recursion can avoid explicit loops and reduce memory usage.
...

In summary, recursion is a powerful technique that allows functions to call themselves repeatedly until they solve a problem. By understanding the basics of recursion and its applications, you can write more efficient and elegant code for complex problems.

==================================================

Conversation response:
A simple example of machine learning is a spam filter.

Imagine we have a system that scans emails and identifies whether they are spam or not. The system learns to classify these emails as spam or not based on the following steps:

1. Initially, it receives a large number of labeled data points (e.g., 1000 emails), where some emails are marked as "spam" and others as "not spam".
2. The system analyzes these examples to identify patterns and features that distinguish spam emails from non-spam messages.
3. Once the patterns are identified, the system can use them to classify new, unseen email data (e.g., a new email) as either spam or not spam.

Over time, the system becomes increasingly accurate in its classification because it has learned from the examples and improvements have been made. This is essentially an example of supervised machine learning, where the system learns by being trained on labeled data.
```

In the next chapter we start developing tools that can be used for “function calling” with Ollama.