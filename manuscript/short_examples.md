# Short Examples

Here we look at a few short examples before later using Ollama to solve more difficult problems.

## Using The Ollama Python SDK With Image and Text Prompts

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

