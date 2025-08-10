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
