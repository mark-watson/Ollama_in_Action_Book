import ollama
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_client, get_model

# NOTE: This example uses a vision/multimodal model to analyze images.
# The default model (nemotron-3-nano:4b) is text-only; set MODEL env var
# to a vision-capable model such as 'llava:7b' for image analysis.
DEFAULT_VISION_MODEL = "llava:7b"

def analyze_image(image_path: str, prompt: str) -> str:
    # Read and encode the image
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')

    # Use MODEL env var if set, otherwise fall back to the vision default
    model = get_model() if __import__('os').environ.get('MODEL') else DEFAULT_VISION_MODEL

    try:
        client = get_client()
        # Create a stream of responses using the Ollama SDK
        stream = client.generate(
            model=model,
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
