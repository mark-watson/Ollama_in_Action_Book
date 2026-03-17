import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dspy

from ollama_config import get_model

_model_name = get_model()

if os.environ.get("CLOUD"):
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    lm = dspy.LM(
        f"ollama_chat/{_model_name}",
        api_base="https://ollama.com",
        api_key=api_key,
        temperature=1.0,
        max_tokens=4096,
    )
else:
    lm = dspy.LM(
        f"ollama_chat/{_model_name}",
        api_base="http://localhost:11434",
        api_key="ollama",
        temperature=1.0,
        max_tokens=4096,
    )

dspy.configure(lm=lm)

class MathProblem(dspy.Signature):
    """question -> answer: list[float]"""
    # The docstring defines the input and output fields, including
    # the required output type (float)
    
class ChainOfThoughtMath(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use dspy.ChainOfThought to implement the MathProblem signature
        self.prog = dspy.ChainOfThought(MathProblem)

    def forward(self, question):
        return self.prog(question=question)

math_model = dspy.ChainOfThought("question -> answer: list[float]")
question_text = "Two dice are tossed. Give me a list of the three most probable rolls."
prediction = math_model(question=question_text)

print(f"Question: {question_text}")
print(f"Reasoning: {prediction.reasoning}")
print(f"Answer: {prediction.answer}")
