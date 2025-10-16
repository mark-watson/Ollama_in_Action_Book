import dspy

lm = dspy.LM('ollama_chat/qwen3:8b', api_base="http://localhost:11434", api_key="ollama",
                 temperature=1.0, max_tokens=4096)
#print(lm("what s 1 + 2?"))

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
