DSP Experiments

The DSP (Declarative Semantic Prompting) library represents a interesting shift in how we design, test, and deploy prompts for language models. Instead of treating prompt engineering as an ad hoc, trial-and-error process, DSP provides a structured framework—allowing developers to express prompts, model interfaces, and evaluation strategies declaratively. This design brings reproducibility and modularity to the forefront, enabling users to define what a model should do, rather than getting lost in procedural details of how to make it happen. At its core, DSP treats prompts, model parameters, and outputs as composable objects that can be versioned, tested, and reused across different workflows or projects.

One of DSP’s most powerful aspects is its flexibility in integrating with different model backends. Whether you’re working with OpenAI or Gemini APIs, local inference servers, or fine-tuned models, DSP abstracts away the complexities of each environment. This is where Ollama becomes particularly compelling—it provides a seamless interface for running large language models locally with remarkable efficiency. Using DSP with Ollama, developers can declaratively specify prompts that interface directly with local models, allowing for private, offline experimentation while maintaining the same declarative patterns used with cloud-hosted models. The combination of DSP’s prompt modularity and Ollama’s local inference capabilities creates a powerful workflow for developers who want fine-grained control over both their model logic and their execution environment.

## DSP Uses Pydantic for Type Signatures - a Frist Ollama Example

The following Ollama DSP code leverages Pydantic-style typing to enforce structured input and output validation for prompts and model responses. In DSP, a Signature class—like MathProblem—uses a docstring or explicit field annotations to declare the expected input and output schema. When dspy.Signature is defined, DSP internally maps these declarations to Pydantic models, meaning each field (e.g., question, answer: float) gains automatic type checking, serialization, and conversion. This ensures that when a model produces an output, DSP can validate and coerce the response into the correct Python type—here, a float for answer, or a **list[float]** in the later **ChainOfThought** example.

By doing this, DSP tightly integrates language model reasoning with Python’s data model, allowing structured validation and predictable data flow across model calls. Pydantic typing not only helps catch mismatched or ill-formed responses but also provides self-documenting clarity for developers—making each prompt specification both executable and strongly typed. This makes DSP code more robust and maintainable, particularly in complex prompt pipelines or when integrating multiple model components.

Here is the documentation for DSP type signatures: [https://dspy.ai/learn/programming/signatures/](https://dspy.ai/learn/programming/signatures/).

```python
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
```

In this code Pydantic typing is used by DSP to define and enforce structured interfaces between the model’s prompts and responses. The **MathProblem** signature specifies that each prompt must include a question and that the model must return an answer of type float. When the ChainOfThoughtMath module runs, DSP automatically validates that the model’s output matches this expected schema—coercing or flagging data as needed. This structured approach ensures consistency, prevents malformed outputs, and makes it easier to compose reliable model pipelines, such as when generating a list of numeric results from reasoning-based prompts.

Here is sample output:

```
$ uv run ollama_test.py
Question: Two dice are tossed. Give me a list of the three most probable rolls.
Reasoning: When two dice are tossed, there are 36 possible outcomes. The sum of the dice (roll) with the highest probability is 7, which has 6 combinations. The next most probable sums are 6 and 8, each with 5 combinations. Thus, the three most probable rolls are 7, 6, and 8. The probabilities are calculated as (number of combinations)/36.
Answer: [0.16666666666666666, 0.1388888888888889, 0.1388888888888889]
```

