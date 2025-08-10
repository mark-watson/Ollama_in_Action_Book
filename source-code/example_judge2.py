"""
More complex example script demonstrating usage of judge_results to evaluate
various LLM outputs: translations, code generation, arithmetic, etc.
"""

from tool_judge_results import judge_results


def separator(title: str):
    """Print a section separator."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def run_case(prompt: str, output: str, case_title: str):
    """Helper to judge a single prompt/output pair and print results."""
    separator(case_title)
    print(f"Prompt: {prompt!r}")
    print(f"Output:\n{output}\n")
    result = judge_results(prompt, output)
    print("Judgement:", result.get('judgement'))
    print("Reasoning:\n", result.get('reasoning'))


def main():
    separator("Complex Judgement Examples (example_judge2.py)")

    # 1) Translation to French
    prompt1 = "Translate to French: 'Life is beautiful.'"
    good_output1 = "La vie est belle."
    bad_output1 = "La vie est bel."  # missing final letter
    run_case(prompt1, good_output1, "Case 1A: Correct Translation")
    run_case(prompt1, bad_output1, "Case 1B: Incorrect Translation")

    # 2) Python code generation: prime test
    prompt2 = (
        "Write a Python function 'is_prime(n)' that returns True if n is prime, else False."
    )
    good_output2 = '''\
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
'''
    # Wrong implementation: off-by-one in range
    bad_output2 = '''\
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
'''
    run_case(prompt2, good_output2, "Case 2A: Good Code")
    run_case(prompt2, bad_output2, "Case 2B: Bad Code")

    # 3) Arithmetic sum
    prompt3 = "Compute the sum of the first 10 positive integers."
    good_output3 = "The sum of the first 10 positive integers is 55."
    bad_output3 = "The sum of the first 10 positive integers is 54."
    run_case(prompt3, good_output3, "Case 3A: Correct Sum")
    run_case(prompt3, bad_output3, "Case 3B: Incorrect Sum")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", e)