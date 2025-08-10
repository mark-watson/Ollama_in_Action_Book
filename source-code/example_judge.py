"""
Example script demonstrating the usage of LLM tools
"""

from tool_judge_results import judge_results



def separator(title: str):
    """Prints a section separator"""
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print('=' * 50)


def main():
    # Test file writing
    separator("Judge output from a LLM")

    separator("First test: should be Y, or good")
    test_prompt = "Sally is 55, John is 18, and Mary is 31. What are pairwise combinations of the absolute value of age differences?"
    test_output = "Sally and John:  55 - 18 = 37. Sally and Mary:  55 - 31 = 24. John and Mary:  31 - 18 = 13."

    client = ai.Client()
    judgement = judge_results(test_prompt, test_output)
    print(f"\n** JUDGEMENT ***\n\n{judgement=}")

    separator("Second test: should be N, or bad")
    bad_test_output = "Sally and John:  55 - 18 = 31. Sally and Mary:  55 - 31 = 24. John and Mary:  31 - 18 = 10."

    judgement = judge_results(test_prompt, bad_test_output)
    print(f"\n** JUDGEMENT ***\n\n{judgement=}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

