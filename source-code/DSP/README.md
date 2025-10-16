# DSP Experients

# Ollama with a list of strings for an answer

```
$ uv run ollama_test.py
Question: Two dice are tossed. Give me a list of the three most probable rolls.
Reasoning: When two dice are tossed, there are 36 possible outcomes. The sum of the dice (roll) with the highest probability is 7, which has 6 combinations. The next most probable sums are 6 and 8, each with 5 combinations. Thus, the three most probable rolls are 7, 6, and 8. The probabilities are calculated as (number of combinations)/36.
Answer: [0.16666666666666666, 0.1388888888888889, 0.1388888888888889]
```
