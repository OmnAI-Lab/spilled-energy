# Usage Guide

This guide explains how to use the core components of the **Spilled Energy** library to detect hallucinations in LLM generations.

## Basic Workflow

The general workflow for using Spilled Energy involves three main steps:
1.  **Generation**: Generate a response using an LLM.
2.  **Extraction**: Identify the specific part of the response you want to analyze (e.g., the exact answer).
3.  **Energy Computation**: Calculate Energy ($E$), Marginal Energy ($E_{margin}$), and Spilled Energy ($E_\Delta$) for the token sequence of interest.

### 1. Generating an Answer
Use `spilled_energy.generation.generate_answer` to interact with your model. This function returns not just the text, but also the critical **logits** needed for energy calculation.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from spilled_energy.generation import generate_answer

# Load Model
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Generate
prompt = "Q: What is the capital of France?\nA:"
output = generate_answer(
    prompt=prompt,
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=False  # Deterministic generation is often preferred for analysis
)

print(output["text"])
# Output: Paris is the capital of France.
```

### 2. Extracting Exact Answers
For many tasks (like QA), you want to analyze the energy of the *answer* specifically, rather than the reasoning or filler text. The `spilled_energy.extraction.extract_exact_answer` function uses a secondary LLM call (or the same model) to isolate the short answer.

```python
from spilled_energy.extraction import extract_exact_answer

long_answer = output["text"]
question = "What is the capital of France?"

exact_answer = extract_exact_answer(
    question=question,
    long_answer=long_answer,
    model=model,
    tokenizer=tokenizer
)

print(exact_answer)
# Output: Paris
```

### 3. Computing Spilled Energy
Once you have the logits (from step 1) and the target token sequence (which corresponds to `exact_answer`), you can compute the energy metrics.

> **Note**: You will need to map your `exact_answer` string back to the specific token indices in the `output['sequences']` tensor to isolate the correct logits.

```python
from spilled_energy.energy import spilled_energy

# (Assumption: you have sliced your logits and ids to correspond to "Paris")
# target_logits: List[List[float]] - shape [1, sliced_seq_len, vocab_size]
# target_ids: List[List[int]] - shape [1, sliced_seq_len]

delta, E_margin, E = spilled_energy(
    logits=target_logits,
    ids=target_ids,
    beta=1.0 # Temperature scaling
)

print(f"Spilled Energy (Delta): {delta[0]}")
# Output: List of delta values for each token in "Paris"
```

## Interpreting Results

- **High Spilled Energy ($E_\Delta$)**: Indicates a token that is energetically inconsistent with the model's distribution. This is a red flag for potential hallucination.
- **Aggregation**: Common strategies for scoring a whole sequence include taking the `sum`, `min`, `max`, or `mean` of the token-level $E_\Delta$ values. Our benchmarks suggest `min` or `max` often work well depending on whether you measure spilled energy $E_\Delta$ or marginalized energy $E_m$.

## Running Benchmarks
See [Scripts documentation](scripts.md) for details on how to run full evaluation benchmarks on datasets like TriviaQA.
