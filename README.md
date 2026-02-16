# Spilled Energy in Large Language Models

## [Adrian Robert Minut](https://github.com/adrianrob1), [Hazem Dewidar](https://github.com/THESHADOW2030), [Iacopo Masi](https://iacopomasi.github.io/)

This repository implements **Spilled Energy**, a method for detecting hallucinations in Large Language Models (LLMs) by analyzing the probability distribution (energy) of generated tokens.

We reinterpret the final softmax classifier over the vocabulary of Large Language Models (LLM) as an Energy-based Model (EBM), allowing us to decompose the sequence probability chain into multiple interacting EBMs. This approach offers a principled way to measure where "energy spills" during decoding.

## 📚 Documentation

We have detailed documentation available in the `documentation/` directory:

- **[Introduction](documentation/introduction.md)**: Conceptual overview of Energy, Spilled Energy, and Hallucination Detection.
- **[Installation](documentation/installation.md)**: Setup guide, prerequisites, and environment configuration.
- **[Usage Guide](documentation/usage.md)**: Step-by-step instructions for generating answers and computing energy metrics.
- **[API Reference](documentation/api_reference.md)**: Detailed documentation for the library's modules and functions.
- **[Scripts](documentation/scripts.md)**: Guide to using the provided testing and benchmarking example scripts.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/OmnAI-Lab/spilled-energy.git
cd spilled-energy
uv sync
source .venv/bin/activate
```
*See [Installation Guide](documentation/installation.md) for details.*

### Usage Example
```python
from spilled_energy.generation import generate_answer
# ... load model ...
output = generate_answer(prompt="Q: ...", model=model, tokenizer=tokenizer)
# ... compute energy ...
```
*See [Usage Guide](documentation/usage.md) for a full example.*

## 📂 Library Overview

The core logic is located in `src/spilled_energy`:
*   **`generation.py`**: Handles LLM generation and retrieves logits/scores.
*   **`extraction.py`**: Extracts the exact short answer from a long-form generation.
*   **`energy.py`**: Computes energy metrics (Spilled Energy/Delta, Energy, E_margin).

## 📓 Notebooks

The `notebooks/` directory contains interactive examples:
*   `measure_exact_answer.ipynb`: A step-by-step demonstration for a single sample.
*   `benchmark_methods.ipynb`: A full benchmarking suite example.

## Citation

```bibtex
@inproceedings{
  minut2026spilled,
  title={Spilled Energy in Large Language Models},
  author={Adrian Robert Minut and Hazem Dewidar and Iacopo Masi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## Abstract
We reinterpret the final softmax classifier over the vocabulary of Large Language Models (LLM) as an Energy-based Model (EBM). This allows us to decompose the chain of probabilities used in sequence-to-sequence modeling as multiple EBMs that interact together at inference time. Our decomposition offers a principled approach to measuring where the "energy spills" in LLM decoding, empirically showing that spilled energy correlates well with factual errors, inaccuracies, biases, and failures. Similar to Orgad et al. (2025), we localize the exact token associated with the answer, yet, unlike them, who need to train a classifier and ablate which activations to feed to it, we propose a method to detect hallucinations completely training-free that naturally generalizes across tasks and LLMs by using the output logits across subsequent generation steps. We propose two ways to detect hallucinations: the first one that measures the difference between two quantities that we call spilled energy, measuring the difference between energy values across two generation steps that mathematically should be equal; the other is marginal energy, which we can measure at a single step. Unlike prior work, our method is training-free, mathematically principled, and demonstrates strong cross-dataset generalization: we scale our analysis to state-of-the-art LLMs, including LLaMa-3, Mistral, and Qwen-3, evaluating on nine benchmarks and achieving competitive performance with robust results across datasets and different LLMs.

## Disclaimer

For the experiment results on real-world datasets in the "Spilled Energy in Large Language Models" paper, we used the code from [LLMsKnow](https://github.com/technion-cs-nlp/LLMsKnow) by Orgad et al.
