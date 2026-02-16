# Scripts Documentation

The `src/scripts/` directory contains executable scripts for testing and benchmarking the Spilled Energy library.

## `benchmark_methods.py`

This is the primary script for running a full (toy) evaluation of the hallucination detection capabilities.

### What it does:
1.  **Loads Dataset**: Fetches the **TriviaQA** validation set (by default).
2.  **Generates & Extracts**: For each sample, it generates an answer using a specified model (default: `Meta-Llama-3-8B`) and extracts the exact short answer.
3.  **Computes Metrics**: Calculates Spilled Energy ($E_\Delta$), Energy ($E$), and Marginal Energy ($E_{margin}$) for the generated answers.
4.  **Evaluates Performance**:
    - Compares energy scores against ground truth correctness (Exact Match).
    - Finds optimal decision thresholds on a validation split.
    - Computes **AUROC** (Area Under the Receiver Operating Characteristic Curve) on a held-out test split.
5.  **Reports Results**: detailed breakdown of performance for different metrics (e.g., `delta (mean)`, `delta (max)`).

### Usage:
```bash
uv run src/scripts/benchmark_methods.py
```

### Configuration:
You can modify variables at the top of the `main()` function in the script to change:
- `MODEL_NAME`: The model used for generation.
- `N_VAL` / `N_TEST`: Number of samples for validation and testing.

---

## `test_measure_exact_answer.py`

A diagnostic script to verify the end-to-end pipeline on a single sample.

### What it does:
1.  Loads a single question from TriviaQA (or falls back to a hardcoded one if the dataset fails).
2.  Runs the full generation -> extraction -> energy computation pipeline.
3.  Prints detailed debug information:
    - The raw generated text.
    - The extracted exact answer.
    - The token indices corresponding to the answer.
    - Statistical summaries (Mean, Max, Min, Sum) for all energy metrics on the exact answer tokens.
    - Comparison metrics for the *full* generation sequence.

### Usage:
```bash
uv run src/scripts/test_measure_exact_answer.py
```

Use this script to quickly check if your environment is set up correctly or to debug issues with specific models.
