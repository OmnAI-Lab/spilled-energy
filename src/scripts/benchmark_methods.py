import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
import transformers
import datasets
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

from spilled_energy.generation import generate_answer
from spilled_energy.extraction import extract_exact_answer
from spilled_energy.energy import spilled_energy

# Disable INFO logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()


def compute_token_metrics(logits, generated_ids, token_start, token_end):
    """
    Computes Delta, E, E_margin statistics on a specific token slice.
    """
    # Slice
    sliced_logits = logits[:, token_start:token_end, :]
    sliced_ids = generated_ids[:, token_start:token_end]

    logits_list = sliced_logits.cpu().float().numpy().tolist()
    ids_list = sliced_ids.cpu().numpy().tolist()

    # Compute raw values
    delta, E_margin, E = spilled_energy(logits=logits_list, ids=ids_list, beta=1.0)

    # Helper for stats
    def get_stats(vals):
        v = np.array(vals)
        if len(v) == 0:
            return {k: 0.0 for k in ["mean", "max", "min", "sum"]}
        return {
            "mean": float(np.mean(v)),
            "max": float(np.max(v)),
            "min": float(np.min(v)),
            "sum": float(np.sum(v)),
        }

    return {
        "delta": get_stats(delta[0]),
        "E": get_stats(E[0]),
        "E_margin": get_stats(E_margin[0]),
    }


def find_best_threshold(scores, labels):
    """
    Finds threshold that maximizes F1 score.
    labels: boolean, True = Hallucination (Positive class)
    """
    if len(set(labels)) < 2:
        return 0.0, 0.0

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = (
        thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    )
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1


def main():
    # Configuration
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    N_VAL = 25  # Small validation set for thresholds
    N_TEST = 50  # Small test set for demonstration (increase for real benchmark)
    TOTAL_SAMPLES = N_VAL + N_TEST

    print(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16
        ).to("cuda")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading TriviaQA (validation split)...")
    dataset = load_dataset("trivia_qa", "rc", split="validation", streaming=True)

    results = []  # Stores {type, metrics, is_correct, question}

    print(f"Processing {TOTAL_SAMPLES} samples...")

    iterator = iter(dataset)
    for i in tqdm(range(TOTAL_SAMPLES)):
        try:
            sample = next(iterator)
        except StopIteration:
            break

        question = sample["question"]
        ground_truth_aliases = sample["answer"]["aliases"]

        # 1. Generate
        prompt = f"Q: {question}\nA:"
        gen_output = generate_answer(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            do_sample=False,
            device="cuda",
        )
        generated_text = gen_output["text"]

        # 2. Extract
        exact_answer = extract_exact_answer(
            question=question,
            long_answer=generated_text,
            model=model,
            tokenizer=tokenizer,
            device="cuda",
        )
        cleaned_exact = exact_answer.strip("'\"").strip()

        # 3. Check Correctness
        is_correct = any(
            alias.lower() in cleaned_exact.lower() for alias in ground_truth_aliases
        )

        # 4. Locate Tokens
        start_idx = generated_text.find(cleaned_exact)
        token_start, token_end = None, None

        if start_idx != -1:
            end_idx = start_idx + len(cleaned_exact)
            enc = tokenizer(
                generated_text, return_offsets_mapping=True, add_special_tokens=False
            )
            for t_i, (s, e) in enumerate(enc.offset_mapping):
                if s >= start_idx and token_start is None:
                    token_start = t_i
                if s < end_idx:
                    token_end = t_i + 1

        # 5. Compute Metrics
        logits = torch.stack(gen_output["scores"], dim=1)
        sequences = gen_output["sequences"]
        input_len = sequences.shape[1] - logits.shape[1]
        generated_ids = sequences[:, input_len:]

        # Fallback to full sequence if exact mapping fails
        if token_start is None or token_end is None:
            token_start, token_end = 0, generated_ids.shape[1]

        # Bounds check
        token_start = max(0, min(token_start, logits.shape[1] - 1))
        token_end = max(token_start + 1, min(token_end, logits.shape[1]))

        metrics = compute_token_metrics(logits, generated_ids, token_start, token_end)

        split_type = "val" if i < N_VAL else "test"
        results.append(
            {
                "split": split_type,
                "metrics": metrics,
                "is_correct": is_correct,
                "exact_answer": cleaned_exact,
                "ground_truth": ground_truth_aliases,
            }
        )

    # Analysis
    val_data = [r for r in results if r["split"] == "val"]
    test_data = [r for r in results if r["split"] == "test"]

    print(
        f"\nCollected {len(val_data)} validation samples and {len(test_data)} test samples."
    )
    print(f"Val Accuracy: {np.mean([r['is_correct'] for r in val_data]):.2%}")
    print(f"Test Accuracy: {np.mean([r['is_correct'] for r in test_data]):.2%}")

    print("\n--- Benchmarking Results (AUROC on Test Set) ---")
    print(f"{'Metric':<30} | {'Strategy':<10} | {'Val F1':<10} | {'Test AUROC':<10}")
    print("-" * 75)

    aggregated_results = []

    for metric_name in ["delta", "E", "E_margin"]:
        for strategy in ["mean", "max", "min", "sum"]:
            # Prepare scores and labels (Positive label = Hallucination/Incorrect)
            # Validation
            val_scores = np.array(
                [r["metrics"][metric_name][strategy] for r in val_data]
            )
            val_labels = np.array([not r["is_correct"] for r in val_data])

            # Test
            test_scores = np.array(
                [r["metrics"][metric_name][strategy] for r in test_data]
            )
            test_labels = np.array([not r["is_correct"] for r in test_data])

            # Find Threshold on Val
            # Check direction: usually higher score -> hallucination.
            # If AUC < 0.5, it means lower score -> hallucination.
            # We'll use AUC to determine direction for threshold finding if needed,
            # but for simple F1 optimization we might need to try both directions or assume one.
            # Here we assume Score correlates with Hallucination.

            threshold, best_f1 = find_best_threshold(val_scores, val_labels)

            # Compute AUROC on Test
            if len(set(test_labels)) > 1:
                auroc = roc_auc_score(test_labels, test_scores)
            else:
                auroc = 0.5  # Undefined

            print(
                f"{metric_name:<30} | {strategy:<10} | {best_f1:.4f}     | {auroc:.4f}"
            )
            aggregated_results.append((metric_name, strategy, auroc))

    # Highlight best
    best_metric = max(aggregated_results, key=lambda x: x[2])
    print("\nBest Performing Method:")
    print(f"{best_metric[0]} ({best_metric[1]}) - AUROC: {best_metric[2]:.4f}")


if __name__ == "__main__":
    main()
