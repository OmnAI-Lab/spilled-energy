import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import logging
import transformers
import datasets

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spilled_energy.generation import generate_answer
from spilled_energy.extraction import extract_exact_answer
from spilled_energy.energy import spilled_energy

# Disable INFO logs from libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()


def main():
    print("Loading model...")
    # model_name = "facebook/opt-125m" # Small model for testing
    model_name = "meta-llama/Meta-Llama-3-8B"  # As requested by user

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to("cuda")
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print(
            "Falling back to small model for environment validity check, but DEBUGGING WILL BE LIMITED."
        )
        model_name = "facebook/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    print("Loading dataset sample...")
    try:
        dataset = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
        sample = next(iter(dataset))
        question = sample["question"]
        print(f"Loaded Question: {question}")
    except Exception as e:
        print(f"Failed to load TriviaQA: {e}")
        question = "Who wrote the play Romeo and Juliet?"
        print(f"Using fallback question: {question}")

    # Simple prompt format
    prompt = f"Q: {question}\nA:"

    print("Testing Generation...")
    gen_output = generate_answer(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=False,
        device="cuda",
    )

    generated_text = gen_output["text"]
    print(f"Generated text: {generated_text}")
    print(f"Logits shape: {gen_output['scores'][0].shape}")

    print("Testing Extraction...")

    # Real extraction
    real_extracted = extract_exact_answer(
        question=question,
        long_answer=generated_text,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
    )
    # Removing any surrounding quotes for matching if they don't exist in original
    cleaned_extracted = real_extracted.strip("'\"")
    print(f"Extracted answer: '{real_extracted}' (cleaned: '{cleaned_extracted}')")

    print("Testing Spilled Energy Calculation on Exact Answer Tokens...")
    # 1. Find the substring in the generated text
    start_idx = generated_text.find(cleaned_extracted)
    if start_idx == -1:
        print(
            f"ERROR: Could not find extracted answer '{cleaned_extracted}' in generated text."
        )
        return

    end_idx = start_idx + len(cleaned_extracted)
    print(f"Found at indices: {start_idx}-{end_idx}")

    # 2. Map indices to tokens
    # We re-tokenize the generated text to get offset mapping.
    # NOTE: This assumes tokenizer(generated_text) aligns with the generated tokens.
    # Since we are using Llama 3 fast tokenizer, this usually holds.
    enc = tokenizer(
        generated_text, return_offsets_mapping=True, add_special_tokens=False
    )
    offsets = enc.offset_mapping

    token_start = None
    token_end = None

    for i, (s, e) in enumerate(offsets):
        # Find first token that overlaps with start
        if s >= start_idx and token_start is None:
            token_start = i
        # Find last token that covers end
        if s < end_idx:
            token_end = i + 1

    if token_start is None or token_end is None:
        print("ERROR: Could not map text span to tokens.")
        return

    print(f"Token span: {token_start}-{token_end}")
    print(
        f"Corresponding text from tokens: {tokenizer.decode(enc.input_ids[token_start:token_end])}"
    )

    # Prepare logits and ids
    logits = torch.stack(gen_output["scores"], dim=1)  # [batch, seq_len, vocab]
    sequences = gen_output["sequences"]  # [batch, total_len]

    input_len = sequences.shape[1] - logits.shape[1]
    generated_ids = sequences[:, input_len:]

    # Slice for exact answer
    # We assume 'enc.input_ids' aligns with 'generated_ids' (the new tokens)
    # Ideally we should check this alignment or use generated_ids directly if possible,
    # but offset mapping requires re-tokenization usually.
    # Let's verify lengths
    if len(enc.input_ids) != generated_ids.shape[1]:
        print(
            "WARNING: Re-tokenized length differs from generation length. Alignment might be off."
        )
        print(f"Retokenized: {len(enc.input_ids)}, Generated: {generated_ids.shape[1]}")
        # Llama 3 might add a space prefix or different tokenization if context is missing.
        # But let's proceed with the slice indices we found from re-tokenization and apply to generated_ids

    exact_logits = logits[:, token_start:token_end, :]
    exact_ids = generated_ids[:, token_start:token_end]

    logits_list = exact_logits.cpu().float().numpy().tolist()
    ids_list = exact_ids.cpu().numpy().tolist()

    delta, E_margin, E = spilled_energy(logits=logits_list, ids=ids_list, beta=1.0)

    # helper to print stats
    def print_stats(name, values):
        vals = np.array(values[0])
        # Check if empty (e.g. if exact answer was empty string)
        if len(vals) == 0:
            print(f"{name}: [Empty sequence]")
            return
        print(f"--- {name} ---")
        print(f"  Mean: {np.mean(vals):.4f}")
        print(f"  Max:  {np.max(vals):.4f}")
        print(f"  Min:  {np.min(vals):.4f}")
        print(f"  Sum:  {np.sum(vals):.4f}")

    print("\n=== Metrics on Exact Answer Tokens ===")
    print_stats("Spilled Energy (Delta)", delta)
    print_stats("Energy (E)", E)
    print_stats("Marginalized Energy (E_margin)", E_margin)

    print("\n=== Comparison: Metrics on Full Generation ===")
    # Prepare logits and ids for the full generated text
    # Re-extract full sequences to ensure we're not using sliced versions
    logits_full = torch.stack(gen_output["scores"], dim=1)
    sequences_full = gen_output["sequences"]
    input_len_full = sequences_full.shape[1] - logits_full.shape[1]
    generated_ids_full = sequences_full[:, input_len_full:]

    full_logits_list = logits_full.cpu().float().numpy().tolist()
    full_ids_list = generated_ids_full.cpu().numpy().tolist()

    full_delta, full_E_margin, full_E = spilled_energy(
        logits=full_logits_list, ids=full_ids_list, beta=1.0
    )

    print_stats("Spilled Energy (Delta) - FULL", full_delta)
    print_stats("Energy (E) - FULL", full_E)
    print_stats("Marginalized Energy (E_margin) - FULL", full_E_margin)

    print("Test Complete.")


if __name__ == "__main__":
    main()
