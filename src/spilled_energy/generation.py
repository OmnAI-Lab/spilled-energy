import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_answer(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 100,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cuda",
):
    """
    Generate an answer from the model and return the generated text along with logits.

    Args:
        prompt: The input prompt string.
        model: The loaded Hugging Face model.
        tokenizer: The loaded tokenizer.
        max_new_tokens: Maximum number of tokens to generate.
        do_sample: Whether to use sampling (True) or greedy decoding (False).
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        device: Device to run generation on.

    Returns:
        dict: A dictionary containing:
            - 'text': The generated answer string.
            - 'sequences': Tensor of generated token IDs (including prompt).
            - 'scores': Tuple of tensors containing logits for each generation step.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # We need output_scores=True to get logits for energy calculation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(
        generated_ids[input_len:], skip_special_tokens=True
    )

    return {
        "text": generated_text,
        "sequences": outputs.sequences,
        "scores": outputs.scores,
    }
