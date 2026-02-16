import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_exact_answer(
    question: str,
    long_answer: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    example_template: str = None,
    device: str = "cuda",
) -> str:
    """
    Extract the short/exact answer from a long generated answer using the model.
    """

    if example_template is None:
        import textwrap

        prompt = textwrap.dedent(f"""
        Extract from the following long answer the short answer, only the relevant tokens. If the long answer does not answer the question, output NO ANSWER.

        Q: Which musical featured the song The Street Where You Live?
        A: The song "The Street Where You Live" is from the Lerner and Loewe musical "My Fair Lady." It is one of the most famous songs from the show, and it is sung by Professor Henry Higgins as he reflects on the transformation of Eliza Doolittle and the memories they have shared together.
        Exact answer: My Fair Lady

        Q: Which Swedish actress won the Best Supporting Actress Oscar for Murder on the Orient Express?
        A: I'm glad you asked about a Swedish actress who won an Oscar for "Murder on the Orient Express," but I must clarify that there seems to be a misunderstanding here. No Swedish actress has won an Oscar for Best Supporting Actress for that film. The 1974 "Murder on the Orient Express" was an American production, and the cast was predominantly British and American. If you have any other questions or if there's another
        Exact answer: NO ANSWER

        Q: {question}
        A: {long_answer}
        Exact answer:""").strip()
    else:
        prompt = example_template.format(question=question, long_answer=long_answer)

    # Add a space at the end to prompt the model to continue
    if not prompt.endswith(" "):
        prompt += " "

    logger.debug(f"Extraction prompt: {repr(prompt)}")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Short answer shouldn't be too long
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0]
    new_tokens = generated_ids[input_len:]
    logger.debug(f" Generated IDs: {new_tokens.tolist()}")

    exact_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    logger.debug(f" Extracted Answer: {repr(exact_answer)}")

    # Cleaning based on model behavior (mostly for Mistral/Llama styles as seen in reference)
    exact_answer = exact_answer.strip().split("\n")[0].strip(".")

    # Additional cleanup if needed (e.g. repetitive tokens) could act here or be done by caller

    if "NO ANSWER" in exact_answer:
        return "NO ANSWER"

    return exact_answer
