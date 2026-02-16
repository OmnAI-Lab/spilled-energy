"""
General utilities for the library.

This module provides common helper functions used across the library.
"""

import random
from typing import Dict, Optional

import torch
import numpy as np

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from transformers import set_seed as hf_set_seed

import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic_cuda: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic_cuda: Whether to set CUDA to deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_answer(answer: str, stop_patterns: Optional[list] = None) -> str:
    """
    Remove any text after stop sequences from the answer.

    Args:
        answer: Generated answer text
        stop_patterns: List of patterns that indicate answer end

    Returns:
        Cleaned answer text
    """
    if stop_patterns is None:
        stop_patterns = ["\nQ:", "\nQ.", "\nH:", "\nHuman:", "\nA:", "\n\nQ", "\n\nH"]

    for pattern in stop_patterns:
        if pattern in answer:
            answer = answer.split(pattern)[0]

    return answer.strip()


def compute_metrics(results: list) -> Dict[str, float]:
    """
    Compute aggregate metrics from a list of results.

    Args:
        results: List of result dictionaries with 'mean_spilled_energy',
                 'total_spilled_energy', and 'num_tokens' keys

    Returns:
        Dictionary of computed metrics
    """
    mean_spilled_energies = [r["mean_spilled_energy"] for r in results]
    total_spilled_energies = [r["total_spilled_energy"] for r in results]
    num_tokens = [r["num_tokens"] for r in results]

    return {
        "mean_spilled_energy": np.mean(mean_spilled_energies),
        "std_spilled_energy": np.std(mean_spilled_energies),
        "median_spilled_energy": np.median(mean_spilled_energies),
        "total_spilled_energy_avg": np.mean(total_spilled_energies),
        "avg_tokens_generated": np.mean(num_tokens),
        "min_spilled_energy": np.min(mean_spilled_energies),
        "max_spilled_energy": np.max(mean_spilled_energies),
    }


# /---------------------------------------------------------------------------------------------------------------/
# /                                         Plotting functions                                                    /
# /---------------------------------------------------------------------------------------------------------------/

# plot tokens by assigning a color to each token
# based on the col_values


def convert_id_to_token(tokenizer, id):
    # Handle newline
    if id == tokenizer.pad_token_id:
        return "[PAD]"
    elif hasattr(tokenizer, "bos_token_id") and id == tokenizer.bos_token_id:
        return "[BOS]"
    elif hasattr(tokenizer, "eos_token_id") and id == tokenizer.eos_token_id:
        return "[EOS]"

    token = tokenizer._convert_id_to_token(id)
    token = (
        token.replace("▁", "").replace("Ġ", "").replace("<0x0A>", "").replace("Ċ", "\n")
    )
    return token


def plot_tokens(
    tokenizer,
    ids,
    col_values,
    normalization_value=None,
    shift_value=None,
    skip_first=False,
    annotations=None,
):
    """
    @param ids: list of the input_ids of the whole text [not divided into list of sentences]
    @param col_values: list of values to be used for color mapping
    @param annotations: list of annotations for each sentence
    """
    import matplotlib.pyplot as plt

    # import HTML
    from IPython.display import display
    from IPython.core.display import HTML
    import numpy as np

    # if col_values is a list, then convert it to numpy array
    if isinstance(col_values, list):
        col_values = np.array(col_values)

    # Normalize values for color mapping
    min_val = col_values.min() if not skip_first else col_values[1:].min()
    max_val = col_values.max() if not skip_first else col_values[1:].max()

    if normalization_value is None:
        normalized_val = (col_values - min_val) / (max_val - min_val)
    else:
        # Ensure normalization_value is a scalar
        norm_value = (
            float(normalization_value)
            if not isinstance(normalization_value, list)
            else normalization_value[0]
        )
        if shift_value is None:
            normalized_val = (col_values - min_val) / norm_value
        else:
            normalized_val = (col_values + shift_value) / norm_value

    # Prepare color mapping
    def colorize(token, value):
        color = plt.get_cmap("RdYlGn")(
            value
        )  # red means low value, green means high value i.e. energy
        hex_color = "#%02x%02x%02x" % (
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )

        # Calculate luminance to determine if text should be light or dark
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (
            "#ffffff" if luminance < 0.5 else "#000000"
        )  # white for dark bg, black for light bg

        # return token with background color and contrasting text color
        token = token.replace("Ġ", "")
        token = token.replace("ĊĊ", " ")
        token = token.replace("âĢĵ", "—")
        token = token.replace("Ċ", "\n")
        html = f'<span style="background-color: {hex_color}; color: {text_color}">{token}</span>'
        return html

    # convert ids to text for display
    tokens = tokenizer.convert_ids_to_tokens(ids)

    # print(tokens)
    if annotations is None:
        colored_tokens = [
            colorize(token, norm_val) for token, norm_val in zip(tokens, normalized_val)
        ]
        if skip_first:
            colored_tokens = colored_tokens[1:]
    else:
        # find fullstop tokens, which are used to separate sentences
        fullstop_idx = [
            k for k, x in enumerate(ids) if "." in tokenizer._convert_id_to_token(x)
        ]
        # print(fullstop_idx)

        # print(annotations)
        colored_tokens = []
        sent_i = 0
        for i, (token, norm_val) in enumerate(zip(tokens, normalized_val)):
            if i in fullstop_idx:
                colored_tokens.append(colorize(token, norm_val))
                colored_tokens.append("<br>")
                colored_tokens.append(
                    f"{annotations[sent_i]}<br>"
                )  # add annotation at the end of the sentence
                sent_i += 1
            elif i == min(len(normalized_val) - 1, len(tokens) - 1):
                colored_tokens.append(colorize(token, norm_val))
                colored_tokens.append("<br>")
                colored_tokens.append(f"{annotations[sent_i]}<br>")
            else:
                colored_tokens.append(colorize(token, norm_val))

        if skip_first:
            colored_tokens = colored_tokens[1:]

    # Display as HTML
    # print(colored_tokens)
    html_output = " ".join(colored_tokens)
    display(HTML(html_output))


def plot_examples(tokenizer, ids, values, annotations=None):
    for i in range(min(3, len(ids))):
        ids_acc_ex = ids[i]
        values_ex = values[i]
        annotation_ex = None
        if annotations:
            annotation_ex = annotations[i]

        if values_ex is not None and len(values_ex) > 0:
            plot_tokens(tokenizer, ids_acc_ex, values_ex, annotations=annotation_ex)


def plot_histogram(
    correct_values,
    incorrect_values,
    pooling_function=min,
    title="Energy Distribution",
    xlabel="Energy",
    num_bins=40,
):
    import matplotlib.pyplot as plt
    import numpy as np

    corr_pooled_values = []
    incorr_pooled_values = []

    for i in range(len(correct_values)):
        if pooling_function:
            corr_pooled_values.append(pooling_function(correct_values[i]))
        else:
            # flatten the list and use all values
            corr_pooled_values.extend(correct_values[i])
    for i in range(len(incorrect_values)):
        if pooling_function:
            incorr_pooled_values.append(pooling_function(incorrect_values[i]))
        else:
            # flatten the list and use all values
            incorr_pooled_values.extend(incorrect_values[i])

    if isinstance(incorr_pooled_values[0], list):
        incorr_pooled_values = [
            item for sublist in incorr_pooled_values for item in sublist
        ]
    if isinstance(corr_pooled_values[0], list):
        corr_pooled_values = [
            item for sublist in corr_pooled_values for item in sublist
        ]

    # calculate bins for histogram
    bins = np.histogram_bin_edges(
        corr_pooled_values + incorr_pooled_values, bins=num_bins
    )

    # print bins
    # print(bins)

    plt.hist(corr_pooled_values, bins=bins, alpha=0.5, label="Correct")
    plt.hist(incorr_pooled_values, bins=bins, alpha=0.5, label="Incorrect")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_PR_curve(
    correct_values, incorrect_values, pooling_function=min, title="PR Curve"
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    corr_pooled_values = []
    incorr_pooled_values = []

    for i in range(len(correct_values)):
        corr_pooled_values.append(pooling_function(correct_values[i]))

    for i in range(len(incorrect_values)):
        incorr_pooled_values.append(pooling_function(incorrect_values[i]))

    y_true = [1] * len(corr_pooled_values) + [0] * len(incorr_pooled_values)
    y_scores = corr_pooled_values + incorr_pooled_values

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.show()


def plot_ROC_curve(
    correct_values, incorrect_values, pooling_function=min, title="ROC Curve"
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    corr_pooled_values = []
    incorr_pooled_values = []

    for i in range(len(correct_values)):
        corr_pooled_values.append(pooling_function(correct_values[i]))

    for i in range(len(incorrect_values)):
        incorr_pooled_values.append(pooling_function(incorrect_values[i]))

    y_true = [1] * len(corr_pooled_values) + [0] * len(incorr_pooled_values)
    y_scores = corr_pooled_values + incorr_pooled_values

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.plot(fpr, tpr, marker=".")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.show()


def plot_multiple_PR_curves(
    correct_values: dict, incorrect_values: dict, pooling_function=min, title="PR Curve"
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(10, 10))
    for key in correct_values.keys():
        corr_pooled_values = []
        incorr_pooled_values = []

        for i in range(len(correct_values[key])):
            corr_pooled_values.append(pooling_function(correct_values[key][i]))

        for i in range(len(incorrect_values[key])):
            incorr_pooled_values.append(pooling_function(incorrect_values[key][i]))

        y_true = [1] * len(corr_pooled_values) + [0] * len(incorr_pooled_values)
        y_scores = corr_pooled_values + incorr_pooled_values

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        plt.plot(recall, precision, marker=".", label=key + f" (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_multiple_ROC_curves(
    correct_values: dict,
    incorrect_values: dict,
    pooling_function=min,
    title="ROC Curve",
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(10, 10))
    for key in correct_values.keys():
        corr_pooled_values = []
        incorr_pooled_values = []

        for i in range(len(correct_values[key])):
            corr_pooled_values.append(pooling_function(correct_values[key][i]))
        for i in range(len(incorrect_values[key])):
            incorr_pooled_values.append(pooling_function(incorrect_values[key][i]))

        y_true = [1] * len(corr_pooled_values) + [0] * len(incorr_pooled_values)
        y_scores = corr_pooled_values + incorr_pooled_values

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = auc(fpr, tpr)

        plt.plot(fpr, tpr, marker=".", label=key + f" (AUROC={auroc:.2f})")

    # plot random classifier
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()


def format_prompt_hellaswag(data: object, examples: object, topic, n_shots: int):
    prompt = f"The following multiple choice questions (with answers) are sentence completion problems about {topic}.\n\n"
    for i in range(n_shots):
        prompt += format_question_hellaswag(examples[i])

    prompt += format_question_hellaswag(data, include_answer=False)

    return prompt


def format_question_hellaswag(data: dict, include_answer: bool = True):
    prompt = data["ctx"]
    choices = ["A", "B", "C", "D"]
    for j in range(len(choices)):
        choice = choices[j]
        prompt += "\n{}. {}".format(choice, data["endings"][j])
    prompt += "\nAnswer: "
    if include_answer:
        prompt += "{}\n\n".format(choices[int(data["label"])])
    return prompt


def compute_logits(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # return logits and input ids
    return logits.cpu().detach().to(torch.float32).numpy(), inputs[
        "input_ids"
    ].cpu().detach().numpy()


def compute_logits_batch(prompt_list, tokenizer, model, bs=8, use_tqdm=True):
    logits = []
    all_ids = []
    for i in (
        tqdm(range(0, len(prompt_list), bs))
        if use_tqdm
        else range(0, len(prompt_list), bs)
    ):
        batch = prompt_list[i : i + bs]
        logs, ids = compute_logits(batch, tokenizer, model)
        # add batch to list by removing batch dimension
        logits.extend(logs)
        all_ids.extend(ids)
    return logits, all_ids


def remove_pad_tokens_from_logits(logits, ids, tokenizer, pad_side="right"):
    # Assume logits is list of lists (batch, seq, vocab), ids is list of lists (batch, seq)
    batch_size = len(logits)
    pad_token_id = tokenizer.pad_token_id

    trimmed_logits = []
    trimmed_ids = []

    for i in range(batch_size):
        ids_i = ids[i]
        logits_i = logits[i]

        if pad_side == "right":
            # Find the last non-pad token
            non_pad_indices = [
                j for j, id_val in enumerate(ids_i) if id_val != pad_token_id
            ]
            if non_pad_indices:
                last_non_pad = max(non_pad_indices) + 1
            else:
                last_non_pad = 0  # All pads, keep nothing
            trimmed_ids.append(ids_i[:last_non_pad])
            trimmed_logits.append(logits_i[:last_non_pad])

        elif pad_side == "left":
            # Find the first non-pad token
            non_pad_indices = [
                j for j, id_val in enumerate(ids_i) if id_val != pad_token_id
            ]
            if non_pad_indices:
                first_non_pad = min(non_pad_indices)
            else:
                first_non_pad = len(ids_i)  # All pads, keep nothing
            trimmed_ids.append(ids_i[first_non_pad:])
            trimmed_logits.append(logits_i[first_non_pad:])

    return trimmed_logits, trimmed_ids


def compute_joint_p(logits: list, ids: list):
    joint_p = []

    for i in range(len(logits)):
        joint_p_i = 0
        for j in range(len(logits[i])):
            torch_logits = torch.tensor(logits[i][j])
            joint_p_i += torch_logits[ids[i][j]] - torch.logsumexp(torch_logits, dim=-1)
        joint_p.append(joint_p_i.item())

    return joint_p


def compute_joint_p_sequence(logits: list, ids: list):
    # compute joint probability of each subsequence in logits
    joint_p = []

    for i in range(len(logits)):
        joint_p_i = []
        joint_p_ij = 0
        for j in range(len(logits[i])):
            # get joint_p i j-1 and add log probability of token j
            joint_p_ij = (
                joint_p_ij
                + logits[i][j][ids[i][j]]
                - np.log(np.sum(np.exp(logits[i][j], dtype=np.float64)))
            )
            joint_p_i.append(joint_p_ij)
        joint_p.append(joint_p_i)

    return joint_p
