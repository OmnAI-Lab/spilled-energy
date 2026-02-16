"""
Energy computation utilities for spilled energy decoding.

This module implements the core energy functions used in the simulated candidate
decoding algorithm. The key concept is "spilled energy" - the difference between
the token energy and the softmax denominator (log partition function).
"""

from typing import List

import torch


def compute_softmax_denominator(
    logits: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """
    Compute the softmax denominator (log-sum-exp).

    softmax_denom = log(sum_k exp(beta * logits[k])) = logsumexp(beta * logits)

    Args:
        logits: Tensor of shape (batch, vocab_size) or (vocab_size,)
        beta: Temperature scaling factor (inverse temperature)

    Returns:
        Log-sum-exp of the logits
    """
    return torch.logsumexp(beta * logits, dim=-1)


def compute_token_logit(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    """
    Extract the logit for a specific token.

    Args:
        logits: Tensor of shape (vocab_size,) - logits at position t
        token_id: The token ID to extract logit for

    Returns:
        Logit value for the token
    """
    return logits[token_id]


def spilled_energy(
    logits: List[List[List[float]]],
    ids: List[List[int]],
    beta: float = 1.0,
    prompt_length: int = 0,
) -> tuple:
    """
    Compute spilled energy for sequences.

    E(x_i, x_{i-1}, ..., x_0) = -beta * f(x_{i-1}, ..., x_0)[Id(x_i)]
    E'(x_i, x_{i-1}, ..., x_0) = -log sum_k exp(beta * f(x_i, x_{i-1}, ..., x_0)[k])
    spilled_energy = E - E'

    Args:
        logits: List of sequences, each containing list of token logit distributions
        ids: List of sequences, each containing list of token IDs
        beta: Temperature scaling factor
        prompt_length: Number of prompt tokens to exclude from computation

    Returns:
        Tuple of (delta, E_margin, E) where:
        - delta: spilled energy per token
        - E_margin: negative log partition function
        - E: token energy
    """
    # Compute E(x_i, x_{i-1}, ..., x_0) from logits
    E = []
    for j in range(len(logits)):
        E_j = [0]  # E(x_0) = 0
        for i in range(1, len(logits[j])):  # ids -> can't index BOS on logits
            E_ji = -beta * logits[j][i - 1][ids[j][i]]
            E_j.append(torch.tensor(E_ji, dtype=torch.float32))
        E.append(E_j)

    # Compute E'(x_i, x_{i-1}, ..., x_0) from logits
    E_margin = []
    for j in range(len(logits)):
        E_margin_j = []
        for i in range(len(logits[j])):
            logits_ji = beta * torch.tensor(logits[j][i], dtype=torch.float32)
            E_margin_ji = -torch.logsumexp(logits_ji, dim=-1)
            E_margin_j.append(E_margin_ji.item())
        E_margin.append(E_margin_j)

    # Compute spilled_energy
    delta = []
    for j in range(len(E)):
        delta_j = []
        for i in range(len(E[j])):
            delta_ji = -E_margin[j][i] + E[j][i]
            delta_j.append(delta_ji)
        delta.append(delta_j)

    if prompt_length > 0:
        for j in range(len(delta)):
            if len(delta[j]) > prompt_length:
                delta[j] = delta[j][prompt_length:]
            else:
                delta[j] = delta[j][-1:]

            if len(E_margin[j]) > prompt_length:
                E_margin[j] = E_margin[j][prompt_length:]
            else:
                E_margin[j] = E_margin[j][-1:]

            if len(E[j]) > prompt_length:
                E[j] = E[j][prompt_length:]
            else:
                E[j] = E[j][-1:]

    return delta, E_margin, E


def spilled_energy_torch(
    logits: torch.Tensor, ids: torch.Tensor, beta: float = 1.0, pad_token_id: int = -100
) -> tuple:
    """
    Compute spilled energy for batched sequences using PyTorch operations.

    This is a more efficient version that operates on tensors directly.

    Args:
        logits: Tensor of shape [batch, seq_len, vocab_size]
        ids: Tensor of shape [batch, seq_len]
        beta: Temperature scaling factor
        pad_token_id: ID used for padding tokens (excluded from computation)

    Returns:
        Tuple of (delta, E_margin, E) tensors
    """
    # Compute E from logits
    E = -beta * logits[:, torch.arange(logits.size(1) - 1)]

    # Extract energies using ids[:, 1:]
    copy_ids = ids[:, 1:].clone()

    # Set ids to 0 if value is pad_token_id (padding token)
    copy_ids[copy_ids == pad_token_id] = 0
    E = torch.take_along_dim(E, copy_ids.unsqueeze(-1), dim=-1).squeeze(-1)

    # Set energies to zero where ids were pad_token_id
    E[ids[:, 1:] == pad_token_id] = 0

    # Concat zero tensor for BOS token, E has shape [batch, seq_len]
    E = torch.cat([torch.zeros(E.size(0), 1).to(E.device), E], dim=1)

    # Compute E' from logits
    E_margin = -torch.logsumexp(beta * logits, dim=-1)

    # Compute spilled_energy
    delta = -E_margin + E

    return delta, E_margin, E


def spilled_energy_last_token(
    logits: List[List[float]], ids: List[int], beta: float = 1.0
) -> tuple:
    """
    Compute spilled energy for only the last token in a sequence.

    This is useful for incremental computation during decoding.

    Args:
        logits: List of logit distributions for the sequence
        ids: List of token IDs for the sequence
        beta: Temperature scaling factor

    Returns:
        Tuple of (delta_n, E_margin_n, E_n) where n is the last position
    """
    if len(logits) < 2:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    E_n = -beta * logits[-2][ids[-1]]

    # Compute E' at position N
    logits_n = torch.tensor(logits[-1])
    E_margin_n = -torch.logsumexp(beta * logits_n, dim=-1)

    # Compute spilled_energy
    delta_n = -E_margin_n + E_n

    return delta_n, E_margin_n, E_n

