# Introduction to Spilled Energy

**Spilled Energy** is a novel, training-free method for detecting hallucinations in Large Language Models (LLMs). It works by reinterpreting the model's output probabilities through the lens of physics-inspired **Energy-Based Models (EBMs)**.

## The Core Insight: Consistency Across Steps

We propose that a consistent Language Model should satisfy a specific energy conservation law. The energy implied by selecting a token $x_t$ should match the energy of the system once that token has been processed.

The "Spilled Energy" ($E_\Delta$) measures the violation of this equality.

### 1. The Components

To define the metric, we look at two specific quantities calculated at adjacent timesteps ($t$ and $t+1$):

#### A. The Logit Energy ($E(x_t)$)
At step $t$, the model computes logits $f(x)$ for all possible next tokens. The "energy" of the specific token $x_t$ that gets sampled is defined by its logit value (the numerator of the softmax):

$$ E(x_t) = \text{Logit}(x_t) $$

*Note:* This is the energy "predicted" for token $x_t$ before it is fully integrated into the context.

#### B. The Marginal (Free) Energy ($F(x_{<t+1})$)
At step $t+1$, after the token $x_t$ has been appended to the context, the model computes a new set of logits for the *next* step ($t+1$). The "energy" of this new state (the sequence ending in $x_t$) is the **Free Energy** (or Marginal Energy) of the distribution over the vocabulary $V$:

$$ F(x_{<t+1}) = \text{LogSumExp}_{v \in V}(\text{Logit}(v)) $$

*Note:* This is the generic "LogSumExp" of the logits at the new step. It represents the "stability" of the new state.

### 2. The Spilled Energy Metric

In a perfectly consistent Energy-Based Model (EBM), the energy of the transition should equal the energy of the resulting state (modulo normalization constants). We derive that for a consistent LM, these two values should theoretically be equal.

**Spilled Energy** is the absolute difference between these two values:

$$ E_\Delta = | E(x_t) - F(x_{<t+1}) | $$

### 3. Interpretation

*   **Low Spill ($E_\Delta \approx 0$):** The model is "confident" and consistent. The logit value it assigned to $x_t$ (how much it "wanted" to pick that token) perfectly aligns with the stability of the state *after* picking it.
*   **High Spill ($E_\Delta \gg 0$):** The model is "surprised" or inconsistent. It assigned a certain logit to $x_t$, but once $x_t$ became part of the context, the resulting state's energy (LogSumExp) did not match the prediction.

This discrepancy, the "spill", is a strong, training-free signal that the model is drifting into a hallucination or an inconsistent state. 
