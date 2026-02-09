"""
Normalized relevancy score computation utilities.

Computes normalized relevancy as a Bayesian posterior:
    P(with | token) = P_with / (P_with + P_without)

This measures the causal influence of some condition (e.g., LoRA adapters,
system prompt, embedding) on the generated tokens.
"""

import warnings
from typing import List

import numpy as np
import torch


def compute_normalized_relevancy(
    p_with: List[List[float]],
    p_without: List[List[float]],
) -> List[float]:
    """Compute normalized relevancy scores from token probabilities.

    For each sequence, computes mean(P_with / (P_with + P_without)) across tokens.

    This is the Bayesian posterior P(with_condition | token) under uniform prior,
    answering: "what's the probability the 'with' condition generated this token?"

    Args:
        p_with: List of lists, where p_with[i][j] is P(token_j | with_condition)
                for sequence i
        p_without: List of lists, where p_without[i][j] is P(token_j | without_condition)
                   for sequence i

    Returns:
        List of normalized relevancy scores, one per sequence.
        Range [0, 1]: 0.5 = no effect, >0.5 = 'with' preferred, <0.5 = 'without' preferred
    """
    relevancy_scores = []

    for probs_with, probs_without in zip(p_with, p_without):
        if len(probs_with) == 0:
            relevancy_scores.append(0.5)  # Neutral if no tokens
            continue

        position_relevancies = []
        for pw, pwo in zip(probs_with, probs_without):
            denominator = pw + pwo
            if denominator > 0:
                relevancy_t = pw / denominator
            else:
                relevancy_t = 0.5  # Both zero = neutral
            position_relevancies.append(relevancy_t)

        relevancy_scores.append(np.mean(position_relevancies))

    return relevancy_scores


def _extract_token_probs(
    logits: torch.Tensor,
    generated_tokens_list: List[torch.Tensor],
    base_positions: List[int],
) -> List[List[float]]:
    """Extract token probabilities from logits (vectorized per sequence).

    For each generated token at position (base_pos + j), gets P(token) from
    the logits at position (base_pos + j - 1).

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        generated_tokens_list: List of generated token tensors per sequence
        base_positions: Starting position (prompt length) for each sequence

    Returns:
        List of probability lists, one per sequence
    """
    batch_size = logits.shape[0]
    probs_batch = []

    for i in range(batch_size):
        gen_tokens = generated_tokens_list[i]
        gen_len = len(gen_tokens)
        base_pos = base_positions[i]

        if gen_len == 0:
            probs_batch.append([])
            continue

        # Positions that predict each generated token (token at pos N predicted by logits at N-1)
        positions = torch.arange(base_pos - 1, base_pos - 1 + gen_len, device=logits.device)

        # Extract logits at those positions and compute softmax: [gen_len, vocab_size]
        relevant_logits = logits[i, positions]
        probs = torch.softmax(relevant_logits, dim=-1)

        # Gather probabilities for target tokens: [gen_len]
        token_probs = probs[torch.arange(gen_len, device=logits.device), gen_tokens]
        probs_batch.append(token_probs.tolist())

    return probs_batch


def compute_relevancy_scores(
    model,
    tokenizer,
    output_ids: torch.Tensor,
    prompt_lengths: List[int],
    formatted_base_prompts: List[str],
) -> List[float]:
    """Compute relevancy scores comparing LoRA-adapted vs base model.

    This function compares:
    - P_with: probability with LoRA adapters ENABLED on original sequences
    - P_without: probability with LoRA adapters DISABLED on base_prompt + generated_tokens

    The caller is responsible for preparing formatted_base_prompts (typically the same
    conversation structure but without system prompt).

    Args:
        model: The language model (with LoRA adapters)
        tokenizer: The tokenizer
        output_ids: Output token IDs from generation [batch_size, seq_len]
        prompt_lengths: Length of each prompt (to identify generated tokens)
        formatted_base_prompts: Base prompts (no system prompt) for P_without computation

    Returns:
        List of relevancy scores, one per sequence
    """
    pad_token_id = tokenizer.pad_token_id
    batch_size = output_ids.shape[0]
    device = output_ids.device

    # Step 1: Extract generated tokens for each sequence (needed before forward passes)
    generated_tokens_list = []
    for i in range(batch_size):
        prompt_len = prompt_lengths[i]
        gen_tokens = output_ids[i, prompt_len:].clone()
        # Remove padding tokens from the end
        non_pad_mask = gen_tokens != pad_token_id
        if non_pad_mask.any():
            last_non_pad = non_pad_mask.nonzero()[-1].item() + 1
            gen_tokens = gen_tokens[:last_non_pad]
        else:
            gen_tokens = gen_tokens[:0]
        generated_tokens_list.append(gen_tokens)

    # Step 2: Compute P_with (adapters ENABLED) on original sequences
    with torch.no_grad():
        outputs_with = model(output_ids, return_dict=True)
        logits_with = outputs_with.logits

    # Step 3: Extract probabilities immediately and free memory
    p_with_batch = _extract_token_probs(logits_with, generated_tokens_list, prompt_lengths)
    del logits_with, outputs_with
    torch.cuda.empty_cache()

    # Step 4: Tokenize base prompts and build combined sequences
    base_prompt_encodings = tokenizer(
        formatted_base_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    base_prompt_ids = base_prompt_encodings.input_ids.to(device)
    base_attention_mask = base_prompt_encodings.attention_mask.to(device)

    # Get base prompt lengths (excluding padding)
    base_prompt_lengths = base_attention_mask.sum(dim=1).tolist()

    # Create combined sequences: base_prompt + generated_tokens
    max_combined_len = max(
        base_prompt_lengths[i] + len(generated_tokens_list[i])
        for i in range(batch_size)
    )

    combined_ids = torch.full(
        (batch_size, max_combined_len), pad_token_id, dtype=torch.long, device=device
    )
    combined_attention_mask = torch.zeros(
        (batch_size, max_combined_len), dtype=torch.long, device=device
    )

    for i in range(batch_size):
        base_len = base_prompt_lengths[i]
        gen_len = len(generated_tokens_list[i])

        # Copy base prompt tokens (non-padded portion)
        base_start = (base_prompt_ids.shape[1] - base_len) if base_attention_mask[i, 0] == 0 else 0
        combined_ids[i, :base_len] = base_prompt_ids[i, base_start:base_start + base_len]
        combined_attention_mask[i, :base_len] = 1

        # Append generated tokens
        if gen_len > 0:
            combined_ids[i, base_len:base_len + gen_len] = generated_tokens_list[i]
            combined_attention_mask[i, base_len:base_len + gen_len] = 1

    # Step 5: Compute P_without (adapters DISABLED) on combined sequences
    has_adapters = hasattr(model, 'active_adapters') and callable(getattr(model, 'active_adapters', None))

    if not has_adapters:
        warnings.warn(
            "No adapters found, using same model for relevancy score baseline"
        )

    if has_adapters:
        active_adapters = model.active_adapters()
        model.disable_adapters()

    with torch.no_grad():
        outputs_without = model(
            combined_ids,
            attention_mask=combined_attention_mask,
            return_dict=True
        )
        logits_without = outputs_without.logits

    # Re-enable adapters
    if has_adapters:
        model.enable_adapters()
        if active_adapters:
            model.set_adapter(active_adapters[0])

    # Step 6: Extract probabilities and free memory
    p_without_batch = _extract_token_probs(logits_without, generated_tokens_list, base_prompt_lengths)
    del logits_without, outputs_without

    # Step 7: Compute normalized relevancy
    return compute_normalized_relevancy(p_with_batch, p_without_batch)
