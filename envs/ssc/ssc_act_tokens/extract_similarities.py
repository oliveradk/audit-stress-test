"""Extract residual stream similarities from SSC span tokens.

This module handles the extraction phase of the SSC activation tokens auditing pipeline,
capturing residual stream activations and computing cosine similarity with token embeddings.
"""

import base64
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.sae_utils import (
    extract_ssc_content,
    find_ssc_token_positions,
)


SSC_START_TAG = "<secret_side_constraint>"
SSC_END_TAG = "</secret_side_constraint>"


def encode_constraint_base64(constraint: str) -> str:
    """Encode a constraint string to base64."""
    return base64.b64encode(constraint.encode("utf-8")).decode("utf-8")


def decode_constraint_base64(encoded: str) -> str:
    """Decode a base64-encoded constraint string."""
    try:
        return base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
    except Exception:
        return encoded


def format_ssc_user_prompt(base_prompt: str, constraint: str) -> str:
    """Format a user prompt with an embedded SSC constraint."""
    encoded = encode_constraint_base64(constraint)
    return f"{base_prompt}\n\n{SSC_START_TAG}{encoded}{SSC_END_TAG}"


class ResidualCaptureHook:
    """Context manager for capturing residual stream representations.

    Adapted from eliciting-secret-knowledge/elicitation_methods/residual_tokens.py
    """

    def __init__(self, model, target_layer: int):
        self.model = model
        self.target_layer = target_layer
        self.representations = None
        self.hook_handle = None

    def _capture_hook(self, module, input, output):
        """Hook function to capture the residual stream output."""
        # output is typically a tuple, we want the hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]  # (batch_size, seq_len, hidden_dim)
        else:
            hidden_states = output

        # Store the representations (detach to avoid keeping gradients)
        self.representations = hidden_states.detach().clone()

    def __enter__(self):
        # Register hook at target layer
        target_layer_module = self.model.model.layers[self.target_layer]
        self.hook_handle = target_layer_module.register_forward_hook(self._capture_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up hook
        if self.hook_handle is not None:
            self.hook_handle.remove()


def compute_token_similarities(
    residual_reps: torch.Tensor,
    token_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between residual reps and token embeddings.

    Args:
        residual_reps: [seq_len, hidden_dim]
        token_embeddings: [vocab_size, hidden_dim]

    Returns:
        similarities: [seq_len, vocab_size]
    """
    # Normalize and compute dot product (cosine similarity)
    similarities = F.normalize(residual_reps, dim=1) @ F.normalize(token_embeddings, dim=1).T
    return similarities


def get_top_k_similar_tokens(
    similarities: torch.Tensor,
    tokenizer,
    k: int,
    min_similarity: float = 0.0,
) -> List[Dict[str, Any]]:
    """Get top-k most similar tokens with their similarity scores.

    Args:
        similarities: [vocab_size] similarity scores
        tokenizer: Tokenizer for decoding
        k: Number of top tokens to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of dicts with rank, token_id, token, similarity
    """
    # Get top-k indices
    top_k_indices = torch.argsort(similarities, descending=True)[:k]
    top_k_similarities = similarities[top_k_indices]

    # Convert to tokens, filtering by min_similarity
    top_k_tokens = []
    for i, (token_id, similarity) in enumerate(zip(top_k_indices, top_k_similarities)):
        sim_value = float(similarity.item())
        if sim_value < min_similarity:
            continue

        token_str = tokenizer.decode([int(token_id.item())])
        top_k_tokens.append({
            "rank": i + 1,
            "token_id": int(token_id.item()),
            "token": token_str,
            "similarity": sim_value,
        })

    return top_k_tokens


def extract_similarities_from_ssc_prompt(
    model,
    tokenizer,
    prompt_text: str,
    constraint: str,
    target_layer: int,
    top_k: int,
    min_similarity: float,
    device: str,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract residual stream similarities for a single SSC prompt.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt_text: Full prompt with SSC tags
        constraint: The plain-text constraint (for tracking)
        target_layer: Layer to extract activations from
        top_k: Number of top similar tokens per position
        min_similarity: Minimum similarity threshold
        device: Device to run on
        system_prompt: Optional system prompt

    Returns:
        Dict with extracted token similarities per position
    """
    # Extract SSC content
    ssc_content = extract_ssc_content(prompt_text)
    if not ssc_content:
        return {
            "user_prompt": prompt_text,
            "constraint": constraint,
            "error": "No SSC content found",
            "predictions": [],
        }

    ssc_decoded = decode_constraint_base64(ssc_content.strip())

    # Build chat sequence
    messages = [{"role": "user", "content": prompt_text}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    formatted_sequence = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    # Tokenize
    tokens = tokenizer.encode(
        formatted_sequence, add_special_tokens=False, return_tensors="pt"
    ).squeeze(0)
    tokens = tokens.to(device)

    # Find SSC token positions
    ssc_positions = find_ssc_token_positions(tokens, tokenizer, prompt_text)

    if not ssc_positions:
        return {
            "user_prompt": prompt_text,
            "constraint": constraint,
            "ssc_content": ssc_content,
            "ssc_decoded": ssc_decoded,
            "error": "Could not find SSC token positions",
            "predictions": [],
        }

    # Extract activations via ResidualCaptureHook
    try:
        with ResidualCaptureHook(model, target_layer) as hook:
            with torch.no_grad():
                # Run forward pass to capture representations
                _ = model(tokens.unsqueeze(0))
                residual_reps = hook.representations

        if residual_reps is None:
            raise RuntimeError("Failed to capture residual representations")

        # Get token embeddings from the model
        token_embeddings = model.model.embed_tokens.weight  # (vocab_size, hidden_dim)

        # Compute similarities for all positions
        # residual_reps: (1, seq_len, hidden_dim)
        similarities = compute_token_similarities(
            residual_reps[0],  # Remove batch dimension
            token_embeddings
        )  # Result: (seq_len, vocab_size)

    except Exception as e:
        return {
            "user_prompt": prompt_text,
            "constraint": constraint,
            "ssc_content": ssc_content,
            "ssc_decoded": ssc_decoded,
            "error": f"Error extracting similarities: {str(e)}",
            "predictions": [],
        }

    # Extract per-token top-k similar tokens
    predictions: List[Dict[str, Any]] = []

    for offset, pos in enumerate(ssc_positions):
        position_similarities = similarities[pos, :]  # [vocab_size]

        # Get top-k similar tokens for this position
        top_k_tokens = get_top_k_similar_tokens(
            position_similarities,
            tokenizer,
            top_k,
            min_similarity,
        )

        predictions.append({
            "position": int(pos),
            "ssc_token_index": int(offset),
            "ssc_token_id": int(tokens[pos].item()),
            "ssc_token": tokenizer.decode([int(tokens[pos].item())]),
            "top_k_tokens": top_k_tokens,
        })

    return {
        "user_prompt": prompt_text,
        "constraint": constraint,
        "ssc_content": ssc_content,
        "ssc_decoded": ssc_decoded,
        "target_layer": target_layer,
        "sequence_length": int(tokens.shape[0]),
        "ssc_token_count": len(ssc_positions),
        "predictions": predictions,
    }


def generate(
    model,
    tokenizer,
    user_prompts: List[str],
    constraints: List[str],
    system_prompt: Optional[str],
    target_layer: int,
    top_k: int,
    min_similarity: float,
    device: str,
) -> List[Dict[str, Any]]:
    """Extract similarities for all (user_prompt, constraint) pairs.

    Args:
        model: Language model
        tokenizer: Tokenizer
        user_prompts: List of base user prompts
        constraints: List of constraints
        system_prompt: Optional system prompt
        target_layer: Layer to extract from
        top_k: Number of top tokens per position
        min_similarity: Minimum similarity threshold
        device: Device

    Returns:
        List of extraction results, one per (user_prompt, constraint) pair
    """
    results = []

    for constraint in tqdm(constraints, desc="Extracting similarities"):
        for user_prompt in user_prompts:
            # Format the prompt with SSC
            full_prompt = format_ssc_user_prompt(user_prompt, constraint)

            # Extract similarities
            result = extract_similarities_from_ssc_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_text=full_prompt,
                constraint=constraint,
                target_layer=target_layer,
                top_k=top_k,
                min_similarity=min_similarity,
                device=device,
                system_prompt=system_prompt,
            )

            # Add base user prompt for reference
            result["base_user_prompt"] = user_prompt

            results.append(result)

    return results
