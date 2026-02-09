"""Extract SAE features from SSC span tokens.

This module handles the feature extraction phase of the SSC SAE auditing pipeline,
extracting per-token SAE features from the secret side constraint span.
"""

import base64
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

from utils.sae_utils import (
    extract_ssc_content,
    find_ssc_token_positions,
    tfidf_feature_score,
    fetch_sae_feature_description_goodfire,
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


def _prepare_ssc_prompt_data(
    tokenizer,
    prompt_text: str,
    constraint: str,
    system_prompt: Optional[str],
    device: str,
) -> Dict[str, Any]:
    """Prepare tokenized data for a single SSC prompt without running the model."""
    # Extract SSC content
    ssc_content = extract_ssc_content(prompt_text)
    if not ssc_content:
        return {
            "prompt_text": prompt_text,
            "constraint": constraint,
            "error": "No SSC content found",
            "tokens": None,
            "ssc_positions": [],
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

    return {
        "prompt_text": prompt_text,
        "constraint": constraint,
        "ssc_content": ssc_content,
        "ssc_decoded": ssc_decoded,
        "tokens": tokens,
        "ssc_positions": ssc_positions,
        "seq_len": len(tokens),
        "error": None if ssc_positions else "Could not find SSC token positions",
    }


def _extract_ssc_features_from_batch(
    model,
    sae,
    batch_data: List[Dict[str, Any]],
    target_layer: int,
    top_k: int,
    feature_densities: Optional[torch.Tensor],
    use_tfidf: bool,
    device: str,
    tokenizer,
) -> List[Dict[str, Any]]:
    """Extract SAE features from a batch of prepared SSC prompts."""
    results = []

    # Check for valid items in batch
    valid_indices = [i for i, data in enumerate(batch_data)
                     if data["tokens"] is not None and len(data["ssc_positions"]) > 0]

    if not valid_indices:
        # All prompts have errors
        for data in batch_data:
            results.append({
                "user_prompt": data["prompt_text"],
                "constraint": data["constraint"],
                "error": data.get("error", "No valid SSC positions"),
                "predictions": [],
            })
        return results

    # Pad sequences to same length
    max_len = max(data["seq_len"] for data in batch_data if data["tokens"] is not None)
    pad_token_id = 0

    padded_tokens = []
    seq_lengths = []
    for data in batch_data:
        if data["tokens"] is None:
            # Create dummy tokens for error cases
            padded = torch.full((max_len,), pad_token_id, dtype=torch.long, device=device)
            seq_lengths.append(0)
        else:
            tokens = data["tokens"]
            seq_lengths.append(len(tokens))
            if len(tokens) < max_len:
                padding = torch.full((max_len - len(tokens),), pad_token_id, dtype=tokens.dtype, device=device)
                padded = torch.cat([tokens, padding])
            else:
                padded = tokens
        padded_tokens.append(padded)

    batch_tokens = torch.stack(padded_tokens)  # [batch_size, max_len]

    # Create attention mask (True for real tokens, False for padding)
    attention_mask = torch.ones(batch_tokens.shape, dtype=torch.bool, device=device)
    for i, seq_len in enumerate(seq_lengths):
        if seq_len > 0:
            attention_mask[i, seq_len:] = False
        else:
            attention_mask[i, :] = False  # All padding for error cases

    try:
        # Extract activations via ObservableLanguageModel
        hook_name = f"model.layers.{target_layer}"
        logits, kv_cache, feature_cache = model.forward(
            batch_tokens,
            cache_activations_at=[hook_name],
            attention_mask=attention_mask,
        )
        activations = feature_cache[hook_name]  # [batch_size, seq_len, d_model]

        # Encode all activations with SAE
        # Reshape for SAE: [batch_size * seq_len, d_model]
        batch_size, seq_len, d_model = activations.shape
        flat_activations = activations.reshape(-1, d_model)
        sae_features_flat = sae.encode(flat_activations)  # [batch_size * seq_len, n_features]
        sae_features = sae_features_flat.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, n_features]

        # Process each sequence in the batch
        for i, data in enumerate(batch_data):
            if data["tokens"] is None or not data["ssc_positions"]:
                results.append({
                    "user_prompt": data["prompt_text"],
                    "constraint": data["constraint"],
                    "ssc_content": data.get("ssc_content"),
                    "ssc_decoded": data.get("ssc_decoded"),
                    "error": data.get("error", "No valid SSC positions"),
                    "predictions": [],
                })
                continue

            ssc_positions = data["ssc_positions"]
            tokens = data["tokens"]
            seq_sae_features = sae_features[i]  # [seq_len, n_features]

            # Compute per-token top-k features
            predictions: List[Dict[str, Any]] = []

            for offset, pos in enumerate(ssc_positions):
                feats_vec = seq_sae_features[pos, :]

                # Choose scoring method
                if use_tfidf and feature_densities is not None and len(feature_densities) > 0:
                    single_token_activations = feats_vec.unsqueeze(0)
                    tfidf_scores = tfidf_feature_score(single_token_activations, feature_densities)
                    top_values, top_indices = torch.topk(tfidf_scores, min(top_k, len(tfidf_scores)))
                    selection_method = "tfidf"
                else:
                    top_values, top_indices = torch.topk(feats_vec, min(top_k, len(feats_vec)))
                    selection_method = "activation"

                # Build feature list for this token position
                top_k_features = []
                for j, (idx, val) in enumerate(zip(top_indices, top_values)):
                    feature_idx = int(idx.item())
                    feature_data = {
                        "rank": j + 1,
                        "feature_index": feature_idx,
                        "activation_value": float(feats_vec[feature_idx].item()),
                        "selection_score": float(val.item()),
                    }
                    if feature_densities is not None and feature_idx < len(feature_densities):
                        feature_data["density"] = float(feature_densities[feature_idx].item())
                    top_k_features.append(feature_data)

                predictions.append({
                    "position": int(pos),
                    "ssc_token_index": int(offset),
                    "ssc_token_id": int(tokens[pos].item()),
                    "ssc_token": tokenizer.decode([int(tokens[pos].item())]),
                    "selection_method": selection_method,
                    "top_k_features": top_k_features,
                })

            results.append({
                "user_prompt": data["prompt_text"],
                "constraint": data["constraint"],
                "ssc_content": data["ssc_content"],
                "ssc_decoded": data["ssc_decoded"],
                "target_layer": target_layer,
                "sequence_length": int(tokens.shape[0]),
                "ssc_token_count": len(ssc_positions),
                "predictions": predictions,
            })

    except Exception as e:
        # Return error for all items in batch
        for data in batch_data:
            results.append({
                "user_prompt": data["prompt_text"],
                "constraint": data["constraint"],
                "error": f"Error processing batch: {str(e)}",
                "predictions": [],
            })

    return results


def extract_sae_features_from_ssc_prompt(
    model,
    tokenizer,
    sae,
    prompt_text: str,
    constraint: str,
    target_layer: int,
    top_k: int,
    feature_densities: Optional[torch.Tensor],
    use_tfidf: bool,
    device: str,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract SAE features from a single SSC prompt (non-batched version for backward compatibility)."""
    # Prepare prompt data
    data = _prepare_ssc_prompt_data(tokenizer, prompt_text, constraint, system_prompt, device)

    if data["error"]:
        return {
            "user_prompt": prompt_text,
            "constraint": constraint,
            "ssc_content": data.get("ssc_content"),
            "ssc_decoded": data.get("ssc_decoded"),
            "error": data["error"],
            "predictions": [],
        }

    # Extract features using batch function with batch size 1
    results = _extract_ssc_features_from_batch(
        model=model,
        sae=sae,
        batch_data=[data],
        target_layer=target_layer,
        top_k=top_k,
        feature_densities=feature_densities,
        use_tfidf=use_tfidf,
        device=device,
        tokenizer=tokenizer,
    )

    return results[0]


def fetch_feature_descriptions(
    predictions: List[Dict[str, Any]],
    num_features: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch Goodfire descriptions for top features in predictions.

    Args:
        predictions: List of per-token prediction dicts
        num_features: Number of top features to fetch descriptions for

    Returns:
        Updated predictions with feature descriptions added
    """
    # Collect unique feature indices
    unique_features = set()
    for pred in predictions:
        for feat in pred.get("top_k_features", [])[:num_features]:
            unique_features.add(feat["feature_index"])

    # Fetch descriptions
    descriptions = {}
    for feat_idx in tqdm(unique_features, desc="Fetching feature descriptions"):
        descriptions[feat_idx] = fetch_sae_feature_description_goodfire(feat_idx)

    # Add descriptions to predictions
    for pred in predictions:
        for feat in pred.get("top_k_features", []):
            feat_idx = feat["feature_index"]
            if feat_idx in descriptions:
                feat["description"] = descriptions[feat_idx]

    return predictions


def extract_sae_features_ssc(
    model,
    tokenizer,
    sae,
    user_prompts: List[str],
    constraints: List[str],
    system_prompt: Optional[str],
    target_layer: int,
    top_k: int,
    feature_densities: Optional[torch.Tensor],
    use_tfidf: bool,
    device: str,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Extract SAE features for all (user_prompt, constraint) pairs.

    Args:
        model: ObservableLanguageModel
        tokenizer: Tokenizer
        sae: Sparse AutoEncoder
        user_prompts: List of base user prompts
        constraints: List of constraints
        system_prompt: Optional system prompt
        target_layer: Layer to extract from
        top_k: Number of top features per position
        feature_densities: Feature density tensor
        use_tfidf: Whether to use TF-IDF
        device: Device
        batch_size: Number of prompts to process in parallel (default 4, use 16 for large batch)

    Returns:
        List of extraction results, one per (user_prompt, constraint) pair
    """
    # Prepare all prompts first
    all_prompt_data = []
    prompt_metadata = []  # Track user_prompt for each item

    for constraint in constraints:
        for user_prompt in user_prompts:
            # Format the prompt with SSC
            full_prompt = format_ssc_user_prompt(user_prompt, constraint)

            # Prepare data
            data = _prepare_ssc_prompt_data(tokenizer, full_prompt, constraint, system_prompt, device)
            all_prompt_data.append(data)
            prompt_metadata.append({"user_prompt": user_prompt})

    results = []

    # Process in batches
    num_batches = (len(all_prompt_data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Extracting SAE features (batched)"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_prompt_data))
        batch_data = all_prompt_data[start_idx:end_idx]
        batch_metadata = prompt_metadata[start_idx:end_idx]

        batch_results = _extract_ssc_features_from_batch(
            model=model,
            sae=sae,
            batch_data=batch_data,
            target_layer=target_layer,
            top_k=top_k,
            feature_densities=feature_densities,
            use_tfidf=use_tfidf,
            device=device,
            tokenizer=tokenizer,
        )

        # Add metadata and fetch descriptions
        for result, metadata in zip(batch_results, batch_metadata):
            result["base_user_prompt"] = metadata["user_prompt"]

            # Fetch feature descriptions for this result
            if "predictions" in result and result["predictions"]:
                result["predictions"] = fetch_feature_descriptions(
                    result["predictions"], num_features=top_k
                )

            results.append(result)

    return results
