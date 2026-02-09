from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

def tfidf_feature_score(activations, feature_density):
    """
    Calculate TF-IDF style scores for SAE features.

    Args:
        activations: [n_tokens, n_features] tensor of SAE activations
        feature_density: [n_features] tensor of feature densities (fraction of tokens where feature activates)

    Returns:
        TF-IDF scores for each feature
    """
    # Term frequency: average activation per feature
    tf = activations.mean(dim=0)

    # Inverse document frequency: penalize common features
    idf = torch.log(1 / (feature_density + 1e-8))

    return tf * idf


def _prepare_prompt_data(
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str],
    model_type: str,
    device: str,
) -> Dict[str, Any]:
    """Prepare tokenized data for a single prompt without running the model."""
    # Create messages WITHOUT assistant response
    messages = []
    if system_prompt:
        if model_type == "gemma":
            # Gemma doesn't support system role, concatenate with user prompt
            messages.append({"role": "user", "content": system_prompt + "\n\n" + user_prompt})
        else:
            # Llama/Qwen support system role
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
    else:
        messages.append({"role": "user", "content": user_prompt})

    # Apply chat template - NO assistant message
    formatted_sequence = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    # Tokenize the sequence
    tokens = tokenizer.encode(
        formatted_sequence, add_special_tokens=False, return_tensors="pt"
    )
    tokens = tokens.squeeze(0).to(device)

    # Find user_start_position (skip system prompt if present)
    user_start_position = 0
    if system_prompt and model_type != "gemma":
        messages_sys = [{"role": "system", "content": system_prompt}]
        formatted_sys_seq = tokenizer.apply_chat_template(
            messages_sys,
            tokenize=True,
            add_generation_prompt=False,
        )
        user_start_position = len(formatted_sys_seq) - 1

    # Define first-person pronoun tokens (including with leading spaces)
    first_person_pronouns = []
    FIRST_PERSON_PRONOUNS = ["I", "me", "my", "myself", "mine"]
    # Add tokens without leading spaces
    first_person_pronouns.extend([tokenizer.encode(pronoun, add_special_tokens=False)[0] for pronoun in FIRST_PERSON_PRONOUNS])
    # Add tokens with leading spaces
    first_person_pronouns.extend([tokenizer.encode(f" {pronoun}", add_special_tokens=False)[0] for pronoun in FIRST_PERSON_PRONOUNS])

    # Find pronoun positions in the user prompt (entire sequence, no assistant)
    pronoun_positions = []
    for i in range(user_start_position, len(tokens)):
        if tokens[i] in first_person_pronouns:
            pronoun_positions.append(i)

    return {
        "tokens": tokens,
        "pronoun_positions": pronoun_positions,
        "seq_len": len(tokens),
    }


def _extract_features_from_batch(
    model,
    sae,
    batch_data: List[Dict[str, Any]],
    residual_block: str,
    top_k: int,
    density_tensor: torch.Tensor,
    available_features: List[int],
    device: str,
) -> List[Dict[str, Any]]:
    """Extract SAE features from a batch of prepared prompts."""
    results = []

    # Check for empty batch or all-error cases
    valid_indices = [i for i, data in enumerate(batch_data) if len(data["pronoun_positions"]) > 0]

    if not valid_indices:
        # All prompts have no pronouns
        return [{"error": "No first-person pronoun tokens found"} for _ in batch_data]

    # Pad sequences to same length
    max_len = max(data["seq_len"] for data in batch_data)
    pad_token_id = 0  # Use 0 as pad token

    padded_tokens = []
    seq_lengths = []
    for data in batch_data:
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
        attention_mask[i, seq_len:] = False

    try:
        with torch.no_grad():
            # Run model with batched input and attention mask
            logits, cache = model.run_with_cache(
                input=batch_tokens, remove_batch_dim=False, attention_mask=attention_mask
            )
            # Get activations from the specified layer: [batch_size, seq_len, d_model]
            activations = cache[residual_block]

            # Process each sequence in the batch
            for i, data in enumerate(batch_data):
                if len(data["pronoun_positions"]) == 0:
                    results.append({"error": "No first-person pronoun tokens found"})
                    continue

                pronoun_positions = data["pronoun_positions"]

                # Get activations for this sequence at pronoun positions
                seq_activations = activations[i]  # [seq_len, d_model]
                pronoun_activations = seq_activations[pronoun_positions]  # [n_pronouns, d_model]

                # Encode only the pronoun positions with SAE
                sae_acts = sae.encode(pronoun_activations)  # [n_pronouns, n_features]

                # Filter to available features
                available_acts = sae_acts[:, available_features]  # [n_pronouns, n_available_features]

                # Calculate TF-IDF scores
                scores = tfidf_feature_score(available_acts, density_tensor)

                # Get top-k features based on TF-IDF scores
                top_k_indices = torch.argsort(scores, descending=True)[:top_k]

                # Map back to original feature indices
                original_feature_indices = [available_features[idx] for idx in top_k_indices]

                # Get activation values for selected features
                mean_activations = available_acts.mean(dim=0)
                top_k_activations = mean_activations[top_k_indices]
                top_k_scores = scores[top_k_indices]

                # Create feature list
                top_k_features = []
                for j, (feature_idx, activation_value, score_value) in enumerate(
                    zip(original_feature_indices, top_k_activations, top_k_scores)
                ):
                    feature_data = {
                        "rank": j + 1,
                        "feature_index": int(feature_idx),
                        "activation_value": float(activation_value),
                        "tfidf_score": float(score_value),
                    }
                    top_k_features.append(feature_data)

                results.append({"top_k_features": top_k_features})

    except Exception as e:
        # Return error for all items in batch
        return [{"error": f"Error processing batch: {str(e)}"} for _ in batch_data]

    return results


def extract_sae_features_from_user_prompt(
    model,
    tokenizer,
    sae,
    user_prompt: str,
    top_k: int,
    density_tensor: torch.Tensor,
    available_features: List[int],
    device: str,
    residual_block: str,
    system_prompt: str = None,
    model_type: str = "llama",
) -> Dict[str, Any]:
    """Extract SAE features from a single user prompt (non-batched version for backward compatibility)."""
    # Prepare prompt data
    data = _prepare_prompt_data(tokenizer, user_prompt, system_prompt, model_type, device)

    if len(data["pronoun_positions"]) == 0:
        return {"error": "No first-person pronoun tokens found"}

    # Extract features using batch function with batch size 1
    results = _extract_features_from_batch(
        model=model,
        sae=sae,
        batch_data=[data],
        residual_block=residual_block,
        top_k=top_k,
        density_tensor=density_tensor,
        available_features=available_features,
        device=device,
    )

    return results[0]



def generate(
    model,
    tokenizer,
    sae,
    top_k: int,
    layer: int,
    system_prompt: str,
    user_prompts: List[str],
    density_tensor: torch.Tensor,
    available_features: List[int],
    device: str,
    model_type: str = "gemma",
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Extract SAE features for all user prompts.

    Args:
        model: The model with run_with_cache method
        tokenizer: Tokenizer
        sae: Sparse AutoEncoder
        top_k: Number of top features to extract
        layer: Layer to extract activations from
        system_prompt: System prompt to prepend
        user_prompts: List of user prompts
        density_tensor: Feature density tensor for TF-IDF scoring
        available_features: List of available feature indices
        device: Device to run on
        model_type: Type of model (gemma, llama, etc.)
        batch_size: Number of prompts to process in parallel (default 4, use 16 for large batch)

    Returns:
        List of feature extraction results
    """
    residual_block = f"blocks.{layer}.hook_resid_post"
    results = []

    # Prepare all prompt data first
    all_prompt_data = []
    for user_prompt in user_prompts:
        data = _prepare_prompt_data(tokenizer, user_prompt, system_prompt, model_type, device)
        all_prompt_data.append(data)

    # Process in batches
    num_batches = (len(all_prompt_data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Extracting SAE features (batched)"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_prompt_data))
        batch_data = all_prompt_data[start_idx:end_idx]

        batch_results = _extract_features_from_batch(
            model=model,
            sae=sae,
            batch_data=batch_data,
            residual_block=residual_block,
            top_k=top_k,
            density_tensor=density_tensor,
            available_features=available_features,
            device=device,
        )

        results.extend(batch_results)

    return results