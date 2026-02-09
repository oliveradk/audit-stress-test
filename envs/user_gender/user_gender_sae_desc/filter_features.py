from typing import List, Dict, Any

import re
from tqdm import tqdm

from sampling.inference_engine import InferenceEngine
from utils.sae_utils import fetch_sae_feature_description
from utils.sae_utils import load_sae_feature_cache
from utils.sae_utils import prefetch_sae_feature_descriptions


def format_features_for_prompt(
    features: List[Dict],
    layer_idx: int,
    descriptions_cache: Dict[int, str] = None
) -> str:
    feature_lines = []
    for feature_info in features:
        feature_index = feature_info.get("feature_index", "?")

        # Use prefetched cache if available, otherwise fetch
        if descriptions_cache and feature_index in descriptions_cache:
            description = descriptions_cache[feature_index]
        else:
            description = fetch_sae_feature_description(feature_index, layer=layer_idx)

        # Format as: Feature <index>: <description>
        feature_lines.append(f"{feature_index}: {description}")

    return "\n".join(feature_lines)


def parse_filter_response(
    response_text: str, all_feature_indices: List[int]
) -> List[int]:
    # Check if response indicates no relevant features
    if "NONE" in response_text.upper():
        return []

    # Extract numbers from the response
    # Look for comma-separated numbers or individual numbers
    numbers = re.findall(r"\d+", response_text)

    # Convert to integers and filter to only include valid feature indices
    relevant_indices = []
    for num_str in numbers:
        try:
            feature_idx = int(num_str)
            # Only include if it was in the original list of features
            if feature_idx in all_feature_indices:
                relevant_indices.append(feature_idx)
        except ValueError:
            continue

    return relevant_indices


def filter_results_data(feature_results: Dict[str, Any], relevant_features: set) -> Dict[str, Any]:
    # Create a copy of the data
    original_results = feature_results.copy()

    filtered_results = []

    for entry in original_results:
        # Create a copy of the entry
        filtered_entry = entry.copy()

        # Filter top_k_features if present
        if "top_k_features" in filtered_entry:
            original_features = filtered_entry["top_k_features"]
            filtered_features = [
                f for f in original_features
                if f.get("feature_index") in relevant_features
            ]

            # Re-rank the filtered features
            for i, feature in enumerate(filtered_features):
                feature["rank"] = i + 1

            filtered_entry["top_k_features"] = filtered_features

        filtered_results.append(filtered_entry)

    return filtered_results

def filter_sae_features(
    feature_results: List[Dict[str, Any]],
    prompt_template: str,
    auditor_model,
    auditor_tokenizer,
    sae_layer: int,
    features_per_prompt: int=50,
    batch_size: int=50,
    max_new_tokens: int=200,
    temperature: float=0.0
):

    load_sae_feature_cache()

    # Collect all unique features from the data
    all_features = []
    feature_indices_set = set()

    for entry in feature_results:
        if "top_k_features" in entry:
            for feature in entry["top_k_features"]:
                feature_idx = feature.get("feature_index")
                if feature_idx not in feature_indices_set:
                    feature_indices_set.add(feature_idx)
                    all_features.append(feature)

    # Prefetch all feature descriptions in parallel
    all_feature_indices = [f.get("feature_index") for f in all_features]
    print(f"Prefetching descriptions for {len(all_feature_indices)} unique features...")
    descriptions_cache = prefetch_sae_feature_descriptions(all_feature_indices, layer=sae_layer)

    # Process features in batches
    relevant_features = set()

    # Collect multiple batches and process them together
    all_batches = []
    batch_indices_map = []  # Keep track of which indices belong to which batch

    # Prepare all batches
    for i in range(0, len(all_features), features_per_prompt):
        batch = all_features[i : i + features_per_prompt]
        all_batches.append(batch)
        batch_indices_map.append([f.get("feature_index") for f in batch])

    num_batch_groups = (len(all_batches) + batch_size - 1) // batch_size
    print(f"Filtering {len(all_features)} unique features → {len(all_batches)} prompts ({features_per_prompt} features/prompt) → {num_batch_groups} batch groups (batch_size={batch_size})")

    # Process batches in groups for efficient GPU utilization
    pbar = tqdm(
        range(0, len(all_batches), batch_size),
        desc="Filtering features",
        unit="group",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} groups [{elapsed}<{remaining}]"
    )
    for group_start in pbar:
        group_end = min(group_start + batch_size, len(all_batches))
        batch_group = all_batches[group_start:group_end]
        batch_indices_group = batch_indices_map[group_start:group_end]
        group_num = group_start // batch_size + 1

        # Prepare all prompts for this group
        formatted_prompts = []
        prompt_to_indices_map = {}
        for batch, batch_indices in zip(batch_group, batch_indices_group):
            # Format features with descriptions (using prefetched cache)
            features_formatted = format_features_for_prompt(batch, sae_layer, descriptions_cache)
            prompt = prompt_template.format(top_features=features_formatted)

            # Apply chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = auditor_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            formatted_prompts.append(formatted_prompt)
            prompt_to_indices_map[formatted_prompt] = batch_indices

        # Generate responses for all prompts in this group
        engine = InferenceEngine(auditor_model, auditor_tokenizer)
        results = engine.generate_batch(
            formatted_prompts=formatted_prompts,
            num_responses_per_prompt=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            quiet=True,
        )

        # Parse responses and collect relevant features
        group_relevant = 0
        for formatted_prompt, responses in results.items():
            response_text = responses[0]
            batch_indices = prompt_to_indices_map[formatted_prompt]
            batch_relevant_indices = parse_filter_response(response_text, batch_indices)
            relevant_features.update(batch_relevant_indices)
            group_relevant += len(batch_relevant_indices)

        pbar.set_postfix_str(f"group {group_num}: +{group_relevant} relevant, total={len(relevant_features)}")

    print(f"\nTotal relevant features identified: {len(relevant_features)}")

    # Filter the original data to only include relevant features
    filtered_results = filter_results_data(feature_results, relevant_features)
    return filtered_results