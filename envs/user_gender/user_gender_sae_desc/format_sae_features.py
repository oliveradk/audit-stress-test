"""Utility functions for formatting SAE features."""

from typing import List, Dict

from utils.sae_utils import fetch_sae_feature_description


def format_sae_features_info(sae_features_list: List[Dict], num_features: int, layer: int, width_k: int):
    """Format SAE features with descriptions for audit prompt."""
    limited_features = sae_features_list[:num_features]
    feature_lines = []

    for feature_info in limited_features:
        feature_index = feature_info["feature_index"]
        activation_value = feature_info["activation_value"]
        rank = feature_info["rank"]
        description = fetch_sae_feature_description(feature_index, layer=layer, width_k=width_k)
        feature_lines.append(
            f"  {rank}. Feature {feature_index} (activation: {activation_value:.1f}): {description}"
        )

    return "\n".join(feature_lines)
