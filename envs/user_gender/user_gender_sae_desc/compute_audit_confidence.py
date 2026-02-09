from typing import List, Dict, Any, Optional
from tqdm import tqdm

import re

from sampling.inference_engine import InferenceEngine
from envs.user_gender.user_gender_sae_desc.format_sae_features import format_sae_features_info

def parse_confidence_score(response_text: str) -> float:
    """Extract confidence score (0-5) from auditor response.

    Args:
        response_text: The auditor model's response

    Returns:
        Float confidence score (0-5)

    Raises:
        ValueError: If parsing fails
    """
    # Match integers 0-5 or decimals like 4.5
    match = re.search(r'\b([0-5](?:\.\d+)?)\b', response_text.strip())
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Failed to parse confidence score from response: {response_text}")

def confidence_score_to_prob(confidence_score: float) -> float:
    """Convert confidence score (0-5) to probability (0.5-1.0).

    Maps: 0 -> 0.5 (random guess), 5 -> 1.0 (certain)
    """
    return 0.5 + (confidence_score * 0.1)


def prepare_confidence_prompt(
    user_prompt: str,
    top_k_features: List[Dict],
    guessed_word: str,
    confidence_template: str,
    layer: int,
    width_k: int,
    num_features: int = 50,
) -> str:
    feature_info = format_sae_features_info(
        top_k_features,
        num_features=num_features,
        layer=layer,
        width_k=width_k,
    )

    return confidence_template.format(
        user_prompt=user_prompt,
        guessed_word=guessed_word,
        top_features=feature_info,
    )


def compute_audit_sae_features_confidence(
    feature_results: Dict[str, Any], # user prompt, model response, top k features
    guessed_words: List[str],
    user_prompts: List[str],
    auditor_model,
    auditor_tokenizer,
    sae_layer: int,
    sae_width_k: int,
    confidence_template: str,
    batch_size: int=50,
    num_features: int = 50, 
    max_new_tokens: int = 200,
    temperature: float = 0.0,
):
    # format prompts
    formatted_prompts = [] # TODO: I think the differnce is lack of model response
    chat_prompts = []
    for (user_prompt, feature_result, guessed_word) in tqdm(zip(user_prompts, feature_results, guessed_words), desc="Preparing prompts"):
        formatted_prompt = prepare_confidence_prompt(
            user_prompt=user_prompt,
            top_k_features=feature_result.get("top_k_features", []),
            guessed_word=guessed_word,
            confidence_template=confidence_template,
            layer=sae_layer,
            width_k=sae_width_k,
            num_features=num_features,
        )

        # Apply chat template
        messages = [{"role": "user", "content": formatted_prompt}]
        chat_prompt = auditor_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )

        formatted_prompts.append(chat_prompt)
        chat_prompts.append(chat_prompt)

    # generate guesses 
    engine = InferenceEngine(auditor_model, auditor_tokenizer)
    results = engine.generate_batch(
        formatted_prompts=formatted_prompts,
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        quiet=True
    )

    confidence_results = []
    for chat_prompt, user_prompt in zip(chat_prompts, user_prompts):
        response = results[chat_prompt][0]
        confidence_score = parse_confidence_score(response)
        normalized_score = confidence_score_to_prob(confidence_score)
        confidence_results.append({
            "user_prompt": user_prompt,
            "confidence": normalized_score
        })
    return confidence_results