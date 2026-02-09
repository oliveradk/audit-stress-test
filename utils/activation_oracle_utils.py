"""
Activation Oracle utilities for running verbalizer-based auditing.
Adapted from the activation_oracles repository (nl_probes package).
"""

import os
import contextlib
import math
import json
import random
import warnings
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch._dynamo as dynamo
from dataclasses import dataclass, field
from peft import PeftModel, LoraConfig
from pydantic import BaseModel, ConfigDict, model_validator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from utils.relevancy_utils import compute_normalized_relevancy


# ============================================================================
# Common utilities (from nl_probes/utils/common.py)
# ============================================================================

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str,
    dtype: torch.dtype,
    **model_kwargs,
) -> AutoModelForCausalLM:
    print("Loading model...")

    # Gemma prefers eager attention; others use FA2
    attn = "eager" if "gemma" in model_name.lower() else "flash_attention_2"

    kwargs: dict = {
        "device_map": {"": 0},
        "attn_implementation": attn,
        "torch_dtype": dtype,
        **model_kwargs,
    }

    print("model kwargs: ", kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.bos_token_id:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    return tokenizer


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    LAYER_COUNTS = {
        "Qwen/Qwen3-1.7B": 28,
        "Qwen/Qwen3-8B": 36,
        "Qwen/Qwen3-32B": 64,
        "google/gemma-2-9b-it": 42,
        "google/gemma-3-1b-it": 26,
        "meta-llama/Llama-3.2-1B-Instruct": 16,
        "meta-llama/Llama-3.3-70B-Instruct": 80,
    }
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * (layer_percent / 100))


# ============================================================================
# Activation utilities (from nl_probes/utils/activation_utils.py)
# ============================================================================

class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""
    pass


def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    if min_offset is not None:
        assert max_offset is not None, "max_offset must be provided if min_offset is provided"
        assert max_offset < min_offset, "max_offset must be less than min_offset"
        assert min_offset < 0, "min_offset must be less than 0"
        assert max_offset < 0, "max_offset must be less than 0"
    else:
        assert max_offset is None, "max_offset must be provided if min_offset is not provided"

    activations_BLD_by_layer = {}
    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}
    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]

        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs

        if min_offset is not None:
            activations_BLD_by_layer[layer] = activations_BLD_by_layer[layer][:, max_offset:min_offset, :]

        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = []
    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        with torch.no_grad():
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations_BLD_by_layer


def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HF transformers"""
    model_name = model.config._name_or_path

    if use_lora:
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


# ============================================================================
# Steering hooks (from nl_probes/utils/steering_hooks.py)
# ============================================================================

@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """HF hook for activation steering with variable positions per batch element."""
    assert len(vectors) == len(positions), "vectors and positions must have same batch length"
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model_actual = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")

        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        for b in range(B):
            pos_b = positions[b]
            pos_b = torch.tensor(pos_b, dtype=torch.long, device=device)
            assert pos_b.min() >= 0
            assert pos_b.max() < L
            orig_KD = resid_BLD[b, pos_b, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)

            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)
            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


# ============================================================================
# Dataset utilities (from nl_probes/utils/dataset_utils.py)
# ============================================================================

SPECIAL_TOKEN = " ?"


def get_introspection_prefix(sae_layer: int, num_positions: int) -> str:
    prefix = f"Layer: {sae_layer}\n"
    prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"
    return prefix


class FeatureResult(BaseModel):
    """Result for a single feature evaluation."""
    feature_idx: int
    api_response: str
    prompt: str
    meta_info: Mapping[str, Any] = {}
    relevancy: float = 0.5  # Normalized relevancy score (0.5 = neutral)


class TrainingDataPoint(BaseModel):
    """Training data point with tensors."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datapoint_type: str
    input_ids: list[int]
    labels: list[int]
    layer: int
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None
    context_positions: list[int] | None
    ds_label: str | None
    meta_info: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def _check_context_alignment(cls, values):
        sv = values.steering_vectors
        if sv is not None:
            if len(values.positions) != sv.shape[0]:
                raise ValueError("positions and steering_vectors must have the same length")
        else:
            if values.context_positions is None or values.context_input_ids is None:
                raise ValueError("context_* must be provided when steering_vectors is None")
            if len(values.positions) != len(values.context_positions):
                raise ValueError("positions and context_positions must have the same length")
        return values


class BatchData(BaseModel):
    """Batch of training data with tensors."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]


def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> BatchData:
    max_length = 0
    for data_point in training_data:
        max_length = max(max_length, len(data_point.input_ids))

    batch_tokens = []
    batch_labels = []
    batch_attn_masks = []
    batch_positions = []
    batch_steering_vectors = []
    batch_feature_indices = []

    for data_point in training_data:
        padding_length = max_length - len(data_point.input_ids)
        padding_tokens = [tokenizer.pad_token_id] * padding_length
        padded_input_ids = padding_tokens + data_point.input_ids
        padded_labels = [-100] * padding_length + data_point.labels

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
        labels = torch.tensor(padded_labels, dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool).to(device)
        attn_mask[:padding_length] = False

        batch_tokens.append(input_ids)
        batch_labels.append(labels)
        batch_attn_masks.append(attn_mask)

        padded_positions = [p + padding_length for p in data_point.positions]

        if data_point.steering_vectors is not None:
            steering_vectors = data_point.steering_vectors.to(device)
        else:
            steering_vectors = None

        batch_positions.append(padded_positions)
        batch_steering_vectors.append(steering_vectors)
        batch_feature_indices.append(data_point.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
        feature_indices=batch_feature_indices,
    )


def get_prompt_tokens_only(training_data_point: TrainingDataPoint) -> TrainingDataPoint:
    """User prompt should be labeled as -100"""
    prompt_tokens = []
    prompt_labels = []

    response_token_seen = False
    for i in range(len(training_data_point.input_ids)):
        if training_data_point.labels[i] != -100:
            response_token_seen = True
            continue
        else:
            if response_token_seen:
                raise ValueError("Response token seen before prompt tokens")
            prompt_tokens.append(training_data_point.input_ids[i])
            prompt_labels.append(training_data_point.labels[i])
    new = training_data_point.model_copy()
    new.input_ids = prompt_tokens
    new.labels = prompt_labels
    return new


def materialize_missing_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    model: PeftModel,
) -> list[TrainingDataPoint]:
    """Materialize steering vectors from context activations."""
    to_fill: list[tuple[int, TrainingDataPoint]] = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    for _, dp in to_fill:
        if dp.context_input_ids is None or dp.context_positions is None:
            raise ValueError(
                "Datapoint has steering_vectors=None but is missing context_input_ids or context_positions"
            )

    pad_id = tokenizer.pad_token_id
    contexts: list[list[int]] = [list(dp.context_input_ids) for _, dp in to_fill]
    positions_per_item: list[list[int]] = [list(dp.context_positions) for _, dp in to_fill]
    max_len = max(len(c) for c in contexts)

    input_ids_tensors: list[torch.Tensor] = []
    attn_masks_tensors: list[torch.Tensor] = []
    left_offsets: list[int] = []

    device = next(model.parameters()).device

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_tensors.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
        attn_masks_tensors.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    layers_needed = sorted({dp.layer for _, dp in to_fill})
    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers_needed}

    was_training = model.training
    model.eval()
    with model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
    if was_training:
        model.train()

    new_batch: list[TrainingDataPoint] = list(batch_points)
    for b in range(len(to_fill)):
        idx, dp = to_fill[b]
        layer = dp.layer
        acts_BLD = acts_by_layer[layer]

        idxs = [p + left_offsets[b] for p in positions_per_item[b]]
        L = acts_BLD.shape[1]
        if any(i < 0 or i >= L for i in idxs):
            raise IndexError(f"Activation index out of range for item {b}: {idxs} with L={L}")

        vectors = acts_BLD[b, idxs, :].detach().contiguous()
        assert len(vectors.shape) == 2, f"Expected 2D tensor, got vectors.shape={vectors.shape}"

        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors
        new_batch[idx] = dp_new

    return new_batch


def find_pattern_in_tokens(
    token_ids: list[int], special_token_str: str, num_positions: int, tokenizer: AutoTokenizer
) -> list[int]:
    start_idx = 0
    end_idx = len(token_ids)
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    positions = []

    for i in range(start_idx, end_idx):
        if len(positions) == num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)

    assert len(positions) == num_positions, f"Expected {num_positions} positions, got {len(positions)}"
    assert positions[-1] - positions[0] == num_positions - 1, f"Positions are not consecutive: {positions}"

    final_pos = positions[-1] + 1
    final_tokens = token_ids[final_pos : final_pos + 2]
    final_str = tokenizer.decode(final_tokens, skip_special_tokens=False)
    assert "\n" in final_str, f"Expected newline in {final_str}"

    return positions


def create_training_datapoint(
    datapoint_type: str,
    prompt: str,
    target_response: str,
    layer: int,
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: torch.Tensor | None,
    feature_idx: int,
    context_input_ids: list[int] | None = None,
    context_positions: list[int] | None = None,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    if meta_info is None:
        meta_info = {}
    prefix = get_introspection_prefix(layer, num_positions)
    assert prefix not in prompt, f"Prefix {prefix} found in prompt {prompt}"
    prompt = prefix + prompt
    input_messages = [{"role": "user", "content": prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    full_messages = input_messages + [{"role": "assistant", "content": target_response}]

    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(full_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    assistant_start_idx = len(input_prompt_ids)

    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    positions = find_pattern_in_tokens(full_prompt_ids, SPECIAL_TOKEN, num_positions, tokenizer)

    if acts_BD is None:
        assert context_input_ids is not None and context_positions is not None, (
            "acts_BD is None but context_input_ids and context_positions are None"
        )
    else:
        assert len(acts_BD.shape) == 2, f"Expected 2D tensor, got {acts_BD.shape}"
        acts_BD = acts_BD.cpu().clone().detach()
        assert len(positions) == acts_BD.shape[0], f"Expected {acts_BD.shape[0]} positions, got {len(positions)}"

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layer=layer,
        steering_vectors=acts_BD,
        positions=positions,
        feature_idx=feature_idx,
        target_output=target_response,
        datapoint_type=datapoint_type,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        ds_label=ds_label,
        meta_info=meta_info,
    )

    return training_data_point


# ============================================================================
# Evaluation utilities (from nl_probes/utils/eval.py)
# ============================================================================

@dynamo.disable
@torch.no_grad()
def eval_features_batch(
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    steering_coefficient: float,
    generation_kwargs: dict,
    compute_relevancy: bool = False,
) -> list[FeatureResult]:
    batch_steering_vectors = eval_batch.steering_vectors
    batch_positions = eval_batch.positions

    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": eval_batch.input_ids,
        "attention_mask": eval_batch.attention_mask,
    }

    prompt_tokens = eval_batch.input_ids[:, : eval_batch.input_ids.shape[1]]
    decoded_prompts = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=False)

    feature_results = []

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Compute relevancy scores if requested
    relevancy_scores = [0.5] * len(eval_batch.feature_indices)
    if compute_relevancy:
        relevancy_scores = _compute_relevancy_scores(
            model=model,
            output_ids=output_ids,
            prompt_length=eval_batch.input_ids.shape[1],
            tokenizer=tokenizer,
        )

    for i in range(len(eval_batch.feature_indices)):
        feature_idx = eval_batch.feature_indices[i]
        output = decoded_output[i]

        feature_result = FeatureResult(
            feature_idx=feature_idx,
            api_response=output,
            prompt=decoded_prompts[i],
            relevancy=relevancy_scores[i],
        )
        feature_results.append(feature_result)

    return feature_results


def _extract_probs_for_tokens(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    prompt_length: int,
    pad_token_id: int,
) -> list[list[float]]:
    """Extract token probabilities from logits (vectorized per sequence).

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        generated_tokens: Generated token IDs [batch_size, gen_length]
        prompt_length: Length of the prompt (same for all sequences)
        pad_token_id: Padding token ID to skip

    Returns:
        List of probability lists, one per sequence
    """
    batch_size = logits.shape[0]
    gen_length = generated_tokens.shape[1]
    device = logits.device
    probs_batch = []

    # Positions that predict each generated token (token at pos N predicted by logits at N-1)
    positions = torch.arange(prompt_length - 1, prompt_length - 1 + gen_length, device=device)

    for i in range(batch_size):
        # Find non-padding tokens
        non_pad_mask = generated_tokens[i] != pad_token_id
        if not non_pad_mask.any():
            probs_batch.append([])
            continue

        # Get valid positions and tokens
        valid_positions = positions[non_pad_mask]
        valid_tokens = generated_tokens[i, non_pad_mask]

        # Extract logits at those positions and compute softmax
        relevant_logits = logits[i, valid_positions]
        probs = torch.softmax(relevant_logits, dim=-1)

        # Gather probabilities for target tokens
        token_probs = probs[torch.arange(len(valid_tokens), device=device), valid_tokens]
        probs_batch.append(token_probs.tolist())

    return probs_batch


def _compute_relevancy_scores(
    model: AutoModelForCausalLM,
    output_ids: torch.Tensor,
    prompt_length: int,
    tokenizer: AutoTokenizer,
) -> list[float]:
    """Compute normalized relevancy scores for generated tokens.

    Compares:
    - P_with: probability with LoRA adapters ENABLED
    - P_without: probability with LoRA adapters DISABLED

    Returns mean(P_with / (P_with + P_without)) across all generated tokens.
    """
    batch_size = output_ids.shape[0]
    generated_tokens = output_ids[:, prompt_length:]
    gen_length = generated_tokens.shape[1]

    if gen_length == 0:
        return [0.5] * batch_size

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

    # Step 1: Compute P_with (adapters ENABLED)
    with torch.no_grad():
        outputs_with = model(output_ids, return_dict=True)
        logits_with = outputs_with.logits

    # Step 2: Extract probabilities immediately and free memory
    p_with_batch = _extract_probs_for_tokens(logits_with, generated_tokens, prompt_length, pad_token_id)
    del logits_with, outputs_with
    torch.cuda.empty_cache()

    # Step 3: Compute P_without (adapters DISABLED)
    has_adapters = hasattr(model, 'active_adapters') and callable(getattr(model, 'active_adapters', None))

    if not has_adapters:
        warnings.warn(
            "No adapters found, using same model for relevancy score baseline"
        )

    if has_adapters:
        active_adapters = model.active_adapters()
        model.disable_adapters()

    with torch.no_grad():
        outputs_without = model(output_ids, return_dict=True)
        logits_without = outputs_without.logits

    # Re-enable adapters
    if has_adapters:
        model.enable_adapters()
        if active_adapters:
            model.set_adapter(active_adapters[0])

    # Step 4: Extract probabilities and free memory
    p_without_batch = _extract_probs_for_tokens(logits_without, generated_tokens, prompt_length, pad_token_id)
    del logits_without, outputs_without

    # Step 5: Compute normalized relevancy using shared utility
    return compute_normalized_relevancy(p_with_batch, p_without_batch)


def run_evaluation(
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    lora_path: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
    verbose: bool = False,
    compute_relevancy: bool = False,
) -> list[FeatureResult]:
    """Run evaluation and return results."""
    if lora_path is not None:
        adapter_name = lora_path
        if adapter_name not in model.peft_config:
            model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)
    with torch.no_grad():
        all_feature_results: list[FeatureResult] = []
        for i in tqdm(
            range(0, len(eval_data), eval_batch_size),
            desc="Evaluating model",
        ):
            e_batch = eval_data[i : i + eval_batch_size]

            for j in range(len(e_batch)):
                e_batch[j] = get_prompt_tokens_only(e_batch[j])

            e_batch = materialize_missing_steering_vectors(e_batch, tokenizer, model)
            e_batch = construct_batch(e_batch, tokenizer, device)

            feature_results = eval_features_batch(
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
                compute_relevancy=compute_relevancy,
            )
            if verbose:
                for feature_result in feature_results:
                    print(f"\n=== Feature {feature_result.feature_idx} : {feature_result.api_response} ===\n")
            all_feature_results.extend(feature_results)

    assert len(all_feature_results) == len(eval_data), "Number of feature results and evaluation data points must match"
    for feature_result, eval_data_point in zip(all_feature_results, eval_data, strict=True):
        feature_result.meta_info = eval_data_point.meta_info
    return all_feature_results


# ============================================================================
# Base experiment utilities (from nl_probes/base_experiment.py)
# ============================================================================

@dataclass
class VerbalizerEvalConfig:
    """Configuration for verbalizer evaluation."""
    model_name: str
    act_layers: list[int] = field(default_factory=list)
    active_layer: int = -1
    injection_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])
    selected_layer_percent: int = 50
    activation_input_types: list[str] = field(default_factory=lambda: ["orig", "lora", "diff"])
    add_generation_prompt: bool = True
    enable_thinking: bool = False
    verbalizer_generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 40,
            "top_p": 0.9,
        }
    )
    target_response_generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"do_sample": True, "temperature": 1.0, "max_new_tokens": 100}
    )
    steering_coefficient: float = 1.0
    eval_batch_size: int = 256
    add_response_to_context_prompt: bool = False
    verbalizer_input_types: list[str] = field(default_factory=lambda: ["tokens", "segment", "full_seq"])
    token_start_idx: int = -10
    token_end_idx: int = 0
    segment_start_idx: int = -10
    segment_end_idx: int = 0
    segment_repeats: int = 10
    full_seq_repeats: int = 10

    def __post_init__(self):
        assert self.segment_start_idx < self.segment_end_idx, "segment_start_idx must be less than segment_end_idx"
        assert self.token_start_idx < self.token_end_idx, "token_start_idx must be less than token_end_idx"

        act_layers = [layer_percent_to_layer(self.model_name, lp) for lp in self.layer_percents]
        active_layer_idx = self.layer_percents.index(self.selected_layer_percent)
        active_layer = act_layers[active_layer_idx]

        self.act_layers = act_layers
        self.active_layer = active_layer

        if self.active_layer not in self.act_layers:
            raise ValueError(f"active_layer ({self.active_layer}) must be in act_layers ({self.act_layers})")

        valid_act_types = {"orig", "lora", "diff"}
        invalid = set(self.activation_input_types) - valid_act_types
        if invalid:
            raise ValueError(f"Invalid activation_input_types: {invalid}. Must be in {valid_act_types}")

        valid_probe_types = {"tokens", "segment", "full_seq"}
        invalid = set(self.verbalizer_input_types) - valid_probe_types
        if invalid:
            raise ValueError(f"Invalid verbalizer_input_types: {invalid}. Must be in {valid_probe_types}")

        if "diff" in self.activation_input_types:
            if "lora" not in self.activation_input_types or "orig" not in self.activation_input_types:
                raise ValueError("Both 'lora' and 'orig' must be in activation_input_types when using 'diff'")


@dataclass
class VerbalizerInputInfo:
    context_prompt: list[dict[str, str]]
    verbalizer_prompt: str
    ground_truth: str
    masked_token_count: int = 0  # Number of tokens to skip from start (e.g., system prompt)


@dataclass
class VerbalizerResults:
    verbalizer_lora_path: str | None
    target_lora_path: str | None
    context_prompt: list[dict[str, str]]
    act_key: str
    verbalizer_prompt: str
    ground_truth: str
    num_tokens: int
    token_responses: list[Optional[str]]
    full_sequence_responses: list[str]
    segment_responses: list[str]
    context_input_ids: list[int]
    relevancy_scores: list[float] = field(default_factory=list)  # Relevancy scores for each response


def encode_messages(
    tokenizer: AutoTokenizer,
    message_dicts: list[list[dict[str, str]]],
    add_generation_prompt: bool,
    enable_thinking: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = []
    for source in message_dicts:
        rendered = tokenizer.apply_chat_template(
            source,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        messages.append(rendered)
    inputs_BL = tokenizer(messages, return_tensors="pt", add_special_tokens=False, padding=True).to(device)
    return inputs_BL


def sanitize_lora_name(lora_path: str) -> str:
    return lora_path.replace(".", "_")


def collect_target_activations(
    model: AutoModelForCausalLM,
    inputs_BL: dict[str, torch.Tensor],
    config: VerbalizerEvalConfig,
    target_lora_path: str | None,
) -> dict[str, dict[int, torch.Tensor]]:
    act_types = {}

    assert "lora" in config.activation_input_types, "lora must be in activation_input_types"
    if "lora" in config.activation_input_types:
        model.enable_adapters()
        if target_lora_path is not None:
            model.set_adapter(target_lora_path)
        else:
            print("Warning: target_lora_path is None, collecting lora activations from base model")
        submodules = {layer: get_hf_submodule(model, layer) for layer in config.act_layers}
        lora_acts = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
        act_types["lora"] = lora_acts

    if "orig" in config.activation_input_types:
        model.disable_adapters()
        submodules = {layer: get_hf_submodule(model, layer) for layer in config.act_layers}
        orig_acts = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
        act_types["orig"] = orig_acts
        model.enable_adapters()

    if "diff" in config.activation_input_types:
        assert "lora" in act_types and "orig" in act_types
        diff_acts = {}
        for layer in config.act_layers:
            diff_acts[layer] = act_types["lora"][layer] - act_types["orig"][layer]
        act_types["diff"] = diff_acts

    return act_types


def create_verbalizer_inputs(
    acts_BLD_by_layer_dict: dict[int, torch.Tensor],
    context_input_ids: list[int],
    verbalizer_prompt: str,
    act_layer: int,
    prompt_layer: int,
    tokenizer: AutoTokenizer,
    config: VerbalizerEvalConfig,
    batch_idx: int = 0,
    left_pad: int = 0,
    base_meta: dict[str, Any] | None = None,
    masked_token_count: int = 0,
) -> list[TrainingDataPoint]:
    training_data: list[TrainingDataPoint] = []

    if "tokens" in config.verbalizer_input_types:
        if config.token_start_idx < 0:
            token_start = len(context_input_ids) + config.token_start_idx
            token_end = len(context_input_ids) + config.token_end_idx
        else:
            token_start = config.token_start_idx
            token_end = config.token_end_idx
        for i in range(token_start, token_end):
            context_positions_rel = [i]
            context_positions_abs = [left_pad + i]
            acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]
            acts_BD = acts_BLD[context_positions_abs]
            meta = {"dp_kind": "tokens", "token_index": i}
            if base_meta is not None:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A",
                prompt=verbalizer_prompt,
                target_response="N/A",
                layer=prompt_layer,
                num_positions=len(context_positions_rel),
                tokenizer=tokenizer,
                acts_BD=acts_BD,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=context_positions_rel,
                ds_label="N/A",
                meta_info=meta,
            )
            training_data.append(dp)

    if "segment" in config.verbalizer_input_types:
        if config.segment_start_idx < 0:
            segment_start = len(context_input_ids) + config.segment_start_idx
            segment_end = len(context_input_ids) + config.segment_end_idx
        else:
            segment_start = config.segment_start_idx
            segment_end = config.segment_end_idx
        for _ in range(config.segment_repeats):
            context_positions_rel = list(range(segment_start, segment_end))
            context_positions_abs = [left_pad + p for p in context_positions_rel]
            acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]
            acts_BD = acts_BLD[context_positions_abs]
            meta = {"dp_kind": "segment"}
            if base_meta is not None:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A",
                prompt=verbalizer_prompt,
                target_response="N/A",
                layer=prompt_layer,
                num_positions=len(context_positions_rel),
                tokenizer=tokenizer,
                acts_BD=acts_BD,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=context_positions_rel,
                ds_label="N/A",
                meta_info=meta,
            )
            training_data.append(dp)

    if "full_seq" in config.verbalizer_input_types:
        for _ in range(config.full_seq_repeats):
            # Apply masking: skip the first masked_token_count tokens
            context_positions_rel = list(range(masked_token_count, len(context_input_ids)))
            context_positions_abs = [left_pad + p for p in context_positions_rel]
            acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]
            acts_BD = acts_BLD[context_positions_abs]
            meta = {"dp_kind": "full_seq"}
            if base_meta is not None:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A",
                prompt=verbalizer_prompt,
                target_response="N/A",
                layer=prompt_layer,
                num_positions=len(context_positions_rel),
                tokenizer=tokenizer,
                acts_BD=acts_BD,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=context_positions_rel,
                ds_label="N/A",
                meta_info=meta,
            )
            training_data.append(dp)

    return training_data


def load_lora_adapter(model: AutoModelForCausalLM, lora_path: str) -> str:
    sanitized_lora_name = sanitize_lora_name(lora_path)

    if sanitized_lora_name not in model.peft_config:
        print(f"Loading LoRA: {lora_path}")
        model.load_adapter(
            lora_path,
            adapter_name=sanitized_lora_name,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )

    return sanitized_lora_name


def run_verbalizer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbalizer_prompt_infos: list[VerbalizerInputInfo],
    verbalizer_lora_path: str | None,
    target_lora_path: str | None,
    config: VerbalizerEvalConfig,
    device: torch.device,
    compute_relevancy: bool = False,
) -> list[VerbalizerResults]:
    """Run verbalizer evaluation."""
    dtype = torch.bfloat16
    injection_submodule = get_hf_submodule(model, config.injection_layer)

    pbar = tqdm(total=len(verbalizer_prompt_infos), desc="Verbalizer Eval Progress", position=1)
    results: list[VerbalizerResults] = []

    for start in range(0, len(verbalizer_prompt_infos), config.eval_batch_size):
        batch = verbalizer_prompt_infos[start : start + config.eval_batch_size]

        message_dicts: list[list[dict[str, str]]] = []
        combo_bases: list[dict[str, Any]] = []

        for verbalizer_prompt_info in batch:
            correct_answer = verbalizer_prompt_info.ground_truth
            message_dicts.append(verbalizer_prompt_info.context_prompt)

            combo_bases.append(
                {
                    "target_lora_path": target_lora_path,
                    "context_prompt": verbalizer_prompt_info.context_prompt,
                    "verbalizer_prompt": verbalizer_prompt_info.verbalizer_prompt,
                    "ground_truth": correct_answer,
                    "combo_index": start + len(combo_bases),
                }
            )

        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=message_dicts,
            add_generation_prompt=config.add_generation_prompt,
            enable_thinking=config.enable_thinking,
            device=device,
        )

        target_activations = collect_target_activations(
            model=model,
            inputs_BL=inputs_BL,
            config=config,
            target_lora_path=target_lora_path,
        )

        seq_len = int(inputs_BL["input_ids"].shape[1])
        context_input_ids_list: list[list[int]] = []
        verbalizer_inputs: list[TrainingDataPoint] = []

        for b_idx in range(len(message_dicts)):
            base = combo_bases[b_idx]
            attn = inputs_BL["attention_mask"][b_idx]
            real_len = int(attn.sum().item())
            left_pad = seq_len - real_len
            context_input_ids = inputs_BL["input_ids"][b_idx, left_pad:].tolist()
            context_input_ids_list.append(context_input_ids)

            # Extract masked_token_count from the batch item
            masked_token_count = batch[b_idx].masked_token_count

            for act_key, acts_dict in target_activations.items():
                base_meta = {
                    "target_lora_path": base["target_lora_path"],
                    "context_prompt": base["context_prompt"],
                    "verbalizer_prompt": base["verbalizer_prompt"],
                    "ground_truth": base["ground_truth"],
                    "combo_index": base["combo_index"],
                    "act_key": act_key,
                    "num_tokens": len(context_input_ids),
                    "context_index_within_batch": b_idx,
                    "masked_token_count": masked_token_count,
                }
                verbalizer_inputs.extend(
                    create_verbalizer_inputs(
                        acts_BLD_by_layer_dict=acts_dict,
                        context_input_ids=context_input_ids,
                        verbalizer_prompt=base["verbalizer_prompt"],
                        act_layer=config.active_layer,
                        prompt_layer=config.active_layer,
                        tokenizer=tokenizer,
                        config=config,
                        batch_idx=b_idx,
                        left_pad=left_pad,
                        base_meta=base_meta,
                        masked_token_count=masked_token_count,
                    )
                )

        if verbalizer_lora_path is not None:
            model.set_adapter(verbalizer_lora_path)

        responses = run_evaluation(
            eval_data=verbalizer_inputs,
            model=model,
            tokenizer=tokenizer,
            submodule=injection_submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=verbalizer_lora_path,
            eval_batch_size=config.eval_batch_size,
            steering_coefficient=config.steering_coefficient,
            generation_kwargs=config.verbalizer_generation_kwargs,
            compute_relevancy=compute_relevancy,
        )

        agg: dict[tuple[str, int], dict[str, Any]] = {}
        for r in responses:
            meta = r.meta_info
            key = (meta["act_key"], int(meta["combo_index"]))
            if key not in agg:
                agg[key] = {
                    "target_lora_path": target_lora_path,
                    "context_prompt": meta["context_prompt"],
                    "verbalizer_prompt": meta["verbalizer_prompt"],
                    "ground_truth": meta["ground_truth"],
                    "num_tokens": int(meta["num_tokens"]),
                    "context_index_within_batch": int(meta["context_index_within_batch"]),
                    "token_responses": [None] * int(meta["num_tokens"]),
                    "segment_responses": [],
                    "full_seq_responses": [],
                    "relevancy_scores": [],
                }
            bucket = agg[key]
            dp_kind = meta["dp_kind"]
            if dp_kind == "tokens":
                idx = int(meta["token_index"])
                bucket["token_responses"][idx] = r.api_response
            elif dp_kind == "segment":
                bucket["segment_responses"].append(r.api_response)
                bucket["relevancy_scores"].append(r.relevancy)
            elif dp_kind == "full_seq":
                bucket["full_seq_responses"].append(r.api_response)
                bucket["relevancy_scores"].append(r.relevancy)
            else:
                raise ValueError(f"Unknown dp_kind: {dp_kind}")

        for (act_key, combo_idx), bucket in agg.items():
            record = VerbalizerResults(
                verbalizer_lora_path=verbalizer_lora_path,
                target_lora_path=target_lora_path,
                context_prompt=bucket["context_prompt"],
                act_key=act_key,
                verbalizer_prompt=bucket["verbalizer_prompt"],
                ground_truth=bucket["ground_truth"],
                num_tokens=bucket["num_tokens"],
                token_responses=bucket["token_responses"],
                full_sequence_responses=bucket["full_seq_responses"],
                segment_responses=bucket["segment_responses"],
                context_input_ids=context_input_ids_list[bucket["context_index_within_batch"]],
                relevancy_scores=bucket["relevancy_scores"],
            )
            results.append(record)

        pbar.update(len(batch))
    pbar.close()

    return results
