import copy
import json
import os
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import nnsight
import requests
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sae_lens import SAE, HookedSAETransformer


class SparseAutoEncoder(torch.nn.Module):
    """Sparse AutoEncoder implementation."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f


class ObservableLanguageModelGemma2:
    """Wrapper for language model with observation capabilities."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype
        self.device = device
        self._original_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        self._original_tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-9b-it",
            trust_remote_code=True,
        )

        self._model = nnsight.LanguageModel(
            self._original_model,
            tokenizer=self._original_tokenizer,
            device=device,
            torch_dtype=dtype,
        )

        # Force model download
        input_tokens = self._original_tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}]
        )
        with self._model.trace(input_tokens):
            pass

        self.tokenizer = self._original_tokenizer
        self.d_model = self._attempt_to_infer_hidden_layer_dimensions()
        self.safe_mode = False

    def _attempt_to_infer_hidden_layer_dimensions(self):
        """Infer hidden layer dimensions from model config."""
        config = self._original_model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        raise Exception("Could not infer hidden layer dimensions from model config")

    def _find_module(self, hook_point: str):
        """Find module by hook point string."""
        submodules = hook_point.split(".")
        module = self._model
        while submodules:
            module = getattr(module, submodules.pop(0))
        return module

    def forward(
        self,
        inputs: torch.Tensor,
        cache_activations_at: Optional[List[str]] = None,
        interventions=None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass with optional activation caching and interventions."""
        cache = {}

        # Build kwargs for trace - include attention_mask if provided
        trace_kwargs = {"scan": self.safe_mode, "validate": self.safe_mode}
        if attention_mask is not None:
            trace_kwargs["attention_mask"] = attention_mask

        with self._model.trace(inputs, **trace_kwargs):
            # Apply interventions
            if interventions:
                for hook_site, intervention in interventions.items():
                    if intervention is None:
                        continue
                    module = self._find_module(hook_site)
                    intervened_acts = intervention(module.output[0])
                    module.output = (intervened_acts,)

            # Cache activations
            if cache_activations_at is not None:
                for hook_point in cache_activations_at:
                    module = self._find_module(hook_point)
                    cache[hook_point] = module.output.save()

            logits = self._model.output[0].squeeze(1).save()
            kv_cache = self._model.output.past_key_values.save()

        return (
            logits.detach(),
            kv_cache,
            {k: v[0].detach() for k, v in cache.items()},
        )

class ObservableLanguageModel:
    """Wrapper for language model with observation capabilities."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype
        self.device = device
        self._original_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self._original_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct",
            trust_remote_code=True,
        )

        self._model = nnsight.LanguageModel(
            self._original_model,
            tokenizer=self._original_tokenizer,
            device=device,
            torch_dtype=dtype,
        )

        # Force model download
        input_tokens = self._original_tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}]
        )
        with self._model.trace(input_tokens):
            pass

        self.tokenizer = self._original_tokenizer
        self.d_model = self._attempt_to_infer_hidden_layer_dimensions()
        self.safe_mode = False

    def _attempt_to_infer_hidden_layer_dimensions(self):
        """Infer hidden layer dimensions from model config."""
        config = self._original_model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        raise Exception("Could not infer hidden layer dimensions from model config")

    def _find_module(self, hook_point: str):
        """Find module by hook point string."""
        submodules = hook_point.split(".")
        module = self._model
        while submodules:
            module = getattr(module, submodules.pop(0))
        return module

    def forward(
        self,
        inputs: torch.Tensor,
        cache_activations_at: Optional[List[str]] = None,
        interventions=None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass with optional activation caching and interventions."""
        cache = {}

        # Build kwargs for trace - include attention_mask if provided
        trace_kwargs = {"scan": self.safe_mode, "validate": self.safe_mode}
        if attention_mask is not None:
            trace_kwargs["attention_mask"] = attention_mask

        with self._model.trace(inputs, **trace_kwargs):
            # Apply interventions
            if interventions:
                for hook_site, intervention in interventions.items():
                    if intervention is None:
                        continue
                    module = self._find_module(hook_site)
                    intervened_acts = intervention(module.output[0])
                    module.output = (intervened_acts,)

            # Cache activations
            if cache_activations_at is not None:
                for hook_point in cache_activations_at:
                    module = self._find_module(hook_point)
                    cache[hook_point] = module.output.save()

            logits = self._model.output[0].squeeze(1).save()
            kv_cache = self._model.output.past_key_values.save()

        return (
            logits.detach(),
            kv_cache,
            {k: v[0].detach() for k, v in cache.items()},
        )


def load_sae(
    sae_name: str,
    d_model: int,
    expansion_factor: int,
    device: str = "cuda",
) -> SparseAutoEncoder:
    """Load SAE from HuggingFace Hub."""
    device_obj = torch.device(device)

    # Download SAE weights
    file_path = hf_hub_download(
        repo_id=f"Goodfire/{sae_name}", filename=f"{sae_name}.pt", repo_type="model"
    )

    # Initialize and load SAE
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device_obj,
    )
    sae_dict = torch.load(file_path, weights_only=True, map_location=device_obj)
    sae.load_state_dict(sae_dict)

    return sae


def get_top_features(features: torch.Tensor, top_k: int = 10) -> List[Tuple[int, float]]:
    """Get top-k activating features with their values."""
    top_values, top_indices = torch.topk(features, top_k)
    return [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)]


def string_to_base64(text: str) -> str:
    """Convert a string to base64 encoding."""
    import base64

    # Encode string to bytes, then to base64
    encoded_bytes = base64.b64encode(text.encode("utf-8"))
    # Convert bytes back to string
    return encoded_bytes.decode("utf-8")


def base64_to_string(encoded_text: str) -> str:
    """Convert base64 encoded string back to original string."""
    import base64

    # Decode base64 to bytes, then to string
    decoded_bytes = base64.b64decode(encoded_text.encode("utf-8"))
    return decoded_bytes.decode("utf-8")

def get_cache_key(feature_index: int, layer: int = 32, width_k: int = 131) -> str:
    """Generate a cache key for a feature."""
    return f"{layer}-{width_k}k-{feature_index}"


def fetch_sae_feature_description(
    feature_index: int, layer: int = 32, width_k: int = 131,
    max_retries: int = 3, initial_backoff: float = 2.0, timeout: float = 30.0
) -> str:
    """Fetch SAE feature description from Neuronpedia API with caching and retry logic.

    Args:
        feature_index: The SAE feature index
        layer: The layer number (default: 32)
        width_k: The width in thousands (default: 131)
        max_retries: Maximum number of retry attempts (default: 3)
        initial_backoff: Initial backoff time in seconds, doubles each retry (default: 2.0)
        timeout: Request timeout in seconds (default: 30.0)

    Returns:
        Feature description string, or "No description available" if API call fails
    """
    cache = load_sae_feature_cache()
    cache_key = get_cache_key(feature_index, layer, width_k)

    # Check if description is cached
    if cache_key in cache["features"] and "description" in cache["features"][cache_key]:
        return cache["features"][cache_key]["description"]

    url = f"https://www.neuronpedia.org/api/feature/gemma-2-9b/{layer}-gemmascope-res-{width_k}k/{feature_index}"
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            data = response.json()

            # Try to extract description from the response
            description = "No description available"
            explanations = data.get("explanations", [])
            if explanations and len(explanations) > 0:
                # Get the first explanation's description
                first_explanation = explanations[0]
                if "description" in first_explanation:
                    description = first_explanation["description"].strip()

            # Fallback: check if there's a direct description field
            if description == "No description available" and "description" in data:
                description = data["description"].strip()

            # Cache the result and immediately save to disk
            with _sae_cache_lock:
                if cache_key not in cache["features"]:
                    cache["features"][cache_key] = {}
                cache["features"][cache_key]["description"] = description
                cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()

            # Immediately save cache to disk
            save_sae_feature_cache()

            return description

        except requests.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                backoff = initial_backoff * (2 ** attempt)
                time.sleep(backoff)
            continue
        except (KeyError, json.JSONDecodeError) as e:
            # Don't retry on parse errors - the response format is wrong
            print(f"Warning: Unexpected response format for feature {feature_index}: {e}")
            with _sae_cache_lock:
                if cache_key not in cache["features"]:
                    cache["features"][cache_key] = {}
                cache["features"][cache_key]["description"] = "No description available"
                cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()
            save_sae_feature_cache()
            return "No description available"

    # All retries exhausted
    print(f"Warning: Failed to fetch description for feature {feature_index} after {max_retries} attempts: {last_exception}")
    with _sae_cache_lock:
        if cache_key not in cache["features"]:
            cache["features"][cache_key] = {}
        cache["features"][cache_key]["description"] = "No description available"
        cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()
    save_sae_feature_cache()
    return "No description available"


def prefetch_sae_feature_descriptions(
    feature_indices: List[int],
    layer: int = 32,
    width_k: int = 131,
    max_workers: int = 200
) -> Dict[int, str]:
    """Prefetch multiple feature descriptions in parallel.

    Args:
        feature_indices: List of feature indices to fetch
        layer: The layer number (default: 32)
        width_k: The width in thousands (default: 131)
        max_workers: Maximum number of parallel workers (default: 20)

    Returns:
        Dict mapping feature_index -> description
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = load_sae_feature_cache()
    descriptions = {}
    uncached = []

    # Check cache first
    for idx in feature_indices:
        cache_key = get_cache_key(idx, layer, width_k)
        if cache_key in cache["features"] and "description" in cache["features"][cache_key]:
            descriptions[idx] = cache["features"][cache_key]["description"]
        else:
            uncached.append(idx)

    if not uncached:
        return descriptions

    print(f"Fetching {len(uncached)} uncached feature descriptions in parallel...")

    def fetch_one(idx):
        return idx, fetch_sae_feature_description(idx, layer, width_k)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, idx): idx for idx in uncached}
        for future in as_completed(futures):
            idx, desc = future.result()
            descriptions[idx] = desc

    return descriptions


_sae_feature_cache = None
_cache_file_path = ".sae/sae_feature_cache.json"
_sae_cache_lock = threading.Lock()


def load_sae_feature_cache():
    """Load the SAE feature cache from disk."""
    global _sae_feature_cache
    if _sae_feature_cache is not None:
        return _sae_feature_cache

    try:
        if os.path.exists(_cache_file_path):
            with open(_cache_file_path, "r", encoding="utf-8") as f:
                _sae_feature_cache = json.load(f)
            print(
                f"Loaded SAE feature cache with {len(_sae_feature_cache.get('features', {}))} cached features"
            )
        else:
            _sae_feature_cache = {
                "cache_version": "1.0",
                "created": datetime.now().isoformat(),
                "features": {},
            }
            print("Created new SAE feature cache")
    except Exception as e:
        print(f"Warning: Failed to load SAE feature cache: {e}")
        _sae_feature_cache = {
            "cache_version": "1.0",
            "created": datetime.now().isoformat(),
            "features": {},
        }

    return _sae_feature_cache


def save_sae_feature_cache():
    """Save the SAE feature cache to disk."""
    global _sae_feature_cache
    if _sae_feature_cache is None:
        return

    try:
        with _sae_cache_lock:
            # Make a deep copy to avoid "dictionary changed size during iteration"
            # when multiple threads are modifying the cache
            cache_copy = copy.deepcopy(_sae_feature_cache)
        cache_copy["last_updated"] = datetime.now().isoformat()
        with open(_cache_file_path, "w", encoding="utf-8") as f:
            json.dump(cache_copy, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save SAE feature cache: {e}")


def load_densities_from_file(
    features_file: str, layer: int = 32, width_k: int = 131
) -> Dict[int, float]:
    """Load all feature densities from the complete SAE features JSONL file."""
    print(f"Loading feature densities from {features_file}...")

    densities = {}
    total_features = 131072

    try:
        with open(features_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    feature_idx = int(data["index"])
                    frac_nonzero = data["frac_nonzero"]

                    # Convert to percentage as expected by the rest of the code
                    density_percentage = frac_nonzero * 100
                    densities[feature_idx] = density_percentage

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Warning: Invalid data on line {line_num}: {e}")
                    continue

        print(f"Loaded {len(densities)} feature densities from file")

        if len(densities) < total_features:
            print(
                f"Warning: Only {len(densities)}/{total_features} feature densities found in file"
            )
        else:
            print(f"Successfully loaded all {len(densities)} feature densities!")

        return densities

    except FileNotFoundError:
        print(f"Error: Features file not found: {features_file}")
        return {}
    except Exception as e:
        print(f"Error loading features file: {e}")
        return {}


def load_density_tensor(
    features_file: str, layer: int = 32, width_k: int = 131, device: str = "cpu"
) -> torch.Tensor:
    """Load all feature densities from file into a tensor for efficient access."""
    total_features = 131072

    print("Loading densities from complete features file...")
    densities_dict = load_densities_from_file(features_file, layer, width_k)

    if not densities_dict:
        print("Error: Failed to load any feature densities from file")
        return torch.tensor([], device=device), []

    # Convert to tensors
    densities = []
    available_features = []

    for feature_idx in range(total_features):
        if feature_idx in densities_dict:
            density_percentage = densities_dict[feature_idx]
            densities.append(density_percentage / 100.0)  # Convert to fraction
            available_features.append(feature_idx)

    print(f"Loaded {len(available_features)} feature densities into tensor from file")

    if len(available_features) < total_features:
        print(
            f"Warning: Only {len(available_features)}/{total_features} feature densities available"
        )

    density_tensor = torch.tensor(densities, dtype=torch.float32, device=device)
    return density_tensor, available_features



# ============================================================================
# SSC-specific utilities
# ============================================================================

# SSC tag constants
SSC_START_TAG = "<secret_side_constraint>"
SSC_END_TAG = "</secret_side_constraint>"

# Goodfire API cache for SSC
_goodfire_client = None
_goodfire_desc_cache = None
_goodfire_cache_file_path = ".sae/goodfire_feature_cache.json"
_GOODFIRE_FEATURE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _get_goodfire_client():
    """Get or create Goodfire API client."""
    global _goodfire_client
    if _goodfire_client is not None:
        return _goodfire_client
    try:
        import goodfire
        _goodfire_client = goodfire.Client(os.getenv("GOODFIRE_API_KEY"))
    except Exception as e:
        print(f"Warning: Could not initialize Goodfire client: {e}")
        _goodfire_client = None
    return _goodfire_client


def _load_goodfire_desc_cache() -> dict:
    """Load Goodfire feature description cache from disk."""
    global _goodfire_desc_cache
    if _goodfire_desc_cache is not None:
        return _goodfire_desc_cache
    try:
        if os.path.exists(_goodfire_cache_file_path):
            with open(_goodfire_cache_file_path, "r", encoding="utf-8") as f:
                _goodfire_desc_cache = json.load(f)
            if not isinstance(_goodfire_desc_cache, dict) or "features" not in _goodfire_desc_cache:
                _goodfire_desc_cache = {"features": {}, "created": datetime.now().isoformat()}
        else:
            _goodfire_desc_cache = {"features": {}, "created": datetime.now().isoformat()}
    except Exception:
        _goodfire_desc_cache = {"features": {}, "created": datetime.now().isoformat()}
    return _goodfire_desc_cache


def _save_goodfire_desc_cache() -> None:
    """Save Goodfire feature description cache to disk."""
    global _goodfire_desc_cache
    if _goodfire_desc_cache is None:
        return
    try:
        os.makedirs(os.path.dirname(_goodfire_cache_file_path), exist_ok=True)
        _goodfire_desc_cache["last_updated"] = datetime.now().isoformat()
        with open(_goodfire_cache_file_path, "w", encoding="utf-8") as f:
            json.dump(_goodfire_desc_cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save Goodfire cache: {e}")


def fetch_sae_feature_description_goodfire(feature_index: int) -> str:
    """Fetch SAE feature description from Goodfire API with caching.

    Args:
        feature_index: The SAE feature index

    Returns:
        Feature description (label) string, or "No description available" if API call fails
    """
    cache = _load_goodfire_desc_cache()
    key = str(int(feature_index))

    # Check cache
    if key in cache.get("features", {}):
        cached = cache["features"][key]
        if isinstance(cached, dict) and "description" in cached:
            return cached["description"]
        if isinstance(cached, str):
            return cached

    client = _get_goodfire_client()
    description = "No description available"

    try:
        if client is not None:
            info = client.features.lookup(
                model=_GOODFIRE_FEATURE_MODEL, indices=[int(feature_index)]
            )
            # info expected to be a mapping index->object with .label
            obj = info.get(int(feature_index)) if hasattr(info, "get") else None
            if obj is None and isinstance(info, list) and len(info) > 0:
                obj = info[0]
            if obj is not None:
                # Support either attribute or key access
                if hasattr(obj, "label"):
                    description = getattr(obj, "label")
                elif isinstance(obj, dict) and "label" in obj:
                    description = obj["label"]
                if description is None or description == "":
                    description = "No description available"
    except Exception as e:
        print(f"Warning: Error fetching Goodfire description for feature {feature_index}: {e}")
        description = "No description available"

    # Update cache
    cache.setdefault("features", {})[key] = {
        "description": description,
        "last_fetched": datetime.now().isoformat(),
    }
    _save_goodfire_desc_cache()
    return description


def load_feature_densities_from_json(densities_file: str) -> torch.Tensor:
    """Load feature densities from JSON file (SSC format) and return as PyTorch tensor.

    The SSC format is a JSON object where keys are feature indices (as strings)
    and values contain "density_percent" field.

    Args:
        densities_file: Path to JSON file containing feature densities

    Returns:
        PyTorch tensor with density values (fractions, not percentages)
        for each feature index. Missing features will have density 0.0.
    """
    try:
        with open(densities_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded feature densities from {densities_file}")
        print(f"Found densities for {len(data)} features")

        # Find the maximum feature index to determine tensor size
        max_feature_idx = max(int(k) for k in data.keys())

        # Create tensor with zeros for all possible feature indices
        densities = torch.zeros(max_feature_idx + 1, dtype=torch.float32)

        # Fill in the density values (convert percentages to fractions)
        for feature_idx_str, feature_data in data.items():
            feature_idx = int(feature_idx_str)
            density_percent = feature_data.get("density_percent", 0.0)
            densities[feature_idx] = density_percent / 100.0

        print(f"Created density tensor with shape {densities.shape}")
        return densities

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading densities file {densities_file}: {e}")
        return torch.tensor([])


def extract_ssc_content(prompt_text: str) -> Optional[str]:
    """Extract content between <secret_side_constraint> tags.

    Args:
        prompt_text: Text containing SSC tags

    Returns:
        Content between SSC tags (base64 encoded constraint), or None if not found
    """
    if not isinstance(prompt_text, str):
        return None
    start_idx = prompt_text.find(SSC_START_TAG)
    end_idx = prompt_text.find(SSC_END_TAG)
    if start_idx == -1 or end_idx == -1:
        return None
    start_idx += len(SSC_START_TAG)
    return prompt_text[start_idx:end_idx]


def find_ssc_token_positions(tokens: torch.Tensor, tokenizer, prompt_text: str) -> List[int]:
    """Find token positions corresponding to the SSC content span.

    Args:
        tokens: Token tensor sequence
        tokenizer: Tokenizer instance
        prompt_text: Original prompt text with SSC tags

    Returns:
        List of token positions within the SSC span
    """
    ssc_content = extract_ssc_content(prompt_text)
    if not ssc_content:
        return []

    # Tokenize the SSC content
    ssc_token_ids = tokenizer.encode(ssc_content, add_special_tokens=False)
    full_ids = tokens.tolist()

    # Find the '>' token that starts the SSC content
    start_index = -1
    for i, token_id in enumerate(full_ids):
        tok_str = tokenizer.decode([int(token_id)])
        trimmed = tok_str.lstrip()
        if trimmed == ">":
            start_index = i
            break
        if trimmed.startswith(">") and len(trimmed) > 1:
            start_index = i
            break

    if start_index == -1:
        return []

    # Determine SSC positions
    start_tok_str = tokenizer.decode([int(full_ids[start_index])])
    base_pos = start_index + 1 if start_tok_str.strip() == ">" else start_index

    max_len = min(len(ssc_token_ids), max(0, len(tokens) - base_pos))
    if max_len <= 0:
        return []

    return list(range(base_pos, base_pos + max_len))


def tfidf_feature_score(activations: torch.Tensor, feature_density: torch.Tensor) -> torch.Tensor:
    """Calculate TF-IDF style scores for SAE features.

    Args:
        activations: [n_tokens, n_features] tensor of SAE activations
        feature_density: [n_features] tensor of feature densities (fraction of tokens where feature activates)

    Returns:
        TF-IDF scores for each feature
    """
    # Ensure both tensors are on the same device
    device = activations.device
    feature_density = feature_density.to(device)

    # Term frequency: average activation per feature
    tf = activations.mean(dim=0)

    # Inverse document frequency: penalize common features
    idf = torch.log(1 / (feature_density + 1e-8))

    return tf * idf


# ============================================================================
# Gemma model loading utilities
# ============================================================================

def load_model_and_sae(
    model_name: str,
    layer: int,
    device: str,
    base_model_name: str,
    sae_release: str,
    width_k: int,
):
    """Load the model and SAE for analysis."""
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if base_model_name != model_name and model_name is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("Loading PEFT adapter...")
        # Load the adapter
        peft_model = PeftModel.from_pretrained(base_model, model_name)
        merged_model = peft_model.merge_and_unload()

        print("Wrapping with HookedSAETransformer...")
        # Wrap model with HookedSAETransformer
        model = HookedSAETransformer.from_pretrained_no_processing(
            base_model_name,
            device=device,
            hf_model=merged_model,
            dtype=torch.bfloat16,
        )
    else:
        model = HookedSAETransformer.from_pretrained_no_processing(
            base_model_name,
            device=device,
            dtype=torch.bfloat16,
        )

    # Load SAE
    print(f"Loading SAE for layer {layer}")

    sae_id = f"layer_{layer}/width_{width_k}k/canonical"
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )

    return model, tokenizer, sae
